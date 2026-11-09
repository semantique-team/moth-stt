/**
 * Browser runner for the Moth smoke page (Task 3.5, docs/06 §2).
 *
 * Loads `release/` (config, model, vocab, mel filters), verifies every `sha256:` line, creates an
 * `onnxruntime-web` WASM session (SIMD, threads only when cross-origin isolated) and exposes
 * `transcribe(pcm)` / `benchmark(pcm, ...)` with per-stage timings and RTFx.
 */

import * as ort from "./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs";

import { beamSearch, greedy } from "./decode.js";
import { loadLM } from "./lm.js";
import { computeMel, N_MELS, N_FFT, SAMPLE_RATE } from "./mel.js";
import { createVad } from "./vad.js";

const FILTER_FLOATS = N_MELS * (N_FFT / 2 + 1);

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function fetchChecked(url, expected) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  if (expected) {
    const actual = await sha256Hex(buffer);
    const wanted = String(expected).replace(/^sha256:/, "");
    if (actual !== wanted) throw new Error(`${url}: sha256 mismatch (${actual} != ${wanted})`);
  }
  return buffer;
}

/** Decode an audio file to 16 kHz mono float PCM (the real mic path adds its own resampler). */
export async function decodeTo16k(url) {
  const bytes = await (await fetch(url)).arrayBuffer();
  // decodeAudioData resamples to the AudioBuffer's context rate: ask for 16 kHz directly.
  const context = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const buffer = await context.decodeAudioData(bytes);
  const mono = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let index = 0; index < mono.length; index += 1) {
      mono[index] += data[index] / buffer.numberOfChannels;
    }
  }
  return mono;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export async function createRunner({ baseUrl = "../release/", threads = null, segment = null } = {}) {
  const root = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;

  ort.env.wasm.wasmPaths = new URL(
    "./node_modules/onnxruntime-web/dist/",
    import.meta.url,
  ).href;
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads =
    threads ??
    (globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1);

  const config = JSON.parse(
    new TextDecoder().decode(await fetchChecked(`${root}config.json`)),
  );
  const files = config.files ?? {};
  const modelBuffer = await fetchChecked(`${root}${config.model.file}`, files[config.model.file]);
  const vocabBuffer = await fetchChecked(
    `${root}${config.tokenizer.vocab}`,
    files[config.tokenizer.vocab],
  );
  const filtersBuffer = await fetchChecked(
    `${root}${config.features.mel_filters}`,
    files[config.features.mel_filters],
  );
  let lm = null;
  if (config.decode?.lm) {
    const lmBuffer = await fetchChecked(`${root}${config.decode.lm}`, files[config.decode.lm]);
    lm = loadLM(lmBuffer);
  }
  let vad = null;
  if (config.vad?.file) {
    const vadBuffer = await fetchChecked(`${root}${config.vad.file}`, files[config.vad.file]);
    vad = await createVad(vadBuffer, { gate: config.vad.threshold });
  }

  const vocabJson = JSON.parse(new TextDecoder().decode(vocabBuffer));
  const maxId = Math.max(...Object.keys(vocabJson).map(Number));
  const pieces = new Array(maxId + 1).fill("");
  for (const [id, piece] of Object.entries(vocabJson)) pieces[Number(id)] = piece;

  const filters = new Float32Array(filtersBuffer);
  if (filters.length !== FILTER_FLOATS) {
    throw new Error(`mel_filters.bin: ${filters.length} floats, expected ${FILTER_FLOATS}`);
  }

  const loadStarted = performance.now();
  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const loadMs = performance.now() - loadStarted;

  async function transcribe(pcm, { segment: segmentOverride = null } = {}) {
    const t0 = performance.now();
    // No-speech guard (D25, E-011) and sentence segmentation from a **single** VAD pass: the
    // per-window probabilities answer both questions, and running the graph twice would double a
    // cost that is already ~80 ms on 18 s of audio.
    const probabilities = vad ? await vad.probabilities(pcm) : null;
    const guarded = probabilities ? !(vad.fraction(probabilities) > vad.gate) : false;
    const t1 = performance.now();
    if (guarded) {
      const audioSeconds = pcm.length / SAMPLE_RATE;
      const totalMs = t1 - t0;
      return {
        text: "",
        frames: 0,
        outFrames: 0,
        vocabSize: pieces.length,
        decoder: "vad-guard",
        guarded: true,
        audioSeconds,
        timings: {
          vad_ms: t1 - t0,
          feature_ms: 0,
          inference_ms: 0,
          decode_ms: 0,
          total_ms: totalMs,
        },
        rtfx: (audioSeconds * 1000) / totalMs,
      };
    }
    // Split a long recording at its pauses before decoding (see runs/audit-2026-11-09/): the model
    // is trained on sentence-sized clips and reads several sentences as one unpunctuated stream,
    // which costs 9.4 WER points on five concatenated sentences. Only split when the VAD finds two
    // or more runs — cutting a single run removes the leading silence the model expects (+1.6 WER).
    const decodeConfig = config.decode ?? {};
    const segmentEnabled = segmentOverride ?? segment ?? decodeConfig.segment ?? true;
    const wantsSegments = segmentEnabled && probabilities !== null;
    const bounds = wantsSegments ? vad.segmentsFrom(pcm.length, probabilities) : [];
    const spans = bounds.length >= 2 ? bounds : [{ start: 0, end: pcm.length }];

    let outFrames = 0;
    let vocabSize = pieces.length + 1;
    let featureMs = 0;
    let inferenceMs = 0;
    let decodeMs = 0;
    const texts = [];
    for (const span of spans) {
      const sliceStart = performance.now();
      const { data, frames } = computeMel(
        span.start === 0 && span.end === pcm.length ? pcm : pcm.subarray(span.start, span.end),
        filters,
      );
      const input = new ort.Tensor("float32", data, [1, N_MELS, frames]);
      featureMs += performance.now() - sliceStart;
      const inferStart = performance.now();
      const outputs = await session.run({ [config.model.input]: input });
      inferenceMs += performance.now() - inferStart;
      const output = outputs[config.model.output];
      const [, framesOut, size] = output.dims;
      outFrames += framesOut;
      vocabSize = size;
      const decodeStart = performance.now();
      const text =
        decodeConfig.default === "beam"
          ? beamSearch(output.data, framesOut, size, pieces, {
              beamWidth: decodeConfig.beam ?? 5,
              topK: decodeConfig.top_k ?? 8,
              lm,
              lmWeight: decodeConfig.lm_weight ?? 0,
              tokenBonus: decodeConfig.token_bonus ?? 0,
            })
          : greedy(output.data, framesOut, size, pieces);
      if (text) texts.push(text);
      decodeMs += performance.now() - decodeStart;
    }
    const text = texts.join(" ");
    const frames = Math.ceil(pcm.length / 160);
    const audioSeconds = pcm.length / SAMPLE_RATE;
    const totalMs = performance.now() - t0;
    return {
      text,
      frames,
      outFrames,
      vocabSize,
      decoder: decodeConfig.default ?? "greedy",
      audioSeconds,
      segments: spans.length,
      timings: {
        vad_ms: t1 - t0,
        feature_ms: featureMs,
        inference_ms: inferenceMs,
        decode_ms: decodeMs,
        total_ms: totalMs,
      },
      rtfx: (audioSeconds * 1000) / totalMs,
    };
  }

  /** 3 warm-ups then the runs of docs/05 §3, reporting medians and RTFx. */
  async function benchmark(pcm, { runs = 5, warmups = 2 } = {}) {
    for (let index = 0; index < warmups; index += 1) await transcribe(pcm);
    const totals = [];
    const vads = [];
    const features = [];
    const inferences = [];
    const decodes = [];
    for (let index = 0; index < runs; index += 1) {
      const result = await transcribe(pcm);
      totals.push(result.timings.total_ms);
      vads.push(result.timings.vad_ms ?? 0);
      features.push(result.timings.feature_ms);
      inferences.push(result.timings.inference_ms);
      decodes.push(result.timings.decode_ms);
    }
    const audioSeconds = pcm.length / SAMPLE_RATE;
    const totalMedian = median(totals);
    const inferenceMedian = median(inferences);
    return {
      runs,
      audio_s: audioSeconds,
      total_ms_median: totalMedian,
      total_ms_min: Math.min(...totals),
      vad_ms_median: median(vads),
      feature_ms_median: median(features),
      inference_ms_median: inferenceMedian,
      decode_ms_median: median(decodes),
      rtfx_median: (audioSeconds * 1000) / totalMedian,
      rtfx_inference: (audioSeconds * 1000) / inferenceMedian,
      num_threads: ort.env.wasm.numThreads,
      cross_origin_isolated: globalThis.crossOriginIsolated,
      simd: ort.env.wasm.simd,
    };
  }

  return {
    config,
    session,
    vad,
    pieces,
    filters,
    loadMs,
    numThreads: ort.env.wasm.numThreads,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    simd: ort.env.wasm.simd,
    transcribe,
    benchmark,
  };
}
