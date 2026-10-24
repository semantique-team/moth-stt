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
import { computeMel, isNoSpeech, N_MELS, N_FFT, SAMPLE_RATE } from "./mel.js";

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

export async function createRunner({ baseUrl = "../release/", threads = null } = {}) {
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

  async function transcribe(pcm) {
    const t0 = performance.now();
    const { data, frames } = computeMel(pcm, filters);
    const t1 = performance.now();
    // No-speech guard (E-011): the LM path invents words on digital silence, so a clip with no
    // speech-like energy never reaches the model. Silence is a feature-only cost.
    if (isNoSpeech(data)) {
      const audioSeconds = pcm.length / SAMPLE_RATE;
      const totalMs = t1 - t0;
      return {
        text: "",
        frames,
        outFrames: 0,
        vocabSize: pieces.length,
        decoder: "no-speech-guard",
        guarded: true,
        audioSeconds,
        timings: {
          feature_ms: t1 - t0,
          inference_ms: 0,
          decode_ms: 0,
          total_ms: totalMs,
        },
        rtfx: (audioSeconds * 1000) / totalMs,
      };
    }
    const input = new ort.Tensor("float32", data, [1, N_MELS, frames]);
    const outputs = await session.run({ [config.model.input]: input });
    const t2 = performance.now();
    const output = outputs[config.model.output];
    const [, outFrames, vocabSize] = output.dims;
    const decodeConfig = config.decode ?? {};
    const text =
      decodeConfig.default === "beam"
        ? beamSearch(output.data, outFrames, vocabSize, pieces, {
            beamWidth: decodeConfig.beam ?? 5,
            topK: decodeConfig.top_k ?? 8,
            lm,
            lmWeight: decodeConfig.lm_weight ?? 0,
          })
        : greedy(output.data, outFrames, vocabSize, pieces);
    const t3 = performance.now();
    const audioSeconds = pcm.length / SAMPLE_RATE;
    const totalMs = t3 - t0;
    return {
      text,
      frames,
      outFrames,
      vocabSize,
      decoder: decodeConfig.default ?? "greedy",
      audioSeconds,
      timings: {
        feature_ms: t1 - t0,
        inference_ms: t2 - t1,
        decode_ms: t3 - t2,
        total_ms: totalMs,
      },
      rtfx: (audioSeconds * 1000) / totalMs,
    };
  }

  /** 3 warm-ups then the runs of docs/05 §3, reporting medians and RTFx. */
  async function benchmark(pcm, { runs = 5, warmups = 2 } = {}) {
    for (let index = 0; index < warmups; index += 1) await transcribe(pcm);
    const totals = [];
    const features = [];
    const inferences = [];
    const decodes = [];
    for (let index = 0; index < runs; index += 1) {
      const result = await transcribe(pcm);
      totals.push(result.timings.total_ms);
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
