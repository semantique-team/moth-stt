/**
 * Browser runner for the Moth smoke page (Task 3.5, docs/06 §2).
 *
 * Loads `release/`, verifies every `sha256:` line (integrity, not publisher authenticity), creates an
 * `onnxruntime-web` WASM session (SIMD, threads only when cross-origin isolated) and exposes
 * `transcribe(pcm)` / `benchmark(pcm, ...)` with per-stage timings and RTFx.
 */

import * as ort from "./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs";

import { ContextualBias, beamSearch, greedy } from "./decode.js";
import { loadLM } from "./lm.js";
import { computeMel, N_MELS, N_FFT, HOP_LENGTH, SAMPLE_RATE } from "./mel.js";
import { createVad } from "./vad.js";

const FILTER_FLOATS = N_MELS * (N_FFT / 2 + 1);

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function fetchChecked(url, expected, root) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  if (root && response.url && !response.url.startsWith(root)) {
    throw new Error(`${url}: response escapes release root`);
  }
  const buffer = await response.arrayBuffer();
  if (!buffer.byteLength) throw new Error(`${url}: empty release file`);
  if (expected) {
    const actual = await sha256Hex(buffer);
    const wanted = expected.slice(7);
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
  const root = new URL(baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`,
    globalThis.document?.baseURI ?? globalThis.location?.href).href;

  ort.env.wasm.wasmPaths = new URL(
    "./node_modules/onnxruntime-web/dist/",
    import.meta.url,
  ).href;
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads =
    threads ??
    (globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1);

  const configBuffer = await fetchChecked(`${root}config.json`, undefined, root);
  const config = JSON.parse(new TextDecoder().decode(configBuffer));
  const object = (value, name) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`${name} must be an object`);
    }
    return value;
  };
  object(config, "config");
  if (config.name !== "moth-stt" || typeof config.version !== "string" || !config.version.trim()) {
    throw new Error("expected moth-stt name and nonempty version");
  }
  if (config.sample_rate !== SAMPLE_RATE) throw new Error(`expected ${SAMPLE_RATE} Hz release sample_rate`);
  const expected = {
    features: { n_fft: N_FFT, hop: HOP_LENGTH, n_mels: N_MELS, window: "hann_periodic",
      center: true, pad_mode: "reflect", drop_last_frame: true, floor_range: 8, scale: "(x+4)/4" },
    model: { input: "mel", output: "log_probs", frame_rate_hz: 50, max_seconds: 30 },
    tokenizer: { blank_id: 0, space_marker: "\u2581" },
    vad: { window: 512, context: 64 },
  };
  for (const [section, fields] of Object.entries(expected)) {
    object(config[section], section);
    for (const [key, value] of Object.entries(fields)) {
      if (config[section][key] !== value) throw new Error(`unsupported ${section}.${key}`);
    }
  }
  const decode = object(config.decode, "decode");
  if (!["beam", "greedy"].includes(decode.default)) throw new Error("invalid decode.default");
  if ("segment" in decode && typeof decode.segment !== "boolean") throw new Error("invalid decode.segment");
  for (const key of ["beam", "top_k", "lm_weight", "token_bonus"]) {
    const value = decode[key];
    if (!Number.isFinite(value) || (["beam", "top_k"].includes(key)
      ? !Number.isSafeInteger(value) || value <= 0 : key === "lm_weight" && value < 0)) {
      throw new Error(`invalid finite decode.${key}`);
    }
  }
  if (!Number.isFinite(config.vad.threshold) || config.vad.threshold < 0 || config.vad.threshold > 1) {
    throw new Error("invalid finite vad.threshold");
  }
  const minSamples = N_FFT + HOP_LENGTH;
  const maxSeconds = config.model.max_seconds;
  const maxSamples = Math.floor(maxSeconds * SAMPLE_RATE);
  const files = object(config.files, "files");
  // The exporter intentionally omits the LM for its zero-weight smoke bundle.
  const hasLM = Object.hasOwn(decode, "lm");
  if (!hasLM && decode.lm_weight !== 0) throw new Error("nonzero decode.lm_weight requires an LM");
  const requiredFiles = [config.model.file, config.tokenizer.vocab, config.features.mel_filters,
    config.vad.file, ...(hasLM ? [decode.lm] : [])];
  const safePath = (file) => {
    if (typeof file !== "string" || !file || /[\\:%?#\x00-\x1f]/.test(file) ||
        file.split("/").some(part => !part || part === "." || part === "..") ||
        /\.(?:pkl|pickle)(?:\.|$)/i.test(file)) throw new Error(`unsafe release path: ${file}`);
  };
  for (const file of [...requiredFiles, ...Object.keys(files)]) safePath(file);
  for (const [index, suffix] of [".onnx", ".json", ".bin", ".onnx", ...(hasLM ? [".bin"] : [])].entries()) {
    if (!requiredFiles[index].endsWith(suffix)) throw new Error(`asset must end with ${suffix}`);
  }
  if (new Set([...requiredFiles, "config.json", "NOTICE"]).size !== requiredFiles.length + 2 || "config.json" in files) {
    throw new Error("runtime assets must be distinct; config cannot hash itself");
  }
  for (const file of requiredFiles) {
    if (!Object.hasOwn(files, file)) throw new Error(`${file}: missing or malformed sha256 hash`);
  }
  for (const file of Object.keys(files)) {
    if (typeof files[file] !== "string" || !/^sha256:[0-9a-f]{64}$/.test(files[file])) {
      throw new Error(`${file}: missing or malformed sha256 hash`);
    }
  }
  const assets = new Map();
  let totalBytes = configBuffer.byteLength;
  for (const file of new Set([...requiredFiles, "NOTICE", ...Object.keys(files)])) {
    const bytes = await fetchChecked(`${root}${file}`, files[file], root);
    totalBytes += bytes.byteLength;
    if (totalBytes > 20_000_000) throw new Error("release exceeds 20000000 byte budget");
    assets.set(file, bytes);
  }
  const modelBuffer = assets.get(config.model.file);
  const vocabBuffer = assets.get(config.tokenizer.vocab);
  const filtersBuffer = assets.get(config.features.mel_filters);
  let lm = null;
  if (hasLM) {
    const lmBuffer = assets.get(decode.lm);
    if (new TextDecoder().decode(lmBuffer.slice(0, 8)) !== "MOTHLMB1") throw new Error("LM must use MOTHLMB1");
    lm = loadLM(lmBuffer);
  }

  const vocabJson = JSON.parse(new TextDecoder().decode(vocabBuffer));
  object(vocabJson, "vocab");
  const entries = Object.entries(vocabJson);
  const pieces = new Array(entries.length + 1).fill("");
  if (!entries.length) throw new Error("vocab must not be empty");
  for (const [id, piece] of entries) {
    if (!/^[1-9][0-9]*$/.test(id) || !Number.isSafeInteger(Number(id)) || Number(id) > entries.length ||
        typeof piece !== "string" || !piece) throw new Error("vocab must map contiguous IDs 1..N to nonempty pieces");
    pieces[Number(id)] = piece;
  }

  const filters = new Float32Array(filtersBuffer);
  if (filters.length !== FILTER_FLOATS) {
    throw new Error(`mel_filters.bin: ${filters.length} floats, expected ${FILTER_FLOATS}`);
  }
  if (!filters.every(Number.isFinite)) throw new Error("mel filters must be finite");
  const vad = await createVad(assets.get(config.vad.file), { gate: config.vad.threshold });

  const loadStarted = performance.now();
  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const loadMs = performance.now() - loadStarted;

  // Opt in per call with CTC IDs from this release's frozen tokenizer, not guessed text.
  // Build once against loaded pieces and reuse for every span; never enable from config.
  async function transcribeUnlocked(pcm, {
    segment: segmentOverride = null,
    contextualSequences = null,
    contextualBonus = 1,
    contextualPrefixBonus = 0,
  } = {}) {
    if (!(pcm instanceof Float32Array)) {
      throw new Error(`pcm must be a mono Float32Array at ${SAMPLE_RATE} Hz`);
    }
    if (pcm.length < minSamples) {
      throw new Error(`clip too short: ${pcm.length} samples < ${minSamples}`);
    }
    for (const sample of pcm) {
      if (!Number.isFinite(sample)) throw new Error("pcm samples must be finite");
    }
    const decodeConfig = config.decode;
    let contextual = null;
    if (contextualSequences !== null) {
      if (!Array.isArray(contextualSequences)) throw new Error("contextualSequences must be an array or null");
      if (decodeConfig.default !== "beam") throw new Error("contextual bias requires the beam decoder");
      contextual = new ContextualBias(contextualSequences, pieces, {
        bonus: contextualBonus,
        prefixBonus: contextualPrefixBonus,
      });
    }
    const t0 = performance.now();
    // No-speech guard (D25, E-011) and sentence segmentation from a **single** VAD pass: the
    // per-window probabilities answer both questions, and running the graph twice would double a
    // cost that is already ~80 ms on 18 s of audio.
    const probabilities = await vad.probabilities(pcm);
    if (!probabilities || probabilities.length !== Math.ceil(pcm.length / 512) ||
        !probabilities.every(value => Number.isFinite(value) && value >= 0 && value <= 1)) {
      throw new Error("invalid finite VAD probabilities");
    }
    const guarded = !(vad.fraction(probabilities) > vad.gate);
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
    const segmentEnabled = segmentOverride ?? segment ?? decodeConfig.segment ?? true;
    const bounds = segmentEnabled ? vad.segmentsFrom(pcm.length, probabilities) : [];
    const spans = bounds.length >= 2 ? bounds : [{ start: 0, end: pcm.length }];
    for (const span of spans) {
      const samples = span.end - span.start;
      if (!Number.isInteger(span.start) || !Number.isInteger(span.end) || span.start < 0 ||
          span.end > pcm.length || samples < minSamples || samples > maxSamples) {
        throw new Error(
          `segment must contain ${minSamples}..${maxSamples} samples (max ${maxSeconds} s); got ${samples}`,
        );
      }
    }

    let outFrames = 0;
    let vocabSize = pieces.length;
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
      if (!Number.isSafeInteger(frames) || frames < 1 || frames > 3000 ||
          !(data instanceof Float32Array) || data.length !== N_MELS * frames || !data.every(Number.isFinite)) {
        throw new Error("invalid finite mel tensor [1, 80, frames]");
      }
      const input = new ort.Tensor("float32", data, [1, N_MELS, frames]);
      featureMs += performance.now() - sliceStart;
      const inferStart = performance.now();
      const outputs = await session.run({ [config.model.input]: input });
      inferenceMs += performance.now() - inferStart;
      const output = outputs[config.model.output];
      if (!output || output.type !== "float32" || output.dims?.length !== 3 ||
          output.dims[0] !== 1 || output.dims[1] !== Math.ceil(frames / 2) ||
          output.dims[2] !== pieces.length || !(output.data instanceof Float32Array) ||
          output.data.length !== output.dims[1] * pieces.length || !output.data.every(Number.isFinite)) {
        throw new Error("invalid finite acoustic output [1, ceil(frames/2), vocab+blank]");
      }
      const [, framesOut, size] = output.dims;
      outFrames += framesOut;
      vocabSize = size;
      const decodeStart = performance.now();
      const text =
        decodeConfig.default === "beam"
          ? beamSearch(output.data, framesOut, size, pieces, {
              beamWidth: decodeConfig.beam,
              topK: decodeConfig.top_k,
              lm,
              lmWeight: decodeConfig.lm_weight,
              tokenBonus: decodeConfig.token_bonus,
              ...(contextual ? { contextual } : {}),
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
      decoder: decodeConfig.default,
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

  let transcribing = false;
  async function transcribe(pcm, options = {}) {
    if (transcribing) throw new Error("transcription already in progress on this runner");
    transcribing = true;
    try {
      return await transcribeUnlocked(pcm, options);
    } finally {
      transcribing = false;
    }
  }

  /** Warm-ups then measured runs, reporting medians and RTFx. */
  async function benchmark(pcm, { runs = 5, warmups = 2 } = {}) {
    if (!Number.isSafeInteger(runs) || runs <= 0 || !Number.isSafeInteger(warmups) || warmups < 0) {
      throw new Error("benchmark runs must be positive and warmups nonnegative integers");
    }
    if (transcribing) throw new Error("transcription already in progress on this runner");
    transcribing = true;
    try {
      for (let index = 0; index < warmups; index += 1) await transcribeUnlocked(pcm);
      const totals = [];
      const vads = [];
      const features = [];
      const inferences = [];
      const decodes = [];
      for (let index = 0; index < runs; index += 1) {
        const result = await transcribeUnlocked(pcm);
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
    } finally {
      transcribing = false;
    }
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
