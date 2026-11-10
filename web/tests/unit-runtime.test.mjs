import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { SourceTextModule, SyntheticModule } from "node:vm";

import { ContextualBias, beamSearch, greedy } from "../decode.js";

// Run with: node --experimental-vm-modules --test web/tests/unit-runtime.test.mjs
// Stub only the runner's asset/compute boundaries; execute its actual orchestration code.
const source = await readFile(new URL("../ort.js", import.meta.url), "utf8");
const pieces = ["", "a", "b", "c"];

async function runner({ maxSeconds = 30, sampleRate = 16000, vad = true,
  vocab = pieces, decode = undefined } = {}) {
  const calls = { vad: 0, mel: [], inference: 0, beam: [] };
  const state = { bounds: [], speech: true, pause: null, fail: null,
    scores: [-10, 0, -10, -10] };
  const config = {
    sample_rate: sampleRate,
    model: { file: "model", input: "mel", output: "log_probs", max_seconds: maxSeconds },
    tokenizer: { vocab: "vocab" },
    features: { mel_filters: "filters" },
    ...(decode ? { decode } : {}),
    ...(vad ? { vad: { file: "vad", threshold: 0.1 } } : {}),
  };
  const fakeVad = {
    gate: 0.1,
    async probabilities() {
      calls.vad += 1;
      if (state.pause) await state.pause;
      if (state.fail) throw state.fail;
      return [state.speech ? 1 : 0];
    },
    fraction: (values) => values[0],
    segmentsFrom: () => state.bounds,
  };
  const dependencies = {
    "./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs": {
      env: { wasm: {} },
      Tensor: class {},
      InferenceSession: {
        async create() {
          return {
            async run() {
              calls.inference += 1;
              if (!vad && state.pause) await state.pause;
              if (state.fail) throw state.fail;
              return { log_probs: { dims: [1, 1, vocab.length], data: state.scores } };
            },
          };
        },
      },
    },
    "./decode.js": { ContextualBias, greedy, beamSearch(...args) {
      calls.beam.push(args[4]);
      return beamSearch(...args);
    } },
    "./lm.js": { loadLM() { throw new Error("unexpected LM asset"); } },
    "./mel.js": {
      N_MELS: 80, N_FFT: 400, HOP_LENGTH: 160, SAMPLE_RATE: 16000,
      computeMel(pcm) {
        calls.mel.push(pcm.length);
        return { data: new Float32Array(80), frames: 1 };
      },
    },
    "./vad.js": { createVad: async () => fakeVad },
  };
  const module = new SourceTextModule(source, {
    initializeImportMeta(meta) { meta.url = new URL("../ort.js", import.meta.url).href; },
  });
  await module.link((specifier) => {
    const exports = dependencies[specifier];
    return new SyntheticModule(Object.keys(exports), function () {
      for (const [name, value] of Object.entries(exports)) this.setExport(name, value);
    });
  });
  await module.evaluate();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => ({
    ok: true,
    arrayBuffer: async () => {
      if (url.endsWith("config.json")) return new TextEncoder().encode(JSON.stringify(config)).buffer;
      if (url.endsWith("vocab")) return new TextEncoder().encode(JSON.stringify(vocab)).buffer;
      if (url.endsWith("filters")) return new Float32Array(80 * 201).buffer;
      return new ArrayBuffer(0);
    },
  });
  try {
    return { runtime: await module.namespace.createRunner({ threads: 1 }), calls, state };
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test("decoder topK counts nonblank plus blank, with stable cutoff ID ties", () => {
  assert.equal(beamSearch([0, -0.1, -0.2, -10], 1, 4, pieces,
    { topK: 1, tokenBonus: 0.15 }), "a");
  for (const topK of [1, 2, 3, 10]) {
    assert.equal(beamSearch([-10, -0.5, -0.5, -0.5], 1, 4, pieces, { topK }), "a");
  }
});

test("decoder blank separates repeats; bonus applies only to emitted tokens", () => {
  const repeated = [-10, 0, -10, 0];
  assert.equal(beamSearch(repeated, 2, 2, pieces, { tokenBonus: 2 }), "a");
  assert.equal(beamSearch([-10, 0, -1, 0, -10, 0], 3, 2, pieces,
    { topK: 1, tokenBonus: 2 }), "aa");
  const ambiguous = [-1, -0.5, -10, -0.5, -10, -1.2];
  assert.equal(beamSearch(ambiguous, 2, 3, pieces, { beamWidth: 4 }), "a");
  assert.equal(beamSearch(ambiguous, 2, 3, pieces,
    { beamWidth: 4, tokenBonus: 1 }), "ab");
});

test("runner rejects malformed/nonfinite/short PCM before VAD, and recovers", async () => {
  const { runtime, calls } = await runner();
  for (const pcm of [null, [], [[0]], new Float64Array(560), { length: 560 }]) {
    await assert.rejects(runtime.transcribe(pcm), /mono Float32Array at 16000 Hz/);
  }
  for (const length of [0, 559]) {
    await assert.rejects(runtime.transcribe(new Float32Array(length)), /clip too short/);
  }
  for (const value of [NaN, Infinity, -Infinity]) {
    const pcm = new Float32Array(560);
    pcm[559] = value;
    await assert.rejects(runtime.transcribe(pcm), /must be finite/);
  }
  assert.equal(calls.vad, 0);
  assert.equal((await runtime.transcribe(new Float32Array(560))).text, "a");
});

test("runner enforces inclusive duration bounds before any acoustic inference", async () => {
  for (const configured of [0.1, 30, 60]) {
    const { runtime, calls } = await runner({ maxSeconds: configured, vad: false });
    const limit = Math.min(configured, 30) * 16000;
    await runtime.transcribe(new Float32Array(limit));
    await assert.rejects(runtime.transcribe(new Float32Array(limit + 1)), /segment must contain/);
    assert.equal(calls.inference, 1);
  }
  for (const maxSeconds of [0, -1, 0.01, "invalid"]) {
    await assert.rejects(runner({ maxSeconds }), /model.max_seconds/);
  }
  await assert.rejects(runner({ sampleRate: 48000 }), /16000 Hz/);
});

test("runner retains >=2 runs policy and checks every span before inference", async () => {
  const { runtime, state, calls } = await runner();
  const pcm = new Float32Array(480001);
  state.bounds = [{ start: 0, end: 560 }];
  await assert.rejects(runtime.transcribe(pcm), /segment must contain/);
  state.bounds = [{ start: 0, end: 560 }, { start: 560, end: pcm.length }];
  assert.equal((await runtime.transcribe(pcm)).segments, 2);
  assert.deepEqual(calls.mel, [560, pcm.length - 560]);
  await assert.rejects(runtime.transcribe(pcm, { segment: false }), /segment must contain/);
  state.bounds = [{ start: 0, end: 560 }, { start: 560, end: 480561 }];
  await assert.rejects(runtime.transcribe(new Float32Array(480561)), /segment must contain/);
  state.bounds = [{ start: 0, end: 560 }, { start: 560, end: 1119 }];
  await assert.rejects(runtime.transcribe(new Float32Array(1120)), /segment must contain/);
  assert.equal(calls.inference, 2);
});

test("runner preserves silence guard without acoustic decoding", async () => {
  const { runtime, state, calls } = await runner();
  state.speech = false;
  assert.equal((await runtime.transcribe(new Float32Array(560))).guarded, true);
  assert.equal(calls.inference, 0);
});

test("overlapping transcribe/benchmark calls reject before shared VAD or session access", async () => {
  for (const vad of [true, false]) {
    const { runtime, state, calls } = await runner({ vad });
    let release;
    state.pause = new Promise((resolve) => { release = resolve; });
    const pending = runtime.transcribe(new Float32Array(560));
    await assert.rejects(runtime.transcribe(new Float32Array(560)), /already in progress/);
    await assert.rejects(runtime.benchmark(new Float32Array(560)), /already in progress/);
    assert.equal(vad ? calls.vad : calls.inference, 1);
    release();
    await pending;
    state.pause = null;
    state.fail = new Error("injected failure");
    await assert.rejects(runtime.transcribe(new Float32Array(560)), /injected failure/);
    state.fail = null;
    assert.equal((await runtime.transcribe(new Float32Array(560))).text, "a");
  }
});

test("contextual sequences use loaded vocabulary and bias every span without persisting", async () => {
  const { runtime, state, calls } = await runner({
    vocab: ["", "\u2581il", "\u2581api"], decode: { default: "beam" },
  });
  state.scores = [-10, 0, -0.5];
  state.bounds = [{ start: 0, end: 560 }, { start: 560, end: 1120 }];
  const pcm = new Float32Array(1120);
  assert.equal((await runtime.transcribe(pcm)).text, "il il");
  assert.ok(calls.beam.every(options => !("contextual" in options)));
  calls.beam.length = 0;
  assert.equal((await runtime.transcribe(pcm, {
    contextualSequences: [[2]], contextualBonus: 1, contextualPrefixBonus: 0.25,
  })).text, "api api");
  assert.equal(calls.beam.length, 2);
  assert.ok(calls.beam[0].contextual instanceof ContextualBias);
  assert.equal(calls.beam[0].contextual, calls.beam[1].contextual);
  assert.equal(calls.beam[0].contextual.prefixBonus, 0.25);
  calls.beam.length = 0;
  assert.equal((await runtime.transcribe(pcm, { contextualSequences: null })).text, "il il");
  assert.ok(calls.beam.every(options => !("contextual" in options)));
});

test("invalid contextual options fail before VAD or inference; runner recovers", async () => {
  const { runtime, calls, state } = await runner({
    vocab: ["", "\u2581il", "\u2581api"], decode: { default: "beam" },
  });
  state.scores = [-10, 0, -0.5];
  const pcm = new Float32Array(560);
  for (const options of [
    { contextualSequences: {} }, { contextualSequences: [[999]] },
    { contextualSequences: [[0]] },
    { contextualSequences: [[2]], contextualBonus: -1 },
    { contextualSequences: [[2]], contextualPrefixBonus: 2 },
  ]) await assert.rejects(runtime.transcribe(pcm, options), /contextual|finite/);
  assert.equal(calls.vad, 0);
  assert.equal(calls.inference, 0);
  assert.equal((await runtime.transcribe(pcm, { contextualSequences: [[2]] })).text, "api");
  state.speech = false;
  assert.equal((await runtime.transcribe(pcm, { contextualSequences: [[2]] })).guarded, true);
  assert.equal(calls.inference, 1);
  const greedyRunner = await runner();
  await assert.rejects(greedyRunner.runtime.transcribe(pcm, { contextualSequences: [] }), /beam decoder/);
  assert.equal(greedyRunner.calls.vad, 0);
});
