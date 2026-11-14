import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test, { after } from "node:test";
import { SourceTextModule, SyntheticModule } from "node:vm";

import { ContextualBias, beamSearch, greedy } from "../decode.js";

// Run with: node --experimental-vm-modules --test web/tests/unit-runtime.test.mjs
// Stub only the runner's asset/compute boundaries; execute its actual orchestration code.
const source = await readFile(new URL("../ort.js", import.meta.url), "utf8");
const pieces = ["", "a", "b", "c"];
// stt.export._config/write_release smoke schema; hashes are filled for synthetic bytes below.
const smokeConfig = JSON.parse(await readFile(new URL("./fixtures/export-smoke-config.json", import.meta.url), "utf8"));
const documentBase = "https://example.test/demo/pages/index.html";
const defaultRoot = "https://example.test/demo/release/";
// Keep VM wrappers alive through suite teardown: early graph collection crashes Node 20 on Windows.
const moduleGraphs = [];
after(() => { moduleGraphs.length = 0; });

async function runner({ maxSeconds = 30, sampleRate = 16000, vad = true,
  vocab = pieces, decode = undefined, hashes = {}, fetches = [], mutate = () => {},
  mutateAssets = () => {}, redirect = null, fixture = null,
  baseURI = documentBase, locationHref = documentBase, baseUrl = "../release/" } = {}) {
  const calls = { vad: 0, mel: [], inference: 0, beam: [] };
  const state = { bounds: [], speech: true, pause: null, fail: null,
    scores: [-10, 0, -10, -10], output: null, mel: null, probabilities: null };
  const config = fixture ? structuredClone(fixture) : {
    name: "moth-stt", version: "test",
    sample_rate: sampleRate,
    model: { file: "model.onnx", input: "mel", output: "log_probs", frame_rate_hz: 50, max_seconds: maxSeconds },
    tokenizer: { vocab: "vocab.json", blank_id: 0, space_marker: "\u2581" },
    features: { mel_filters: "filters.bin", n_fft: 400, hop: 160, n_mels: 80,
      window: "hann_periodic", center: true, pad_mode: "reflect", drop_last_frame: true,
      floor_range: 8, scale: "(x+4)/4" },
    decode: { default: "greedy", beam: 5, top_k: 8, lm_weight: 0, token_bonus: 0,
      lm: "lm.bin", segment: vad, ...decode },
    vad: { file: "vad.onnx", threshold: 0.1, window: 512, context: 64 },
  };
  const assets = {
    [config.model.file]: new Uint8Array([1]).buffer,
    [config.tokenizer.vocab]: new TextEncoder().encode(JSON.stringify(Array.isArray(vocab)
      ? Object.fromEntries(vocab.slice(1).map((piece, index) => [index + 1, piece])) : vocab)).buffer,
    [config.features.mel_filters]: new Float32Array(80 * 201).buffer,
    [config.vad.file]: new Uint8Array([1]).buffer,
    ...(config.decode.lm ? { [config.decode.lm]: new TextEncoder().encode("MOTHLMB1").buffer } : {}),
    NOTICE: new TextEncoder().encode("test notice").buffer,
  };
  mutateAssets(assets);
  config.files = Object.fromEntries(Object.entries(assets).filter(([name]) => !fixture || name !== "NOTICE").map(([name, bytes]) =>
    [name, `sha256:${createHash("sha256").update(new Uint8Array(bytes)).digest("hex")}`]));
  Object.assign(config.files, hashes);
  mutate(config);
  const fakeVad = {
    gate: 0.1,
    async probabilities(pcm) {
      calls.vad += 1;
      if (vad && state.pause) await state.pause;
      if (vad && state.fail) throw state.fail;
      return state.probabilities ?? new Array(Math.ceil(pcm.length / 512)).fill(state.speech ? 1 : 0);
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
              return { log_probs: state.output ?? { type: "float32", dims: [1, 1, vocab.length],
                data: Float32Array.from(state.scores) } };
            },
          };
        },
      },
    },
    "./decode.js": { ContextualBias, greedy, beamSearch(...args) {
      calls.beam.push(args[4]);
      return beamSearch(...args);
    } },
    "./lm.js": { loadLM: () => ({}) },
    "./mel.js": {
      N_MELS: 80, N_FFT: 400, HOP_LENGTH: 160, SAMPLE_RATE: 16000,
      computeMel(pcm) {
        calls.mel.push(pcm.length);
        return state.mel ?? { data: new Float32Array(80), frames: 1 };
      },
    },
    "./vad.js": { createVad: async () => fakeVad },
  };
  const module = new SourceTextModule(source, {
    initializeImportMeta(meta) { meta.url = "https://example.test/static/modules/ort.js"; },
  });
  const graph = [module];
  moduleGraphs.push(graph);
  await module.link((specifier) => {
    const exports = dependencies[specifier];
    const dependency = new SyntheticModule(Object.keys(exports), function () {
      for (const [name, value] of Object.entries(exports)) this.setExport(name, value);
    });
    graph.push(dependency);
    return dependency;
  });
  await module.evaluate();
  const originalFetch = globalThis.fetch;
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, "location");
  Object.defineProperty(globalThis, "document", { configurable: true, value: baseURI ? { baseURI } : undefined });
  Object.defineProperty(globalThis, "location", { configurable: true, value: { href: locationHref } });
  const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: {
    subtle: { async digest(algorithm, bytes) {
      assert.equal(algorithm, "SHA-256");
      return Uint8Array.from(createHash("sha256").update(new Uint8Array(bytes)).digest()).buffer;
    } },
  } });
  globalThis.fetch = async (url) => {
    fetches.push(url);
    return {
      ok: true,
      url: (typeof redirect === "function" ? redirect(url) : redirect) ?? new URL(url, baseURI ?? locationHref).href,
      arrayBuffer: async () => {
        if (url.endsWith("config.json")) return new TextEncoder().encode(JSON.stringify(config)).buffer;
        return assets[url.split("/").at(-1)];
      },
    };
  };
  try {
    return { runtime: await module.namespace.createRunner({ threads: 1, baseUrl }), calls, state };
  } finally {
    globalThis.fetch = originalFetch;
    if (originalDocument) Object.defineProperty(globalThis, "document", originalDocument);
    else delete globalThis.document;
    if (originalLocation) Object.defineProperty(globalThis, "location", originalLocation);
    else delete globalThis.location;
    if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
    else delete globalThis.crypto;
  }
}

test("release fetches use the document base, not the module directory, with real response URLs", async () => {
  for (const options of [
    { baseURI: "https://example.test/site/base/page.html", locationHref: documentBase,
      expectedRoot: "https://example.test/site/release/" },
    { baseURI: null, locationHref: "https://example.test/other/page.html",
      expectedRoot: "https://example.test/release/" },
    { baseUrl: "https://assets.test/bundle", expectedRoot: "https://assets.test/bundle/" },
  ]) {
    const fetches = [];
    const { runtime } = await runner({ ...options, fetches });
    assert.deepEqual(fetches, ["config.json", "model.onnx", "vocab.json", "filters.bin",
      "vad.onnx", "lm.bin", "NOTICE"].map(name => options.expectedRoot + name));
    assert.equal((await runtime.transcribe(new Float32Array(560))).text, "a");
  }
  await assert.rejects(runner({ redirect: url => url.endsWith("model.onnx")
    ? "https://example.test/demo/release-other/model.onnx" : url }), /escapes release root/);
});

test("exporter smoke schema loads and transcribes without an LM, retaining all feature and VAD checks", async () => {
  const fetches = [];
  const { runtime, calls } = await runner({ fixture: smokeConfig, fetches });
  assert.equal(Object.hasOwn(runtime.config.decode, "lm"), false);
  assert.deepEqual(Object.keys(runtime.config.files).sort(),
    ["model.int8.onnx", "vocab.json", "mel_filters.bin", "silero_vad.onnx"].sort());
  assert.deepEqual(fetches, ["config.json", "model.int8.onnx", "vocab.json", "mel_filters.bin",
    "silero_vad.onnx", "NOTICE"].map(name => defaultRoot + name));
  assert.equal((await runtime.transcribe(new Float32Array(560))).text, "a");
  assert.equal(calls.beam[0].lm, null);
  assert.equal(calls.beam[0].lmWeight, 0);
  assert.equal(calls.vad, 1);
  for (const mutate of [config => { config.features.hop = 80; },
    config => { delete config.features.drop_last_frame; }, config => { delete config.vad; }]) {
    await assert.rejects(runner({ fixture: smokeConfig, mutate }));
  }
});

test("LM omission requires explicit zero weight in beam or greedy; declared LMs remain mandatory", async () => {
  for (const decoder of ["beam", "greedy"]) {
    await runner({ fixture: smokeConfig, mutate: config => { config.decode.default = decoder; } });
    for (const weight of [0.55, undefined, null, "0", false]) {
      const fetches = [];
      await assert.rejects(runner({ fixture: smokeConfig, fetches, mutate: config => {
        config.decode.default = decoder;
        config.decode.lm_weight = weight;
      } }), /lm_weight/);
      assert.deepEqual(fetches, [`${defaultRoot}config.json`]);
    }
    for (const lm of [null, "", "../lm.bin", "missing.bin"]) {
      await assert.rejects(runner({ fixture: smokeConfig, mutate: config => {
        config.decode.default = decoder;
        config.decode.lm = lm;
      } }), /unsafe release path|missing or malformed sha256/);
    }
  }
  await runner({ decode: { default: "beam", lm_weight: 0.55 } });
});

test("runner validates every required hash before fetching any asset", async () => {
  for (const file of ["model.onnx", "vocab.json", "filters.bin", "lm.bin", "vad.onnx"]) {
    for (const hash of [undefined, null, "", 123, "sha256:abc", "g".repeat(64)]) {
      const fetches = [];
      await assert.rejects(runner({ hashes: { [file]: hash }, fetches }),
        /missing or malformed sha256/);
      assert.deepEqual(fetches, [`${defaultRoot}config.json`]);
    }
  }
  await runner();
});

test("runner rejects an asset whose bytes do not match its valid hash", async () => {
  await assert.rejects(runner({ hashes: { "model.onnx": "sha256:" + "0".repeat(64) } }), /sha256 mismatch/);
});

test("release semantics reject missing weighted LM/VAD, unsupported features and invalid finite decoder fields", async () => {
  for (const mutate of [
    config => { delete config.decode.lm; config.decode.lm_weight = 0.55; }, config => { delete config.vad; },
    config => { config.name = "other"; }, config => { config.version = " "; },
    config => { config.features.center = 1; }, config => { config.features.floor_range = Infinity; },
    config => { config.tokenizer.blank_id = 1; }, config => { config.model.frame_rate_hz = 100; },
    config => { config.decode.default = "unknown"; }, config => { config.decode.segment = 1; },
    ...["beam", "top_k", "lm_weight", "token_bonus"].flatMap(key => [null, true, "1", Infinity]
      .map(value => config => { config.decode[key] = value; })),
    config => { config.decode.beam = 0; }, config => { config.decode.top_k = 1.5; },
    config => { config.decode.lm_weight = -1; }, config => { config.vad.threshold = 1.1; },
  ]) {
    const fetches = [];
    await assert.rejects(runner({ mutate, fetches }));
    assert.deepEqual(fetches, [`${defaultRoot}config.json`]);
  }
});

test("release paths stay inside the asset root, including extras and redirects", async () => {
  for (const path of ["../evil.onnx", "/evil.onnx", "a//b.onnx", "a/./b.onnx",
    "a\\b.onnx", "https://evil/model.onnx", "%2e%2e/evil.onnx", "a?x.onnx", "a#x.onnx",
    "a\u0000.onnx", "a.pkl.onnx"]) {
    for (const mutate of [config => { config.model.file = path; },
      config => { config.files[path] = "sha256:" + "0".repeat(64); }]) {
      const fetches = [];
      await assert.rejects(runner({ mutate, fetches }), /unsafe release path/);
      assert.deepEqual(fetches, [`${defaultRoot}config.json`]);
    }
  }
  await assert.rejects(runner({ redirect: "https://evil.test/config.json" }), /escapes release root/);
  await assert.rejects(runner({ mutate: config => { config.vad.file = config.model.file; } }), /distinct/);
  await assert.rejects(runner({ mutate: config => { config.files["config.json"] = "sha256:" + "0".repeat(64); } }), /hash itself/);
});

test("vocab, filters, LM signature, NOTICE and extra declared assets are required and validated", async () => {
  for (const vocab of [[], {}, { 0: "blank", 1: "a" }, { 2: "a" }, { "01": "a" },
    { "1e309": "a" }, { 1: null }, { 1: 123 }, { 1: "" }]) {
    await assert.rejects(runner({ vocab }), /vocab/);
  }
  for (const mutateAssets of [
    assets => { new Float32Array(assets["filters.bin"])[0] = NaN; },
    assets => { assets["filters.bin"] = new Float32Array(1).buffer; },
    assets => { assets["lm.bin"] = new TextEncoder().encode("not-an-lm").buffer; },
    assets => { assets.NOTICE = new ArrayBuffer(0); },
    assets => { assets.extra = new ArrayBuffer(0); },
    assets => { assets.extra = new ArrayBuffer(20_000_000); },
  ]) await assert.rejects(runner({ mutateAssets }), /finite|floats|MOTHLMB1|empty|budget/);
});

test("invalid feature tensors, acoustic tensors and VAD values reject before decode and recover", async () => {
  const { runtime, state } = await runner();
  const pcm = new Float32Array(560);
  for (const mel of [ { frames: NaN, data: new Float32Array(80) },
    { frames: 1, data: new Float32Array(79) }, { frames: 1, data: new Float32Array(80).fill(Infinity) }]) {
    state.mel = mel;
    await assert.rejects(runtime.transcribe(pcm), /mel tensor/);
  }
  state.mel = null;
  for (const output of [{}, { type: "float32", dims: [2, 1, 4], data: new Float32Array(4) },
    { type: "float32", dims: [1, 2, 4], data: new Float32Array(8) },
    { type: "float32", dims: [1, 1, 3], data: new Float32Array(3) },
    { type: "float32", dims: [1, 1, 4], data: new Float32Array(3) },
    { type: "float64", dims: [1, 1, 4], data: new Float64Array(4) },
    { type: "float32", dims: [1, 1, 4], data: new Float32Array(4).fill(-Infinity) },
    { type: "float32", dims: [1, 1, 4], data: new Float32Array(4).fill(NaN) }]) {
    state.output = output;
    await assert.rejects(runtime.transcribe(pcm), /acoustic output/);
  }
  state.output = null;
  for (const probabilities of [[], [1], [1, NaN], [1, 1.1], [1, -0.1]]) {
    state.probabilities = probabilities;
    await assert.rejects(runtime.transcribe(pcm), /VAD probabilities/);
  }
  state.probabilities = null;
  assert.equal((await runtime.transcribe(pcm)).text, "a");
});

test("benchmark owns the runner for all iterations and releases it after failure", async () => {
  const { runtime, state } = await runner();
  const pcm = new Float32Array(560);
  for (const options of [{ runs: 0 }, { runs: 1.5 }, { warmups: -1 }, { warmups: NaN }]) {
    await assert.rejects(runtime.benchmark(pcm, options), /integers/);
  }
  let release;
  state.pause = new Promise(resolve => { release = resolve; });
  const pending = runtime.benchmark(pcm, { runs: 2, warmups: 0 });
  await assert.rejects(runtime.transcribe(pcm), /already in progress/);
  await assert.rejects(runtime.benchmark(pcm, { runs: 1, warmups: 0 }), /already in progress/);
  state.fail = new Error("benchmark failure");
  release();
  await assert.rejects(pending, /benchmark failure/);
  state.pause = state.fail = null;
  assert.equal((await runtime.transcribe(pcm)).text, "a");
  const competing = [];
  state.pause = { then(resolve) {
    competing.push(assert.rejects(runtime.transcribe(pcm), /already in progress/));
    resolve();
  } };
  assert.equal((await runtime.benchmark(pcm, { runs: 2, warmups: 1 })).runs, 2);
  await Promise.all(competing);
  assert.equal(competing.length, 3);
});

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
  const { runtime, calls } = await runner({ vad: false });
  await runtime.transcribe(new Float32Array(480000));
  await assert.rejects(runtime.transcribe(new Float32Array(480001)), /segment must contain/);
  assert.equal(calls.inference, 1);
  for (const maxSeconds of [0, -1, 0.01, 60, "invalid"]) {
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
