import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const root = fileURLToPath(new URL("../../", import.meta.url));
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
const release = path.resolve(process.env.MOTH_RELEASE_DIR || path.join(root, "../moth-stt"));
const reportDir = process.env.MOTH_ACCEPTANCE_REPORT_DIR
  ? path.resolve(process.env.MOTH_ACCEPTANCE_REPORT_DIR)
  : path.join(root, "runs/browser-acceptance-2026-11-10");

test("original release executes real WASM with guards, recovery and contextual timing", async ({ page }, info) => {
  const { isolated, threads } = info.project.metadata;
  const report: any = {
    project: info.project.name,
    passed: false,
    timestamp: new Date().toISOString(),
    node: process.version,
    platform: `${process.platform}/${process.arch}`,
    code_sha: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(),
    source_sha256: {},
    assets: [],
    audio: [],
    accounting: "Fresh browser context; routing disables HTTP cache. Fulfilled release bodies are raw bytes, not HTTP-compressed sizes. Resource Timing is browser-reported, not wire billing.",
    memory_scope: "Optional performance.memory.usedJSHeapSize snapshots only; not RSS, peak memory, or total WASM memory.",
  };
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
  try {
    for (const name of ["web/ort.js", "web/decode.js", "web/lm.js", "web/mel.js", "web/vad.js", "web/server.py", "web/package-lock.json", "web/playwright.acceptance.config.ts", "web/acceptance/real-wasm.spec.ts"]) {
      report.source_sha256[name] = hash(readFileSync(path.join(root, name)));
    }
    report.ort_package_version = JSON.parse(readFileSync(path.join(root, "web/node_modules/onnxruntime-web/package.json"), "utf8")).version;
    report.playwright_version = JSON.parse(readFileSync(path.join(root, "web/node_modules/@playwright/test/package.json"), "utf8")).version;
    const configBytes = readFileSync(path.join(release, "config.json"));
    const config = JSON.parse(configBytes.toString());
    report.release_version = config.version;
    report.decode_config = config.decode;
    const assets = new Map<string, Buffer>([["config.json", configBytes]]);
    const releaseRoot = realpathSync(release) + path.sep;
    for (const [name, expected] of Object.entries(config.files)) {
      const filename = realpathSync(path.resolve(release, name));
      expect(filename.startsWith(releaseRoot), "release allowlist must remain inside release").toBe(true);
      const bytes = readFileSync(filename);
      expect(`sha256:${hash(bytes)}`, name).toBe(expected);
      assets.set(name, bytes);
    }
    for (const name of [config.model.file, config.tokenizer.vocab, config.features.mel_filters, config.decode.lm, config.vad.file]) {
      expect(assets.has(name), "all runtime assets require a configured hash").toBe(true);
    }
    for (const [name, bytes] of assets) report.assets.push({ name, bytes: bytes.length, sha256: hash(bytes), fulfilled_responses: 0, fulfilled_raw_bytes: 0 });
    const headers: Record<string, string> = { "Cross-Origin-Resource-Policy": "same-origin", "Cache-Control": "no-store" };
    if (isolated) Object.assign(headers, { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" });
    await page.route("**/acceptance.html", (route) => route.fulfill({
      contentType: "text/html", headers,
      body: '<!doctype html><meta charset="utf-8"><title>Real WASM acceptance</title><script type="module">import "/web/ort.js";</script>',
    }));
    await page.route("**/acceptance-release/**", async (route) => {
      const name = new URL(route.request().url()).pathname.slice("/acceptance-release/".length);
      const body = assets.get(name);
      if (!body) return route.fulfill({ status: 404, body: "Not allowlisted", headers });
      const row = report.assets.find((asset: any) => asset.name === name);
      row.fulfilled_responses += 1;
      row.fulfilled_raw_bytes += body.length;
      await route.fulfill({ body, headers, contentType: name.endsWith(".json") ? "application/json" : "application/octet-stream" });
    });
    const fixture = readFileSync(path.join(root, "web/tests/fixtures/speech.wav"));
    report.audio.push({ label: "versioned-public-speech", sha256: hash(fixture), bytes: fixture.length });
    const audioUrls = [{ label: "versioned-public-speech", url: "/web/tests/fixtures/speech.wav" }];
    const manifestPath = path.resolve(process.env.MOTH_ACCEPTANCE_AUDIO || path.join(root, "../moth-stt-experiments/evaluation/manifest.json"));
    if (process.env.MOTH_ACCEPTANCE_AUDIO || existsSync(manifestPath)) {
      const manifestBytes = readFileSync(manifestPath);
      const manifest = JSON.parse(manifestBytes.toString());
      report.audio_manifest_sha256 = hash(manifestBytes);
      const audioRoot = realpathSync(path.dirname(manifestPath)) + path.sep;
      for (const group of ["it_it", "en_us", "spliced_it_en"]) {
        const entry = manifest.find((row: any) => row.group === group);
        expect(entry, `manifest group ${group}`).toBeTruthy();
        const filename = realpathSync(path.resolve(path.dirname(manifestPath), entry.path.replace(/\\/g, "/")));
        expect(filename.startsWith(audioRoot), "audio must remain inside manifest directory").toBe(true);
        const body = readFileSync(filename);
        if (entry.sha256) expect(hash(body), `audio hash ${group}`).toBe(entry.sha256.replace(/^sha256:/, ""));
        const url = `/acceptance-audio/${group}.wav`;
        await page.route(`**${url}`, (route) => route.fulfill({ body, contentType: "audio/wav", headers }));
        audioUrls.push({ label: group, url });
        report.audio.push({ label: group, sha256: hash(body), bytes: body.length, manifest_hash_verified: Boolean(entry.sha256) });
      }
      report.optional_audio_coverage = "First IT, EN and spliced entries in existing manifest; reused public evaluation audio, not natural code-switching evidence.";
    } else {
      report.optional_audio_coverage = "Not available; public versioned Italian fixture only.";
    }
    await page.goto("/acceptance.html");
    report.environment = await page.evaluate(() => ({
      userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency,
      crossOriginIsolated: globalThis.crossOriginIsolated,
      sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    }));
    expect(report.environment.crossOriginIsolated).toBe(isolated);
    expect(report.environment.sharedArrayBuffer).toBe(isolated);
    report.results = await page.evaluate(async ({ threads, audioUrls }) => {
      // @ts-ignore Browser imports are served by the repository-root servers.
      const { createRunner, decodeTo16k } = await import("/web/ort.js");
      // @ts-ignore Browser-only module.
      const ort = await import("/web/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs");
      const heap = () => (performance as any).memory?.usedJSHeapSize ?? null;
      const heapBefore = heap();
      const start = performance.now();
      const runner = await createRunner({ baseUrl: "/acceptance-release/", threads });
      const coldLoadMs = performance.now() - start;
      const heapAfterLoad = heap();
      const pcm = await decodeTo16k(audioUrls[0].url);
      const single = await runner.transcribe(pcm);
      const baseline = await runner.benchmark(pcm, { warmups: 1, runs: 3 });
      const rejection = async (input: any, options = {}) => {
        try { await runner.transcribe(input, options); return "ACCEPTED"; }
        catch (error) { return (error as Error).message; }
      };
      const badFinite = new Float32Array(16000); badFinite[100] = NaN;
      const infinity = new Float32Array(16000); infinity[100] = Infinity;
      const invalid = {
        type: await rejection([0, 1]),
        short: await rejection(new Float32Array(559)),
        nan: await rejection(badFinite),
        infinity: await rejection(infinity),
      };
      const pending = runner.transcribe(pcm);
      const overlap = await rejection(pcm);
      const firstAfterOverlap = await pending;
      const recovered = await runner.transcribe(pcm);
      const silence = await runner.transcribe(new Float32Array(3 * 16000));
      const tonePcm = new Float32Array(3 * 16000);
      for (let i = 0; i < tonePcm.length; i++) tonePcm[i] = 0.1 * Math.sin(2 * Math.PI * 440 * i / 16000);
      const tone = await runner.transcribe(tonePcm);
      const longSpeech = new Float32Array(31 * 16000);
      for (let i = 0; i < longSpeech.length; i++) longSpeech[i] = pcm[i % pcm.length];
      const longRejection = await rejection(longSpeech, { segment: false });
      const afterLong = await runner.transcribe(pcm);
      const marker = runner.config.tokenizer.space_marker;
      const words = new Set(single.text.split(/\s+/));
      const id = runner.pieces.findIndex((piece: string, id: number) => id > 0 &&
        piece.startsWith(marker) && words.has(piece.slice(marker.length)) &&
        /^[a-z\u00e0\u00e8\u00e9\u00ec\u00ed\u00f2\u00f3\u00f9\u00fa']+$/.test(piece.slice(marker.length)));
      if (id <= 0) throw new Error("Positive contextual coverage requires a verified single-piece output word");
      const sequences = [[id]];
      const contextOptions = { contextualSequences: sequences, contextualBonus: 1.75, contextualPrefixBonus: 0 };
      // Observe the same class used by the real runner; never replace scores or acoustic outputs.
      // @ts-ignore Browser-only module.
      const { ContextualBias } = await import("/web/decode.js");
      const originalDelta = ContextualBias.prototype.delta;
      const originalFinish = ContextualBias.prototype.finish;
      const scorerUse: Record<string, any> = {};
      let active: any;
      for (const method of ["delta", "finish"] as const) {
        const original = method === "delta" ? originalDelta : originalFinish;
        ContextualBias.prototype[method] = function (...args: any[]) {
          const score = original.apply(this, args);
          active[`${method}_calls`] += 1;
          if (score > 0) active.positive_scores += 1;
          const settings = { bonus: this.bonus, prefixBonus: this.prefixBonus, terminals: [...this.terminals].sort() };
          if (!active.settings.some((row: any) => JSON.stringify(row) === JSON.stringify(settings))) active.settings.push(settings);
          return score;
        };
      }
      const observe = async (label: string, options: any) => {
        active = scorerUse[label] = { delta_calls: 0, finish_calls: 0, positive_scores: 0, settings: [] };
        return await runner.transcribe(pcm, options);
      };
      let noOptions, emptyContext, zeroBonus, contextual;
      try {
        noOptions = await observe("no_options", {});
        emptyContext = await observe("empty_context", { contextualSequences: [] });
        zeroBonus = await observe("zero_bonus", { ...contextOptions, contextualBonus: 0 });
        contextual = await observe("positive_bonus", contextOptions);
      } finally {
        ContextualBias.prototype.delta = originalDelta;
        ContextualBias.prototype.finish = originalFinish;
      }
      const measure = async (options: any) => {
        await runner.transcribe(pcm, options);
        const samples = [];
        for (let run = 0; run < 3; run++) {
          const started = performance.now();
          const result = await runner.transcribe(pcm, options);
          samples.push({ wall_ms: performance.now() - started, timings: result.timings, text: result.text });
        }
        const median = [...samples].map((row) => row.wall_ms).sort((a, b) => a - b)[1];
        return { warmups: 1, repeats: 3, wall_ms_median: median, rtfx: pcm.length / 16 / median, samples };
      };
      const pairedTimings = { without_bias: await measure({}), with_bias: await measure(contextOptions) };
      const extraAudio = [];
      for (const audio of audioUrls.slice(1)) {
        const decoded = await decodeTo16k(audio.url);
        extraAudio.push({ label: audio.label, result: await runner.transcribe(decoded) });
      }
      return {
        ort_versions: ort.env.versions,
        cold_load_ms: coldLoadMs, session_load_ms: runner.loadMs,
        threads: runner.numThreads, single, baseline, invalid, overlap, firstAfterOverlap, recovered,
        silence, tone, longRejection, afterLong, extraAudio,
        contextual: { coverage: "Verified single-piece output word and observed real scorer use; synthetic glossary, not natural benefit", sequences, piece: runner.pieces[id], noOptions, emptyContext, zeroBonus, result: contextual, scorerUse },
        pairedTimings,
        heap: { before: heapBefore, after_load: heapAfterLoad, after_calls: heap() },
        resources: performance.getEntriesByType("resource").map((entry: any) => ({
          path: new URL(entry.name).pathname, transferSize: entry.transferSize,
          encodedBodySize: entry.encodedBodySize, decodedBodySize: entry.decodedBodySize,
          duration_ms: entry.duration,
        })),
      };
    }, { threads, audioUrls });
    const result = report.results;
    report.ort_loaded_assets = result.resources
      .filter((row: any) => row.path.startsWith("/web/node_modules/onnxruntime-web/dist/"))
      .map((row: any) => {
        const bytes = readFileSync(path.join(root, row.path.slice(1)));
        return { path: row.path, bytes: bytes.length, sha256: hash(bytes) };
      });
    const expectedThreads = threads ?? (isolated ? Math.min(4, report.environment.hardwareConcurrency || 1) : 1);
    expect(result.threads).toBe(expectedThreads);
    expect(result.baseline.num_threads).toBe(expectedThreads);
    expect(result.baseline.cross_origin_isolated).toBe(isolated);
    expect(result.baseline.runs).toBe(3);
    expect(result.baseline.simd).toBe(true);
    expect(result.single.text.trim().length).toBeGreaterThan(0);
    expect(result.single.outFrames).toBeGreaterThan(0);
    expect(result.single.decoder).toBe("beam");
    expect(result.single.guarded).not.toBe(true);
    expect(result.invalid.type).toMatch(/Float32Array/);
    expect(result.invalid.short).toMatch(/too short/);
    expect(result.invalid.nan).toMatch(/finite/);
    expect(result.invalid.infinity).toMatch(/finite/);
    expect(result.overlap).toMatch(/already in progress/);
    for (const row of [result.firstAfterOverlap, result.recovered, result.afterLong, result.contextual.noOptions, result.contextual.emptyContext, result.contextual.zeroBonus]) expect(row.text).toBe(result.single.text);
    for (const row of [result.silence, result.tone]) {
      expect(row.text).toBe("");
      expect(row.guarded).toBe(true);
      expect(row.outFrames).toBe(0);
    }
    expect(result.longRejection).toMatch(/segment.*max 30 s.*496000/);
    for (const row of result.extraAudio) {
      expect(row.result.text.trim().length, row.label).toBeGreaterThan(0);
      expect(row.result.outFrames, row.label).toBeGreaterThan(0);
    }
    expect(result.contextual.result.text.trim().length).toBeGreaterThan(0);
    expect(result.contextual.result.outFrames).toBeGreaterThan(0);
    const scorerUse = result.contextual.scorerUse;
    expect(scorerUse.no_options).toEqual({ delta_calls: 0, finish_calls: 0, positive_scores: 0, settings: [] });
    for (const [label, bonus, terminals] of [
      ["empty_context", 1, []],
      ["zero_bonus", 0, result.contextual.sequences.map((ids: number[]) => ids.join(","))],
      ["positive_bonus", 1.75, result.contextual.sequences.map((ids: number[]) => ids.join(","))],
    ] as const) {
      expect(scorerUse[label].delta_calls, `${label} delta use`).toBeGreaterThan(0);
      expect(scorerUse[label].finish_calls, `${label} finish use`).toBeGreaterThan(0);
      expect(scorerUse[label].settings).toEqual([{ bonus, prefixBonus: 0, terminals }]);
    }
    expect(scorerUse.empty_context.positive_scores).toBe(0);
    expect(scorerUse.zero_bonus.positive_scores).toBe(0);
    expect(scorerUse.positive_bonus.positive_scores, "real scorer must award a positive completion bonus").toBeGreaterThan(0);
    for (const timing of Object.values(result.pairedTimings) as any[]) {
      expect(timing.samples).toHaveLength(3);
      for (const sample of timing.samples) expect(Number.isFinite(sample.wall_ms) && sample.wall_ms > 0).toBe(true);
    }
    for (const asset of report.assets) expect(asset.fulfilled_responses, asset.name).toBe(1);
    expect(result.resources.some((row: any) => row.path.endsWith(".wasm"))).toBe(true);
    expect(errors).toEqual([]);
    report.passed = true;
  } finally {
    report.browser_errors = errors;
    // Keep only portable measurements; never serialize environment variables, local paths or stacks.
    const json = JSON.stringify(report, null, 2) + "\n";
    expect(json).not.toMatch(/[A-Za-z]:\\|\/Users\/|\/home\//);
    mkdirSync(reportDir, { recursive: true });
    writeFileSync(path.join(reportDir, `${info.project.name}.json`), json);
    await info.attach("real-wasm-acceptance", { body: json, contentType: "application/json" });
  }
});
