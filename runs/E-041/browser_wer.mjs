/**
 * E-041 — decode T clips with the pinned 0.8.1 bundle and exact local browser runtime.
 *
 * Serves the repository with `web/server.py` (COOP/COEP, WASM threads), loads `web/ort.js` in a
 * headless Chromium page, and transcribes every clip with the release's own config (beam 12,
 * top-k 10, LM 0.55, bonus 2.7, VAD gate 0.1). Writes hypotheses and per-clip timings as JSONL.
 *
 *   node runs/E-041/browser_wer.mjs --name browser-1 [--limit 3] [--release runs/E-031/release]
 *
 * Read-only: nothing is trained, tuned or written outside the run directory.
 */

import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { createServer } from "node:net";
import { createHash } from "node:crypto";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../..");
const expected = JSON.parse(readFileSync(path.join(repo, "runs/E-031/release-0.8.1.json"), "utf8"));
const require = createRequire(path.join(repo, "web/package.json"));
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const name = option("name", null);
const limit = Number(option("limit", "0"));
const release = option("release", "runs/E-031/release");
const port = Number(option("port", "8781"));
if (!name || /[/\\]/.test(name) || name === "." || name === "..") {
  console.error("usage: node runs/E-041/browser_wer.mjs --name <dir> [--limit N] [--release DIR]");
  process.exit(2);
}
if (!Number.isInteger(limit) || limit < 0 || !Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error("--limit must be nonnegative and --port must be 1..65535 integers");
}
const releaseDir = path.resolve(repo, release);
if (releaseDir === repo || !releaseDir.startsWith(`${repo}${path.sep}`)) {
  throw new Error("--release must be a directory inside the repository");
}
const releaseBase = `/${path.relative(repo, releaseDir).split(path.sep).join("/")}/`;

const outDir = path.join(here, "readout", name);
if (existsSync(outDir)) {
  console.error(`${outDir} exists; use a new --name`);
  process.exit(2);
}

const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const manifestPath = "recordings/manifests/T.jsonl";
const manifestBytes = readFileSync(path.join(repo, manifestPath));
const manifest = manifestBytes.toString("utf8")
  .split("\n")
  .filter((line) => line.trim())
  .map((line) => JSON.parse(line));
const rows = limit > 0 ? manifest.slice(0, limit) : manifest;
const runtimeReceipts = new Map([
  "runs/E-041/browser_wer.mjs", "runs/E-041/score.py", "web/server.py",
].map((file) => [file, { path: file, sha256: digest(readFileSync(path.join(repo, file))) }]));
const audioReceipts = new Map();
const localPython = path.join(repo, process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python");
const python = existsSync(localPython) ? localPython : process.platform === "win32" ? "python" : "python3";

await new Promise((resolve, reject) => {
  const probe = createServer();
  probe.once("error", (error) => reject(new Error(`port ${port} is unavailable: ${error.message}`)));
  probe.listen({ host: "127.0.0.1", port, exclusive: true }, () => probe.close(resolve));
});
const server = spawn(
  python,
  ["web/server.py", "--root", ".", "--host", "127.0.0.1", "--port", String(port)],
  { cwd: repo, stdio: ["ignore", "pipe", "pipe"] },
);
const base = `http://127.0.0.1:${port}`;
// server.py announces readiness only after binding: an occupied port must fail, not be reused.
const serverReady = new Promise((resolve, reject) => {
  let stdout = "";
  let stderr = "";
  const timeout = setTimeout(() => reject(new Error("server did not start")), 10000);
  server.stdout.on("data", (chunk) => {
    stdout += chunk;
    if (stdout.includes(` on ${base}`)) {
      clearTimeout(timeout);
      resolve();
    }
  });
  server.stderr.on("data", (chunk) => { stderr += chunk; });
  server.once("error", (error) => { clearTimeout(timeout); reject(error); });
  server.once("exit", (code) => {
    clearTimeout(timeout);
    reject(new Error(`server exited (${code}); port may be occupied: ${stderr}`));
  });
});

const hypotheses = [];
const timings = [];
let browser;
let failures = 0;
try {
  await serverReady;
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ serviceWorkers: "block" });
  const errors = [];
  // Hash and fulfill the same bytes: a second fetch would not bind what was actually decoded.
  await context.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    const file = decodeURIComponent(url.pathname).replace(/^\//, "");
    if (url.origin !== base || !(
      file.startsWith("recordings/audio/") ||
      (file.startsWith("web/") && /\.(?:js|mjs|wasm)$/.test(file))
    )) return route.continue();
    try {
      const response = await route.fetch();
      if (!response.ok()) throw new Error(`${file}: HTTP ${response.status()}`);
      const body = await response.body();
      const receipts = file.startsWith("recordings/audio/") ? audioReceipts : runtimeReceipts;
      const receipt = { path: file, sha256: digest(body) };
      if (receipts.has(file) && receipts.get(file).sha256 !== receipt.sha256) {
        throw new Error(`${file}: changed during measurement`);
      }
      receipts.set(file, receipt);
      await route.fulfill({ response, body });
    } catch (error) {
      errors.push(`receipt: ${error.message}`);
      await route.abort();
    }
  });
  const page = await context.newPage();
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  await page.route("**/e041.html", (route) =>
    route.fulfill({
      contentType: "text/html",
      // The static server sends these on real files; a fulfilled route must repeat them, otherwise
      // SharedArrayBuffer stays disabled and ORT-Web falls back to a single thread.
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Cache-Control": "no-store",
      },
      body: "<!doctype html><meta charset=utf-8><title>E-041 browser WER</title>",
    }),
  );
  await page.goto(`${base}/e041.html`);

  const provenance = await page.evaluate(async ({ releaseBase, expected }) => {
    const response = await fetch(`${releaseBase}config.json`);
    if (!response.ok) throw new Error("cannot fetch release config");
    const bytes = await response.arrayBuffer();
    const configSha256 = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
      .map((byte) => byte.toString(16).padStart(2, "0")).join("");
    if (configSha256 !== expected.files["config.json"].sha256) {
      throw new Error("expected immutable E-031 0.8.1 config");
    }
    const { createRunner } = await import("/web/ort.js");
    const runner = await createRunner({ baseUrl: releaseBase });
    if (JSON.stringify(runner.config) !== JSON.stringify(JSON.parse(new TextDecoder().decode(bytes)))) {
      throw new Error("release config changed while loading runner");
    }
    globalThis.__e041 = runner;
    return {
      version: runner.config.version,
      decode: runner.config.decode,
      model: runner.config.model.file,
      files: runner.config.files,
      config_sha256: configSha256,
      vad: runner.config.vad,
      segmentation: runner.config.decode.segment ?? true,
      user_agent: navigator.userAgent,
      hardware_concurrency: navigator.hardwareConcurrency,
      cross_origin_isolated: globalThis.crossOriginIsolated,
    };
  }, { releaseBase, expected });
  if (provenance.version !== "0.8.1" || provenance.model !== "model.int8.onnx" ||
      !provenance.cross_origin_isolated || provenance.segmentation !== true ||
      !isDeepStrictEqual(provenance.vad, expected.vad)) {
    throw new Error("expected isolated E-031 0.8.1 runner with shipped VAD/segmentation");
  }
  for (const [key, value] of Object.entries(expected.decode)) {
    if (provenance.decode[key] !== value) throw new Error(`unexpected decoder setting: ${key}`);
  }
  for (const file of ["model.int8.onnx", "lm_bpe3.bin", "vocab.json", "mel_filters.bin", "silero_vad.onnx"]) {
    if (provenance.files[file]?.replace(/^sha256:/, "") !== expected.files[file].sha256) {
      throw new Error(`expected immutable E-031 0.8.1 ${file}`);
    }
  }

  const started = Date.now();
  for (const [index, row] of rows.entries()) {
    const url = `/recordings/audio/${row.audio.split("/").map(encodeURIComponent).join("/")}`;
    const result = await page.evaluate(async ({ audioUrl }) => {
      const { decodeTo16k } = await import("/web/ort.js");
      const pcm = await decodeTo16k(audioUrl);
      const before = performance.now();
      const transcription = await globalThis.__e041.transcribe(pcm);
      return {
        text: transcription.text,
        wall_ms: performance.now() - before,
        timings: transcription.timings ?? null,
        audio_seconds: pcm.length / 16000,
      };
    }, { audioUrl: url });
    if (!result || typeof result.text !== "string") {
      failures += 1;
      throw new Error(`${row.id}: transcribe returned no text`);
    }
    hypotheses.push({ id: row.id, hyp: result.text, speaker: row.speaker });
    timings.push({
      id: row.id,
      speaker: row.speaker,
      audio_seconds: result.audio_seconds,
      wall_ms: result.wall_ms,
      rtfx: result.audio_seconds / (result.wall_ms / 1000),
      timings: result.timings,
    });
    if ((index + 1) % 25 === 0 || index + 1 === rows.length) {
      const elapsed = (Date.now() - started) / 1000;
      const audio = timings.reduce((sum, item) => sum + item.audio_seconds, 0);
      console.log(
        `${index + 1}/${rows.length} clips | ${elapsed.toFixed(1)} s wall | ` +
          `audio ${audio.toFixed(1)} s | RTFx ${(audio / elapsed).toFixed(2)} | errors ${errors.length}`,
      );
    }
    if (errors.length) throw new Error(`${row.id}: ${errors[0]}`);
  }

  const ortRoot = "web/node_modules/onnxruntime-web/dist/";
  for (const file of ["web/ort.js", "web/mel.js", "web/decode.js", "web/lm.js", "web/vad.js",
    `${ortRoot}ort.wasm.min.mjs`]) {
    if (!runtimeReceipts.has(file)) throw new Error(`missing loaded runtime receipt: ${file}`);
  }
  if (![...runtimeReceipts.keys()].some((file) => file.startsWith(ortRoot) && file.endsWith(".wasm"))) {
    throw new Error("missing loaded ORT WASM receipt");
  }
  if (![...runtimeReceipts.keys()].some((file) => file.startsWith(ortRoot) &&
      /ort-wasm.*\.mjs$/.test(file))) {
    throw new Error("missing loaded ORT WASM loader receipt");
  }
  if (errors.length) throw new Error(errors[0]);
  const measurement = {
    manifest: { path: manifestPath, sha256: digest(manifestBytes) },
    audio: rows.map((row) => {
      const receipt = audioReceipts.get(`recordings/audio/${row.audio}`);
      if (!receipt) throw new Error(`${row.id}: missing decoded audio receipt`);
      return { id: row.id, ...receipt };
    }),
    runtime: [...runtimeReceipts.values()].sort((a, b) => a.path.localeCompare(b.path)),
    runtime_note: "exact local runtime file snapshot; not immutable deployed code",
  };
  for (const receipt of [measurement.manifest, ...measurement.audio, ...measurement.runtime]) {
    if (digest(readFileSync(path.join(repo, receipt.path))) !== receipt.sha256) {
      throw new Error(`${receipt.path}: local file changed during measurement`);
    }
  }
  mkdirSync(outDir, { recursive: true });
  writeFileSync(
    path.join(outDir, "browser.hyp.jsonl"),
    hypotheses.map((row) => JSON.stringify(row)).join("\n") + "\n",
    "utf8",
  );
  writeFileSync(
    path.join(outDir, "browser.timings.jsonl"),
    timings.map((row) => JSON.stringify(row)).join("\n") + "\n",
    "utf8",
  );
  const summary = timings.map((row) => row.rtfx).sort((a, b) => a - b);
  writeFileSync(
    path.join(outDir, "browser.run.json"),
    JSON.stringify(
      {
        name,
        release,
        provenance,
        measurement,
        clips: rows.length,
        failures,
        wall_seconds: (Date.now() - started) / 1000,
        audio_seconds: timings.reduce((sum, item) => sum + item.audio_seconds, 0),
        rtfx: {
          median: summary[Math.floor(summary.length / 2)],
          min: summary[0],
          max: summary[summary.length - 1],
          mean: summary.reduce((sum, value) => sum + value, 0) / summary.length,
        },
        note: "exact local browser runtime + pinned 0.8.1 bundle with VAD guard/segmentation; E-037 uses cached ungated mel clips; not immutable deployed code, isolated acoustic parity or a phone/Safari claim",
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );
  console.log(JSON.stringify({ clips: rows.length, wall_seconds: (Date.now() - started) / 1000 }));
} finally {
  if (browser) await browser.close();
  server.kill("SIGTERM");
}
