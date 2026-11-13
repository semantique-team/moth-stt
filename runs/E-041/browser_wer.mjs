/**
 * E-041 — decode T clips through the shipped 0.8.1 release in a real browser.
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

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../..");
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

const outDir = path.join(here, "readout", name);
if (existsSync(outDir)) {
  console.error(`${outDir} exists; use a new --name`);
  process.exit(2);
}
mkdirSync(outDir, { recursive: true });

const manifest = readFileSync(path.join(repo, "recordings/manifests/T.jsonl"), "utf8")
  .split("\n")
  .filter((line) => line.trim())
  .map((line) => JSON.parse(line));
const rows = limit > 0 ? manifest.slice(0, limit) : manifest;
const python = existsSync(path.join(repo, ".venv/bin/python"))
  ? path.join(repo, ".venv/bin/python")
  : "python3";

const server = spawn(
  python,
  ["web/server.py", "--root", ".", "--host", "127.0.0.1", "--port", String(port)],
  { cwd: repo, stdio: ["ignore", "pipe", "pipe"] },
);
const base = `http://127.0.0.1:${port}`;

async function waitForServer() {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const response = await fetch(`${base}/web/ort.js`);
      if (response.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("server did not start");
}

const hypotheses = [];
const timings = [];
let browser;
let failures = 0;
try {
  await waitForServer();
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
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

  const provenance = await page.evaluate(async ({ releaseBase }) => {
    const { createRunner } = await import("/web/ort.js");
    const runner = await createRunner({ baseUrl: releaseBase });
    globalThis.__e041 = runner;
    return {
      version: runner.config.version,
      decode: runner.config.decode,
      model: runner.config.model.file,
      files: runner.config.files,
      user_agent: navigator.userAgent,
      hardware_concurrency: navigator.hardwareConcurrency,
      cross_origin_isolated: globalThis.crossOriginIsolated,
    };
  }, { releaseBase: `/${release.replace(/^\/+|\/+$/g, "")}/` });

  const started = Date.now();
  for (const [index, row] of rows.entries()) {
    const url = `/recordings/audio/${row.audio}`;
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
        note: "browser-only measurement: headless Chromium + WASM, not a phone or Safari claim",
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
