# E-041 — Browser acoustic WER/ETA harness

**Status:** pre-registered 2026-11-13, **before** the first measurement run. Read-only: it decodes
existing T clips through the shipped release in a real browser. No training, no data change, no
release change.

## Question

Every published Moth quality number comes from the **Python** ONNX runner. The artifact users
actually run is the same INT8 graph inside ONNX Runtime Web with a **JavaScript** mel front end,
the JS decoder and the same LM. Acceptance measured 4/4 clips and RTFx 46.69 / 46.23 / 25.10 /
24.83, and decoder parity is exact on identical logits, but **acoustic WER/ETA has never been
measured in the browser**. The closure plan lists this as a missing closure evidence.

> With the shipped 0.8.1 bundle (INT8 `7bbfad7e…`, A3 LM `3be1177b…`, decoder 12/10/0.55/2.7),
> decoded in Headless Chromium via ONNX Runtime Web WASM, what are T-600 WER/ETA and per-clip RTFx,
> and how far are they from the Python INT8 numbers in E-037?

## Harness Under Test

- `runs/E-041/browser_wer.mjs` — Playwright + Chromium headless, serves the repository through
  `web/server.py` (COOP/COEP headers, WASM threads), loads `web/ort.js` in the page and calls
  `createRunner({ baseUrl: "/runs/E-031/release/" })`, i.e. the **0.8.1 bundle**
  (config SHA-256 and model/LM/config identities asserted against the committed immutable E-031
  `release-0.8.1.json` record, not merely the selected bundle's self-declared `config.files`).
- Runtime code is the **exact local runtime file snapshot**, not immutable deployed code. The run
  records SHA-256 of the raw manifest, each actual served/decoded audio file, runner/mel/decode/LM/
  VAD modules and the ORT JS/WASM paths actually requested, plus harness/scorer/server code. Scoring
  requires these local files to remain byte-identical and checks E-037's retained manifest digest
  when recorded. Existing E-037 T metrics do not record one, so historic audio equality is not proven.
- Audio: the 600 T clips served from `recordings/audio/<speaker>/<id>.wav`; the reference is
  `recordings/manifests/T.jsonl`. The browser decodes each clip with `decodeTo16k` and transcribes
  it with the release's own config (beam 12 / top-k 10 / LM 0.55 / bonus 2.7, VAD gate 0.1).
- The harness records, per clip: hypothesis, wall milliseconds, audio milliseconds, RTFx, and any
  page error. A page error aborts the run; nothing is silently skipped.

## Protocol

1. Smoke: 3 clips, harness must reproduce the known fixture behaviour (text non-empty, no page
   error), then the full run.
2. Full run: **T-600**, one pass, no retries, no parameter changes. Clip order is the manifest order.
3. Scoring: `runs/E-041/score.py` scores the browser hypotheses with `stt.eval.score_rows` and
   compares them with the committed E-037 INT8 Python hypotheses using a **paired per-utterance
   bootstrap** (1,000 resamples, seed 0) for WER and ETA.
4. Reported: WER, ETA, word counts, RTFx median/mean/min per clip, wall time total, count of clips
   whose hypothesis differs from the Python INT8 one, and the paired Δ with 95% CI.

## Interpretation Rule

- The browser path is a **different implementation of the same model**: a JS mel front end, JS
  decode and ORT-Web kernels, **including the shipped VAD no-speech guard and pause segmentation**.
  E-037 instead decodes **cached, ungated whole-clip mel features**, without that VAD/segmentation
  path. The comparison therefore measures the end-to-end deployment gap, **not isolated acoustic
  kernel/front-end parity**. A gap is a **deployment property**, not a model change; it must be
  reported as such and cannot be "fixed" here.
- A Δ whose CI contains zero is a **tie**, not exact equality; the number of differing hypotheses is
  reported separately so a small WER move with many text changes is visible.
- No tuning: decoder parameters come from the released `config.json`. No re-selection on these
  results.
- RTFx here is measured on this machine with headless Chromium + WASM threads; it is **not** a
  mobile, Safari, or end-user latency claim.

## What This Does And Does Not Establish

- **Does**: give the first acoustic WER/ETA measured through the deployed browser runtime, closing
  the last "same logits only" gap in the evidence for the shipped artifact.
- **Does not**: validate mobile browsers, Safari, real microphone input, VAD in live capture, or
  user-perceived latency. It also does not make T untouched or speaker-independent.

## Artifacts And Cost

`runs/E-041/` — `PLAN.md`, `browser_wer.mjs`, `score.py`, `readout/browser-1/browser.hyp.jsonl`
and `browser.timings.jsonl`, `metrics.json`, `RESULT.md`, `EXPERIMENTS.md` entry. Cost: a few
minutes of browser decoding on the local M5 (bench, not a phone), 0 USD, no downloads (Playwright
and `onnxruntime-web` are already in `web/node_modules`).
