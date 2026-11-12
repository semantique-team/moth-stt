# 0.8.0 — real Chromium/WASM acceptance

> Run 2026-11-12 on the published 0.8.0 package with decoder **beam 12 / top-k 10 / LM 0.55 /
> bonus 2.7**. **4/4 projects passed.** Full per-project JSON is in this directory.

## Environment

| Item | Value |
|---|---|
| Package | 19,440,761 bytes |
| Model | `model.int8.onnx`, sha256 `7bbfad7e288640538c5c0147f81738bfd63021727b8760b1f3cbbcc4f939764d` |
| Decoder | beam 12 / top-k 10 / LM 0.55 / token bonus 2.7 |
| Browser | HeadlessChrome 156.0.8078.4, Chromium via Playwright |
| Platform | macOS Darwin, 10 logical cores |
| Code | `3f5a4711e7dd636b63f3a2b2f388f2c859fcb39e` + the release commit |
| Audio | versioned public Italian fixture `web/tests/fixtures/speech.wav` (96,078 bytes) |

## Results

| Project | Isolated | Threads | RTFx median | Single-clip RTFx | Passed |
|---|---|---:|---:|---:|---|
| isolated-auto | yes | 4 | 46.52 | 24.66 | yes |
| isolated-1 | yes | 1 | 25.66 | 19.14 | yes |
| isolated-4 | yes | 4 | 46.42 | 25.17 | yes |
| nonisolated-auto | no | 1 | 25.40 | 19.05 | yes |

RTFx is the browser-reported median over 3 runs on the 3 s fixture; it is not a full release
quality benchmark. The acceptance also checks hash verification of every runtime asset, input
guards (type/length/NaN/Infinity), overlap rejection and recovery, silence/tone behaviour, the
30 s `segment: false` rejection, and contextual-scorer instrumentation. See the per-project JSON
for the raw timings, asset byte counts, heap snapshots and contextual counters.

## Limits

- One public 3 s Italian fixture; the optional evaluation-audio manifest was not present on this
  machine, so no IT/EN/spliced extra clips were exercised.
- Runtime acceptance is not a WER/ETA measurement and does not replace T, S3 or a natural
  mixed-dev.
- The non-isolated server has no COOP/COEP, so it falls back to 1 thread, as expected.

## Reproduction

```bash
cd web
PYTHON=python3 \
MOTH_RELEASE_DIR="$(cd .. && pwd)/release" \
MOTH_ACCEPTANCE_REPORT_DIR="$(cd .. && pwd)/runs/E-028/browser-acceptance" \
npm run test:acceptance
```

The report directory defaults to `runs/browser-acceptance-2026-11-10`; this run used
`MOTH_ACCEPTANCE_REPORT_DIR` so the 0.7.0 reports stay untouched.
