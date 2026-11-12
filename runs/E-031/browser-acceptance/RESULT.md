# 0.8.1 release candidate — real Chromium/WASM acceptance

> Run 2026-11-12 on the staged **0.8.1** bundle (0.8.0 assets + the E-031 A3 LM) with decoder
> **beam 12 / top-k 10 / LM 0.55 / bonus 2.7**. **4/4 projects passed.**

## Environment

| Item | Value |
|---|---|
| Package | 19,706,289 bytes |
| Release dir | `runs/E-031/release/` (staged, not the published `release/`) |
| Model | `model.int8.onnx`, sha256 `7bbfad7e288640538c5c0147f81738bfd63021727b8760b1f3cbbcc4f939764d` (unchanged) |
| LM | `lm_bpe3.bin`, sha256 `3be1177b0ca9f18343efb8d00f0d96365226595c9de32d992720aa4e0b0f16f1`, 5,381,992 B |
| Config | version `0.8.1`, LM hash updated; all runtime asset hashes verified by the spec |
| Browser | HeadlessChrome 156.0.8078.4, Chromium via Playwright |
| Platform | macOS Darwin, 10 logical cores |
| Audio | versioned public Italian fixture `web/tests/fixtures/speech.wav` (96,078 bytes) |

## Results

| Project | Isolated | Threads | RTFx median | Single-clip RTFx | Passed |
|---|---|---:|---:|---:|---|
| isolated-auto | yes | 4 | 46.69 | 25.30 | yes |
| isolated-1 | yes | 1 | 25.10 | 19.07 | yes |
| isolated-4 | yes | 4 | 46.23 | 25.37 | yes |
| nonisolated-auto | no | 1 | 24.83 | 18.86 | yes |

All four projects produced the **same transcript** for the fixture:
`le diffuseo proteste iniziate a ottobrek con testa`. RTFx is the browser-reported median over 3
runs on the 3 s fixture; it is not a full release-quality benchmark. The run also checks hash
verification of every runtime asset, input guards, overlap/recovery, silence/tone gating, the 30 s
`segment: false` rejection, and contextual-scorer instrumentation.

## Parity

`runs/E-031/browser-parity.json` and `decoder-parity/parity.json`:

- browser ↔ Python on the fixture is **not exact** (browser `…con testa`, Python `…conte`,
  2/8 words, 25% WER on this 3 s clip);
- the **0.8.0 control run shows the identical mismatch** (`browser-parity-0.8.0.json`), so the
  new LM does not introduce it;
- on **identical Python logits**, the JS beam search + LM reader and Python `beam_search` +
  `BinaryBPELM` produce **exactly the same text** (150 frames, decoder parity true).

The residual difference is in the browser acoustic path (JS mel / ORT / VAD), pre-existing and
already measured historically as a tie on 200 T clips in E-022. It is not an LM binary issue.

## Limits

- One public 3 s Italian fixture; no IT/EN/spliced evaluation manifest was available on this
  machine. Runtime acceptance is not a WER/ETA measurement.
- No independent natural mixed-dev; the 0.8.1 candidate still relies on T/S3 evidence.
- The non-isolated server has no COOP/COEP, so it falls back to 1 thread, as expected.

## Reproduction

```bash
uv run python runs/E-031/stage_release.py
cd web
PYTHON=python3 \
MOTH_RELEASE_DIR="$(cd .. && pwd)/runs/E-031/release" \
MOTH_ACCEPTANCE_REPORT_DIR="$(cd .. && pwd)/runs/E-031/browser-acceptance" \
npm run test:acceptance

uv run python runs/E-031/browser_parity.py
uv run python runs/E-031/decoder_parity.py
```
