# 0.8.1 candidate — browser ↔ Python parity

> Run 2026-11-12 on the staged 0.8.1 bundle and on the published 0.8.0 control. **The LM binary
> and the decoder are parity-clean; the single-fixture transcript difference is pre-existing and
> belongs to the browser acoustic path.**

## Results

| Run | Browser text | Python text | Exact | WER (browser as ref) |
|---|---|---|---|---|
| **0.8.1 candidate** | `le diffuseo proteste iniziate a ottobrek con testa` | `le diffuseo proteste iniziate a ottobrek conte` | no | 25.0% (2/8) |
| **0.8.0 control** | `le diffuseo proteste iniziate a ottobrek con testa` | `le diffuseo proteste iniziate a ottobrek conte` | no | 25.0% (2/8) |

The two runs differ only in `lm_bpe3.bin`. The mismatch is **identical** with the shipped 0.8.0
LM, so the E-031 LM does not change the browser↔Python relationship.

## Decoder-level parity

`decoder-parity/parity.json` runs the **same 150×1024 Python ONNX logits** through:

- JS `beamSearch` + `loadLM` (`web/decode.js`, `web/lm.js`), and
- Python `beam_search` + `BinaryBPELM` (`stt/decode.py`, `stt/lm.py`),

with beam 12 / top-k 10 / LM 0.55 / bonus 2.7. **Both produce exactly
`le diffuseo proteste iniziate a ottobrek conte`** (`exact_match: true`). The LM binary format,
reader and beam decoder therefore carry over.

## Interpretation

The residual browser-vs-Python difference is in the acoustic path (JS mel, ORT/WASM execution,
VAD segmentation), not in the new LM. This is consistent with E-022, where browser↔Python on 200
T clips was a **tie** (−0.09 WER, 77% identical transcripts). This one 3 s public fixture is a
runtime parity check, not a WER benchmark; it does not establish browser quality parity on a
larger sample.

## Artifacts

- `runs/E-031/browser-parity.json` — 0.8.1 candidate, four projects, all identical text.
- `runs/E-031/browser-parity-0.8.0.json` — control.
- `runs/E-031/browser-acceptance/` and `runs/E-031/browser-acceptance-0.8.0-control/` — raw
  per-project acceptance JSON (4/4 each).
- `runs/E-031/decoder-parity/` — logits, meta, parity JSON.
