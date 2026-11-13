# E-037 — Paired fp32/INT8 readout with the shipped A3 LM

**Status:** pre-registered 2026-11-13, **before** exporting fp32 or decoding any set. Decode-only:
no training, no weights, no new data. This is closure evidence, not an adoption experiment.

## Question

The published 0.8.1 artifact is the **INT8** graph (11,910,677 B, sha `7bbfad7e…`) with the E-031
A3 LM. The fp32 graph is 46.37 MB and cannot ship (package ~53.9 MB), so the release has always
assumed a small quantization cost. E-028 measured that cost on CV-IT/FLEURS dev with the 0.8.0 LM
(30.93 → 31.14, +0.21 pp); it was never re-measured with the **shipped A3 LM** on the product
sets. This experiment quantifies it:

> With the same E-028 checkpoint, the same A3 LM, the same decoder and the same clips, how much
> WER/ETA does the shipped INT8 graph cost relative to its fp32 export?

The answer is a **measurement**. INT8 remains the only deployable format; there is no gate that
can switch the release to fp32, and no recipe is tuned here.

## Assets

| Role | Asset | Identity |
|---|---|---|
| Weights | `runs/E-028/checkpoints/ckpt_best.pt` | sha256 `01b83cc3…` |
| INT8 graph | `runs/E-028/release/model.int8.onnx` | 11,910,677 B, sha256 `7bbfad7e…` (the 0.8.1 graph) |
| fp32 graph | fresh `stt.export.export_fp32` of the same checkpoint | size/hash recorded; not shipped |
| LM | `runs/E-031/build/frozen_1_2_terms.bin` | 5,381,992 B, sha256 `3be1177b…` |
| Vocab / mel | `release/vocab.json`, `release/mel_filters.bin` | identical for both graphs |
| Decoder | beam 12 / top-k 10 / LM 0.55 / bonus 2.7 | E-030/E-031/E-035 |

The fp32 export is verified against PyTorch with `stt.export.verify_fp32` (the same export gate:
median ≤1e-3, worst ≤1e-2 on the synthetic 1/5/10/29 s draws). The INT8 graph is the shipped one
and is **not** re-quantized, so the comparison is about the artifact users download.

## Sets And Protocol

Both graphs are decoded in one session with the same A3 LM and decoder:

- **dev** (secondary, larger sample): `mls_it_dev` 1,248 + `ls_en_dev` 2,691 = 3,939 clips;
  macro WER and paired within-language bootstrap.
- **product** (primary): T-600 (600), T-S3 (S3 subset), FLEURS-IT test (697), FLEURS-EN test
  (525), noise (457), rare 47 and the natural/gold term groups.

Metrics: pooled WER/ETA, per-language dev WER, rare aligned ETA (81 span tokens), natural/gold
counts, noise words/non-empty clips, and a **paired per-utterance bootstrap** (1,000 resamples,
seed 0) for INT8 − fp32 on every WER/ETA. Positive Δ means INT8 is worse. The logit-level gap is
also measured on the frozen `web/tests/fixtures/speech.wav` (max abs, top-1 agreement) and on a
small fixed dev sample, to separate decoder behaviour from acoustic quantization.

## Interpretation Rule

- Report the INT8 cost with 95% CIs on every metric; a tie is a tie, not "zero cost".
- **Do not change the release**: fp32 is 46.37 MB and cannot fit 20 MB; INT8 remains. If the cost
  is significant, the result goes into the model card/paper caveat, not into a new artifact.
- No selection, no tuning, no product claim beyond the measured sets. T and S3 are reused
  evidence; this experiment does not make them pristine.

## Expected Size

E-028's dev comparison with the older LM was **+0.21 pp** (30.93 → 31.14). With A3 the acoustic
graph is unchanged, so the expected product cost is of the same order (≈0.1–0.5 pp WER), with the
largest effect likely on T-600 and the rare terms. A larger or smaller value is reported as
measured; no threshold is pre-committed.

## Artifacts And Cost

`runs/E-037/` — `PLAN.md`, `export_fp32.py`, `fp32.json` (size/hash/verification), `readout.py`,
`readout.json`, hypotheses/metrics for both graphs, `RESULT.md`, and the `EXPERIMENTS.md` entry.
The 46 MB fp32 ONNX is reproducible and gitignored. Cost: one export + two decodes over dev and
product, estimated **~30–40 min**, no training, no download, 0 USD.

Pre-registration commit: the commit that adds this file, before the export and the first decode.
