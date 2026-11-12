# E-030 — Decoder recalibration on the E-028 INT8 artifact

**Status:** executed 2026-11-11. Dev-selected arm; T read once. Result: 5 of 6 E-028 gates pass;
the primary rare-ETA gate is missed by two aligned occurrences. Not published.

## Question

E-028 INT8 misses the E-028 adoption gates by small margins (`runs/E-028/RESULT.md`):
rare aligned ETA 29.6% vs ≥30%, T-600 ETA 53.50% vs ≥53.7%, natural 428 vs ≥431. The cause is
mostly quantization: E-028 fp32 passes all three (30.9%, 54.65%, 438). Can a decoder-only change
recover the margin without touching the weights?

## Hypothesis

E-028 INT8 deletes more words than 0.7.0 INT8 on T-600 (8.49% vs 7.88%) and inserts fewer
(1.03% vs 1.29%). The shipped decode arm (beam 12 / top-k 10 / LM 0.5 / bonus 2.4) was tuned
for 0.7.0, so E-028 may prefer a slightly different operating point. A higher `token_bonus`
counteracts deletions; a higher LM weight may help fluency, at the risk of suppressing rare
English tokens.

## Protocol

- **Model:** `runs/E-028/release/model.int8.onnx`, unchanged.
- **Selection:** 12 arms over beam / top-k / LM weight / token bonus, decoded on
  CV-IT dev 600 + FLEURS-IT dev 334 + FLEURS-EN dev 343. Primary criterion: combined dev WER;
  per-set WER, deletions and insertions reported as guards.
- **Readout:** the selected arm is decoded once on T-600 and T-S3; 0.7.0 INT8 is decoded with the
  same arm for a fair comparison. T is not used for selection.
- **Gates:** the E-028 pre-registered gates: rare aligned ETA ≥30%, T-600 ETA ≥53.7%, T-600
  WER ≤25.7%, natural ≥431/817, FLEURS-IT/EN not worse by >0.5 with a 95% CI excluding zero,
  dev-selected checkpoint not the step-20 baseline.
- **Artifact:** the package is unchanged (19,440,760 bytes); only decoder settings/config change.
  Browser acceptance must be re-run if the arm is adopted.

## Commands

```bash
uv run python runs/E-030/decode_sweep.py
uv run python runs/E-030/eval_decode.py
uv run python runs/E-015/eval_terms.py \
    --hyp runs/E-030/eval/e028-int8/t600.hyp.jsonl \
    --baseline runs/E-024/eval/gate/int8/t600.hyp.jsonl \
    --out runs/E-030/eval/e028-vs-070-old-arm.json
```

The head-fp32 check used `runs/E-028/quant/int8_head_fp32.onnx` copied into
`runs/E-030/head_fp32_release/`; it is over the 20 MB budget and was evaluated only to see whether
reducing head quantization error recovers the rare terms.
