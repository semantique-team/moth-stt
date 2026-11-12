# E-032 — targeted rare addendum with S1 and a gentle fine-tune from 0.8.0

**Status:** pre-registered 2026-11-12, before recording and before training. One arm, no sweep.
This experiment exists because E-026 found six terms acoustically pruned (`startup`, `prototype`,
`chromecast`, `charger`, `exam`, `mondays`) and because 0.8.0 misses the primary rare gate by two
aligned occurrences (23/81 vs 25/81).

## Why this arm exists

E-028 already added 50 S1 and 50 S2 term sentences at x20 and a gentle recipe; the int8 artifact
reaches **28.4%** rare aligned ETA. The remaining six terms are not reachable in the top-k at all,
so a decode-side change cannot recover them. This arm adds **new contexts** for those six terms
(four each) and one new context for every other never-seen term, recorded by the only training
speaker currently available, **S1**.

**Declared limitation, before the data exists:** S1 is in training and two thirds of T-600 are
S1/S2. A pass on T therefore does **not** establish transfer to a new voice. S3 has recorded
audio but is repeatedly inspected; its rare slice (2/18 at 0.7.0) is reported, not gated. A pass
here authorizes **a candidate**, not a release; release still requires the independent mixed-dev
and a new-voice rare readout described in `docs/16`.

## Data

- 50 new sentences, ids `R-S1-0692`–`R-S1-0741`, candidate rows in `runs/E-032/candidates.jsonl`
  and sheet in `runs/E-032/RECORDING.md`. All 44 terms once; the six pruned terms four times
  each (62 aligned occurrences).
- Human review before recording; then `record.py --speaker S1 --kind R --start 692`,
  `build_manifest.py build --kind R --speaker S1`.
- Term subset manifest `recordings/manifests/R-S1-terms3.jsonl` (rows 692–741), mel shard
  `data/shards/R-S1-terms3`, source `data/shards/gold-terms3-x20`.
- Base mixture unchanged from E-028 (589 h, old gold x50, S1 terms x20, S2 terms x20).
- **Built 2026-11-12:** recording validated 741/741 R-S1 clips (0 errors, 0 warnings); the new
  subset is 50 clips / 2.8 min, x20 = 0.93 h; mel shard 50 clips, 0 errors.

## Recipe

| Item | Value |
|---|---|
| Init | `runs/E-028/checkpoints/ckpt_best.pt` (0.8.0) |
| Architecture | Whisper-tiny encoder + 2 blocks + linear CTC, vocab 1024 |
| Epochs | 1 |
| Batch | 192 s / 32 clips |
| Optimizer | AdamW, head 1e-4, encoder 1e-5, betas 0.9/0.98, eps 1e-8, wd 0.01 |
| Schedule | warmup 500, min_factor 0.05 |
| Augmentation | full, bf16 autocast, grad clip 1.0 |
| Eval | greedy dev every 1,000 steps; checkpoints every 10 min, keep 3 |
| Seed | 0 |

## Pre-registered gates

Baselines are the published 0.8.0 int8 numbers. The candidate is decoded with the **same int8
graph and decoder** (beam 12 / top-k 10 / LM 0.55 / bonus 2.7), so the only new variable is the
checkpoint.

1. **Primary:** rare aligned ETA on the 47 T rows **>= 30.0% (>= 25/81)**, and above 0.8.0's 23/81.
2. T-600 WER **<= 25.21** and aligned ETA **>= 53.67**.
3. T-S3 WER **<= 25.71** and aligned ETA **>= 49.64**.
4. Natural terms **>= 431/817**.
5. FLEURS-IT/EN not worse than +0.5 with a 95% CI excluding zero.
6. The dev-selected checkpoint is **not** the step-20 baseline.
7. Package bytes unchanged (no LM or asset change in this arm).

Report, do not gate: S3 rare, per-speaker rare, the 16-term regression check, gold-only terms.

## Decision rule

Evaluate the dev-selected checkpoint and the final checkpoint if they differ; report both. If gate
1 passes and gates 2–7 pass, the arm is a **candidate** for a new-voice readout; it is not
published as 0.8.1 on this evidence. If gate 1 fails, record the negative and stop; do not iterate
the recipe on T.

## Cost

~2 h training + ~20 min evaluation, no downloads, 0 USD. `caffeinate -i` for the run.

## Artifacts

`runs/E-032/` (`PLAN.md`, `make_terms3.py`, `candidates.jsonl`, `RECORDING.md`, `config.yaml`,
`prepare_terms.py`, `train.log.jsonl`, `eval/`, `RESULT.md`).
