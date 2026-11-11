# E-028 — E-015 second arm: S2 as a second voice + a gentler fine-tune

**Status:** pre-registered 2026-11-11, before the training run. One arm, no sweep.
This is the second arm of the E-015 gold-term question; it is a new experiment ID because it
changes **two** variables at once and cannot isolate either cause.

## Why this arm exists

E-015 (first arm) added 50 S1 sentences covering the 44 never-seen terms, x20, one epoch at the
E-017 learning rates (head 3e-4 / encoder 5e-5). It was killed:

- rare presence **17.0% → 23.4%** (8/47 → 11/47, CI crosses zero);
- by speaker **S1 +1, S2 +2, held-out S3 0** — the gain stayed on the voices the model had heard;
- natural-term occurrences **439 → 423**; FLEURS-IT **+0.72**, FLEURS-EN **+1.31** significantly
  worse;
- dev was shocked **30.81 → 33.64** at step 2,000 and ended **31.22**; the best checkpoint was
  step 20, i.e. the baseline, so the run selected no new model.

E-026 corrected the readout: on the same 47 rows, aligned ETA was **23.5% → 30.9%**, while full
T-600 ETA fell **54.17% → 53.40%**. The rare gain was real but paid for on natural terms.
E-026 also found the reference path inside top-k 10 for **41/47** rows; only six are acoustically
pruned. E-027 tested a contextual bias and did not move the rare rows, so the next lever is the
acoustic/data one.

This arm therefore changes exactly the two variables the first arm could not test together:

1. **S2 as the second voice** — 50 new S2 sentences covering the same 44 terms, 2.1 min, x20
   (**0.70 h**). The S1 term source stays at x20 (0.84 h); the old gold stays at x50.
2. **A gentler fine-tune** — head **1e-4**, encoder **1e-5** (the first arm used 3e-4 / 5e-5),
   one epoch, same schedule shape. Dev is evaluated every 1,000 steps instead of 2,000 so the
   shock and recovery are visible.

The two variables are confounded by design: this is a product arm, not a causal ablation. If it
works, a later arm can separate voice count from learning rate.

## Data

Same base mixture as E-024/E-015, with the two term sources kept separate:

- Common Voice IT clean/noisy 50/50, MLS IT, LibriSpeech EN, spliced IT+EN, old gold x50;
- `data/shards/gold-terms-x20` (50 S1 sentences, 0.84 h);
- `data/shards/gold-terms-S2-x20` (50 S2 sentences, 0.70 h);
- dev: MLS-IT; dev_en: LibriSpeech-EN, unchanged.

The S2 recordings were validated with `recordings/build_manifest.py` (50/50 new clips; the
pre-existing 0.39 s `S2_0001.wav` remains the one known invalid take and is not in the manifest).
The S2 terms are **not** folded into `gold-x50`, so the old gold remains the historical snapshot.

## Recipe

| Item | Value |
|---|---|
| Init | `runs/E-024/checkpoints/ckpt_best.pt` (0.7.0) |
| Architecture | Whisper-tiny encoder + 2 extra blocks + linear CTC, vocab 1024 |
| Epochs | 1 |
| Batch | 192 s / 32 clips |
| Optimizer | AdamW, head **1e-4**, encoder **1e-5**, betas 0.9/0.98, eps 1e-8, wd 0.01 |
| Schedule | warmup 500, min_factor 0.05 |
| Augmentation | full (noise, tempo-safe, SpecAugment) |
| Precision | bf16 autocast, gradient clip 1.0 |
| Eval | greedy dev every 1,000 steps; checkpoints every 10 min, keep 3 |
| Seed | 0 |

## Pre-registered gates

**Primary: aligned ETA on the 47 rare T utterances** (the canonical ETA, not presence). The arm
passes only if:

1. rare aligned ETA **≥ 30%** (at least the first arm's rare-subset result; baseline 23.5%);
2. full T-600 aligned ETA **≥ 53.7%** (no worse than 0.7.0's 54.17% by more than 0.5 point);
3. T-600 WER **≤ 25.7%** (no worse than 25.20% by more than 0.5);
4. natural-term occurrences **≥ 431/817** (no more than 8 below the 439 baseline);
5. FLEURS-IT/EN not worse by more than 0.5 with a 95% CI excluding zero;
6. the dev-selected checkpoint is **not** the step-20 baseline; a run that selects no new model is
   a kill even if the final checkpoint looks better.

Report, but do not gate on, S3 rare (baseline 2/18), gold-only terms and the 16-term regression
check. S3 never contributes training audio.

## Evaluation

Clean gate `runs/E-022/eval.py` on T-600, T-S3, FLEURS-IT, FLEURS-EN, CV-IT dev; rare readout with
`runs/E-015/eval_terms.py` (aligned ETA primary, presence diagnostic); natural/gold/never groups and
per-speaker breakdown. If the dev curve selects a checkpoint other than step 20, evaluate that
checkpoint and the final checkpoint; otherwise evaluate the final checkpoint as the arm that saw
the data and record that the run selected no model.

## Cost

~2 h training + ~15 min evaluation, no teacher, no downloads, 0 USD. Long run on power with
`caffeinate -i`; checkpoints every 10 min.

## Artifacts

`runs/E-028/` (`PLAN.md`, `config.yaml`, `prepare_terms.py`, `train.log.jsonl`, `train.out`,
`eval/`, `RESULT.md`), `recordings/manifests/R-S2-terms.jsonl`, `data/shards/R-S2-terms`,
`data/shards/gold-terms-S2-x20`.
