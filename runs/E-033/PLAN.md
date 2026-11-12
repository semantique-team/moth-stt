# E-033 — InterCTC fine-tune: one auxiliary loss, two paired arms

**Status:** planned 2026-11-12, **not executed**. Awaiting owner approval of the gates below.
This is step 2 of the [model-improvement strategy](../../docs/plans/2026-11-12-model-improvement-strategy.md);
it changes **one** training variable and no data, tokenizer, decoder or asset.

## Why this arm exists

The strategy puts InterCTC first because it is the only listed option that is training-only:
the auxiliary head is created by the trainer (`stt/train.py`, "Training-owned auxiliary head;
never registered on the deployable model"), the exported graph and the 19,706,289-byte package
cannot change, and the recipe is already implemented and tested. The historical cost probe
measured **+4.4% step time** ([E-024 PLAN](../E-024/PLAN.md), `probe_cost.py`); that is a cost
measurement, not a WER result, and there is no evidence yet that InterCTC helps Moth.

E-015 and E-032 both bought target gains with general regressions by adding or reweighting
**data**. E-033 asks a different question: with the data held fixed, does an auxiliary CTC loss
on the encoder output improve the fine-tune enough to survive a paired comparison? A negative
answer closes this option cheaply; a positive answer is a recipe change that costs no bytes.

## Design

Two arms, same data, same recipe, same seed, same init; only `train.interctc` differs.

| Arm | Config | `train.interctc` |
|---|---|---|
| control | `runs/E-033/control/config.yaml` | absent |
| interctc | `runs/E-033/interctc/config.yaml` | `weight: 0.3`, `tap: encoder_output` |

Both are re-run here rather than compared to the retained E-028 run: E-028 and E-033 would
otherwise differ by session, torch build and non-determinism, and the whole point of this arm is
a paired dev comparison. The retained E-028 result (**dev macro 31.01**, and the published 0.8.1
INT8 readout under the E-030 decoder and E-031 LM) stays as a historical reference, not as this
arm's control.

The **E-024 best checkpoint (0.7.0)** is also a candidate in every selection: a short fine-tune
is allowed to conclude that the initial checkpoint is still the best model. Both arms start from
it, so "no new model" is a legitimate outcome of the arm and of the control.

## Data

Identical to E-028, unchanged:

- Common Voice IT clean/noisy 50/50, MLS IT, LibriSpeech EN, spliced IT+EN, old gold x50;
- `data/shards/gold-terms-x20` (50 S1 sentences, 0.84 h);
- `data/shards/gold-terms-S2-x20` (50 S2 sentences, 0.70 h);
- dev: MLS-IT dev (1,248 clips); dev_en: LibriSpeech-EN dev (2,691 clips);
- no T/A/S3/FLEURS audio; no new recordings; the S1 E-032 addendum stays out of this run.

## Recipe

| Item | Value |
|---|---|
| Init | `runs/E-024/checkpoints/ckpt_best.pt` (0.7.0) |
| Architecture | Whisper-tiny encoder + 2 blocks + linear CTC, vocab 1024 |
| Epochs | 1 (≈13,283 steps at the E-028 batch plan) |
| Batch | 192 s / 32 clips |
| Optimizer | AdamW, head 1e-4, encoder 1e-5, betas 0.9/0.98, eps 1e-8, wd 0.01 |
| Schedule | warmup 500, min_factor 0.05 |
| Augmentation / precision | full, bf16 autocast, grad clip 1.0 |
| Eval | greedy dev every 1,000 steps; checkpoints every 10 min, keep 3 |
| Seed | 0 |
| InterCTC (arm only) | aux CTC after the encoder, weight 0.3, tap `encoder_output` |

## Selection and gates

**Selection is dev-only**: macro WER on MLS-IT dev (1,248) + LibriSpeech-EN dev (2,691), the
same metric E-028 selected on. Dev is development data, not a test, and it has informed previous
runs; it cannot certify a product gain.

1. The **arm advances** only if its best dev macro is lower than the control's best dev macro by
   **at least 0.05 pp** and the paired per-utterance bootstrap (1,000 resamples, seed 0) has an
   upper 95% bound **< 0**. A tie or a control win closes the InterCTC option for this recipe.
2. If the arm advances (or the initial checkpoint wins both), the retained readout runs with the
   **published 0.8.1 stack**: INT8 export, E-030 decoder (beam 12 / top-k 10 / LM weight 0.55 /
   token bonus 2.7), E-031 LM. Comparison against 0.8.1 on the same sets, paired CIs:

| Gate | Required |
|---|---|
| T-600 WER | upper 95% paired CI of the difference **< +0.30 pp** |
| T-S3 WER | upper 95% paired CI **< +0.50 pp** |
| FLEURS-IT / FLEURS-EN | no significant regression (paired CI excluding zero against us fails) |
| Natural terms | **≥ 431 / 817** (0.8.1 is 432) |
| Rare presence | **≥ 24 / 47** (not worse) |
| All-span rare ETA | reported, not gated (38/81 in 0.8.1) |
| Noise (E-011 set) | music and non-music reported separately; non-music emitted words **not higher** than 0.8.1, or the difference declared a tie |
| Package | unchanged **19,706,289 B**; no new asset, no LM retrain |

These margins are proposals for approval, not results, and they are deliberately narrow because
the intervention is supposed to be free: a training-only loss that costs general WER has no
product argument. Aligned target-only ETA is reported next to the all-span number, never instead
of it.

## Non-goals

No data or tokenizer change, no new LM, no decoder retune, no quantization change, no browser
re-measurement beyond the existing acceptance path, and no claim about new speakers. InterCTC
does not address the mixed-dev gap; that remains the strategy's first priority.

## Cost

≈13.3k steps per arm; at the E-024 measured pace (~0.5 s/step, +4.4% for the arm) that is
**≈2 h per arm on the historical M5**, plus dev evals and the INT8 readout, so **≈5 h** for both
arms. Measure it; do not quote it as a result. Run with `caffeinate -i` on macOS, checkpoints
every 10 min.

## Decision

- Arm fails step 1: report "no dev benefit from InterCTC" and close the option; nothing is
  exported.
- Arm advances but fails a product gate: reject the arm (as E-015/E-032) and keep 0.8.1.
- Arm advances and passes every gate: adopt only after the retained readout is rerun and recorded.

## Artifacts

`runs/E-033/` — `PLAN.md`, `control/` and `interctc/` (config, `train.log.jsonl`, `train.out`,
checkpoints gitignored), `preflight.log`, and the selection readout. The `EXPERIMENTS.md` entry is
written when the run exists, not before. Pre-registration commit: the commit that adds this file.
