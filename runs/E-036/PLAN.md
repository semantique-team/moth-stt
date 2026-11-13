# E-036 — Conservative Fine-Tune: Frozen Layers, Replay And Distillation

**Status:** pre-registered 2026-11-13, **before** training. Two arms, one seed each, no sweep of
data or decoder. Training data is the E-032 mixture; no new audio is recorded or downloaded.

## Question

E-032 added 50 S1 sentences for the never-seen terms and improved rare aligned ETA from 23/81 to
26/81 (0.8.0 LM) / 42/81 (E-031 LM), but a full fine-tune regressed T-S3, FLEURS and the natural
term group. This experiment asks the protection question directly:

> If the encoder is frozen (or only its last two blocks move), the full E-028 mixture is replayed
> in every epoch, and a frozen E-028 teacher distills its output distribution into the student,
> can the S1 rare-term gain survive **without** the E-032 product regressions?

This is the training-side closure question. It is not a new-data experiment and not a sweep of
the S1 sentences: the data and the seed are fixed, and only the protection regime changes.

## Data And Recipe

Same mixture as E-028/E-032: CV-IT clean/noisy, MLS-IT, LS-EN, spliced IT/EN, gold ×50, E-015/
E-028 term sources ×20 and the 50 S1 sentences of `gold-terms3-x20`. The replay is therefore the
full 589 h mixture, not an S1-only pass. 13,314 steps per epoch, exactly as E-032.

| Item | Value |
|---|---|
| Init | `runs/E-028/checkpoints/ckpt_best.pt` (0.8.1 acoustic) |
| Teacher | the same E-028 checkpoint, frozen, eval, never registered on the student |
| Architecture | Whisper-tiny encoder + 2 blocks + linear CTC, vocab 1024 |
| Epochs | 1 (13,314 steps) |
| Batch | 192 s / 32 clips |
| Optimizer | AdamW, head 1e-4, encoder 1e-5, betas 0.9/0.98, eps 1e-8, wd 0.01 |
| Schedule | warmup 500, min_factor 0.05 |
| Augmentation | full, bf16 autocast, grad clip 1.0 |
| Eval | greedy dev every 1,000 steps; checkpoints every 10 min, keep 3 |
| Seed | 0 |
| Distillation | KL(teacher ‖ student) per utterance, masked to true frames, T = 2, weight 20 |

## Arms

| Arm | Trainable parameters | Purpose |
|---|---|---|
| `E-036-frozen` | 2 head blocks + CTC head; encoder, conv and final norm frozen | most conservative: only the output stack adapts |
| `E-036-last2` | last 2 of the 4 encoder blocks + 2 head blocks + CTC head; conv, final norm and first 2 encoder blocks frozen | allows acoustic adaptation while protecting most of the encoder |

Both arms use the same data, teacher, KD weight and seed. The initial E-028 checkpoint is
**eligible** as a candidate: an arm that only reproduces it cannot win. The teacher is loaded from
disk, `requires_grad_(False)`, excluded from optimizer groups and checkpoints by construction;
`distill_loss` is logged separately at every step.

### Distillation scale (pilot, not a result)

A 100-step training-loss pilot of the frozen arm measured a raw per-utterance KL of
**0.0007–0.0014** against CTC losses of **0.15–0.89**. A weight of 0.5 would contribute ~0.2% of
the loss and be inert. The weight is therefore frozen at **20**, which contributes **~2–5%** at
steps 10–100 and grows if the student diverges. The pilot reads no dev or product data, is not
used for selection, and its smoke logs are gitignored.

## Protocol

- Dev only for selection sanity: `mls_it_dev` + `ls_en_dev`, 3,939 clips, greedy, no LM, macro WER
  with a paired within-language bootstrap (1,000 resamples, seed 0). Report both per-language
  numbers and the initial E-028 checkpoint decoded in the same session.
- A run whose dev macro is worse than the initial checkpoint by more than **0.50 pp** is declared
  broken and gets **no** product readout; this is a guard against divergence, not a tuning gate.
- Product readout: each remaining arm is exported once to int8 (`stt.export`, same 2 head blocks,
  same vocab, `--version e036-<arm>`) and decoded once on T-600, T-S3, FLEURS-IT/EN, the 47 rare
  rows, the natural/gold groups and the noise set, with the **same 0.8.1 A3 LM** and decoder
  (beam 12 / top-k 10 / LM 0.55 / bonus 2.7). The baseline hypotheses are the committed 0.8.1
  int8 readout in `runs/E-035/readout/readout-1/baseline/`, so both arms share one baseline and
  no product set is re-read for the baseline.
- The readout is a single pass per arm; no checkpoint or recipe is changed after seeing T/FLEURS.

## Pre-Registered Gates (per arm, against 0.8.1)

All absolute gates from E-031/E-032:

1. package ≤ **20,000,000 B** (the LM is unchanged; only the int8 graph may differ by a few KiB);
2. T-600 WER ≤ **25.21%**, ETA ≥ **53.67%**;
3. T-S3 WER ≤ **25.71%**, ETA ≥ **49.64%**;
4. FLEURS-IT ≤ **20.75%**, FLEURS-EN ≤ **50.31%**;
5. natural term count ≥ **431/817**;
6. noise: `non_music` and `music` words and non-empty clips ≤ baseline (same policy as E-035);
7. decoder-level parity of the exported graph on the frozen fixture, and no non-finite training.

**Rare target:** aligned ETA over all 81 span tokens in the 47 rare rows must reach **≥ 42/81
(51.9%)**, the point level E-032 reached with the A3 LM. 40–41/81 is reported as insufficient, not
rounded up; target-only ETA and presence are reported separately.

**No significant regression:** no primary product metric may have its paired 95% CI entirely on
the worse side of zero: T-600 WER/ETA, T-S3 WER/ETA, FLEURS-IT/EN WER, rare aligned ETA.

## Decision Rule

An arm is a **candidate** only if all absolute gates pass, rare aligned ETA is ≥ 42/81, and no
primary metric is significantly worse than 0.8.1. A candidate is **not released**: it earns the
independent new-voice rare readout and the mixed-dev, which do not exist yet. If both arms fail,
E-036 is negative, the acoustic model stays at 0.8.1 and the freeze is recorded with these
numbers. No gate is renegotiated after the readout.

## Expected Size Of The Effect

The conservative arm can plausibly **preserve** the product sets but may not reach 42/81, because
the six acoustically pruned terms need encoder changes; the `last2` arm is the one with a real
chance to move them. The prior is that `last2` trades a little general performance for rare
evidence, and that `frozen` is a tie or a small rare gain. A null is useful: it says the S1
addendum cannot be made to work with this protection, and the model freezes at 0.8.1.

## What This Does Not Test

- New voices or a natural mixed-dev: neither exists; T is reused evidence and two thirds of it are
  S1/S2, so a T pass is not a generalization claim.
- A different KD weight/temperature, layer selection or data mixture: each would be a new arm.
- A release: adoption requires the independent readout after this experiment.
- Browser acoustic WER: only decoder-level parity and the package size are checked.

## Cost

Two arms × ~2 h on the M5 (13,314 steps, `caffeinate -i`), two dev evaluations (~4 min each), two
int8 exports (~2–3 min each) and two product readouts (~5 min each): **~4.5–5 h**, no downloads,
0 USD. The teacher adds one forward pass per step; the frozen encoder removes its backward pass,
so the step time is expected to stay near E-028/E-033 (~0.52 s).

## Artifacts

`runs/E-036/` — `PLAN.md`, `frozen/config.yaml`, `last2/config.yaml`, `preflight.log`, training
logs and checkpoints (checkpoints gitignored), `dev/` selection, `release-<arm>/` exports
(gitignored binaries), `eval/` readout hypotheses/metrics, `readout.json`, `RESULT.md`, and the
`EXPERIMENTS.md` entry written with the verdict. Pre-registration commit: the commit that adds
this file, before the first training step.
