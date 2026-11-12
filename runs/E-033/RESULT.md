# E-033 — RESULT: InterCTC at weight 0.3 gives no dev benefit

> Executed 2026-11-12 from `runs/E-024/checkpoints/ckpt_best.pt`, two paired arms, one variable.
> Pre-registered in `PLAN.md` before the run. **No arm advances.** Nothing was exported, no
> product readout was run, T/T-S3/FLEURS/noise and the browser path were not read.

## Verdict

**Rejected.** With the E-028 recipe and one epoch, adding a training-only auxiliary CTC loss on
the encoder output (weight 0.3) does not beat an identical control: paired dev macro WER
**31.0350% vs 31.0101%** (arm − control **+0.0249 pp**, 95% CI **[−0.0998, +0.1557]**), a tie
that points the wrong way. The pre-registered rule required a lower point estimate and a CI
upper bound below zero, so the InterCTC option is **closed for this recipe and weight**.

## The comparison is genuinely paired

The control re-ran E-028's recipe from the same checkpoint and reproduced it almost exactly:

| | E-028 retained | E-033 control |
|---|---|---|
| step loss 500 / 1.000 / 1.500 | 0.7717 / 0.6943 / 0.7474 | 0.7717 / 0.6943 / 0.7474 |
| dev macro at step 1.000 | 31.331% | 31.331% |
| best dev macro (final step) | 31.0077% | 31.0077% |
| `skipped_nonfinite` | 0 | 0 |

Same machine, torch 2.14.1, seed 0, same commit (`767122b`), same parity gate (CPU/MPS CTC loss
difference 4.6e-6). The arm differs only in `train.interctc`.

## Readout (full dev, 3.939 clips, greedy, no LM; paired per-utterance bootstrap within language)

| Arm | Best dev macro | Δ vs control (pp) | 95% CI (pp) | IT | EN | Final step |
|---|---:|---:|---|---:|---:|---:|
| `control` | **31.0101** | — | — | 36.291 | 25.730 | 13,283 |
| `interctc` (w 0.3) | 31.0350 | **+0.0249** | [−0.0998, +0.1557] | 36.273 | 25.797 | 13,283 |

Pooled WER agrees (+0.0308 pp, CI [−0.0923, +0.1657]). The trainer's own in-run dev curve says the same: the arm starts clearly worse
(**32.137% vs 31.331% at step 1.000**) and closes the gap without crossing it —
31.160 vs 31.036 at 10.000, 31.051 vs 31.019 at 12.000, **31.0335 vs 31.0077** at the end.
`select.py` re-decodes both best checkpoints independently and reproduces the training numbers
within 0.003 pp (MPS kernel nondeterminism on a handful of utterances), so the paired interval
uses real per-utterance hypotheses rather than two aggregates.

## What the auxiliary loss did

- `aux_loss` starts at **66.9** (random 1024-way head over CTC) and learns down to **0.72**
  (min 0.35) — the head is real and trained, not inert.
- With `loss = 0.7·main + 0.3·aux`, the early gradient is dominated by the auxiliary term:
  grad-norm mean **7.15** vs **6.60** for the control, max **66.0** vs **18.9**, and the dev shock
  at step 1.000 is 0.81 pp worse. The arm recovers as the head converges but never converts that
  into a gain.
- The auxiliary head is **394,625 training-only parameters** (3.4% more trainable weights),
  never registered on the deployable model: graph, package (19,706,289 B) and decoder are
  untouched, which is why the arm was cheap enough to test at all.

Cost: control **1 h 56**, arm **1 h 53** wall on the M5 — the +4.4% step-time probe from E-024
did not reproduce as a slowdown on this run (both arms within noise of each other).

## What this does and does not establish

- **Does**: at weight 0.3 with this recipe and one seed, InterCTC does not improve dev; the
  option has no free lunch for Moth and should not be promoted to a product readout.
- **Does not**: refute InterCTC in general. A lower weight, an aux-head warm-up, a different tap
  or a length-normalized auxiliary loss are untested and would each be a **new pre-registered
  arm** — not a re-tune of this one. One seed on one selection set cannot resolve a 0.02 pp
  difference in either direction; the honest reading is "no signal", not "InterCTC hurts".

Dev is the same MLS-IT + LibriSpeech-EN read-speech set that has informed every previous
selection round. No mixed-dev, no new-voice evidence and no browser measurement exists here.

## Artifacts

`runs/E-033/control/` and `runs/E-033/interctc/` — `config.yaml`, `train.log.jsonl`, `train.out`,
`dev/{dev.hyp.jsonl,dev.metrics.json}`, checkpoints (gitignored). `preflight.log` (both configs
validated before the run), `select.py` + `selection.json` (paired comparison), and this report.
