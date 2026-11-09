# E-015 — RESULT: the first gold addendum did not teach the rare terms

> Executed 2026-11-09. Plan and target: `PLAN.md` (addendum written before the sentences existed).
> **Killed** by the criterion fixed there. 0.7.0 stays the shipped model; nothing is published.

## Verdict

The 50 S1 sentences covering the 44 never-seen terms moved the primary readout from **17.0% to
23.4%** (8/47 → 11/47), a **+6.4-point** difference whose 95% CI crosses zero
**[−4.3, +17.0]**. The target was **≥50%**; the kill threshold was **<30%**. Both the primary
target and the clean gate fail, so the arm is not adopted.

## Primary readout: the 47 rare utterances

| Arm | Correct | Accuracy |
|---|---|---|
| 0.7.0 (baseline) | 8/47 | **17.0%** |
| E-015 final checkpoint | 11/47 | **23.4%** |
| Δ | +3 utterances | **+6.4 [−4.3, +17.0]**, not significant |

By speaker, and this is the important part:

| Speaker | 0.7.0 | E-015 | Δ |
|---|---|---|---|
| S1 (the recorded voice, training speaker) | 6/14 | 7/14 | +1 |
| S2 (training speaker) | 0/15 | 2/15 | +2 |
| **S3 (held out, never trained on)** | **2/18** | **2/18** | **0** |

**The gain is entirely on the two speakers the model has already heard; the held-out speaker does
not move.** One recorded voice did not produce a speaker-independent term mapping.

The flips: T-0055 `startup`, T-0112 `screenshot` (S1), T-0086 `popup`, T-0329 `ipad` (S2),
T-0069 `checkout` (S3) became correct; T-0346 `cadbury` (S1) and T-0309 `invision` (S3) regressed.

## What happened to the rest of the terms

| Group | 0.7.0 | E-015 | Δ |
|---|---|---|---|
| Natural training text (817 occurrences) | 439 (53.7%) | 423 (51.8%) | **−16** |
| Gold-only (179) | 123 (68.7%) | 129 (72.1%) | +6 |
| Never seen (47) | 8 (17.0%) | 11 (23.4%) | +3 |
| **All terms (1,043)** | **570 (54.65%)** | **563 (53.98%)** | **−0.67** |
| 18 gold-covered terms (regression check) | 20/27 (74.1%) | 19/27 (70.4%) | −1 |

The model traded **16 natural-term occurrences** for 3 rare-term utterances. That is not a good
trade, and it is the opposite of the intended one.

## Clean gate (the second reason to stop)

| Set | 0.7.0 | E-015 | Δ [95% CI] |
|---|---|---|---|
| T-600 | 25.20 | **24.91** | −0.29 [−1.21, +0.63] — tie |
| T-S3 | 26.92 | **25.55** | −1.36 [−3.16, +0.25] — tie |
| FLEURS-IT | 20.14 | 20.86 | **+0.72 [+0.32, +1.12] — worse** |
| FLEURS-EN | 49.14 | 50.45 | **+1.31 [+0.68, +1.97] — worse** |
| CV-IT dev | 19.85 | 19.97 | +0.12 [−0.48, +0.73] — tie |

The recorded mixed set improves (T-S3 −1.36, not significant), but the public read-speech sets
regress significantly. The plan's kill rule — clean WER worse by >0.5 with a CI excluding zero —
fires on FLEURS-IT and FLEURS-EN.

## The dev curve: the recipe was too aggressive

| Step | Dev WER |
|---|---|
| 20 (pre-fine-tune) | **30.81** |
| 2,000 | 33.64 |
| 4,000 | 33.54 |
| 6,000 | 32.79 |
| 8,000 | 32.11 |
| 10,000 | 31.48 |
| 12,000 | 31.21 |
| 13,252 (final) | 31.22 |

The fine-tune **shocked** the model by almost 3 dev points and spent the whole epoch recovering;
it never got back to the start. Under the project's model-selection rule the best checkpoint is
**step 20**, i.e. the baseline itself. The run therefore selects no new model at all, and the
final checkpoint is evaluated only as the arm that saw the data.

## Reading

Three things are true at once:

1. **The data is not disproven; this recipe is.** 50 sentences in one voice, repeated x20, at the
   E-017 learning rates (head 3e-4, encoder 5e-5) for one full epoch, is too much update for a
   converged model. The dev shock and the natural-term loss are optimization damage, not evidence
   that the terms cannot be learned.
2. **One voice is not enough for a speaker-independent mapping.** The gain is S1 +1, S2 +2, S3 0.
   The held-out speaker is exactly where the product claim lives, and it did not move. The original
   50 S1 + 50 S2 design was right; the S1-only compromise was not.
3. **The term target was ambitious but honest.** The direct ceiling was +3.7 ETA; the intervention
   did not approach it, and the overall term accuracy went down, not up.

## Next arm, if we continue

Not another 2-hour run at the same recipe. The next one should change two variables at once and is
therefore a new experiment, not a rerun:

- **S2 as the second voice**: record the 50 S2 sentences already in the script (`--start 624`);
  this is the variable the S1-only run could not test.
- **A gentler fine-tune**: lower LRs (e.g. head 5e-5 / encoder 1e-5), or freeze the encoder and
  train the head only, and/or a shorter schedule with the term source at a higher share. The dev
  curve says the model needs a smaller step, not more data.
- **Keep the same kill rule** and the 47-utterance readout; do not lower the bar because this arm
  failed.

The 50 S1 sentences are recorded, validated and reusable; the `gold-terms-x20` source is built.
No audio or data work is lost.

## Cost

Data prep + mel + x20 **2 min**; training **1 h 50** (13,252 steps, 0 non-finite); gate + rare-term
evaluation **~3 min**. **No teacher, no downloads, 0 USD.**

## Artifacts

- `PLAN.md` — the pre-registered target and kill rule (addendum)
- `RECORDING.md`, `recordings/manifests/R-S1.jsonl` — the 50 recorded S1 sentences
- `config.yaml`, `train.log.jsonl`, `train.out` — the run
- `eval/final/`, `eval/gate/`, `eval/terms.json` — gate and rare-term readout
- `eval_terms.py`, `prepare_terms.py` — the readout and the data source
- `checkpoints/` — gitignored, as always
