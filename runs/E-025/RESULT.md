# E-025 — RESULT: the noise gain survives a room the model never heard

> Executed 2026-11-09. Plan and decision rule: `PLAN.md`, written before the recording existed.

## Verdict

**The gain transfers, and it is larger on the unseen room.** E-024's improvement is not an artifact
of training and evaluating on the same 217 clips: 0.7.0 beats 0.6.0 at **every** noise condition on
a room neither model has ever heard, and all five differences are significant. The card's
"upper bound" caveat is closed and replaced by this measurement.

| Condition | 0.6.0 | **0.7.0** | Δ WER [95% CI] | |
|---|---|---|---|---|
| clean (harness check) | 22.24 | 23.94 | +1.70 [−0.34, +3.72] | tie |
| 30 dB | 28.52 | **24.62** | **−3.90 [−7.20, −0.83]** | |
| 20 dB | 38.71 | **31.92** | **−6.79 [−10.84, −3.15]** | |
| 15 dB | 47.71 | **36.50** | **−11.21 [−14.69, −7.48]** | |
| 10 dB | 63.33 | **44.99** | **−18.34 [−22.78, −13.40]** | |
| 5 dB | 81.32 | **66.72** | **−14.60 [−18.84, −10.47]** | |

Mean gain over 10–30 dB: **−10.06 points**, against **−6.18** on the same-pool curve.

The clean row is not a result: the audio is byte-identical to E-024's clean condition, so it is a
harness check. It reproduces 22.24 and 23.94 to within **1e-5 WER**, which is what makes the rest of
the table trustworthy.

## Side by side with the same-pool curve

| Condition | same pool (E-024) | unseen room (E-025) |
|---|---|---|
| 30 dB | −2.72, tie | **−3.90, significant** |
| 20 dB | −0.68, tie | **−6.79, significant** |
| 15 dB | −6.28 | **−11.21** |
| 10 dB | −11.54 | **−18.34** |
| 5 dB | −9.68 | **−14.60** |

The unseen room is **harsher at the same nominal SNR**: 0.6.0 scores 47.7 at 15 dB there against
33.6 on the pool. That is the whole explanation of the larger gain: a model that collapses further
has more room to be rescued. The absolute 0.7.0 number is therefore worse on this room (36.5 at
15 dB) than on the pool (27.3) — the *gain* is bigger, the *performance* is not. Any sentence that
quotes one without the other is misleading.

## What the holdout room actually is

A 78.7 s recording on the built-in microphone, in a noisier room than the first take, and in a
**different physical room from the room tone in the training pool** (confirmed by the author; the
spectra differ as expected: the training room tone is 91% below 120 Hz with a 66–69 Hz peak, this
take is broadband with 33% below 200 Hz and peaks at 47/62 Hz). The first take was rejected: it was dominated by 50 Hz mains hum from the charger (93% of the energy below 200 Hz),
which would have tested electrical hum, not a room. The used take:

- **trimmed to 40–78 s**: three loud events at 12.1, 16.9 and 31.0 s (objects/doors, up to
  −19.6 dBFS) were excluded; the trimmed segment has one 50 ms transient and **VAD speech fraction
  0.0** (max window probability 0.11);
- RMS **−52.6 dBFS**, peak −29.1, level range 9.8 dB across 10 s blocks (an appliance cycling);
- spectral centroid **1.05 kHz**, 33% of the energy below 200 Hz, 47% between 250 and 1000 Hz,
  10% above 1 kHz — **broadband**, unlike the rejected take;
- peaks at 47 and 62 Hz exist but carry only 11.5% of the energy.

## The prediction was wrong again, in the same direction

`PLAN.md` predicted attenuation (−3 to −6 at 15 dB, −6 to −10 at 10 dB, ties at 30/20). Measured:
−11.21, −18.34, and both 30 and 20 dB significant. **All five conditions beat the upper end of the
prediction.**

The reason is in the table above: I anchored the prediction to the same-pool curve, which implicitly
assumed the two noises were equally damaging at a given nominal SNR. They are not. The new room is
harsher, so the baseline falls further and the gain grows. **A gain at fixed nominal SNR is not a
property of the model alone; it depends on the noise.** That is the lesson, and it is the second
time in two experiments that a prediction was too conservative about the size of the effect (E-024
was right on the mean and wrong on the shape; this one is wrong on the magnitude at every point).

## Decision (against the rule fixed in PLAN.md)

The rule was: transfers if 15 dB ≤ −3 and 10 dB ≤ −5, both CIs excluding 0. Measured −11.21 and
−18.34, both CIs far from 0. **Transfers.** The card's caveat is downgraded from "upper bound for an
unfamiliar room" to "measured on one unseen room", with this table. No new weights, no version bump:
0.7.0 is the same artifact, with a better-supported claim.

## What is still not true

- **One room, one take, 38 s.** The noise offsets are different parts of the same recording, not
  independent rooms. A café, a car and a kitchen are not covered.
- **Steady ambience only.** The loud events were trimmed out, so this says nothing about door slams,
  cutlery or a passing voice — the VAD is the only defence there.
- **The SNR is synthetic.** It is RMS-relative mixing, not a calibrated acoustic SNR; the room is
  quiet (−52.6 dBFS), so 5 dB means amplifying its floor by a large factor.
- **The room is harsher than the training pool.** Absolute WER at 15 dB is 36.5, not 27.3. The gain
  is not a promise of good absolute performance.
- **Single seed**, as everywhere in this project.

## Cost

Recording 3 min (including the rejected take), analysis 2 min, paired evaluation **20 s** (the models
are small; the earlier 10-minute E-024 run was dominated by the three clean sets, not by the curve).
**No training, 0 USD.**

## Artifacts

- `PLAN.md` — prediction and decision rule, fixed before recording
- `record_room.py`, `room_report.json` — the used take, trimmed, with VAD/spectrum/transient checks
- `room_report_take1_rejected.json` — the first take, dominated by 50 Hz mains hum, kept as the
  reason for the redo
- `room_report_take2_untrimmed.json` — the second take before trimming, with the three loud events
- `holdout_eval.py`, `eval/` — paired curve for both arms, per-condition hypotheses and metrics
- `data/raw/noise/room_holdout/` — audio, gitignored as always

Card updated in place: `runs/E-024/MODEL_CARD.md`; the public card is re-uploaded without touching
the weights.
