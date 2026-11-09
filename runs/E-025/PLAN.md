# E-025 — does the noise gain survive a room the mixer has never heard?

> Written 2026-11-09, **before recording and before measuring**. The point of this file is that
> the prediction cannot be adjusted after seeing the number.

## Why this experiment

E-024 shipped 0.7.0 with a noise curve measured on a pool of 217 ten-second clips (120 MUSAN
composites + 97 room-tone clips). The **same pool was used for training**, so the gain is an upper
bound for an unfamiliar room. The card says so; this experiment is the control that turns the
bound into a measurement.

The control cannot be an 80/20 split of the 217 clips. The training build cycled all of them
(`pool[noise_index % len(pool)]`), so every clip was heard, and the original MUSAN audio was deleted
after the noise set was built, so the split cannot be made at the level of the underlying source
recordings either. Any 80/20 on the composites is leaky by construction.

## Design — one genuinely new noise source

A **single new room**: the room this project is developed in, recorded 2026-11-09 on the MacBook Pro
built-in microphone (48 kHz, mono, 120 s, no speech). The model has never seen this recording, this
microphone or this room. It is also the exact deployment condition of the demo — the whole reason
E-024 exists.

- **Speech set**: the same first 100 held-out-speaker (S3) clips as E-024's curve, so the two
  experiments are comparable clip by clip.
- **Conditions**: clean + 30 / 20 / 15 / 10 / 5 dB, mixed with the same `mix_at_snr` rule as E-024
  (RMS-relative, tiled, peak-clipped only above 1.0). Noise offset drawn per clip from the 120 s
  recording by a deterministic seed, so each mixture sees a different part of the room.
- **Arms**: `runs/E-017/checkpoints/ckpt_best.pt` (0.6.0, no noise training) against
  `runs/E-024/checkpoints/ckpt_best.pt` (0.7.0), same decoder (beam 12, top-k 10, LM 0.5,
  token bonus 2.4), same mixed audio — paired.
- **Clean is a harness check, not a result**: the audio is identical to E-024's clean condition, so
  the numbers must reproduce 22.24 (0.6.0) and 23.94 (0.7.0) on these 100 clips. If they do not,
  the harness is wrong and the curve is not read.

## Prediction (recorded before the recording exists)

The new room is a normal quiet room with a laptop fan; it is not MUSAN, and the model has never
heard it. The gain should survive, attenuated:

| SNR | same-pool gain (E-024) | predicted on the new room |
|---|---|---|
| 30 dB | −2.72 (tie) | tie |
| 20 dB | −0.68 (tie) | tie |
| 15 dB | −6.28 | **−3 to −6** |
| 10 dB | −11.54 | **−6 to −10** |
| 5 dB | −9.68 | **−5 to −9** |

The attenuation is predicted because part of the same-pool gain may be familiarity with those
specific recordings; the rest is the ability to ignore noise, which should transfer.

## Decision rule (fixed now)

- **Transfers** (15 dB ≤ −3 and 10 dB ≤ −5, both CIs excluding 0): the card's caveat is downgraded
  from "upper bound" to "measured on one unseen room", with this table added. No new weights, no
  version bump; the card is edited in place.
- **Partially transfers** (15 dB between −1 and −3, or a CI that includes 0): the card keeps the
  upper-bound language and gains this measurement as the honest lower bound.
- **Does not transfer** (15 dB worse than −1, or 0.7.0 worse than 0.6.0): the card must say that the
  gain is specific to the training noise family, and 0.7.0's noise claim is weakened to what was
  actually measured. Consider a second room before concluding — one room is one room.

## Cost and risks

Recording ~2 min, analysis ~2 min, evaluation ~20 min (5 SNRs × 100 clips × 2 arms, plus clean).
**No training, 0 USD.**

Risks, stated up front: one room is not a corpus; the microphone is the same one the demo uses, so
this is a *deployment* test and not a *generalisation* test; the room tone may contain a tonal fan
component that makes it easier or harder than MUSAN; a passing event (door, voice) would contaminate
the recording, so it is VAD-checked and any speech region is trimmed before use.

## Artifacts

- `record_room.py` — records, VAD-checks and writes `data/raw/noise/room_holdout/` (audio is
  gitignored, as always) plus `room_report.json`
- `holdout_eval.py` — the paired curve for both arms
- `compare.py` — paired deltas with CIs
- `RESULT.md` — the measurement against this plan
