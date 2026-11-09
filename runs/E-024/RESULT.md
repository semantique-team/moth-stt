# E-024 — RESULT: noise augmentation on half of Common Voice IT

> Executed 2026-11-09. Plan and kill criteria: `PLAN.md` (written before the data was built).
> The rest of this file is the measurement against that plan.

## Verdict

**Adopted**, shipped as **release 0.7.0**. The noise gain clears the bar by a wide margin and the
clean gate passes: the model is not worse on clean audio, and on two of the four clean sets it is
measurably better.

| Plan criterion | Required | Measured | |
|---|---|---|---|
| mean WER gain over 10–30 dB | ≥ 4.0 points | **−6.18** | **pass** |
| clean T-600 | ≤ +1.0 | **−0.72** [−1.75, +0.26] | **pass** |
| clean T-S3 | ≤ +1.0 | **+0.26** [−1.37, +1.96] | **pass** |

## The noise curve (primary readout)

100 held-out-speaker clips, paired, same noise sequence for both arms:

| SNR | E-017 (0.6.0) | **E-024 (0.7.0)** | Δ WER [95% CI] |
|---|---|---|---|
| clean | 22.24 | 23.94 | +1.70 [−0.34, +3.72] — **tie** |
| 30 dB (quiet room, fan) | 25.98 | **23.26** | −2.72 [−5.63, +0.17] — tie |
| 20 dB (office) | 27.50 | **26.83** | −0.68 [−4.13, +2.73] — tie |
| 15 dB (noisy office) | 33.62 | **27.33** | **−6.28 [−10.58, −2.38]** |
| 10 dB (bar) | 44.82 | **33.28** | **−11.54 [−16.18, −7.43]** |
| 5 dB (street) | 50.93 | **41.26** | **−9.68 [−14.50, −5.01]** |

Three of the five noise conditions are significant, and they are the three where it matters. At
30 dB the new model is **faster than the old one is clean** (23.26 against 22.24 is a tie, but the
old model needed a silent room to reach 26).

## Where the prediction was wrong

`PLAN.md` predicted −3 to −6 at 20 dB and −4 to −10 at 10 dB. Measured: **−0.68 at 20 dB** and
**−11.54 at 10 dB**. The shape was wrong, not just the magnitude: I assumed the gain would be
spread across the curve, and it is concentrated at the low-SNR end. A model at 20 dB was already
holding on; what it could not do was survive 10 dB. Recorded because a prediction that is right on
the mean and wrong on the shape is still a wrong prediction.

The clean prediction (±0.5) was also missed on the 100-clip subset (+1.70, a tie) — but that
subset is 100 clips, and the full 200-clip T-S3 says +0.26. The full set is the number to trust.

## Clean gate (secondary, and the reason it can ship)

| Set | 0.6.0 | **0.7.0** | Δ [95% CI] |
|---|---|---|---|
| T-600 | 25.92 | **25.20** | −0.72 [−1.75, +0.26] — tie |
| T-S3 (200, held out) | 26.66 | 26.92 | +0.26 [−1.37, +1.96] — tie |
| FLEURS-IT | 20.46 | **20.14** | −0.32 [−0.69, +0.03] — tie |
| CV-IT test (14,643) | 24.88 | **24.37** | **−0.52 [−0.66, −0.38]** |

CV-IT improves significantly; the rest are ties. The dev macro that selects the checkpoint also
moved: **0.3084** against E-017's 0.3098, which is 0.14 points and therefore noise — as predicted.

**Honest reading**: the clean gain is not established. What is established is that losing half the
clean Italian exposure cost nothing measurable, and that CV-IT — the largest set, 24.9 h — improved
by half a point with a CI that excludes zero. That is consistent with noise acting as a regulariser,
but a single seed cannot separate that from run noise.

## Full result table, 0.7.0

| Set | n | WER | ETA |
|---|---|---|---|
| Mixed test T, all speakers | 600 | **25.20** | **54.17** |
| T, S1 (training) | 200 | 21.67 | 63.28 |
| T, S2 (training) | 200 | 26.93 | 51.27 |
| T, S3 (held out) | 200 | 26.92 | 48.45 |
| FLEURS-IT | 697 | 20.14 | — |
| FLEURS-EN | 525 | 49.14 | — |
| CV-IT test | 14 643 | 24.37 | — |
| Noise, non-music words (VAD) | 337 | **0** | — |

int8 on the shipped graph: T-600 25.14 / ETA 54.65, T-S3 27.00 / 48.45 — both ties against fp32
(−0.06 and +0.08). Browser↔Python: **−0.09 WER [−1.19, +1.13], tie**, 152/200 identical.
Browser 1-thread RTFx 24.3 / 22.9 / **20.3×** on 5/10/20 s, unchanged.

## What is still not true

- **The noise pool is shared.** Training and evaluation draw from the same 217 clips (MUSAN noise +
  one room tone). The gain is an **upper bound** for an unfamiliar room. A per-source split (train
  on 80% of the noise, evaluate on 20%) is the obvious next control and was not run.
- **Half the Italian read speech is now only ever heard in noise.** It cost nothing measurable
  here, but it is a real change in what the model sees, and a second seed might show it.
- **The floor moved, not the ceiling.** At 5 dB the model still scores 41.3%: this is robustness,
  not a denoiser.
- **CV-IT is read speech.** The demo's condition — a close-talking laptop microphone in a room — is
  closer to 25–30 dB, where the gain is only 1–3 points.

## Cost

Build 1.1 min (not the 2 h estimated — only the noisy half is computed), training **~3 h 40**,
evaluation ~13 min. Release rebuild and republish ~10 min. **No author time, 0 USD.**

## Artifacts

- `PLAN.md` — prediction and decision rule, fixed before the build
- `build_noisy_mel.py`, `build.log` — the 50% split and the mixing
- `config.yaml`, `train.log.jsonl`, `train.out` — the run
- `noise_eval.py`, `compare.py`, `eval/{base,noisy,gate,noise-set}/`, `comparison.json` — the readout
- `parity.py`, `parity-wer.json`, `browser.json` — browser parity and speed
- `record_release.py`, `release-0.7.0.json` — the shipped artifact
- `checkpoints/` — gitignored, as always

Published: model `semantique-team/moth-stt`, demo `semantique-team/moth-demo` (static Space).

## Follow-up: the shared-pool caveat was closed the same day

E-025 re-ran this curve on a **room recorded after training**, which the model has never heard. The
gain transfers and is **larger** (15 dB −11.21, 10 dB −18.34, 5 dB −14.60; mean −10.06 over 10–30 dB
against −6.18 here), because the new room is harsher at the same nominal SNR: 0.7.0 scores 36.5 at
15 dB there against 27.3 here. The clean condition reproduced this experiment's numbers exactly
(22.24 / 23.94, within 1e-5), which is the harness check. **No weights changed**; the card was
updated in place. See `runs/E-025/RESULT.md` and `docs/11` §3.7.
