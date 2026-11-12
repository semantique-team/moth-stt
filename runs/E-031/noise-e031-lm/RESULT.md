# E-031 — noise curve with the E-031 LM (0.8.1 candidate)

> Run 2026-11-12. Same E-028 fp32 checkpoint, decoder 0.55/2.7, 100 held-out-speaker S3 clips,
> E-024 pool and seed as the 0.8.0 control; **only the LM changes**. **No regression: every
> condition is a statistical tie, and the 5–30 dB mean is slightly better.**

## Protocol

- Harness: `runs/E-024/noise_eval.py`, the same pool rule the training noise used (MUSAN + room
  tone), 100 S3 clips, clean + 30/20/15/10/5 dB SNR.
- New arm: `--lm runs/E-031/build/frozen_1_2_terms.pkl.gz`.
- Control: `runs/E-028/noise-0.8.0/` (E-028 checkpoint, shipped LM, same decoder and clips).
- Python fp32 model + pickle trigram LM, not the int8/browser path; paired utterance bootstrap,
  1,000 resamples, seed 0.

## Results

| SNR | 0.8.0 LM WER | **E-031 LM WER** | Δ | 95% CI | Tie? |
|---|---:|---:|---:|---|---|
| clean | 20.37 | **20.03** | −0.34 | [−1.36, +0.70] | yes |
| 30 dB | 21.90 | **21.22** | −0.68 | [−1.85, +0.50] | yes |
| 20 dB | 25.47 | **24.28** | −1.19 | [−2.53, +0.00] | yes |
| 15 dB | 27.33 | 27.67 | +0.34 | [−0.67, +1.53] | yes |
| 10 dB | 34.63 | **33.79** | −0.85 | [−2.22, +0.34] | yes |
| 5 dB | 40.92 | **40.41** | −0.51 | [−1.19, +0.00] | yes |

Mean WER 5–30 dB: **29.47 vs 30.05**. Five of six point estimates are better; 15 dB is +0.34 and
inside zero. No condition is a significant regression.

## Interpretation

The LM reweighting does not trade noise robustness for rare terms. The absolute curve is the
same-pool upper bound measured in E-028, on one speaker; it is not the E-025 unseen-room curve.
The int8/browser path was not re-measured here, but the acoustic graph is unchanged and the
browser acceptance passes 4/4.

## Artifacts

`runs/E-031/noise-e031-lm/` — per-SNR hypotheses/metrics, clean T-600/FLEURS diagnostics,
`result.json`; comparison in `compare.json` and `runs/E-031/noise_compare.py`. Control:
`runs/E-028/noise-0.8.0/`.
