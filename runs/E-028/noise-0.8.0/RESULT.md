# 0.8.0 — noise check against 0.7.0 with the same decoder (fp32 checkpoints)

> Run 2026-11-12. **No noise regression detected.** 0.8.0 is a tie at 10–30 dB, a tie at 15 dB,
> and significantly better on clean audio (−2.72 WER [−4.42, −1.02]).

## Protocol

- Harness: `runs/E-024/noise_eval.py`, the same pool rule the training noise used (MUSAN +
  room tone), 100 held-out-speaker S3 clips.
- Conditions: clean, 30, 20, 15, 10, 5 dB SNR; paired by clip; 1,000-resample utterance
  bootstrap.
- **Both checkpoints use the published 0.8.0 decoder** (beam 12 / top-k 10 / LM 0.55 /
  bonus 2.7), so the comparison isolates the acoustic change. The historical 0.7.0 card curve
  used 0.5 / 2.4 and is not the fair comparison.
- Python fp32 model + pickle trigram LM, not the int8/browser path.

## Results

| SNR | 0.7.0 WER | **0.8.0 WER** | Δ | 95% CI | Tie? | ETA Δ |
|---|---:|---:|---:|---|---|---:|
| clean | 23.09 | **20.37** | **−2.72** | [−4.42, −1.02] | **no** | +2.03, tie |
| 30 dB | 23.43 | **21.90** | −1.53 | [−2.92, 0.00] | yes | +1.35, tie |
| 20 dB | 26.66 | **25.47** | −1.19 | [−2.53, 0.17] | yes | −1.35, tie |
| 15 dB | 27.33 | 27.33 | 0.00 | [−1.53, 1.54] | yes | −2.03, tie |
| 10 dB | 33.62 | 34.63 | +1.02 | [−1.04, 3.32] | yes | −0.68, tie |
| 5 dB | 41.94 | 40.92 | −1.02 | [−2.76, 0.67] | yes | +0.68, tie |

Mean WER 10–30 dB: **27.76 → 27.33** (a tie, slight improvement). No condition is a significant
regression; clean is significantly better. The 10 dB point is +1.02 but its CI includes zero.

## Interpretation

The 0.7.0 noise claim is not undone by the 0.8.0 fine-tune: the model is non-inferior from 10 to
30 dB and better clean. The result is a same-pool upper bound, not the E-025 unseen-room curve,
and one speaker (S3) is used. It is a post-release check, not a pre-registered gate.

## Artifacts

- `runs/E-028/noise-0.8.0/` — `result.json`, per-SNR metrics/hypotheses.
- `runs/E-028/noise-0.7.0-same-decoder/` — the fair control.
- Repro:

```bash
uv run python runs/E-024/noise_eval.py \
    --checkpoint runs/E-028/checkpoints/ckpt_best.pt \
    --config runs/E-028/config.yaml \
    --out runs/E-028/noise-0.8.0 --clips 100 --skip-clean \
    --lm-weight 0.55 --token-bonus 2.7
```
