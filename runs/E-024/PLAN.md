# E-024 — noise augmentation on the read-speech half

> Written 2026-11-09, before the data is built. Replaces the planned InterCTC arm: a feasibility
> probe (`probe_cost.py`) showed InterCTC is affordable (+4.4% step time, ~3 h 30 for the run) but
> the audit of 2026-11-09 measured a weakness **an order of magnitude larger**, so the budget goes
> there. InterCTC is not rejected as an idea, only postponed.

## Why this and not something else

The demo was tested with a real laptop microphone and the transcripts were poor. Measuring it
(`runs/audit-2026-11-09/noise_curve.py`, 100 held-out-speaker clips, MUSAN mixed at a controlled
SNR):

| SNR | WER | ETA | Δ WER vs clean |
|---|---|---|---|
| clean | 22.24 | 47.30 | — |
| 30 dB (quiet room, fan) | 26.99 | 42.57 | +4.75 [+2.17, +7.58] |
| 20 dB (office) | 32.09 | 36.49 | +9.85 [+6.34, +13.89] |
| 15 dB (noisy office) | 38.37 | 35.14 | +16.13 [+11.82, +20.84] |
| 10 dB (bar) | 48.05 | 24.32 | +25.81 [+20.31, +31.44] |
| 5 dB (street) | 58.23 | 20.27 | +35.99 [+31.07, +41.02] |

Deletions go from 5.6% of reference words to 24.6%: the model does not mis-hear words under noise,
it **drops** them. That is the signature of training on clean read speech, which is what CV, MLS
and LibriSpeech are.

For scale: at 15 dB the model scores 38.4 — worse than Whisper-tiny's 32.1 on clean audio. The
project's whole advantage over Whisper is contingent on the room being quiet.

## Hypothesis

The training mixture contains no noise whatsoever (H3's noise augmentation was listed in `docs/10`
§4.2.6 and `docs/11` §7.4 and never run). Adding noise to part of it should buy most of the curve
back, and it should cost little or nothing on clean audio.

## Design — one variable, same mixture size

The only source with raw audio still on disk is **Common Voice IT** (`data/raw/cv_it_27`, 167.8 h).
MLS and LibriSpeech audio was deleted after the mel cache and would need a ~22 GB re-download, so
this experiment uses CV-IT alone and says so.

CV-IT's 113,118 training clips are split deterministically, by id hash, into two halves:

| Arm | Clips | How | Shards |
|---|---|---|---|
| `cv_it_train_clean50` | 56,559 (83.9 h) | untouched, **existing cache reused** | `data/shards/cv_it_train` |
| `cv_it_train_noisy50` | 56,559 (83.9 h) | mixed with noise at a per-clip SNR | `data/shards/cv_it_noisy50` (new, ~4.8 GB) |

**The mixture is therefore exactly E-017's 589.6 h**: CV 167.8 (half clean, half noisy) + MLS 169.9
+ LS 100 + spliced 110 + gold 41.87. No source changes size, no source is added or dropped. Each CV
clip is still heard once per epoch. The only difference from E-017 is that **half of the Italian
read speech is heard through noise.**

Rejected, with reasons: *all CV noised* (removes clean Italian, and clean WER is a gate); *all CV
clean plus a noisy copy* (mixture grows to 757 h and CV's share from 28.5% to 44.3%, confounding
"noise" with "CV oversampling"); *noising MLS/LS too* (needs a re-download, deferred).

### The noise and the SNR

- **Sources**: MUSAN `noise` (120 clips, CC-BY 4.0) + our own room tone (97 clips) — both already in
  `data/raw/noise/`. Music is excluded: it has harmonic structure and the VAD already reports it
  separately.
- **SNR**: uniform in dB over **[10, 30]**, drawn per clip from a hash of the clip id, so the dataset
  is reproducible and a resume rebuilds byte-identically. The range is the one the curve shows is
  damaging; below 10 dB the model is a lost cause without a denoiser.
- **Mixing**: on the raw audio, noise tiled to length and scaled on RMS relative to the speech,
  peak-clipped only if it would exceed 1.0. Same rule as `noise_curve.py`, so the training and
  evaluation conditions are the same *kind* of noise — note this is a weakness: the test noise and
  the training noise come from the same pool, so the gain measured here is an upper bound for a new
  room.

### Recipe

Identical to E-017 except the data: init `runs/E-014/checkpoints/ckpt_best.pt`, 2 epochs, head LR
3e-4 / encoder 5e-5, cosine 500-step warm-up, bf16, truncated input, augmentation `full`, batch
192 s / 32 clips, seed 0. **26,442 steps**, ~3 h 21 at E-017's measured throughput, plus the
augmented data being the same size so no schedule change is needed.

## Prediction (recorded before the build)

| Condition | clean (E-017) | predicted E-024 | predicted Δ |
|---|---|---|---|
| clean | 22.24 | 22.0–23.0 | ±0.5 |
| 30 dB | 26.99 | 23.5–25.5 | **−2 to −3.5** |
| 20 dB | 32.09 | 26.0–29.5 | **−3 to −6** |
| 15 dB | 38.37 | 29.5–33.5 | **−5 to −9** |
| 10 dB | 48.05 | 38.0–44.0 | **−4 to −10** |
| 5 dB | 58.23 | 48.0–54.0 | **−4 to −10** |

The prediction is deliberately wide: augmentation has never been run in this repo, the SNR range is
a guess, and the noise pool is shared with the evaluation.

## Decision rule (fixed now)

Primary readout: the **noise curve** (mean WER over 10/15/20/30 dB), paired over the same 100
clips. Secondary: clean WER on T-600, T-S3, FLEURS-IT, CV-IT test — because a model that wins on
noise and loses on clean is not shippable.

- **Adopt** if the mean improvement over 10–30 dB is **≥ 4 points** *and* clean WER degrades by
  **≤ 1.0 point** on T-600 and T-S3.
- **Partial**: ≥ 4 points on noise but clean worse by > 1.0 → retry with 25% of CV noised (N2),
  since the damage is likely the lost clean exposure.
- **Kill**: < 3 points at 15 dB → augmentation at this scale does not pay; record it and stop.

## Risks

- **The evaluation noise is the training noise's source.** The measured gain is an upper bound; a
  new room will get less. Stated in the result, and the mitigation is a per-source split of the
  noise pool (train on 80% of the noise clips, evaluate on the other 20%) — worth adding if the
  numbers look too good.
- **CV-IT is read speech from many accents.** Noise robustness learned there may not transfer to a
  close-talking laptop microphone, which is the demo's condition.
- **Half the clean Italian is lost.** That is the risk the ≤1.0-point clean gate exists for.
- **Disk**: 18 GB free, ~4.8 GB needed. The clean half reuses the existing cache precisely so this
  fits.
- Empty-target noise clips (`docs/11` §7.4) are **out of scope**: `train.py::_encode_targets`
  rejects empty targets, so it is a code change, not a data one. The VAD already covers the gate.

## Artifacts (to create)

```text
runs/E-024/
  PLAN.md                      # this file
  probe_cost.py                # InterCTC feasibility (done, kept for the postponed arm)
  build_noisy_mel.py           # the 50% split + mixing, reuses stt.features.log_mel
  config.yaml                  # E-017's recipe with the split manifests
  train.log.jsonl
  noise_eval.py                # the curve, paired, for any checkpoint
  eval/                        # noise curve + clean gate sets
  RESULT.md
```

## Cost

Build: ~2 h (83.9 h of audio through noise mixing + mel, 8 workers). Training: ~3 h 25. Evaluation:
~30 min. **~6 h of machine time, 0 USD, no author time.**
