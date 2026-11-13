# E-038 — RESULT: the pooled T-600 number hides a 5.3 pp WER and 12.0 pp ETA spread

> Executed 2026-11-13. Decode-free: the same 600 hypotheses already published in E-037, re-scored
> by speaker. No decode, no training, no new audio, no release change.

## Verdict

The shipped 0.8.1 scores **20.9802% WER / 63.2836% ETA on S1**, **26.3249% / 51.5581% on S2** and
**24.9574% / 51.2676% on S3**. The pooled **24.0826% / 55.2253%** is arithmetically correct and
descriptively misleading: the three voices span **5.3447 pp of WER** and **12.0160 pp of ETA**.

Two consequences:

- **The published pooled number is not a speaker-independent estimate.** Any statement about "the
  model" on T must carry the per-voice range, or it overstates precision.
- **The never-trained voice is not the worst one.** S3 (no acoustic training) lands next to S2
  (trained) and far from S1 (trained, same microphone family as S3). The gap therefore is not
  explained by exposure to training audio in any simple way; it is a speaker/session effect that
  three voices cannot resolve.

## Numbers

Reference: `recordings/manifests/T.jsonl` (600 rows, sha `11e002d88dbfaee9…`). Hypotheses:
`runs/E-037/readout/paired-1/product/int8/t600.hyp.jsonl` (sha `a07fc630eccb5bdb…`), the shipped
INT8 graph + A3 LM. Per-utterance bootstrap, 1,000 resamples, seed 0.

| Voice | Role | n | Ref words | Sub | Del | Ins | WER | 95% CI | ETA | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| S1 | acoustic training | 200 | 1,163 | 153 | 78 | 13 | **20.9802** | [18.0851, 23.7882] | **63.2836** | [57.3134, 69.3252] |
| S2 | acoustic training | 200 | 1,151 | 186 | 106 | 11 | **26.3249** | [22.8669, 29.5713] | **51.5581** | [44.7439, 57.7259] |
| S3 | never trained, observed | 200 | 1,174 | 187 | 97 | 9 | **24.9574** | [21.6512, 28.1488] | **51.2676** | [44.5652, 57.6803] |
| pooled | — | 600 | 3,488 | 526 | 281 | 33 | 24.0826 | [22.0392, 25.9442] | 55.2253 | [51.7241, 59.1919] |

Dispersion: WER min 20.9802, max 26.3249, **range 5.3447 pp**, mean 24.0885; ETA min 51.2676,
max 63.2836, **range 12.0160 pp**, mean 55.3698. Descriptive only.

## What This Does And Does Not Establish

- **Consistency check**: the pooled WER **24.0826** and ETA **55.2253** reproduce E-037's T-600
  INT8 figures exactly, so the re-scoring introduces no drift and the per-speaker split is the only
  change.
- **Does**: quantify how much a single pooled number hides, give the per-voice unit the paper needs,
  and give an empirical band (5.3 pp WER, 12.0 pp ETA) for how much an unseen voice could plausibly
  differ on this protocol.
- **Does not**: resolve the S1/S2/S3 differences (their CIs overlap; 200 clips each is a small
  sample), estimate between-speaker variance, or provide a speaker-cluster CI. With three voices
  that variance is not estimable — the intervals above are **within-speaker, per-utterance**.
- **Does not** make any voice a holdout. S1 and S2 are training speakers; S3 was inspected
  repeatedly. T remains reused evidence.
- **Does not** touch rare terms, noise, FLEURS or the release.

## Why It Matters For The Paper

The honest sentence becomes: *on the three available voices the model ranges from 20.98% to 26.32%
WER and from 51.27% to 63.28% ETA (T-600, 0.8.1); the pooled 24.08% / 55.23% is an average over
voices, not a speaker-independent estimate.* A fourth voice is not available in this repository, so
no unseen-speaker claim is made.

## Artifacts And Cost

`runs/E-038/` — `PLAN.md`, `per_speaker.py`, `metrics.json` (assets and hashes included), this
report. Cost: seconds of CPU, no decode, no download, 0 USD.

Reproduction: `uv run python runs/E-038/per_speaker.py`.
