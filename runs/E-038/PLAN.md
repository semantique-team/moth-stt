# E-038 — Per-speaker T readout of 0.8.1 from retained hypotheses

**Status:** pre-registered 2026-11-13, **before** scoring. Decode-free: it re-scores hypotheses
that are already committed in E-037. No decode, no training, no new audio, no release change.

## Question

The published T-600 numbers are pooled over S1, S2 and S3. The workflow guide warns that the
scorer's `S3+S4` aggregate covers every non-S1/S2 voice and must not be read as held-out, and the
closure plan lists "bootstrap per cluster (speaker/stanza)" as a gap that depends on more speakers.
With only three voices available, what does the 0.8.1 error look like **per speaker**, and how much
do the three voices differ?

> For the shipped 0.8.1 configuration on T-600, report WER and ETA separately for S1, S2 and S3,
> with per-utterance bootstrap CIs, plus the observed dispersion across the three voices.

The point is **honesty and variance**: a fourth voice is not available, so the closest defensible
substitute is to show how much the three existing voices already differ, and to stop reporting a
single pooled number as if it described all speakers.

## Assets

| Role | Asset | Note |
|---|---|---|
| References | `recordings/manifests/T.jsonl` | 600 rows, `speaker` = S1/S2/S3, `norm`, `en_spans` |
| Hypotheses | `runs/E-037/readout/paired-1/product/int8/t600.hyp.jsonl` | the shipped INT8 graph + A3 LM, 600 ids |
| Scorer | `stt.eval.score_rows` | one text rule, aligned ETA |

The hypotheses are the ones already published in E-037 (`sha256` recorded in `metrics.json`), so
this experiment cannot introduce harness drift: it only changes the grouping.

## Roles (explicit, frozen here)

| Voice | Role | What a number on it can support |
|---|---|---|
| S3 | never in acoustic training; inspected repeatedly during development | the strongest available voice evidence, **not** an untouched test |
| S1, S2 | acoustic training speakers | dev/diagnostic; no generalization claim |

No speaker in this set is a holdout. A fourth voice does not exist in this repository.

## Protocol

- Score the **same 600 hypotheses** four ways: S1 (200), S2 (200), S3 (200), pooled (600).
- Metrics: pooled WER and aligned ETA per speaker, with **per-utterance** bootstrap (1,000
  resamples, seed 0); word-level error counts (substitutions/insertions/deletions) per speaker;
  category counts alongside denominators.
- Dispersion: min, max and range of the three per-speaker WERs and ETAs. Descriptive only.
- **No cluster bootstrap**: three speakers cannot estimate a between-speaker variance; the
  per-utterance CI is a within-speaker interval and is labelled as such.

## Interpretation Rule

- Per-speaker numbers **replace the pooled number as the reported unit**; the pooled figure stays
  for continuity with earlier reports.
- A large spread across S1/S2/S3 is the honest bound on how a new voice could differ. It is not a
  prediction and it is not a generalization test.
- **No selection, no tuning, no new claim about rare terms or noise.** T is reused evidence.
- If a speaker's CI is much wider than the others, that is reported as measured (200 clips per
  speaker is a small sample).

## Artifacts And Cost

`runs/E-038/` — `PLAN.md`, `per_speaker.py`, `metrics.json`, `RESULT.md`, and the `EXPERIMENTS.md`
entry. Cost: seconds of CPU, no decode, no download, 0 USD.
