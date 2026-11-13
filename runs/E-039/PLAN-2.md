# E-039 — second pass: raw corpus audio and cross-corpus near duplicates

**Status:** pre-registered 2026-11-13, **before** the second pass runs. Read-only, no training, no
decode, no data change. It extends [E-039](PLAN.md), whose first pass explicitly excluded these two
checks.

## Question

The first pass covered every text source, our own recorded audio, corpus id splits and term
provenance. It **excluded**:

1. re-hashing `data/raw` (13 GB, 287,429 audio files) — so "no audio appears in two sets" was a
   claim about our recordings only;
2. near-duplicate text across **external** corpora — so the 42 CV-IT dev/test shared sentences were
   the only corpus-level observation.

> Does any external-corpus audio file appear under two sources (e.g. the same clip downloaded into
> CV-IT train and dev, or into the noise pool twice)? And are there near-duplicate texts (word
> trigram Jaccard > 0.8) between the evaluation corpora and the training corpora, or across a
> corpus's own splits, beyond the exact matches already reported?

## Design

- **Audio**: sha256 of every file under `data/raw` with an audio extension (`.wav`, `.flac`,
  `.mp3`, `.m4a`), recorded with size and source label (first path component under `data/raw`,
  second too when present). Duplicates are reported as hashes appearing in **more than one
  source**, and separately as repeated hashes inside a source. Parallel hashing, deterministic
  output.
- **Text**: MinHash + LSH candidates over word trigrams (64 permutations, 16 bands of 4), then an
  **exact** Jaccard verification with the same 0.8 threshold as the first pass. Scope: external
  training corpora (`cv_it_train`, `mls_it_train`, `ls_en_train`, `spliced_it_en`) against external
  evaluation corpora (`cv_it_dev/test`, `mls_it_dev/test`, `ls_en_dev`, FLEURS IT/EN dev/test),
  plus within-corpus split pairs. Our own scripts/manifests are excluded: the first pass already
  measured them exactly.
- Buckets larger than a fixed cap are skipped and counted, so the report never silently claims
  completeness it does not have.

## Rules

- Report counts and examples only; nothing is deleted, re-split or fixed.
- A shared text inside one corpus's own splits is **not** audio leakage by itself: crowd-sourced
  corpora reuse prompts across speakers, and the splits are speaker-disjoint (established in the
  first pass). The report states that distinction instead of collapsing it.
- Deterministic seeds; sorted output.

## Artifacts And Cost

`runs/E-039/audit2.py`, `audit2.json`, and a section appended to `RESULT.md`. Cost: a few minutes of
local I/O and CPU, no network, 0 USD.
