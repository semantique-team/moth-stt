# E-039 — RESULT: no literal leakage, and the "never seen" term label is a 0.7.0 property

> Executed 2026-11-13. Read-only audit: no training, no decode, no data change. Every finding below
> is reported, not fixed; nothing in the repository was modified by the audit itself except this
> run directory and the documentation notes listed at the end.

## Verdict

- **Text**: 2,055 normalized texts appear in more than one source. Every pair is either
  **structural** (anchor reads T; T manifest is the recorded T script; R addenda are part of the R
  scripts) or **corpus text repetition** (42 sentences shared between CV-IT dev and test, which is
  how Common Voice works: the same prompts are read by different speakers). **Zero unexpected
  pairs.** Near duplicates at trigram Jaccard > 0.8 **within our own data: 0 pairs**.
- **Audio**: 2,209 wav files, **0 repeated hashes**, **0 cross-set collisions**, **0 missing**
  files referenced by `recordings/manifests/`.
- **Ids**: train/dev/test id sets are disjoint inside CV-IT, MLS-IT and LS-EN (0 shared ids in every
  comparison).
- **Terms**: the 44 "never seen" terms are never seen **by 0.7.0**, not by 0.8.1. Against the
  training text of E-028 (S1 and S2 term addenda included) the split is natural 461, gold 150,
  **addendum 44, never 0**. The 47 T utterances and 44 terms reproduce E-015's boundary exactly,
  and `terms3` (E-036 only, rejected) introduces no additional term.

The second point is the one that changes documentation: **"rare terms never seen in any training
text" is false for the shipped 0.8.1** and must be written as "never seen by the 0.7.0 baseline".
The metric itself (rare ETA 38/81, rare WER) is unaffected — it is computed on the same 47
utterances — but its label was wrong.

## Text Duplicates

| Pair | Texts | Class | Reason |
|---|---:|---|---|
| `script/A` ↔ `script/T` | 60 | structural | anchor A is drawn from 60 reviewed T sentences |
| `rec/T` ↔ `script/T` | 600 | structural | T manifest is the recorded T script |
| `rec/R-S1` ↔ `script/R-S1` | 741 | structural | R-S1 manifest is the recorded R-S1 script |
| `rec/R-S2` ↔ `script/R-S2` | 672 | structural | R-S2 manifest is the recorded R-S2 script |
| `rec/A-S*` ↔ `rec/T`, `script/T`, `script/A` | 60 each | structural | each anchor reads the same T sentences |
| `rec/A-S1` ↔ `rec/A-S2` ↔ `rec/A-S3` | 60 | structural | all anchors read the same 60 sentences |
| `rec/R-S1-terms(+terms3)` ↔ `rec/R-S1` | 50 each | structural | addendum takes are part of the R script |
| `rec/R-S2-terms` ↔ `rec/R-S2` | 50 | structural | addendum takes are part of the R script |
| `rec/T` ↔ `rec/T-S2`, `rec/T-S2` ↔ `script/T` | 200 | structural | T-S2 is the S2 subset of T |
| `data/cv_it_dev` ↔ `data/cv_it_test` | 42 | corpus text repeat | crowd-sourced prompts reused across speakers |
| anything else | — | **none** | — |

Near duplicates (word trigram Jaccard > 0.8) inside `script/*` and `recordings/manifests/*`:
**0 pairs** out of **10,918 candidate pairs actually evaluated**, so the null result is not vacuous.
The highest sub-threshold pair is exactly **0.8000** (`rec/R-S1:R-S1-0418` "ho pushato la
correzione sul ramo sbagliato" vs `script/R-S1:R-S1-0618` "ho pushato la correzione sul ramo"): a
shared prefix between two distinct sentences, not a duplicate. T and R are disjoint beyond exact
match, which is stronger than the script importer requires.

## Audio And Ids

- Registry: 2,209 wav files hashed by set; **no hash appears in two different sets**, so no take was
  copied between T, R, anchor or speaker folders. Every `audio` path referenced by every
  `recordings/manifests/*.jsonl` resolves under its proper root (T at `recordings/audio/`, anchors
  under `anchor/`, R sets under `R-S1/` and `R-S2/`).
- Id disjointness: CV-IT train/dev/test 0 shared; MLS-IT train/dev/test 0 shared; LS-EN train/dev 0
  shared.

## Term Provenance

| Classification | Terms | Meaning |
|---|---:|---|
| natural | 461 | present in CV-IT, MLS-IT or spliced training text |
| gold (pre-addendum) | 150 | present in R-S1/R-S2 before the term addenda |
| **never by 0.7.0** | **44** | absent from all pre-addendum text; **47 T utterances** contain one |
| addendum (E-015/E-028) | 44 | exactly the 44 above, present in the S1/S2 addenda |
| never by 0.8.1 | **0** | — |
| terms3-only | 0 | E-036's third addendum adds no new term |

## What This Does And Does Not Establish

- **Does**: mechanically confirm T/R/pool separation, anchor reuse, absence of duplicated audio,
  corpus split disjointness, and the exact provenance of every English term in T.
- **Does**: correct the "never seen" label for the shipped model, which is a factual error in any
  document that attaches it to 0.8.1 numbers.
- **Does not**: re-hash `data/raw` (13 GB) or `data/shards` (49 GB); external corpora are used as
  published and checked at id level, not by audio content.
- **Does not**: cover upstream contamination (Whisper/Whistle/Parakeet overlap), consent/rights, or
  annotation validity. Those remain listed gaps.
- **Does not**: turn near-duplicate candidates into decisions. There are none in our own data, but
  external-corpus near-dups were out of scope.

## Documentation Follow-Ups Applied

- `docs/03-data.md`: the hygiene bullet now says T text never appears in R or pool and that the
  anchor repeats 60 T sentences by design.
- `docs/13-current-model-and-paper-readiness.md`: note that the 44 terms were never seen by 0.7.0,
  were introduced by the E-015/E-028 addenda, and are therefore **seen** by 0.8.1.
- `docs/plans/2026-11-13-model-closure.md`: leakage audit marked as first pass done, with the
  remaining human-review scope named.

## Artifacts And Cost

`runs/E-039/` — `PLAN.md`, `audit.py`, `audit.json` (full registry, hashes, per-source row counts,
term counts), this report. Cost: ~3 s of I/O plus hashing 492 MB, no decode, no training, 0 USD.

Reproduction: `uv run python runs/E-039/audit.py`.
