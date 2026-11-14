# E-039 — RESULT: retained audit with corrected scope and unverified conclusions

## No-Training Correction (2026-11-13)

This notice supersedes the negative cross-corpus MinHash conclusion and the `never44`
taxonomy below. The historical tables and JSON numbers are retained unchanged, not reproduced
by this correction. No full corpus audit, decode, training or download was run.

- **MinHash negative result is unverified**: the old `31 * hash + seed` signatures used one
  multiplier and different offsets, largely preserving the same shingle minimum in every
  column. They were not independent permutations; absence from those candidates does not
  establish absence of high-overlap pairs. `audit2.py` now uses 64 deterministic independently
  drawn affine multiplier/offset pairs modulo the 32-bit prime `2**32 - 5`. CRC32 inputs and
  bounded coefficients keep multiplication plus addition within uint64, without overflow.
  Regression tests recover high-overlap pairs with different original minima against a small
  exhaustive synthetic sample. This is not a corpus-level recall estimate. LSH remains
  approximate, oversized buckets are skipped, and CRC32 shingles can collide.
- **`never44` taxonomy is superseded/unverified**: the old natural counter omitted
  `data/manifests/ls_en_train.jsonl`, although E-014, E-024 and E-028 explicitly train on it.
  This is clean LibriSpeech train-clean-100, not merely the English clips sampled into splicing.
  The corrected natural coverage includes CV-IT, MLS-IT, clean LS-EN and spliced IT+EN.
  E-024/E-028 use CV clean/noisy halves with unchanged parent transcripts; provenance counts
  the CV parent once and spliced reuse separately, not epoch-weighted exposures. E-024
  initializes from E-014, which initializes from E-006, not the E-012 train360 experiment;
  `ls_en_train360` is not part of this accepted acoustic lineage. The old 461/150/44 split
  and claims that all 44 were unseen by 0.7.0 must not be reused as verified taxonomy until
  recomputation. Historical fixed 47-utterance/81-span readouts are not rescored or changed.
  This provenance concerns acoustic source text, not Whisper pretraining, tokenizer or LM
  exposure. `terms3` was used in rejected E-032/E-036, not in shipped E-028.
- **Raw-audio scope is limited**: the retained hashes describe supported loose audio files
  present under `data/raw` at that execution, grouped by their first two path components.
  In particular, `librispeech/LibriSpeech` groups train and dev together, so zero cross-source
  collisions does not certify train/dev disjointness. Archives, deleted/missing raw audio,
  `recordings/audio`, mel shards and decoded-signal equivalence are outside this pass.
  No exhaustive corpus or combined training/evaluation audio-leakage claim follows.

The remaining sections are the **historical execution record**, subject to this notice.

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

## Second Pass: Raw Corpus Audio And Cross-Corpus Near Duplicates

Pre-registered in [PLAN-2](PLAN-2.md); the first pass had explicitly excluded both checks.

### Audio (`data/raw`, 287,429 files, 12.8 GB)

- **0 cross-source collisions** in retained loose-file byte hashes grouped by the first two
  path components; this does not establish train/dev/test or full training/noise disjointness.
- **2 repeated hashes, both inside one source**:
  - `noise_silence`: **120 byte-identical 10 s silence clips**. The "120 silence clips" used by
    E-036 (and the noise gate) are therefore **one unique signal repeated 120 times**, not 120
    independent silences. The finding (`x` on every clip) stands, but its effective sample is one.
  - `room_holdout`: `room_16k.wav` and `room_2026-11-09_16k.wav` are the **same recording under two
    names**. Neither hash appears in `noise_room_tone` (97 distinct files), so the E-025 held-out
    room is genuinely absent from the training noise pool — the "unseen room" claim survives, and
    the duplicate naming is documented instead of being mistaken for a leak.

### Text (247,510 documents, 136,179 MinHash candidates, exact Jaccard verified)

| Pair | Pairs > 0.8 | Class |
|---|---:|---|
| `ls_en_train` ↔ `spliced_it_en` | 5,026 | **by construction**: splicing concatenates MLS-IT and LS-EN clips, so the spliced training body repeats LS-EN sentences |
| `mls_it_train` ↔ `spliced_it_en` | 146 | same construction |
| `cv_it_train` ↔ `spliced_it_en` | 6 | same construction |
| `cv_it_dev` ↔ `cv_it_test` | 44 | Common Voice public-domain prompts reused across speakers |
| `cv_it_test` ↔ `cv_it_train` | 8 | same, near matches (1-word variants) |
| `cv_it_dev` ↔ `cv_it_train` | 6 | same |
| own structural pairs (`rec/*` ↔ `script/*`, anchor ↔ T) | 741/672/600/120 | already explained in the first pass |

The historical candidate search returned zero pairs involving T/external corpora or FLEURS.
That negative conclusion is **superseded/unverified** by the MinHash defect described above;
it cannot establish textual separation from training corpora.

The `spliced_it_en` result is not test leakage but it is a **training-signal** fact worth stating:
the spliced mixture re-uses LS-EN and MLS-IT material that is already in the mixture, so those
sentences are effectively counted twice (once clean, once spliced).

### What The Second Pass Adds

- The retained pass hashed **287,429 supported loose files then present under `data/raw`**;
  it is not an exhaustive audio-leakage audit.
- The E-025 held-out room claim is verified rather than assumed.
- The silence probe's effective sample size (1) is documented before anyone quotes "120 clips".
- Corpus prompt reuse is separated from leakage: text repeats inside a speaker-disjoint corpus are
  a property of the corpus, not a split error.
- It does **not** cover upstream model contamination (Whisper/Whistle/Parakeet) or consent/rights.
