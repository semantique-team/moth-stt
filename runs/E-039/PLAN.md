# E-039 — Leakage audit (first pass)

**Status:** pre-registered 2026-11-13, **before** running the audit. Read-only: no training, no
decode, no data change. Produces a registry and a list of findings for human review.

## Question

The closure plan lists a **leakage audit** as a paper gap: "hash audio per source, near-dup
testuale su train/dev/test/T/R/pool, provenienza dei termini". Nothing has ever been checked
end-to-end. The repository repeatedly claims that

- T and R scripts are disjoint,
- A reuses T sentences (intended),
- S3 recorded no R and no acoustic training used S3,
- the 44 "never" rare terms appear in no training text,

and those claims are load-bearing for the paper. This audit tests them mechanically:

> Do any evaluation texts or clips also appear in training text, training scripts, or another
> evaluation set; do any audio files repeat across sets; are the corpus train/dev/test id splits
> disjoint; and does the natural/gold/never term classification still hold against the actual
> training text?

## Scope Of This Pass

**Included**

1. **Exact normalized text duplicates** across every text source: `recordings/scripts/{T,A,R-S1,
   R-S2,pool}.jsonl`, `recordings/manifests/*.jsonl`, and all `data/manifests/{cv_it_*,mls_it_*,
   ls_en_*,fleurs_*,noise,spliced_it_en}.jsonl`.
2. **Near duplicates** (word-trigram Jaccard > 0.8) **within our own data only**: scripts and
   recording manifests. Cross-corpus near-dups are out of scope for this pass and are stated as
   such.
3. **Audio registry**: sha256 of every wav under `recordings/audio/`, grouped by set; cross-set
   hash collisions; presence check for every `audio` path in `recordings/manifests/`.
4. **Id disjointness** inside each external corpus (train vs dev/test) and between our T/A/R sets.
5. **Term provenance**: recompute the natural/gold/never classification of T English spans from the
   actual training text, including the E-015/E-028 addenda boundary (`R-S1-0642`, `R-S2-0624`),
   and verify the 44 never terms are still absent from all training text.

**Excluded, and why**

- Re-hashing `data/raw` (13 GB) and `data/shards` (49 GB): external corpora audio is used as
  downloaded and split by the source; the risk audited here is our own recordings and texts.
  Corpus splitting is checked at id level instead.
- Transcript-aware filtering or any change to the data: this pass only reports.
- Consent/rights verification: a human/legal matter, not a hash.

## Rules

- Report **every** finding with the ids involved; do not auto-delete or auto-fix anything.
- Expected overlaps (A ⊂ T by construction; term addenda inside R) are reported separately from
  unexpected ones, so the reader can see the distinction.
- A finding is **not** a conclusion: near-dups and shared prompts need human review, and the report
  says so.
- Deterministic: fixed seeds, sorted output, no network.

## Artifacts And Cost

`runs/E-039/` — `PLAN.md`, `audit.py`, `audit.json`, `RESULT.md`, `EXPERIMENTS.md` entry. Cost:
minutes of local I/O, no GPU, no download, 0 USD.
