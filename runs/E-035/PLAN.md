# E-035 — Spending The Last 293,711 Bytes On Single-Count Trigrams

**Status:** pre-registered 2026-11-13, **before** the build and before any decode. Decode-only:
no training, no new data, no checkpoint or weight change. T, T-S3, FLEURS, the natural set and the
noise pool are **not read here**; this is development data only.

## Question

0.8.1 ships **19,706,289 B** of required assets and its LM binary is **5,381,992 B**, leaving
**293,711 B** below the 20 MB ceiling. E-034 showed that full trigram coverage is a real dev lever
(`(1,1)`: **−0.644 pp** macro, paired CI excluding zero) but needs **12.3 MB**, which cannot ship
here. This experiment asks the only remaining decode-side question inside the ceiling:

> Can the 293,711 B of headroom buy a measurable dev gain by adding **single-count trigrams** on
> top of the shipped `(1,2)` + term-×3 LM, selected by the frequency of their preceding bigram?

If the answer is no, the headroom is not worth spending and the LM is frozen at the E-031 arm. If
the answer is yes, the arm gets a separate product readout; it is not adopted from dev.

## What The Bytes Buy (Layout Arithmetic, Not A Result)

The browser format is described in `stt/lm.py::save_browser` and was cross-checked against E-034:

- trigram entry: `u2` token + `i2` quantized log-probability = **4 B**;
- new trigram context: one `u4` key + one extra `u4` offset = **8 B** on top of its first entry;
- bigram section unchanged (all bigrams are already kept).

E-034 confirms the model on its own arms: `(2,1)` vs `(1,1)` differ only by 69,859 bigrams and
279,436 B = 4 × 69,859, and the trigram sections differ by 6,931,448 B =
4 × 1,477,990 entries + 8 × 127,436 contexts.

Shipped baseline and the build inputs:

| Quantity | Value |
|---|---:|
| 0.8.1 package | 19,706,289 B |
| shipped LM binary (E-031 `frozen_1_2_terms`) | 5,381,992 B |
| non-LM assets (`19,706,289 − 5,381,992`) | 14,324,297 B |
| headroom below 20,000,000 B | 293,711 B |
| **maximum admissible LM binary** | **5,675,703 B** |

On the frozen E-031 corpus at term weight ×3 the full trigram set has 265,001 contexts. The
shipped set keeps **802,437 trigrams** (count ≥ 2) across **137,565 contexts**. The remaining
**1,477,990 single-count trigrams** are the candidate pool; 268,221 of them sit in the 127,436
contexts that have no kept trigram, while **1,209,769 sit in the 137,565 contexts that already
have a kept trigram** and therefore cost exactly 4 B each.

At 4 B/entry, the budget buys **73,427** such entries: `5,381,992 + 4 × 73,427 = 5,675,700 B`,
package **19,999,997 B**. The build verifies the real binary and trims at the cap if the layout
arithmetic is off; the predicted count is arithmetic, not a measurement.

## Arms

Built from the same frozen corpus, receipt, tokenizer, backoff 0.4 and `BPETrigramLM` code path as
E-031/E-034. The control is rebuilt in this run, not reused blindly.

| Arm | Definition | Purpose |
|---|---|---|
| `A3_shipped` | frozen corpus, all bigrams, trigrams ≥ 2, term weight ×3 | control: must reproduce the shipped binary byte for byte |
| `B1_budgeted` | A3 plus single-count trigrams chosen by the rule below, binary ≤ 5,675,703 B | the only candidate |

`B1` is **one arm, not a sweep**. E-034 already established that extra coverage helps dev; this
experiment asks whether the affordable slice of it helps enough to matter. Alternative rankings
(KL/entropy pruning), replacing existing trigrams, or a different binary layout would each be a
new pre-registered arm and are **not** tested here.

## Selection Rule (Frozen Now)

1. Start from the shipped set: all bigrams, trigrams with count ≥ 2, term weight ×3.
2. Rank every single-count trigram by **the occurrence count of its context**
   (`sum(counts of the context row)`, i.e. how often the preceding bigram occurs), descending.
3. Break ties deterministically by `(prev2, prev1, token)` ascending, so the selection is
   reproducible from the corpus hash and the builder hash alone.
4. Add entries in rank order while the predicted binary size
   `5,381,992 + 4 × entries + 8 × new_contexts` stays ≤ **5,675,703 B**; skip an entry that would
   not fit and continue. Stop within 4 B of the cap.
5. Build the LM once, verify the actual binary size and the package arithmetic, and if the actual
   size exceeds the cap, drop the lowest-ranked selected entries and rebuild until it fits.

Pre-run corpus arithmetic puts the cut at contexts occurring **≥ 698** times and touches **1,359**
contexts. This is a **frequency heuristic**, not entropy pruning; it rewards single-count
continuations of frequent contexts and ignores their bigram-fallback probability.

## Protocol

- Corpus `runs/E-031/frozen/lm-corpus.jsonl` (receipt
  `runs/E-031/frozen/lm-corpus.receipt.json`, sha256 `c8d94192…`), tokenizer
  `data/tokenizer/moth_bpe.model` (sha256 `d6e1054e…`), term weight ×3, backoff 0.4.
- Control check: `A3_shipped` must reproduce binary **5,381,992 B** and sha256
  **`3be1177b…f16f1`**. If it does not, the run stops before decoding (harness drift), exactly as
  E-034's control reproduced E-031 byte for byte.
- Decoder frozen as in E-030/E-031/E-034: beam 12 / top-k 10 / LM weight 0.55 / token bonus 2.7,
  fp32 E-028 checkpoint (`runs/E-028/checkpoints/ckpt_best.pt`), the same `MelDataset` and batch
  plan (`max_seconds=192`, `max_clips=32`, seed 0, epoch 0).
- Dev only: `mls_it_dev` (1,248 clips) + `ls_en_dev` (2,691 clips) = **3,939 clips / 10.48 h**.
- Both arms are decoded in this run, control first, so they share session and code.
- Harness check: the control's 3,939 hypotheses are compared with E-034's retained `A3_shipped`
  file. Identity is recorded in `sweep.json`; a mismatch does not invalidate the within-run pairing
  but is reported and must be explained before claiming continuity with E-034.
- Metrics: pooled WER/ETA, per-language WER, equal-language **macro** WER, and the paired
  per-utterance bootstrap (1,000 resamples, seed 0, within language, `macro_wer_bootstrap`).

## Pre-Registered Decision Rule

`B1_budgeted` **advances** only if all of the following hold:

1. its dev macro WER point estimate is **lower than `A3_shipped`**;
2. the paired 95% CI upper bound of the macro difference is **< 0**;
3. its browser binary is **≤ 5,675,703 B** and the package is **≤ 20,000,000 B**.

Nothing advances on a tie, on pooled WER alone, or on one language improving while the other
regresses. There is no minimum-effect clause beyond the interval: if the interval excludes zero in
the right direction, the effect is large enough to be interesting at this sample size; if it does
not, the result is a tie and the RESULT will recommend closing the headroom question for this LM
format. The freeze itself remains the owner's decision.

If `B1` advances, the **product readout is a separate step and is not authorized here**: one
int8 run on T-600, T-S3, FLEURS-IT/EN, the 47 rare rows, the natural set and the noise pool with
the 0.8.1 assets, plus browser parity and latency. Only a candidate that passes that readout would
become a decode-only 0.8.2.

## Expected Size Of The Effect

E-034's full extra coverage is 1,477,990 entries for −0.598 pp macro; 73,427 entries are **4.97%**
of that, so a proportional expectation is **≈ −0.03 pp**, below the arm-to-arm band E-031 itself
reported (−0.011 to −0.097 pp). That estimate is not confounded by term weighting: E-034's extra
arms used ×1, but E-031 measured the same pruning at ×1 vs ×3 (A2 vs A3) as a **0.0047 pp tie**, so
the 0.6 pp is coverage, not the term weight. The prior is therefore a **tie**. The experiment is worth its
~15 minutes because it answers the last in-budget decode question with a number, and a null is the
useful outcome: it says the 293 KB should not be spent on this format.

## What This Does Not Test

- Replacing or dropping existing trigrams (rank-all/entropy pruning), which could reallocate bytes
  rather than only spend them.
- Four-grams or a different binary layout/quantization, both of which need runtime changes.
- Term weighting (E-034: ×5 is a tie), a larger LM (E-034: over budget), or any training-side
  change (E-036 is the separate arm).
- Browser load time, memory and latency for a 5.68 MB LM, and any product-set behaviour; those
  are measured only if the arm advances.
- A new dev set: this is the same MLS-IT/LS-EN read speech that has informed every previous
  selection round. A dev gain is a selection signal, not a generalization claim.

## Artifacts And Cost

`runs/E-035/` — `PLAN.md`, `build_budgeted.py` + `build/` (pickles and binaries, gitignored),
`build.json`, `dev_sweep.py`, `dev/<run>/{A3_shipped,B1_budgeted}/{dev.hyp.jsonl,dev.metrics.json}`
+ `sweep.json`, `decide.py` + `decision.json`, `RESULT.md`, and the `EXPERIMENTS.md` entry written
with the result.

Cost: build ~15 s for both LMs, two dev decodes ~8 min (E-034 measured 240 s/arm), no training, no
download, 0 USD. Pre-registration commit: the commit that adds this file, before any build or
decode.
