# E-032 — RESULT: the targeted addendum moves rare terms but costs the product gates

> Executed 2026-11-12 from the E-028 checkpoint with 50 new S1 sentences (new contexts, six
> pruned terms x4). Planned and pre-registered in `PLAN.md`. **Rejected**: it improves the rare
> slice but fails the T-600, T-S3, natural and FLEURS-EN gates. 0.8.1 remains the E-031 LM
> candidate; no acoustic change is adopted.

> **Documentary correction (2026-11-12):** original measured pairs and CIs below are preserved.
> T-S3 is **26.75 / 46.76 with the 0.8.0 LM**, **26.41 / 47.61 with E-031**, not the mixed
> 26.75 / 47.61 pair copied into summaries. The original gate table contains **six FAIL rows**,
> not five. "Rare aligned ETA" includes all **81 annotated span tokens** in 47 rare-bearing
> utterances (47 target + 34 co-occurring tokens); target-only ETA is a separate diagnostic —
> computed for 0.8.1 in `runs/E-031/audit-readout.json` (24/47 vs 11/47), not for E-032 — and
> not a replacement gate. E-031 is now the published 0.8.1, not just a candidate.

## Training

| Item | Value |
|---|---|
| Steps | 13,314 (1 epoch) |
| Best checkpoint | **final, step 13,314**; dev **31.06%** (E-028 31.01%) |
| Non-finite steps | 0 |
| New gold | 50 S1 clips, 2.8 min, x20 = 0.93 h |
| Export (diagnostic) | int8 11.91 MB; fp32 median 2.06e-4, worst 1.526e-3 (2/80 over 1e-3); top-1 100% |

The dev curve dipped to 31.56% at step 4,000, recovered to 31.12% at step 9,000 and finished at
31.06%; the final checkpoint is the dev best, so the run did select a new model (not step 20).

## Readout, shipped 0.8.0 LM (the pre-registered baseline)

| Set | 0.8.0 | **E-032** | Δ | 95% CI | Tie? |
|---|---:|---:|---:|---|---|
| T-600 WER | 24.71 | 25.23 | +0.52 | [−0.17, +1.26] | yes |
| T-600 ETA | 54.17 | 53.21 | −0.96 | [−2.58, +0.51] | yes |
| T-S3 WER | 25.21 | 26.75 | **+1.53** | [+0.34, +2.80] | **no** |
| T-S3 ETA | 50.14 | 46.76 | **−3.38** | [−6.50, −0.51] | **no** |
| FLEURS-IT | 20.25 | 20.66 | **+0.41** | [+0.08, +0.73] | **no** |
| FLEURS-EN | 49.81 | 50.47 | **+0.66** | [+0.15, +1.14] | **no** |
| Natural | 436/817 | **424/817** | −12 | gate ≥431 | **fail** |
| Rare aligned ETA | 28.4% (23/81) | **32.1% (26/81)** | +3.7 | [0.0, +9.0] | borderline |

## Readout, E-031 A3 LM (Now Published 0.8.1)

| Set | E-031 | **E-032** | Δ | 95% CI | Tie? |
|---|---:|---:|---:|---|---|
| T-600 WER | 24.08 | 24.83 | **+0.75** | [+0.03, +1.48] | **no** |
| T-600 ETA | 55.23 | 53.88 | −1.34 | [−2.89, +0.28] | yes |
| T-S3 WER | 24.96 | 26.41 | **+1.45** | [+0.09, +2.90] | **no** |
| T-S3 ETA | 51.27 | 47.61 | **−3.66** | [−6.76, −0.65] | **no** |
| FLEURS-IT | 20.16 | 20.54 | **+0.37** | [+0.05, +0.68] | **no** |
| FLEURS-EN | 49.81 | 50.28 | +0.47 | [−0.02, +0.95] | yes |
| Natural | 432/817 | **415/817** | −17 | gate ≥431 | **fail** |
| Rare aligned ETA | 46.9% (38/81) | **51.9% (42/81)** | +4.9 | [0.0, +12.0] | borderline |

The rare direction is consistent (presence 24/47 → 29/47 with the E-031 LM), but it is not a
clean win: the CI touches zero and the natural group loses 17 occurrences.

## Gates (pre-registered, vs 0.8.0)

| Gate | Required | Result | Verdict |
|---|---|---|---|
| Rare aligned ETA | >= 30.0% | 32.1% | PASS |
| T-600 WER | <= 25.21 | 25.23 | **FAIL** (0.02) |
| T-600 ETA | >= 53.67 | 53.21 | **FAIL** |
| T-S3 WER | <= 25.71 | 26.75 | **FAIL** |
| T-S3 ETA | >= 49.64 | 46.76 | **FAIL** |
| Natural | >= 431 | 424 | **FAIL** |
| FLEURS-IT | <= 20.75 | 20.66 | PASS |
| FLEURS-EN | <= 50.31 | 50.47 | **FAIL** |
| Dev-selected checkpoint | not step-20 | final, 31.06% | PASS |
| Package | unchanged | no release | — |

## Interpretation

- The 50 new S1 contexts do give the decoder more rare-term evidence: +3.7 pp over 0.8.0 and
  +4.9 pp over E-031, with presence 11→14 and 24→29 of 47. But the size is small and the CI
  touches zero; it is not the +18 pp observed for E-031's composite corpus/pruning/weighting change.
- The acoustic fine-tune costs the general model: T-S3 WER and ETA regress significantly against
  both baselines, FLEURS-IT/EN regress, and natural terms fall below the 431 gate.
- The S1-only addendum cannot separate speaker adaptation from term learning; the declared
  limitation was not overcome, and this arm does not even produce a rare gain large enough to
  trade against the regressions.

## Decision

**Rejected as a product arm.** Keep the E-031 LM candidate (same acoustic weights, rare gate
closed at 46.9%) as the 0.8.1 direction. Do not iterate the recipe on T; the data remain a
reusable S1 addendum for a future arm with an independent selection set, not a reason to tune
again on this test.

## Artifacts

`runs/E-032/` — `PLAN.md`, `make_terms3.py`, `candidates.jsonl`, `RECORDING.md`, `DATA.md`,
`config.yaml`, `train.out`, `train.log.jsonl`, `checkpoints/` (gitignored), `release/`
(gitignored, diagnostic export), `evaluate.py`, `eval/` (two LM readouts), `eval.json`.
