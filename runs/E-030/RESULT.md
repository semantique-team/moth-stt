# E-030 — RESULT: decoder recalibration brings E-028 INT8 to 5 of 6 gates

> Executed 2026-11-11 on the unchanged E-028 INT8 graph; planned in `PLAN.md`. **Published
> 2026-11-12 as 0.8.0 with this decoder.** The artifact passes five of the six E-028 gates; the
> primary rare-ETA gate is missed by two aligned occurrences and is a documented exception.

## Verdict

A decoder-only change recovers most of the E-028 INT8 gap:

- T-600 **24.71 WER / 54.17 ETA** (E-028 old arm 25.49 / 53.50; 0.7.0 shipped 25.14 / 54.65);
- T-S3 **25.21 / 50.14** (E-028 old 25.89 / 49.01; 0.7.0 27.00 / 48.45);
- natural **436/817** (gate ≥431; E-028 old 428; 0.7.0 441);
- rare presence **11/47** and aligned ETA **28.4% (23/81)**: against 0.7.0 shipped (8/47,
  22.2%) that is **+6.2 points [+1.1, +11.8]**; against 0.7.0 with the same arm (7/47, 21.0%)
  **+7.4 [+1.3, +13.6]**; both CIs exclude zero;
- FLEURS-IT/EN **+0.29 / +0.42** vs 0.7.0 with the same arm, both ties and inside the 0.5 gate.

The remaining miss is the primary gate: rare aligned ETA **28.4% vs ≥30%**, two aligned
occurrences short (23/81 vs the required 25/81). E-028 fp32 has 25/81, so the loss is quantization,
not the decoder; the head-fp32 variant does not recover it (also 23/81).

## Dev selection (no T used)

12 arms, dev = CV-IT 600 + FLEURS-IT 334 + FLEURS-EN 343. `frozen` is the shipped arm.

| Arm | beam / top-k / LM / bonus | Dev WER | Del % | Ins % |
|---|---|---:|---:|---:|
| **lm_0.55_b2.7** | **12 / 10 / 0.55 / 2.7** | **30.949** | 5.50 | 3.12 |
| lm_0.55_b3.0 | 12 / 10 / 0.55 / 3.0 | 31.019 | 4.94 | 3.56 |
| beam16_b2.7 | 16 / 10 / 0.50 / 2.7 | 31.049 | 5.02 | 3.42 |
| frozen | 12 / 10 / 0.50 / 2.4 | 31.138 | 5.72 | 2.93 |
| topk12_b2.7 | 12 / 12 / 0.50 / 2.7 | 31.203 | 5.03 | 3.38 |
| bonus_2.7 | 12 / 10 / 0.50 / 2.7 | 31.208 | 5.04 | 3.40 |
| bonus_2.0 | 12 / 10 / 0.50 / 2.0 | 31.268 | 6.74 | 2.30 |
| bonus_3.0 | 12 / 10 / 0.50 / 3.0 | 31.383 | 4.43 | 3.87 |
| lm_0.45_b2.7 | 12 / 10 / 0.45 / 2.7 | 31.488 | 4.62 | 3.66 |
| cheap_b2.7 | 8 / 8 / 0.50 / 2.7 | 31.522 | 5.18 | 3.38 |
| bonus_3.3 | 12 / 10 / 0.50 / 3.3 | 31.892 | 3.95 | 4.42 |
| lm_0.45_b3.0 | 12 / 10 / 0.45 / 3.0 | 31.922 | 4.11 | 4.25 |

`lm_0.55_b2.7` is 0.189 points better than frozen and improves all three dev sets
(CV-IT 20.02 vs 20.15, FLEURS-IT 20.57 vs 20.69, FLEURS-EN 50.99 vs 51.30). It cuts deletions
(5.72 → 5.50) while adding some insertions (2.93 → 3.12), consistent with the length hypothesis.
The gain is small; T is the decisive readout.

## T-600 / T-S3 (one readout)

| System / arm | T-600 WER | T-600 ETA | T-S3 WER | T-S3 ETA |
|---|---:|---:|---:|---:|
| 0.7.0 INT8, shipped arm | 25.14 | **54.65** | 27.00 | 48.45 |
| 0.7.0 INT8, E-030 arm | 24.94 | 54.75 | 27.09 | 48.45 |
| E-028 INT8, old arm | 25.49 | 53.50 | 25.89 | 49.01 |
| **E-028 INT8, E-030 arm** | **24.71** | **54.17** | **25.21** | **50.14** |
| E-028 fp32 (reference) | 24.66 | 54.65 | 25.55 | 49.01 |

Paired bootstrap, E-028 E-030 arm vs:

| Comparison | WER Δ | 95% CI | Tie? | ETA Δ | 95% CI | Tie? |
|---|---:|---|---|---:|---|---|
| 0.7.0, same arm | −0.23 | [−1.06, +0.55] | yes | −0.58 | [−2.36, +1.21] | yes |
| 0.7.0, shipped arm | −0.43 | [−1.28, +0.40] | yes | −0.48 | [−2.29, +1.35] | yes |
| E-028, old arm | −0.77 | [−1.23, −0.32] | **no** | +0.67 | [−0.10, +1.52] | yes |

So against the published model T-600 is a statistical tie (WER slightly better, ETA slightly
worse), while T-S3 is clearly better. Against the old E-028 arm the WER gain is significant.

## Terms

| Metric | 0.7.0, same arm | E-028 old arm | **E-028 E-030 arm** |
|---|---:|---:|---:|
| Natural | 440/817 | 428/817 | **436/817** |
| Gold-only | 130/179 | 124/179 | 126/179 |
| Rare presence | 7/47 | 12/47 | 11/47 |
| Rare aligned ETA | 21.0% (17/81) | 29.6% (24/81) | **28.4% (23/81)** |
| 16 gold terms | 22/27 | 19/27 | 19/27 |

Rare aligned ETA difference (paired bootstrap over the 47 rare utterances):

- vs 0.7.0, same arm: **+7.4 [1.3, 13.6]**, significant;
- vs 0.7.0, shipped arm: **+6.2 [1.1, 11.8]**, significant;
- vs E-028 old arm: **−1.2 [−4.4, 0.0]**, tie (one occurrence).

## FLEURS-IT/EN

| Set | 0.7.0, same arm | E-028 E-030 arm | Δ | 95% CI | Tie? |
|---|---:|---:|---:|---|---|
| FLEURS-IT test | 19.96 | 20.25 | +0.29 | [−0.01, +0.60] | yes |
| FLEURS-EN test | 49.39 | 49.81 | +0.42 | [−0.09, +0.92] | yes |

Both are inside the 0.5 non-inferiority margin and statistically ties.

## Gates (E-028 pre-registered)

| Gate | Required | Result | Verdict |
|---|---|---|---|
| Rare aligned ETA | ≥ 30.0% | 28.4% (23/81) | **FAIL** (2 occurrences short) |
| T-600 aligned ETA | ≥ 53.7% | 54.17% | PASS |
| T-600 WER | ≤ 25.7% | 24.71% | PASS |
| Natural terms | ≥ 431/817 | 436 | PASS |
| FLEURS-IT/EN not worse >0.5 | CI check | +0.29 / +0.42, ties | PASS |
| Dev-selected checkpoint | not step-20 | E-028, dev-selected | PASS |
| Package | ≤ 20,000,000 B | 19,440,760 B | PASS |
| Browser acceptance | runtime | not run with new arm | to do if adopted |

The first six rows are the E-028 gates; package and browser are release checks. Five of the six
E-028 gates pass; only the primary rare-ETA gate is missed.

## Head-fp32 check (over budget, diagnostic only)

The `int8_head_fp32` graph with the E-030 arm gives T-600 **24.54 / 54.36**, T-S3
**25.13 / 49.86**, FLEURS-IT/EN **20.35 / 49.62**, natural **438**, rare **23/81 (28.4%)**.
It improves T-600 slightly and natural by two terms, but **does not recover the rare gate**, and
its package is 20.62 MB. It is not worth the size work for this arm.

## Interpretation

- The decoder recovers most of the old E-028 INT8 gap: +0.78 WER / +0.67 ETA on T-600, +8 natural
  terms, and T-S3 improves further. The rare gain over 0.7.0 remains significant.
- The only gate miss is the primary rare ETA, by two aligned occurrences. The loss is
  quantization: E-028 fp32 has 25/81, the INT8 graph 23/81. It is not a decoder problem (head-fp32
  does not recover it) and not a QAT problem (E-029 closed the dev gap but degraded the product).
- Against the published 0.7.0 the case is: T-600 tie, T-S3 better, rare significantly better,
  natural 4–5 terms lower (within the 431 gate), FLEURS ties. This is a much stronger tradeoff
  than the old E-028 INT8 arm, but it still misses the pre-registered primary gate.

## Decision

**Published 2026-11-12 as 0.8.0 with the E-030 decoder.** The package is **19,440,761 bytes**
(config grew one byte), browser acceptance is **4/4** with decoder 0.55 / 2.7, and the HF model +
Space were verified live. The rare shortfall is documented in the model card and release record.
The remaining work is a data/model problem: targeted rare data, a same-size LM, or a natural
mixed-dev for future selection.

## Artifacts

- `decode_sweep.py`, `sweep.json`, `sweep.out` — 12-arm dev selection.
- `eval_decode.py`, `eval/` — T-600/T-S3/FLEURS readout for 0.7.0 and E-028 with the new arm.
- `compare.json` — paired WER/ETA/rare and FLEURS comparisons.
- `head_fp32/` — diagnostic head-fp32 readout; `head_fp32_release/` is gitignored (ONNX).
- The E-028 package itself is unchanged: `runs/E-028/release/` (gitignored).
