# E-031 — RESULT: an intermediate LM clears the rare gate inside the 20 MB ceiling

> Executed 2026-11-12 on the unchanged 0.8.0 int8 graph. Planned and pre-registered in `PLAN.md`
> before the frozen corpus, the arms and any decode. **Adopted and published as 0.8.1 on
> 2026-11-12**, after the browser acceptance, the full noise curve and the decoder-parity check.
> Independent natural mixed-dev is still missing and is declared as such.

## Verdict

The winning arm is **A3**: frozen corpus, all bigrams + trigrams with count >= 2, and the 143
frozen sentences containing one of the 44 never-seen terms repeated x3. Browser binary
**5,381,992 B**; package **19,706,289 B** (limit 20,000,000). Decoder unchanged
(beam 12 / top-k 10 / LM 0.55 / bonus 2.7).

Against the published 0.8.0 int8, on the same graph:

| Set | 0.8.0 | **E-031 A3** | Δ | 95% CI | Tie? |
|---|---:|---:|---:|---|---|
| T-600 WER | 24.71 | **24.08** | **−0.63** | [−1.11, −0.20] | no |
| T-600 ETA | 54.17 | **55.23** | **+1.06** | [+0.09, +2.04] | no |
| T-S3 WER | 25.21 | **24.96** | −0.26 | [−1.02, +0.51] | yes |
| T-S3 ETA | 50.14 | **51.27** | +1.13 | [−0.83, +3.23] | yes |
| FLEURS-IT | 20.25 | **20.16** | −0.09 | [−0.21, +0.04] | yes |
| FLEURS-EN | 49.81 | **49.81** | 0.00 | [−0.19, +0.20] | yes |

**Rare gate (primary): aligned ETA 28.4% (23/81) → 46.9% (38/81), +18.5 [+9.6, +31.0].**
The pre-registered >=30% gate now passes with 13 occurrences to spare; the 0.8.0 exception is
closed by a decode-side change.

| Terms | 0.8.0 | **E-031 A3** |
|---|---:|---:|
| Natural | 436/817 | **432/817** (gate >=431) |
| Gold-only | 126/179 | **127/179** |
| Never-seen | 11/47 | **24/47** |
| 16 gold terms | 19/27 | 19/27 |
| T-600 aligned ETA | 565/1043 | **576/1043** |

Noise check (E-011 set, no VAD, relative only): candidate **559** emitted words vs baseline
**567** on 457 clips; no regression. The absolute count is not the VAD-gated product metric.
The full E-024 SNR curve was then re-run with the new LM on the same checkpoint, decoder, 100 S3
clips and seed: clean 20.03 vs 20.37, 30/20/15/10/5 dB **21.22/24.28/27.67/33.79/40.41** vs
**21.90/25.47/27.33/34.63/40.92**; every condition is a tie, mean 5–30 dB **29.47 vs 30.05**
(`runs/E-031/noise-e031-lm/RESULT.md`).

## Dev selection (no T used)

Full dev `mls_it_dev` + `ls_en_dev`, 3,939 clips, macro WER, E-030 decoder.

| Arm | Counts | Dev macro | Δ vs A0 | 95% CI | Advance |
|---|---|---:|---:|---|---|
| A0 shipped | (2,2) | 22.9463 | — | — | control |
| A1 frozen | (2,2) | 22.9352 | −0.010 | [−0.031, +0.015] | no (tie) |
| A2 frozen | (1,2) | 22.8541 | **−0.084** | [−0.140, −0.035] | yes |
| **A3 frozen + terms x3** | **(1,2)** | **22.8494** | **−0.091** | **[−0.147, −0.040]** | **yes** |

A2 and A3 both pass the dev rule; A3 has the lower macro and is the arm the plan prefers on
non-inferiority, so it is the single gate readout. The dev gain is small; the product readout is
decisive.

## Why it works

The shipped LM was built on 2026-10-30, before the E-015/E-028 term sentences existed. The frozen
corpus includes them (training-side text, never T/A), and A3 additionally repeats each
term-containing sentence three times. Duplicated sentences add no n-gram types under `(1,2)` /
`(2,2)`, so the reweighting costs 3 KB over A2 (5,378,740 → 5,381,992), while giving the decoder a
prior for exactly the terms E-026 found reachable but under-weighted. This is the LM lever E-026
identified, exercised at build time rather than as decode-time contextual bias (E-027, rejected).

## Gates (pre-registered)

| Gate | Required | Result | Verdict |
|---|---|---|---|
| Package | <= 20,000,000 B | 19,706,289 B | PASS |
| T-600 WER | <= 25.21 | 24.08 | PASS |
| T-600 ETA | >= 53.67 | 55.23 | PASS |
| T-S3 WER | <= 25.71 | 24.96 | PASS |
| T-S3 ETA | >= 49.64 | 51.27 | PASS |
| FLEURS-IT | <= 20.75 | 20.16 | PASS |
| FLEURS-EN | <= 50.31 | 49.81 | PASS |
| Rare aligned ETA | >= 30.0% | 46.9% | PASS |
| Natural | >= 431 | 432 | PASS |
| Noise non-regression | relative | 559 <= 567 | PASS |

## Decision

**Adopted** under `PLAN.md` rule 6: all gates pass and both rare ETA and T-600 WER improve with a
paired CI excluding zero. The artifact is a **decode-side release candidate**: same acoustic
weights, same decoder settings, one new `lm_bpe3.bin`.

Done before publication:

- browser (JS/WASM) decode: real-Chromium acceptance **4/4** on the staged bundle, and
  **decoder-level JS↔Python parity exact** on identical logits. The single 3 s fixture differs
  from Python by two words with **both** the 0.8.0 and 0.8.1 LM, so the residual is the browser
  acoustic path, not the new LM (`runs/E-031/browser-parity/RESULT.md`);
- full E-024 noise curve with the new LM: all conditions tie, mean 5–30 dB **29.47 vs 30.05**
  (`runs/E-031/noise-e031-lm/RESULT.md`);
- model card `runs/E-031/MODEL_CARD.md`, release record `runs/E-031/release-0.8.1.json`
  (`published: true`), staged bundle `runs/E-031/release/` validated at 19,706,289 B;
- **published 2026-11-12**: HF rev `1257327e…`, Space rev `9dfe70df…` `RUNNING`, remote hashes
  verified.

Still required for the next evidence level, and **blocked**:

- independent natural mixed-dev and a new-voice rare readout. No new speaker is available at this
  date; the published card states that the evidence remains T/S3.

## Artifacts

`runs/E-031/` — `PLAN.md`, `probe_lm_bytes.py` + `probe/`, `freeze.sh` + `frozen/` (receipt;
the 24 MB corpus itself is gitignored and reproducible), `build_lm.py` + `build/` + `build.json`,
`dev_sweep.py` + `dev/` + `sweep.json`, `gate.py` + `gate.json` + `gate.log`,
`eval/frozen_1_2_terms/` (hyps, metrics, terms), `stage_release.py`, `release-0.8.1.json`,
`MODEL_CARD.md`, `browser-acceptance/`, `browser-parity/`, `decoder-parity/`,
`noise-e031-lm/` + `noise_compare.py`.
