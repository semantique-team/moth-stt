# E-031 — RESULT: an intermediate LM clears the rare gate inside the 20 MB ceiling

> Executed 2026-11-12 on the unchanged 0.8.0 int8 graph. Planned and pre-registered in `PLAN.md`
> before the frozen corpus, the arms and any decode. **Adopted and published as 0.8.1 on
> 2026-11-12**, after the browser acceptance, the full noise curve and the decoder-parity check.
> Independent natural mixed-dev is still missing and is declared as such.

> **Documentary correction, 2026-11-12 audit:** measured figures below are retained; PLAN and
> historical JSON are not rewritten. "Rare aligned ETA" is all **81 annotated span tokens in
> 47 rare-bearing utterances** (47 target tokens + 34 co-occurring tokens), not target-only ETA.
> Presence **24/47** is a separate anywhere-in-hypothesis readout. The original PLAN threshold
> is **0.284 (28.4%)**; the later report/record used the historical E-028/E-030 **0.30** target.
> **38/81 passes both**, but 30% was not E-031's pre-registered threshold. The new target-only
> diagnostic and stratified macro bootstrap tooling have not produced new results in this audit.

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
The original E-031 >=28.4% gate passes, as does the historical 30% target (13 hits above
the 25-hit minimum for 30%); the 0.8.0 publication exception is closed by a decode-side change.

| Terms | 0.8.0 | **E-031 A3** |
|---|---:|---:|
| Natural | 436/817 | **432/817** (gate >=431) |
| Gold-only | 126/179 | **127/179** |
| Never-seen | 11/47 | **24/47** |
| 16 gold terms | 19/27 | 19/27 |
| T-600 aligned ETA | 565/1043 | **576/1043** |

Historical noise check (E-011 set, no VAD, relative only): candidate **559** emitted words vs
baseline **567** on 457 clips. This aggregate is not a split music/non-music gate certificate
or the VAD-gated product metric; the audit tooling now requires the split evidence.
The full E-024 SNR curve was then re-run with the new LM on the same checkpoint, decoder, 100 S3
clips and seed: clean 20.03 vs 20.37, 30/20/15/10/5 dB **21.22/24.28/27.67/33.79/40.41** vs
**21.90/25.47/27.33/34.63/40.92**; every condition is a tie, mean 5–30 dB **29.47 vs 30.05**
(`runs/E-031/noise-e031-lm/RESULT.md`).
The mean is descriptive, with no mean CI; condition-level ties do not prove equivalence or
non-inferiority. Shared-pool evidence is not a rigorous upper bound.

## Dev selection (no T used)

Full dev `mls_it_dev` + `ls_en_dev`, 3,939 clips, macro WER, E-030 decoder.

| Arm | Counts | Dev macro | Macro Δ vs A0 (pp) | Historical pooled Δ / 95% CI (pp) | Historical advance |
|---|---|---:|---:|---|---|
| A0 shipped | (2,2) | 22.9463 | — | — | control |
| A1 frozen | (2,2) | 22.9352 | −0.01114 | −0.010 / [−0.031, +0.015] | no (tie) |
| A2 frozen | (1,2) | 22.8541 | **−0.09226** | −0.084 / [−0.140, −0.035] | yes |
| **A3 frozen + terms x3** | **(1,2)** | **22.8494** | **−0.09696** | **−0.091 / [−0.147, −0.040]** | **yes** |

Historically A2 and A3 advanced and A3 was the single gate readout, preferred under the PLAN's
point-estimate +0.10 pp rule. The retained `sweep.json` CIs are **pooled WER**, not macro WER;
they do not certify the pre-registered macro-CI selection rule. Macro deltas above are arithmetic
from retained language scores. The separate [audit readout](audit-readout.json) now reports
paired within-language macro CIs: A1 [-0.03777, +0.01701], A2 [-0.14910, -0.03680],
A3 [-0.15519, -0.04066] pp (1,000 resamples, seed 0). These post-hoc development intervals
do not replace the original sweep or establish independent-test confirmation.

## Why it works

The shipped LM was built on 2026-10-30, before the E-015/E-028 term sentences existed. The frozen
corpus includes them (training-side text, never T/A), and A3 additionally repeats each
term-containing sentence three times. Repetition creates no new raw n-gram types, but raises
counts across pruning thresholds: A2 retains **801,858** trigrams, A3 **802,437**, or **579
additional retained trigrams**, for **3,252 B** (5,378,740 → 5,381,992). Relative to the shipped
LM this is a composite intervention (corpus refresh, bigram pruning and term weighting), not an
isolated causal estimate for x3 or the LM prior. It is build-time weighting, not decode-time
contextual bias (E-027, rejected).

## Gates (Historical Report)

The table preserves recorded measurements and verdicts. The rare row's 30% requirement is the
later historical target, not the original E-031 PLAN's 28.4%; the aggregate noise row also lacks
the split evidence now required. No gate was re-executed by this documentary correction.

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

**Historically adopted** under the reported interpretation of `PLAN.md` rule 6: reported gates pass and both rare ETA and T-600 WER improve with a
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

**Current reproduction boundary:** full git history is required (normal clone, or explicit
`git fetch --unshallow` for a shallow clone; current CI uses `fetch-depth: 0`). The corpus
reproducer verifies pinned code/version, all 22 input hashes and the original corpus hash.
The stricter historical term fingerprint rejects current R-S1 drift from E-032: a gate rerun
requires a separate disposable checkout with inputs at `bd13c19ad45cef63c4588b73e5c8e001d0b4b2da`
and reviewed changed code, or exact pinned inputs restored safely in a disposable checkout.
It is not a generic gate for today's main inputs. This task ran no full LM build/staging,
quality gate or model release; see [workflow prerequisites](../../docs/guides/repository-workflows.md#current-081-reproduction).

`runs/E-031/` — `PLAN.md`, `probe_lm_bytes.py` + `probe/`, `freeze.sh` + `frozen/` (receipt;
the 24 MB corpus itself is gitignored; `reproduce_corpus.py` pins pre-E-032 git inputs and the
historical normalizer/receipt), `build_lm.py` + `build/` + `build.json`,
`dev_sweep.py` + `dev/` + `sweep.json`, `gate.py` + `gate.json` + `gate.log`,
`eval/frozen_1_2_terms/` (hyps, metrics, terms), `stage_release.py`, `release-0.8.1.json`,
`MODEL_CARD.md`, `browser-acceptance/`, `browser-parity/`, `decoder-parity/`,
`noise-e031-lm/` + `noise_compare.py`.
