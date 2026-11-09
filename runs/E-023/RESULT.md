# E-023 — RESULT: the LM was already tuned right, and the one improvement does not fit

> Executed 2026-11-09. Plan and decision rule: `PLAN.md` (written before the sweep).
> **Decode-only**, like E-022: no training, no checkpoint change, selection on **full dev**
> (`mls_it_dev` 1,248 + `ls_en_dev` 2,691 = 3,939 clips / 10.48 h), gate sets read once.

## Verdict

**Nothing adopted.** E-022's decode settings are already the LM's optimum on a selection set ten
times wider than the one E-008 used, and the single improvement found — an LM with count pruning
≥ 1 — does not fit the package. Recorded as a **negative result with a quantified boundary**: it
tells us exactly what the 20 MB ceiling costs, and the answer is *less than it looks*.

| Phase | Question | Answer |
|---|---|---|
| 1 — `lm_weight` × `gamma` (20 configs) | is E-022's cell the optimum on full dev? | **yes, exactly**: best is w 0.5 / γ 2.4, macro **22.98** |
| 2 — `backoff` (5 values) | is 0.4 still right? | **yes**: 0.4 → 22.98, 0.3 → 23.01, 0.5 → 23.06, 0.2 → 23.09, 0.6 → 23.23 |
| 3 — count pruning (4 arms) | can the LM shrink, or is there quality left? | **both**: (1,1) gains 0.65 dev points but does not fit; (3,3) saves 1.8 MB at a 0.56-point cost |

## Phase 1 — the weight was already right

Full-dev macro WER, best five of twenty:

| `lm_weight` | `gamma` | dev macro (it / en) |
|---|---|---|
| **0.5** | **2.4** | **22.98** (28.16 / 17.80) |
| 0.5 | 2.0 | 23.13 |
| 0.5 | 2.8 | 23.17 |
| 0.4 | 2.0 | 23.20 |
| 0.6 | 2.8 | 23.23 |

The basin E-022 found on 400 clips is the basin full dev finds on 3,939. Worth stating plainly:
**E-008 tuned this parameter to 0.2 under a miscalibrated decoder and got the wrong answer; E-022
re-tuned it to 0.5 on 400 clips and got the right one.** The second tuning was not lucky — a wider
selection set reproduces it exactly.

## Phase 3 — the one real finding, and why it is not adopted

| LM | pickle | browser binary | dev macro | Δ vs current |
|---|---|---|---|---|
| **(1,1)** | 6.93 MB | **12.31 MB** | **22.33** | **−0.65** |
| (2,2) = shipped | 3.55 MB | 5.12 MB | 22.98 | — |
| (2,3) | 2.76 MB | 3.42 MB | 23.50 | +0.52 |
| (3,3) | 2.67 MB | 3.28 MB | 23.54 | +0.56 |

The (1,1) LM keeps every n-gram the corpus contains, and it is **measurably better on the gate
sets** (paired, 1,000 resamples, vs 0.6.0):

| Set | 0.6.0 | (1,1) | Δ [95% CI] |
|---|---|---|---|
| T-600 | 25.92 | **25.14** | **−0.77** [−1.40, −0.11] |
| FLEURS-IT | 20.46 | **19.74** | **−0.72** [−1.01, −0.42] |
| CV-IT dev (full, 14,643) | 23.52 | **22.87** | **−0.65** [−0.75, −0.55] |
| FLEURS-EN | 49.63 | 49.27 | −0.37 [−0.81, +0.04], tie |
| **T-S3 (held-out)** | 26.66 | 27.09 | **+0.43** [−0.61, +1.62], **tie** |
| Noise, non-music words | 0 | 0 | — |

**Not adopted, for one reason: it does not fit.** 12.31 MB of LM plus the 11.91 MB int8 and the
2.33 MB VAD is **26.63 MB** against a hard ceiling of 20 (`docs/01`, D4). The package would grow by
7.2 MB.

**And the gain is not where the project's value is.** Look at which set does *not* improve: T-S3,
the held-out speaker, is a tie — marginally worse. The 0.65 dev points and the significant gate
gains all land on **general Italian** (CV-IT −0.65) and on FLEURS-IT, not on code-switching. So
relaxing the ceiling would buy a better Italian model, not a better *mixed-speech* model — which is
the thing this repository exists to build. That reframes the constraint: the 20 MB ceiling is not
costing the project its core claim; it is costing it some out-of-domain Italian WER it never
promised.

## What this entry settles

1. **E-022's decode parameters are the LM's optimum.** Three sweeps over weight, bonus, backoff and
   pruning all point at the shipped configuration. The 5.12 MB LM is paying its way.
2. **The 20 MB ceiling has a price, and it is now measured**: ~0.7 WER on FLEURS-IT and CV-IT,
   ~0.8 on T-600, **zero** on the held-out code-switching speaker. If the ceiling is ever relaxed,
   the first thing to buy is `lm-prune-1-1.pkl.gz` (kept in this directory, with its gate read).
3. **E-008's tuning was not unlucky, it was mis-specified** — one parameter, two jobs. E-022 fixed
   the job; E-023 confirms the result holds at ten times the selection size.

## Cost

**No training.** ~2 h 20 of CPU (sweep 2 h, gate read 4 min, LM rebuilds 30 s), **0 USD**.

## Artifacts

- `sweep.py`, `sweep.log`, `sweep.json` — every number in this entry
- `gate.py`, `eval/lm-1-1/`, `eval/shipped-0.6.0/`, `gate-diffs.json` — the one-time gate read
- `lm-prune-1-1.pkl.gz` — the candidate that does not fit (kept for a future ceiling decision)
- `PLAN.md` — the decision rule, fixed before the sweep

## Not done, on purpose

- A **4-gram LM**: needs a new browser binary format and a `web/lm.js` parser, so it is a code
  change, not a sweep. If the ceiling is ever relaxed, that is the experiment to run instead of (1,1).
- Adding the **spliced-set transcripts** to the LM corpus: legitimate training-side text and the
  only code-switched pattern the LM has never seen, but it is a different question from "are the
  knobs right?" and would deserve its own entry.
