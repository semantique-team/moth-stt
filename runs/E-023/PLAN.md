# E-023 — re-tune the LM now that the decoder is calibrated

> Written 2026-11-09, before any sweep. **Decode-only**, like E-022: no training, no checkpoint
> change. Selection is on **full dev** (`mls_it_dev` 1,248 + `ls_en_dev` 2,691 clips, 10.5 h) —
> wider than E-022's 400-clip subset, because a tuning sweep that overfits its selection set is the
> main risk here. T, A and S3 are never read for selection.

## Why this exists

E-022 found that `lm_weight` was doing two jobs (language modelling *and* length control) and that
its 1-D optimum was the optimum of neither. The bonus `gamma` now does the length job, so the LM's
own parameters are being asked, for the first time, to do only one job. Three things may therefore
have moved:

1. **`lm_weight`**: E-022's grid was coarse (0.4 / 0.5 / 0.6) on 400 dev clips. The optimum may sit
   between, or higher.
2. **`backoff`** (default 0.4, from E-008): tuned under the same broken calibration.
3. **count pruning** (bigram/trigram ≥ 2, from E-008): the LM is **5.12 MB of the 19.44 MB
   package**, the second largest file. A lighter LM that holds quality frees budget; a heavier one
   may now pay for itself.

The LM corpus is unchanged (CV/MLS/LS train transcripts + R-S1/R-S2/pool — never T/A), and so is
the model: still E-017 `ckpt_best.pt`.

## Arms, in order, and what each answers

| # | Sweep | Values | Answers |
|---|---|---|---|
| 1 | `lm_weight` × `gamma` on **full dev** | w ∈ {0.3, 0.4, 0.5, 0.6, 0.7} × γ ∈ {1.6, 2.0, 2.4, 2.8}, beam 12 / top-k 10 | is E-022's cell still the optimum on a bigger selection set? |
| 2 | `backoff` | 0.2 / 0.3 / 0.4 / 0.5 / 0.6 at the winning (w, γ) | is the stupid-backoff factor still right? |
| 3 | count pruning | (1,1), (2,2) = current, (3,3), (2,3) | can the LM shrink, or is there quality left in it? |

Phase 1 and 2 reuse the current LM artifact; phase 3 rebuilds it via the existing
`runs/E-008/build_lm_augmented.py` flags, so the corpus is byte-identical and only the knobs move.

## Decision rule (fixed now)

- Selection metric: **dev macro** (unweighted mean of IT and EN WER), as in E-008's tuning.
- **Adopt** a change only if dev macro improves by **more than 0.3 points** (the width of the
  200-clip subset's noise) *or* the LM **shrinks by ≥ 0.5 MB** at a dev cost of ≤ 0.3 points.
- The gate sets (FLEURS IT/EN, CV-IT dev, noise with VAD, T-600, T-S3) are read **once**, at the
  end, for the winning arm — never during selection.
- A change that improves dev but hurts a gate set beyond its CI is **not adopted**, even if dev
  says so. Dev is read speech; the gate sets are the ones that ship.

## Prediction

- The dev optimum for `lm_weight` moves **up** relative to E-008's 0.2 but stays near E-022's 0.5
  (the E-022 grid already found the basin; this refines it).
- `backoff` stays within 0.3–0.5: it was a reasonable default and the bonus does not change how
  backoff behaves.
- Count pruning ≥ 3 costs more than 0.3 dev points: the LM's tail is exactly where the English
  terms live, and the terms are rare by construction. The interesting arm is therefore **(1,1)** —
  a bigger LM, whose size cost has to be checked against the 20 MB ceiling (19.44 + ~1–2 MB would
  still fit, but it must be measured).

## What would make this a failure

- No arm clears the 0.3-point bar → **nothing is adopted**, E-022's settings stand, and the entry
  records a negative result. That is a legitimate outcome: it would mean E-022 already sat at the
  LM's optimum and the 5.12 MB are paying their way.
- A dev gain that does not survive the gate sets → not adopted, and the discrepancy goes in the
  entry (it would be evidence that full dev is drifting from the product's domain).

## Cost

- Phase 1: 20 configs × full dev ≈ **70 min**. Phase 2: 9 ≈ **35 min**. Phase 3: 4 LM rebuilds
  (CPU, minutes each) + 4 evals ≈ **20 min**. Gate read ≈ **5 min**. Total ≈ **2 h 15**, all CPU,
  **0 USD**, no training.
