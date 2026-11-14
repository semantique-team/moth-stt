# E-040 — RESULT: the rare-term shortfall reproduces, the silence hallucination does not

> Executed 2026-11-13/14. Two arms × seeds 1, 2, 3, plus seed 0 (E-036) as the fourth replicate;
> only `seed` and `run` differ from E-036. One epoch, 13,314 steps per run, no non-finite step,
> MPS/CPU parity ≤ 5.6e-6. Product readouts use the shipped INT8 path and the A3 LM.

## Verdict

E-036's rejection was two claims, and multi-seed separates them:

- **Rare terms: confirmed.** Across eight replicates no run reaches the 42/81 target —
  frozen 38, 38, 38, 39; last2 41, 39, 38, 39. The shortfall is a **property of the recipe**, not
  a bad draw. 0.8.1 stays at 38/81 with an unchanged rule.
- **Silence: not reproducible.** 0.8.1 emits nothing on all 120 silence clips. The replicates are
  **bimodal**: frozen 120 / 120 / **0** / 120 non-empty clips, last2 120 / **0** / **0** / **0**.
  E-036's "`x` on all 120 in both arms" was the **seed-0 draw**, not a property of the recipe.
  Half the replicates are clean, half hallucinate on every silence clip — so the recipe does not
  fix silence either, it only moves the probability.
- **Dev and product WER: noise.** last2 is 0.0568 pp better on dev and 0.1577 pp worse on T-600.
  Frozen's own seed range is **0.4587 pp on T-600**, three times the arm gap. The E-036 arm
  ordering carried no information.

Nothing is adopted; **0.8.1 remains the release**.

## Dev (best checkpoint, selection metric)

| Arm | seed 0 | seed 1 | seed 2 | seed 3 | mean | range |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 30.9890 | 30.9714 | 30.9237 | 30.9667 | **30.9627** | 0.0653 |
| last2 | 30.9683 | 30.8638 | 30.8946 | 30.8967 | **30.9059** | 0.1045 |

Reference 0.8.1 dev is 31.0101; every replicate is below it, and the four frozen replicates differ
among themselves by more than the arm gap (0.0568 pp).

## Product (INT8 + A3 LM, same decoder 12/10/0.55/2.7)

| Arm | seed | T-600 WER | T-600 ETA | T-S3 WER | FLEURS-IT | FLEURS-EN | rare | natural | gold | silence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0 | 24.3406 | 55.4171 | 25.1278 | 20.2869 | 49.9126 | 39/81 | 435 | 125 | 120/120 ✗ |
| frozen | 1 | **23.8819** | 55.3212 | 24.7871 | 20.3368 | 49.5629 | 38/81 | 435 | 126 | 120/120 ✗ |
| frozen | 2 | 23.9679 | 55.2253 | 24.5315 | 20.3243 | 49.6241 | 38/81 | 432 | 127 | 0/120 |
| frozen | 3 | 24.0252 | 55.2253 | 24.5315 | 20.2120 | 49.7552 | 38/81 | 432 | 127 | 120/120 ✗ |
| last2 | 0 | 24.2259 | 55.3212 | 25.2129 | 20.2495 | 49.8426 | 41/81 | 434 | 123 | 120/120 ✗ |
| last2 | 1 | 24.1972 | 55.2253 | 25.0426 | 20.2370 | 49.7377 | 39/81 | 433 | 125 | 0/120 |
| last2 | 2 | 24.1399 | 55.6088 | 24.8722 | 20.2806 | 49.6241 | 38/81 | 438 | 125 | 0/120 |
| last2 | 3 | 24.2833 | 55.2253 | 25.1278 | 20.2682 | 49.7727 | 39/81 | 431 | 126 | 0/120 |
| **0.8.1** | — | 24.0826 | 55.2253 | 24.9574 | 20.1621 | 49.8077 | 38/81 | 432 | 127 | **0/120** |

T-600 range: frozen **0.4587 pp** [23.8819, 24.3406], last2 **0.1434 pp** [24.1399, 24.2833].
Rare range: frozen 38–39, last2 38–41. Natural terms 431–438 against 432 for 0.8.1.

## What This Establishes

- **Does**: separate a recipe property from a seed artifact inside the same experiment. Rare-term
  weakness is stable and confirms E-036; silence hallucination is a coin flip and E-036's wording
  was too strong. Both conclusions are now stated with eight replicates instead of one.
- **Does**: give the paper a defensible dispersion number for this recipe family — about
  **0.46 pp T-600 WER** between seeds of the same arm.
- **Does not**: estimate a variance (four replicates), adopt anything, or make 0.8.1 better.
- **Does not**: change the fact that no replicate is a candidate: none reaches the rare target, and
  half still hallucinate on silence.

## Artifacts And Cost

`runs/E-040/` — `PLAN.md`, six `config.yaml`, `run_queue.sh`, `readout.py`, `readout_all.sh`,
`summarize.py`, per-run `train.log.jsonl`/`train.out`/`preflight.json`, `checkpoints/` (gitignored),
`eval/<name>/` (hypotheses, metrics, `terms.json`, `readout.json`), `summary.json`, `queue.log`,
`readout.log`. Seeds 2 and 3 ran from `19:50` to `04:06` (six runs total ≈ 12 h of machine time),
readouts 04:15–04:29. INT8 exports are reproducible and gitignored.
