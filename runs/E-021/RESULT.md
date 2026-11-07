# E-021 — where the fp32 export gap comes from

**Question.** E-016 recorded `fp32 max |Δ| = 1.051e-3` on a synthetic 29 s clip against a gate of
`1e-3`, and 0.3.0/0.4.0/0.5.0 were all built with `--no-verify` because of it. The same run
measured 5.46e-4 on 20 real clips with top-1 agreement 100%. Is the export wrong, or is the
fixture?

**Method.** `runs/E-021/fp32_gap.py` exports the E-017 checkpoint to fp32 ONNX and measures the
same quantity against the PyTorch reference under four fixtures, plus a seed sweep.

## Result

| Fixture | max abs |Δ| | vs gate 1e-3 |
|---|---|---|
| Gate fixture as written (`N(0,1)·0.1`, 1/5/10/29 s, seed 0) | **2.43e-4** | passes |
| Real T clips at the same lengths | **3.05e-4** | passes |
| Synthetic at the natural mel scale (`N(0,1)·1.0`) | 1.37e-4 | passes |
| Real clip zero-padded to 29 s | 2.21e-4 | passes |
| **20 seeds × 4 lengths (80 draws)** | median ≈ 1.8e-4, **worst 1.28e-3** | **1/80 over** |

The single over-gate draw is a **29 s** clip at seed 2. At 1/5/10 s no draw of 80 ever crosses.

## Reading

**The gate is one random draw, not a statistic.** Its typical value is ~1.8e-4 — 5.5× under the
threshold — and the tail reaches 1.3e-3 about once in 80 draws, always at the longest clip. E-016
happened to sample that tail at seed 0 (`1.051e-3`); the E-017 checkpoint at seed 0 gives `2.43e-4`
and at seed 2 gives `1.282e-3`. So the value is a property of **(checkpoint, seed, length)**, and a
single-shot gate on it is a coin flip at the ~1% level. That is why the same checkpoint passed at
seed 0 in this run and "failed" in E-016, and why E-016 also reported seeds 2–4 passing.

What is **not** happening: the export is not wrong. Real speech, including a real clip padded to the
29 s length that failed, is at 2.2–3.1e-4 — a third of the gate — and top-1 agreement is 1.0.
The int8 artifact built from the same fp32 graph measures +0.17 WER on T-600 (E-020), well inside
the noise.

**Recommendation (not implemented here, it changes the gate):** replace the single-draw gate with
a seeded sweep (e.g. 20 seeds, assert the **median** ≤ 1e-3 and the worst ≤ 1e-2), so the check
tests the export and not the luck of one `default_rng(0)` call. Until then `--no-verify` is not
needed for correctness, but the release build should record which fixture and seed it ran.

## Reproduce

```bash
uv run python runs/E-021/fp32_gap.py
```

Output: `runs/E-021/fp32_gap.json` (per-fixture gaps, per-length seed sweep, real clip ids).