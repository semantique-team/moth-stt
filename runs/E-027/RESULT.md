# E-027 — RESULT: the contextual-bias pilot did not move the rare terms

> Pre-registered in `PLAN.md`; executed 2026-11-11. **No training, no weights, no release changes.**
> T is reused test evidence; this is a negative pilot, not a product claim.

## Verdict

**Rejected.** The pre-registered 24-term product glossary changed T-600 WER by **−0.03 points**
(95% CI includes zero) and aligned ETA by **+0.10 points** (CI includes zero), while the 47 rare
utterances did **not move at all** (ETA 22.22% → 22.22%, presence 8/47 → 8/47). The gate required
**≥ +1.0 aligned-ETA point**; it fails by an order of magnitude. The irrelevant-glossary control
also produced a small, non-specific change, so there is no evidence the product glossary is doing
the work.

## T-600 (600 clips, whole-clip decode, no VAD)

| Arm | WER | Aligned ETA | Insertion rate |
|---|---:|---:|---:|
| Baseline | 25.32% | 54.75% | 1.29 |
| Product glossary 1.0 / 0.25 | **25.29%** | **54.84%** | 1.29 |
| Product glossary 0.5 / 0.00 | 25.32% | 54.75% | 1.29 |
| Irrelevant glossary 1.0 / 0.25 | 25.29% | 54.84% | 1.29 |
| Product glossary, zero bonus | 25.32% | 54.75% | 1.29 |

Paired product-bias vs baseline: WER **−0.03 [−0.12, +0.06]**, ETA **+0.10 [−0.20, +0.42]**,
both ties. The absolute baseline is 25.32 rather than the official 25.20 because this pilot
decodes whole float32-resampled clips without VAD; the paired deltas are the comparison.

## The 47 rare utterances

| Arm | WER | Aligned ETA | Presence |
|---|---:|---:|---:|
| Baseline | 40.93% | 22.22% | 8/47 (17.0%) |
| Product 1.0 | 40.93% | 22.22% | 8/47 (17.0%) |
| Product 0.5 | 40.93% | 22.22% | 8/47 (17.0%) |
| Irrelevant 1.0 | 40.54% | 23.46% | 8/47 (17.0%) |
| Zero bonus | 40.93% | 22.22% | 8/47 (17.0%) |

The product glossary does not recover a single rare occurrence. The irrelevant glossary moves one
term by chance (ETA +1.23 points, CI [0.00, +3.49]), which is precisely why the control exists.

## Controls and safety

| Set | Baseline WER | Product 1.0 WER | Irrelevant 1.0 WER |
|---|---:|---:|---:|
| T `control_it` (30) | 13.24% | 13.24% | 13.24% |
| CV-IT dev (100) | 21.06% | 21.06% | 20.95% |
| FLEURS-EN test (100) | 47.28% | 47.18% | 47.32% |

Product-glossary hits: T-600 **35 → 37** (two extra occurrences, WER-neutral), T `control_it`
**0 → 0**, CV-IT **0 → 0**, FLEURS-EN **2 → 3**. There are no false insertions on the Italian
controls; the two extra T-600 hits are not enough to change the term readout. `zero_bonus` is
byte-identical to baseline, confirming the scorer path is a no-op when disabled.

## Gates

| Gate | Result |
|---|---|
| T-600 aligned ETA gain ≥ 1.0 point | **FAIL** (+0.10) |
| T-600 WER not worse by >0.5 with CI excluding zero | PASS (tie) |
| Insertion rate not worse by >0.5 | PASS (unchanged) |
| Pure IT/EN control regression ≤ 0.5 | PASS |
| `zero_bonus` identical; irrelevant glossary no meaningful gain | `zero_bonus` PASS; irrelevant shows +1.23 rare ETA, so **non-specific** |

## Reading

1. **The mechanism is sound but the glossary is not the missing lever.** The bounded scorer is a
   no-op at zero bonus, does not insert terms into Italian controls, and is byte-identical to
   baseline when disabled. It simply does not change the 47 rare outputs at bonus 1.0 / prefix 0.25.
2. **The terms in the pre-registered product glossary are too common or already handled by the LM.**
   They are work/meeting words (`meeting`, `sprint`, `deploy`, …), not the consumer/tech nouns that
   fail in T. A future glossary must come from product requirements and a real mixed-dev, not from
   T errors; this pilot does not license mining the 47 failures.
3. **The E-026 headroom is real but not automatically reachable.** 41/47 reference paths survive
   top-k, yet a 24-term bias does not make the decoder choose them. The next interventions remain
   (a) a gentler S2 fine-tune for the six acoustically pruned rows and the broader mapping, and
   (b) a real mixed-dev to select and calibrate any bias at all.

## Caveats

- T is reused test evidence; no independent dev or holdout was used.
- Absolute numbers differ from the official eval (no VAD, float32 resampling, whole clips); only
  paired comparisons are interpreted.
- The pilot fixes one glossary and two bonus settings; it does not prove that no contextual bias
  can work. It does show that this pre-registered one does not.
- The six top-k-pruned rows (`startup`, `prototype`, `chromecast`, `charger`, `exam`, `mondays`)
  cannot be recovered by this scorer at all.

## Artifacts

- `PLAN.md`, `eval_bias.py`, `metrics.json`, `hyp/<set>.<arm>.jsonl`.
