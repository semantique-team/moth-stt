# E-035 — READOUT: the dev gain does not transfer; 0.8.1 stays

> Executed 2026-11-13 with the rule frozen in `READOUT-PLAN.md` before the first product decode.
> One readout, no tuning loop. **The candidate is not adopted**: every primary product metric is
> a tie against 0.8.1, and the only absolute-gate failure is a two-word music-noise difference.
> 0.8.1 remains the release.

## Verdict

`B1_budgeted` **does not ship**. Its **−0.0645 pp** dev macro gain is real on the selection set,
but on the product sets it is indistinguishable from zero: T-600, T-S3, FLEURS-IT/EN and the rare
47 all tie, the natural and gold term counts are identical, and the one absolute-gate miss is
`noise_music` (299 → 301 words on 120 music clips, non-empty clips 84 → 82). Under the frozen rule
the candidate fails that gate, and independently it has **no significant product gain**, which is
required for adoption. Both routes lead to the same decision.

## Readout (int8 0.8.1 assets; paired per-utterance bootstrap, 1,000 resamples, seed 0)

| Set | n | Baseline 0.8.1 | Candidate | Δ (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|---|
| T-600 WER | 600 | 24.0826 | 24.1112 | **+0.0287** | [−0.1732, +0.2300] |
| T-600 ETA | 600 | 55.225 | 55.225 | **0.000** | [−0.2817, +0.2844] |
| T-S3 WER | 200 | 24.9574 | 24.8722 | **−0.0852** | [−0.5853, +0.3393] |
| T-S3 ETA | 200 | 51.268 | 50.986 | **−0.2817** | [−0.8902, 0.0000] |
| FLEURS-IT WER | 697 | 20.1621 | 20.1684 | **+0.0062** | [−0.0880, +0.1003] |
| FLEURS-EN WER | 525 | 49.8077 | 49.7639 | **−0.0437** | [−0.2325, +0.1498] |
| Rare 47 aligned ETA | 47 | 46.9136 (38/81) | 46.9136 (38/81) | **0.000** | [0.0000, 0.0000] |

- **No primary metric is significantly better or significantly worse.** The largest movement is
  T-S3 WER (−0.085 pp) and its interval crosses zero; T-600 moves +0.029 pp.
- **Rare terms do not move at all**: aligned ETA 38/81 in both arms, target-only 24/47 in both.
  Natural terms 432/817, gold 127/179, never 24/47: identical.
- **Noise**: non-music words 258 → 257 and non-empty clips 146 → 142 (slightly better); music words
  299 → **301** and non-empty clips 84 → 82. The `noise_music` gate requires words ≤ baseline, so
  it fails by two words. This is a trivial difference, but the gate was pre-registered fail-closed
  and is reported as a fail, not waived.

## Gates And Rule

| Check | Result |
|---|---|
| package 19,999,997 B ≤ 20,000,000 | pass |
| T-600 WER ≤ 25.21, ETA ≥ 53.67 | pass |
| T-S3 WER ≤ 25.71, ETA ≥ 49.64 | pass |
| FLEURS-IT ≤ 20.75, EN ≤ 50.31 | pass |
| rare aligned ETA ≥ 28.4% | pass (46.9%) |
| natural ≥ 431 | pass (432) |
| noise `non_music` | pass |
| noise `music` | **fail** (301 vs 299 words) |
| decoder parity | **exact** (`le diffuseo proteste iniziate a ottobrek conte`, 0.123 s Node) |
| no significant product regression | pass (all seven primary metrics tie) |
| at least one significant product gain | **fail** (none) |

The adoption rule required all absolute gates, no significant regression, and at least one
significant gain. The candidate fails the noise gate and the gain condition; the verdict is
**not adopted**, both by the letter of the rule and because a tie is not evidence for a product
change.

## What This Establishes

- **The dev gain is selection evidence, not a product gain.** A 73,427-entry, frequency-ranked
  slice of single-count trigrams helped MLS-IT/LS-EN dev (−0.0645 pp) but does not move T-600,
  FLEURS or the rare/natural term groups at detectable size.
- **The 293,711 B question is now answered negatively for this LM format.** Spending the headroom
  exactly is safe by bytes and parity, but it does not buy a product improvement, so **0.8.1 is
  frozen**: no 0.8.2, no package change, no model-card change.
- **The noise gate is doing its job.** A +2-word music difference is not a quality claim, but a
  decode-side change with no product upside should not be shipped to find out.
- **Not established**: a browser acoustic WER (still a separate closure gap); anything about a
  larger LM or a differently ranked slice; whether a much larger LM would transfer (E-023/E-034
  suggest it may, but it cannot ship here).

## Artifacts

`runs/E-035/readout/readout-1/` —
`baseline/{t600,fleurs_it_test,fleurs_en_test,noise}/{hyp,metrics}` and the same under
`candidate/` (T-S3 is scored from the T-600 hypotheses), `terms.json`, `parity.json`,
`readout.json`, and `readout.log`.

Reproduction: `uv run python runs/E-035/readout.py --name readout-1`. Assets are verified against
`runs/E-031/release-0.8.1.json`; the candidate hash is `458441ab…`. Cost ~5 min of int8 CPU
decoding, no training, no download, 0 USD.

The pre-registered rule is in [`READOUT-PLAN.md`](READOUT-PLAN.md); the dev result that motivated
it is in [`RESULT.md`](RESULT.md).
