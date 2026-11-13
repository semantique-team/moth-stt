# E-037 — RESULT: the INT8 graph costs a small, measurable WER, and it is not uniform

> Executed 2026-11-13. Decode-only closure evidence: the same E-028 checkpoint through the fp32
> export and the shipped INT8 graph, with the same A3 LM, decoder and clips. **No release change**:
> fp32 is 54,164,659 B and cannot fit 20 MB. This measurement replaces the assumption that an
> unchanged INT8 graph has an unchanged quantization cost.

## Verdict

The INT8 graph is **not free**:

- **Dev** (3,939 clips): macro **22.8596% → 23.0170%**, **+0.1574 pp**, paired CI
  **[+0.0539, +0.2664]** — a significant cost. Pooled +0.1489 pp [+0.0413, +0.2620]; IT +0.2177,
  EN +0.0969.
- **T-600**: **23.5952 → 24.0826**, **+0.4874 pp**, CI [−0.0290, +0.9963] — a tie by CI but a
  consistent point cost; T-600 ETA **56.088 → 55.225** (−0.8629 pp, CI [−1.9743, +0.1910]),
  natural terms **444 → 432** (−12).
- **FLEURS-EN**: **49.3618 → 49.8077**, **+0.4459 pp**, CI **[+0.0087, +0.8652]** — significant.
- **T-S3**: +0.4259 pp [−0.4160, +1.2153], tie; ETA +0.2817 [−1.1905, +1.7857], tie.
- **FLEURS-IT**: **−0.1559 pp** [−0.3950, +0.0739], a tie in the *opposite* direction — the
  quantization cost is not uniform across sets.
- **Rare terms do not move**: aligned ETA 38/81 in both; rare WER −0.386 pp (tie); gold 124 → 127
  (+3); never 24/47 in both.
- **Noise is slightly better under INT8**: non-music 266 → 258 words, 153 → 146 non-empty; music
  304 → 299 words, 84 non-empty in both.

**INT8 remains the only deployable format**: the fp32 package is **54,164,659 B** versus
**19,706,289 B**, over the 20 MB ceiling by 34.2 MB. The cost is documented, not "fixed".

## Dev (MLS-IT + LS-EN, greedy? no: beam 12/top-k 10/LM 0.55/bonus 2.7)

| Graph | Macro | IT | EN | Pooled | Seconds |
|---|---:|---:|---:|---:|---:|
| fp32 | **22.8596** | 27.8863 | 17.8330 | 22.1527 | 304.8 |
| INT8 | 23.0170 | 28.1040 | 17.9299 | 22.3016 | 306.3 |
| Δ (INT8 − fp32) | **+0.1574** | +0.2177 | +0.0969 | +0.1489 | — |
| 95% CI (macro) | **[+0.0539, +0.2664]** | — | — | [+0.0413, +0.2620] | — |

## Product

| Set | fp32 | INT8 | Δ (pp) | 95% CI (pp) |
|---|---:|---:|---:|---|
| T-600 WER | 23.5952 | 24.0826 | +0.4874 | [−0.0290, +0.9963] |
| T-600 ETA | 56.088 | 55.225 | −0.8629 | [−1.9743, +0.1910] |
| T-S3 WER | 24.5315 | 24.9574 | +0.4259 | [−0.4160, +1.2153] |
| T-S3 ETA | 50.986 | 51.268 | +0.2817 | [−1.1905, +1.7857] |
| FLEURS-IT | 20.3181 | 20.1621 | −0.1559 | [−0.3950, +0.0739] |
| FLEURS-EN | 49.3618 | 49.8077 | **+0.4459** | **[+0.0087, +0.8652]** |
| Rare 47 WER | 28.5714 | 28.1853 | −0.3861 | [−1.8868, +0.7937] |
| Rare aligned ETA | 38/81 | 38/81 | 0.000 | [0.0000, 0.0000] |

Term groups on T-600: natural **444 → 432** (−12), gold **124 → 127** (+3), never 24 → 24.
T-600 aligned ETA over all terms: 585/1043 (56.09%) → 576/1043 (55.23%).

## Why It Happens

The INT8 side reproduces the committed E-035 baseline exactly (T-600 24.0826, FLEURS-IT 20.1621,
FLEURS-EN 49.8077, rare 38/81, natural 432), so the comparison has no harness drift; the fp32 side
differs from the PyTorch dev number by ~0.01 pp, inside the export gate.

The logit gap on the frozen 3 s fixture is **max |Δ| 2.3503**, mean **0.2418**, with **top-1
agreement 98.67%** over 150 frames. A few frames flip; that is enough to change words on a small
fraction of clips, which is exactly the observed pattern: a significant dev cost, a large T-600
point cost, a significant FLEURS-EN cost, and no rare-term change.

The earlier E-028 measurement with the 0.8.0 LM was **+0.21 pp dev** (30.93 → 31.14). The new
number with the shipped A3 LM is **+0.16 pp** on a different dev set (MLS/LS), so the two are
consistent: the acoustic INT8 cost is of order 0.15–0.2 pp on dev, and larger on individual
product sets.

## What This Establishes

- **Does**: quantify the price of the 20 MB ceiling on the shipped artifact. The model card and
  paper can now state a measured INT8 penalty instead of assuming it is negligible, and can name
  where it lands (T-600 point WER/ETA, FLEURS-EN, natural terms; not rare terms, not noise).
- **Does not**: change the release, justify fp32, or say anything about a different quantization
  recipe. It also does not measure browser acoustic WER; the gap is measured through the Python
  ONNX runner on the same cached mel, not through the browser/ORT-Web path.
- **Does not** make T/S3 pristine: they are reused product evidence, as in every prior readout.

## Artifacts And Cost

`runs/E-037/` — `PLAN.md`, `export_fp32.py`, `fp32.json` (46,369,047 B, sha `3fd1492e…`;
verification median 1.82e-4, 1/80 over 1e-3), `readout.py`, `readout/paired-1/` (both graphs'
dev and product hypotheses/metrics, `terms.json`, `readout.json`), `export.log`, `readout.log`, and
this report. The 46 MB fp32 ONNX is reproducible and gitignored.

Reproduction: `uv run python runs/E-037/export_fp32.py`, then
`uv run python runs/E-037/readout.py --name paired-1`. Cost ~15 min of CPU decoding, no training,
no download, 0 USD.
