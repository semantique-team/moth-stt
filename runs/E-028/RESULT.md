# E-028 — RESULT: S2 + a gentler fine-tune is the first arm that passes the gates

> Executed 2026-11-11, pre-registered in `PLAN.md`. Training 1 h 47, 13,283 steps, 0 non-finite.
> **No weights have been published.** The result is a candidate for a product decision, not an
> automatic release.

## Verdict

**Passes the six pre-registered gates.** This is the first E-015 arm that improves the mixed test
and the held-out speaker while keeping the natural terms essentially intact:

- rare **presence 17.0% → 27.7%** (8/47 → 13/47, **+10.6 [+2.1, +21.3]**, CI excludes zero);
- rare **aligned ETA 23.5% → 30.9%** (19/81 → 25/81, **+7.4 [0.0, +15.4]**, borderline);
- T-600 **WER 25.20 → 24.66**, **ETA 54.17 → 54.65**;
- T-S3 **26.92 → 25.55**; S3 rare **2/18 → 3/18**;
- natural-term occurrences **439 → 438** (the first arm lost 16);
- FLEURS-EN and CV-IT ties; FLEURS-IT **+0.33 [+0.05, +0.67]**, significant but inside the
  pre-registered 0.5 margin.

The caveats are real: dev is **31.01 vs 30.81** (+0.20), S1 WER is slightly worse (**21.67 →
22.10**), and the historical 16-term gold check drops **20/27 → 18/27** (`milestone`, `paypal`).
The verdict is “candidate for 0.8.0”, not “clean ship”.

## Training and dev

| Item | Value |
|---|---|
| Init | `runs/E-024/checkpoints/ckpt_best.pt` (0.7.0) |
| Steps | 13,283 (1 epoch) |
| LRs | head 1e-4 / encoder 1e-5 |
| Final loss | 0.4557 |
| Final dev | **31.01%** (IT 36.29 / EN 25.73) |
| Selected checkpoint | final step 13,283 (best dev inside the run) |

Dev curve:

| Step | Dev WER |
|---|---:|
| 0.7.0 baseline | **30.81** |
| 1,000 | 31.33 |
| 2,000 | 31.45 |
| 3,000 | 31.53 |
| 4,000 | 31.41 |
| 5,000 | 31.35 |
| 6,000 | 31.27 |
| 7,000 | 31.27 |
| 8,000 | 31.01 |
| 9,000 | 31.07 |
| 10,000 | 31.04 |
| 11,000 | 31.03 |
| 12,000 | 31.02 |
| 13,283 | **31.01** |

The shock is **much smaller** than the first arm (33.64 at step 2,000), but the model never gets
below the 0.7.0 dev. Under the historical dev-only selection rule the baseline would still win;
this is exactly the A1 misalignment described in the audit. The test evidence is stronger than the
dev evidence, and a real mixed-dev is still missing.

## Clean gate

| Set | 0.7.0 | E-028 | Δ [95% CI] |
|---|---:|---:|---|
| T-600 WER | 25.20 | **24.66** | −0.54 [−1.38, +0.28] — tie |
| T-600 ETA | 54.17 | **54.65** | +0.48 [−1.25, +2.06] — tie |
| T-S3 WER | 26.92 | **25.55** | −1.36 [−3.00, +0.33] — tie |
| T-S3 ETA | 48.45 | **49.01** | +0.56 [−2.83, +3.47] — tie |
| FLEURS-IT | 20.14 | 20.47 | **+0.33 [+0.05, +0.67] — worse** |
| FLEURS-EN | 49.14 | 49.38 | +0.24 [−0.27, +0.79] — tie |
| CV-IT dev | 19.85 | 19.80 | −0.05 [−0.53, +0.46] — tie |

By speaker on T-600:

| Speaker | 0.7.0 WER / ETA | E-028 WER / ETA |
|---|---|---|
| S1 | 21.67 / 63.28 | 22.10 / 62.39 |
| S2 | 26.93 / 51.27 | **26.32 / 52.97** |
| S3 | 27.00 / 48.45 | **25.55 / 49.01** |

The overall gain comes from S2 and S3, not S1: S1 is slightly worse on the general mixed set,
although its rare terms improve (below).

## Rare terms

Primary readout, aligned ETA on the 47 rare utterances:

| Metric | 0.7.0 | E-028 | Δ |
|---|---:|---:|---:|
| Aligned ETA (81 English occurrences) | 19/81 = **23.5%** | 25/81 = **30.9%** | **+7.4 [0.0, +15.4]** |
| Presence (47 rare terms) | 8/47 = **17.0%** | 13/47 = **27.7%** | **+10.6 [+2.1, +21.3]** |
| Rare WER | 40.54% | **35.91%** | −4.63 [−8.98, −0.75] |

By speaker (presence / aligned ETA on that speaker’s rare rows):

| Speaker | 0.7.0 | E-028 |
|---|---|---|
| S1 | 6/14; ETA 38.1% | **8/14; ETA 47.6%** |
| S2 | 0/15; ETA 20.8% | **2/15; ETA 33.3%** |
| S3 | 2/18; ETA 16.7% | **3/18; ETA 19.4%** |

Five positive flips, no negative rare flip: `startup` (T-0055), `screenshot` (T-0112), `popup`
(T-0086), `exam` (T-0593), `checkout` (T-0069, S3). The gain is no longer confined to S1/S2.

## Term groups and the gold regression

| Group | 0.7.0 | E-028 | Δ |
|---|---:|---:|---:|
| Natural training text (817) | 439 (53.7%) | 438 (53.6%) | −1 |
| Gold-only (179) | 123 (68.7%) | 126 (70.4%) | +3 |
| Never seen (47) | 8 (17.0%) | 13 (27.7%) | +5 |
| All terms (1,043) | 570 (54.65%) | 577 (55.32%) | +0.67 |
| 16 gold terms (regression) | 20/27 (74.1%) | 18/27 (66.7%) | −2 |

The two gold regressions are `milestone` (→ “mileston”) and `paypal` (→ “papal”). This is small in
absolute terms but below the old 70% regression bar. It is a caveat for adoption, not a
pre-registered E-028 gate.

## Gates

| Gate | Result |
|---|---|
| Rare aligned ETA ≥ 30% | **PASS** (30.9%) |
| T-600 aligned ETA ≥ 53.7% | **PASS** (54.65%) |
| T-600 WER ≤ 25.7% | **PASS** (24.66%) |
| Natural terms ≥ 431/817 | **PASS** (438) |
| FLEURS-IT/EN not worse by >0.5 with CI excluding zero | **PASS by margin** (IT +0.33, CI excludes zero; EN tie) |
| Dev-selected checkpoint is not the step-20 baseline | **PASS** (final step 13,283) |

The dev caveat remains: the final checkpoint is best inside the run, but **not better than the
0.7.0 dev**. The baseline was not a candidate in this run’s selection; a true product-selection
comparison needs a mixed-dev (A1/A3).

## Reading

1. **The two changes work together.** S2 provides the second voice and the lower LRs avoid the
   first arm’s optimization damage: natural terms are preserved and the rare gain is larger.
2. **The gain is cross-voice.** S3 rare improves 2/18 → 3/18; S2 0/15 → 2/15; S1 6/14 → 8/14.
   The first arm’s gain was confined to S1/S2.
3. **The product metrics improve, the dev does not.** T-600 and T-S3 improve; dev is +0.20. This
   is an A1 case study: greedy IT/EN dev is not the mixed product metric.
4. **The remaining regressions are small but real.** S1 general WER +0.43; FLEURS-IT +0.33
   significant; two gold terms lost. A 0.8.0 decision should weigh these against the rare-term
   and held-out-speaker gains.

## Recommendation

**Candidate for 0.8.0.** I would not publish automatically: the dev caveat and the FLEURS-IT
regression need an explicit product decision, and a new mixed-dev remains the correct selection
instrument. The next no-training step is to freeze this checkpoint, export INT8, run the size and
browser acceptance gates, and compare against 0.7.0 on the same T/S3 readouts before any upload.

## Cost and artifacts

Training **1 h 47**, evaluation ~4 min, no teacher, no downloads, 0 USD. Artifacts:
`runs/E-028/` (`PLAN.md`, `config.yaml`, `train.log.jsonl`, `train.out`, `eval/`),
`recordings/manifests/R-S2-terms.jsonl`, `recordings/manifests/R-S2.jsonl` (672 rows),
`data/shards/R-S2-terms`, `data/shards/gold-terms-S2-x20`.
