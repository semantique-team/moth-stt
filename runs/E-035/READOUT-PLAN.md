# E-035 — Product Readout Plan (Frozen Before Reading The Product Sets)

**Status:** pre-registered 2026-11-13, **before** decoding T-600, T-S3, FLEURS, noise or the
browser path. The dev arm already advanced (`runs/E-035/RESULT.md`); this is the single readout,
not a tuning loop. No training, no weight change, no export before the verdict.

## Baseline And Candidate

| Role | Assets | LM | Package |
|---|---|---|---:|
| **Baseline 0.8.1** | shipped int8 graph, vocab, mel filters, VAD, config, NOTICE | `runs/E-031/build/frozen_1_2_terms.bin`, **5,381,992 B**, sha `3be1177b…` | **19,706,289 B** |
| **Candidate** | the same assets | `runs/E-035/build/B1_budgeted.bin`, **5,675,700 B**, sha `458441ab…` | **19,999,997 B** |

The 0.8.1 asset hashes are in `runs/E-031/release-0.8.1.json`; the readout verifies the model,
vocab, mel filters, VAD, config and baseline LM against that record before decoding. The candidate
binary is the one built and decided on dev; it is not rebuilt here.

Both arms use the shipped int8 graph through `OnnxMoth`, decoder **beam 12 / top-k 10 / LM 0.55 /
token bonus 2.7**, the same mel shards and the same session. Only the LM binary differs.

## Sets Read Once

| Set | Rows | Purpose |
|---|---:|---|
| `t600` | 600 | primary product WER/ETA; also the term groups |
| `t-s3` | S3 subset of T | held-out-speaker slice |
| `fleurs_it_test` | 697 | Italian generalization |
| `fleurs_en_test` | 525 | English generalization |
| `noise` | 457 | non-speech words and non-empty clips |
| rare 47 | 47 of T | aligned ETA over all 81 span tokens, plus target-only diagnostic |

Dev is **not** re-read. T and S3 have informed development repeatedly; this readout is a product
comparison, not a pristine test, and will be described as such.

## Metrics And Comparisons

- WER and ETA per set, paired per-utterance bootstrap, 1,000 resamples, seed 0
  (`_compare`, candidate − baseline), reported with 95% CI.
- Rare aligned ETA on the 47 rare rows, paired CI; target-only ETA as a diagnostic.
- Term groups on T-600 through `runs/E-015/eval_terms.py` (natural, gold, never), candidate vs
  baseline, including the natural count.
- Noise: clips, words and non-empty clips per group, candidate vs baseline.
- Package size and candidate binary hash.
- **Decoder-level parity**: `web/decode.js` + `web/lm.js` with the candidate LM must reproduce the
  Python beam output **exactly** on the frozen fixture logits. Node time is recorded as a smoke,
  not as a browser latency claim. Browser acoustic WER remains a separate closure gap and is not
  measured here.

## Absolute Gates (All Required)

Same thresholds as E-031; 0.8.1 already passes them.

1. package ≤ **20,000,000 B**;
2. T-600 WER ≤ **25.21%**, ETA ≥ **53.67%**;
3. T-S3 WER ≤ **25.71%**, ETA ≥ **49.64%**;
4. FLEURS-IT WER ≤ **20.75%**, FLEURS-EN WER ≤ **50.31%**;
5. rare aligned ETA ≥ **28.4%**;
6. natural term count ≥ **431**;
7. noise: for `non_music` and `music`, clips > 0, clips equal to baseline, words ≤ baseline and
   non-empty clips ≤ baseline;
8. decoder-level parity **exact**.

## Non-Inferiority Against 0.8.1

No primary metric may be **significantly worse** than 0.8.1: the paired 95% CI must not lie
entirely on the worse side of zero.

Primary metrics: T-600 WER and ETA, T-S3 WER and ETA, FLEURS-IT WER, FLEURS-EN WER, rare aligned
ETA. For WER (lower is better) the rejection condition is CI **lower** > 0; for ETA (higher is
better) it is CI **upper** < 0. Natural/gold counts and noise are reported and covered by the
absolute gates above.

## Adoption Rule

**Adopt the candidate as a decode-only 0.8.2 only if all three hold:**

1. every absolute gate passes;
2. no primary metric is significantly worse than 0.8.1;
3. at least one **significant gain** (mirrors E-031): T-600 WER CI upper < 0, or FLEURS-IT WER CI
   upper < 0, or rare aligned ETA CI lower > 0.

If the gates and non-inferiority pass but no metric improves significantly, the verdict is
**product tie**: do not ship, keep 0.8.1, and record that the 0.0645 pp dev gain did not transfer
detectably to the product sets. If a primary metric is significantly worse, reject even if
another improves. No metric may be promoted after seeing its value; this rule is fixed now.

## Artifacts And Cost

`runs/E-035/readout/<name>/` — `baseline/<set>/{hyp,metrics}`, `candidate/<set>/{hyp,metrics}`,
`terms.json`, `parity/{logits.bin,meta.json,parity.json}`, `readout.json`, and a `READOUT.md`
verdict appended to the experiment record. The pre-registration commit is the commit that adds
this file, before the first product decode.

Cost: three product sets plus noise, twice, on the int8 CPU graph; estimated tens of minutes, no
training, no download, 0 USD.
