# E-034 — LM pruning and weighting on dev only: is any headroom worth reclaiming?

**Status:** pre-registered 2026-11-12 **before** the run. Decode-only: no training, no new data,
no weights. T, T-S3, FLEURS and the noise pool are **not read here**; this is development data.

## Question

0.8.1 ships 19,706,289 bytes with **293,711 bytes** of headroom below the 20 MB ceiling, and
E-031 showed the LM is a real lever (decode-side rare gain). The strategy's LM options are
"budgeted pruning on training-side text" and "common/technical term balance", but it warns that a
bigger LM needs bytes reclaimed elsewhere. Before spending effort on byte accounting, test
whether extra n-grams or extra term weighting move **dev** at all.

Two hypotheses, both cheap to falsify:

1. Relaxing trigram pruning (keeping trigrams seen once) improves dev macro WER; if yes, a
   budgeted-pruning follow-up is worth designing, if no there is nothing to reclaim bytes for.
2. Raising the term-sentence weight above ×3 improves dev macro WER without costing bytes that
   matter; if no, the shipped ×3 is not an under-weighted corner.

## Arms

Built from the same frozen corpus as E-031 (`runs/E-031/frozen/lm-corpus.jsonl`, receipt
committed), same tokenizer, same backoff 0.4, same builder code path (`BPETrigramLM`).

| Arm | Bigrams | Trigrams | Term weight | Purpose |
|---|---|---|---|---|
| `A3_shipped` | ≥ 1 | ≥ 2 | ×3 | control: the published 0.8.1 LM |
| `N1_2_1` | ≥ 2 | ≥ 1 | ×1 | relaxed trigrams, diagnostic size |
| `N2_1_1` | ≥ 1 | ≥ 1 | ×1 | maximum coverage, diagnostic size |
| `N3_1_2_terms5` | ≥ 1 | ≥ 2 | ×5 | term-weighting lever at unchanged pruning |

`N1` and `N2` are expected to exceed the byte budget; that is a **size** result, not a reason to
skip the decode, because "does extra coverage help dev at all" decides whether reclaiming bytes
is worth anything. They cannot be adopted even if they win.

## Protocol

- Decoder frozen as in E-030 and E-031: beam 12 / top-k 10 / LM weight 0.55 / token bonus 2.7,
  fp32 E-028 checkpoint (`runs/E-028/checkpoints/ckpt_best.pt`), the same `MelDataset` and batch
  plan as the retained E-031 sweep.
- Dev only: `mls_it_dev` (1,248 clips) + `ls_en_dev` (2,691 clips), 3,939 clips, 10.48 h.
- Every arm is decoded in this run, including the control, so all arms share session and code.
- Metrics: pooled WER/ETA plus per-language WER and the equal-language **macro** WER; paired
  per-utterance bootstrap, 1,000 resamples, seed 0, within language (`macro_wer_bootstrap`).
- Package arithmetic uses the 0.8.1 baseline: `19,440,761 − 5,116,464 + binary_bytes`.

## Pre-registered decision rule

An arm **advances** only if all of the following hold:

1. dev macro WER point estimate is **lower than the control**;
2. the paired 95% CI upper bound of the macro difference is **< 0**;
3. its browser binary fits the package budget: **binary_bytes ≤ 5,675,703** (20,000,000 − the
   0.8.1 non-LM bytes).

Nothing advances on a tie, on pooled WER alone, or on per-language improvement in one language
with a regression in the other. An over-budget arm that wins on dev is reported as **motivation
for a budgeted-pruning follow-up**, never as a shipped candidate and never as a gate result.

A positive result changes the LM only after the full product readout on T/T-S3/FLEURS/natural and
the browser path; that readout is a separate step and is not authorized here.

## Expected size of the effect

E-031's own dev deltas were −0.011 to −0.097 pp between arms. A macro gain below ~0.05 pp is
inside that noise band and will be reported as a tie, not as a trend. This experiment is cheap
enough that a null result is the useful outcome: it tells us not to spend the 293 KB question.

## Artifacts

`runs/E-034/` — `PLAN.md`, `build_variants.py` + `build/` (pickles, browser binaries, `build.json`),
`dev_sweep.py` + `dev/<arm>/{dev.hyp.jsonl,dev.metrics.json}` + `sweep.json`, `RESULT.md`.
The `EXPERIMENTS.md` entry is written with the result. Pre-registration commit: the commit that
adds this file.
