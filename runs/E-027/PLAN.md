# E-027 — contextual bias pilot on 0.7.0 (no training)

**Status:** pre-registered 2026-11-11 before running. No training, no weight changes, no release
changes. T is reused evidence; this is an exploratory decoding pilot, not a product selection.

## Question

Can a small, bounded, word-aware contextual glossary recover English terms that 0.7.0 already has
inside the acoustic top-k but chooses wrongly, without increasing insertions or hurting pure
Italian/English speech?

E-026 found the reference path inside top-k 10 for **41/47** rare T utterances. Only 6/47 are
completely pruned. This pilot tests whether the PR's `ContextualBias` can exploit that headroom.

## Pre-registered glossary

The product glossary is the **first 24 single-word entries under the “work & meetings” section of
`data/en_terms.txt`, in file order**. It was fixed before looking at per-term E-027 results and is
not selected from T errors:

```
meeting call team sprint ticket deploy deployment release deadline backlog roadmap standup
feedback update report budget forecast target task project workflow pipeline feature bug
```

The irrelevant-glossary control uses 24 common English words unrelated to Italian work/tech speech
(`banana`, `guitar`, `mountain`, …); the exact list is in `eval_bias.py`.

## Arms

| Arm | Glossary | Bonus | Prefix bonus |
|---|---|---:|---:|
| `baseline` | none | 0 | 0 |
| `bias_1.0` | product, 24 terms | 1.0 | 0.25 |
| `bias_0.5` | product, 24 terms | 0.5 | 0.00 |
| `irrelevant_1.0` | irrelevant, 24 terms | 1.0 | 0.25 |
| `zero_bonus` | product, 24 terms | 0 | 0 |

`zero_bonus` must reproduce `baseline` exactly; any difference is a bug. `irrelevant_1.0` is the
false-insertion control. All arms use the release decoder: beam 12, top-k 10, LM weight 0.5,
token bonus 2.4, release `lm_bpe3.bin`.

## Data

- **Primary readout:** T-600 (mixed, reused test) — WER, aligned ETA, insertion rate, 47-rare
  aligned ETA and presence.
- **Pure-Italian controls:** T `control_it` rows and the first 100 CV-IT dev clips.
- **Pure-English control:** the first 100 FLEURS-EN test clips.
- Clips are read with `load_audio_16k`, decoded whole, no VAD/segmentation. Clips over 30 s are
  skipped and counted; no truncation.

## Gates (for “worth a dev experiment”, not adoption)

A bias arm passes only if all hold:

1. T-600 aligned ETA improves by **≥ 1.0 point** over baseline.
2. T-600 WER does not worsen by more than **0.5 point** with a paired 95% CI excluding zero.
3. T-600 insertion rate does not worsen by more than **0.5 words per 100 reference words**.
4. No pure-Italian or pure-English control regresses by more than **0.5 WER point**.
5. `zero_bonus` is byte-identical to `baseline`; `irrelevant_1.0` shows no meaningful ETA gain.

Even a pass is only a reason to collect a real mixed-dev and rerun; T reuse forbids a product claim.
If bias increases false insertions or only moves terms already handled by the LM, reject it.

## Artifacts

- `eval_bias.py` — reproducible evaluation, writes `hyp/`, `metrics.json`.
- `RESULT.md` — results, gates, decision.
