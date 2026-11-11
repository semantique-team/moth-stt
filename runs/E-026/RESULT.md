# E-026 — RESULT: the rare terms are mostly reachable; the search/prior chooses wrong

> Diagnostic on 47 T utterances, 81 English reference occurrences, release 0.7.0 int8.
> **No training, no adoption, no model selection.** T is reused evidence, not an untouched test.

## What was measured

| Decoder | WER | Aligned ETA | Insertion rate |
|---|---:|---:|---:|
| Greedy | **48.65%** | **17.28%** | 3.47 |
| Beam, no LM (bonus 0) | 49.81% | 16.05% | 3.86 |
| Beam + LM (release 12/10/0.5/2.4) | **40.54%** | **23.46%** | 4.63 |

The LM is the lever that matters on this subset: it recovers about 7.4 aligned-ETA points over
greedy and lowers WER by 8.1 points. Beam without LM is slightly worse than greedy, so “more search”
alone is not the answer.

## Aligned ETA vs the historical presence readout

The E-015 result used term **presence** (the term appears anywhere in the hypothesis), not the
canonical aligned ETA. On the same 47 rows:

| Metric | 0.7.0 baseline | E-015 arm |
|---|---:|---:|
| Presence on the 47 rare terms | 8/47 = **17.0%** | 11/47 = **23.4%** |
| Aligned ETA on the 81 English occurrences | 19/81 = **23.5%** | 25/81 = **30.9%** |
| Full T-600 aligned ETA | 565/1043 = **54.17%** | 557/1043 = **53.40%** |
| Full T-600 WER | **25.20%** | **24.91%** |

So the E-015 arm improved the rare subset by more than the presence number suggested (+7.4
aligned-ETA points), but lost more on the full set’s natural terms: the full T-600 ETA fell 0.77
points. The kill verdict stands (FLEURS-IT/EN regressed significantly and held-out S3 did not
move), but **the next arm should use aligned ETA as the primary readout and keep presence as a
diagnostic**. This is a correction of interpretation, not a rewrite of the E-015 result.

## Forced-reference support

| Check | Result |
|---|---:|
| Full CTC path reachable | **47/47** |
| Reference reachable under top-k = 10 | **41/47 (87.2%)** |
| Reference completely pruned by top-k | **6/47 (12.8%)** |
| Retained probability fraction, 41 reachable | median **0.9999**, mean 0.953, min 0.0018 |
| Reachable rows retaining < 99% of mass | 6/41 |

The six top-k-pruned references are:

| id | reference | 0.7.0 beam+LM |
|---|---|---|
| T-0055 | la startup ha raccolto un finanziamento | la stup ha raccolto un fi finanziamento |
| T-0054 | ho visto il prototype del nuovo prodotto | ho visto il proto type del nuovo prodotto |
| T-0333 | la tv è un chromecast | la via è un crucast |
| T-0544 | il mio telefono è scarico do you have a charger | il mref scarico do charger |
| T-0593 | i have to study for the exam | i have to studi forxer |
| T-0594 | the restaurant is closed on mondays | the rest untiss onmandays |

For 41 of 47 rows the acoustic model already puts the reference path inside the top-k mask; the
wrong output is chosen by the acoustic ranking plus the LM/prior. For those, a bounded contextual
bias or a better prior is a plausible intervention. For the other six, bias cannot recover what
top-k removed: those need acoustic/data work.

## Reading

1. **Top-k is not the main bottleneck** for this sample: 41/47 references survive the mask.
2. **The LM is doing the heavy lifting**, but it also chooses wrong alternatives; a word-aware
   contextual bias is the cheapest next test because it does not touch the acoustic model.
3. **The E-015 readout was too pessimistic on the rare subset and too optimistic on the product
   metric**: aligned ETA on the 47 improved +7.4 points, while full T-600 ETA fell 0.77. Any next
   arm must report both and protect the natural terms.
4. **Six rows are acoustic/data failures** (`startup`, `prototype`, `chromecast`, `charger`,
   `exam`, `mondays`): S2 audio and a gentler fine-tune remain the right training response.

## Caveats

- The diagnostic uses the int8 release, while E-015 was evaluated from a fp32 checkpoint.
- Audio was FFT-resampled to 16 kHz; the production eval may use the same `load_audio_16k` path,
  but this run did not use VAD or segmentation.
- The 47 rows are a small, repeatedly inspected subset of T. The aligned-ETA difference is
  descriptive, not a new significance claim.
- Forced-reference support is an oracle; it does not prove the decoder would find the reference.

## Reproduction

```bash
uv run python runs/E-026/reproduce.py --out /tmp/e026
uv run python -m stt.diagnose release \
    --manifest /tmp/e026/rare47-16k.jsonl \
    --audio-root /tmp/e026/audio \
    --tokenizer data/tokenizer/moth_bpe.model \
    --out /tmp/e026/diagnose
```

The manifest SHA-256 is pinned by `reproduce.py`; `report.json` records the release, tokenizer and
per-clip audio hashes. The committed hypotheses are `hyp/greedy.jsonl`,
`hyp/beam_acoustic0.jsonl` and `hyp/beam_lm.jsonl`.
