# E-032 — data provenance

**Recorded 2026-11-12 by S1.** 50 sentences, ids `R-S1-0692`–`R-S1-0741`, appended to
`recordings/scripts/R-S1.jsonl` after review (`reviewed: true`, reviewer `tancredi`).
The rows are the candidate set in `candidates.jsonl`; no text was edited after review.

## Validation

| Check | Result |
|---|---|
| R-S1 manifest | 741/741 clips, 0 errors, 0 warnings |
| New subset | 50/50 clips, 167.1 s (2.8 min) |
| Mel shard `R-S1-terms3` | 50 clips, 1 file, 0 errors |
| x20 source `gold-terms3-x20` | 1,000 rows, 0.93 h, shard copied once |
| Recording log | `recordings/audio/R-S1/S1/session.log.jsonl`, last entry `R-S1-0741` |

## SHA-256

**Audit correction (2026-11-12):** the original table copied the script hash to the manifest
and called them byte-identical. They are different schemas/files; only the shared annotation
fields (`id`, `text`, `category`, `en_spans`) match; the manifest also stores `norm`. That does not certify span validity;
later annotation review may still be needed. Other entries below remain historical recorded
prefixes, not new hash verification.

| File | sha256 |
|---|---|
| `recordings/scripts/R-S1.jsonl` | `f137e5a6…` |
| `recordings/manifests/R-S1.jsonl` | Original copied hash withdrawn; matching shared annotation fields, not same bytes |
| `recordings/manifests/R-S1-terms3.jsonl` | `9a3e8bdb…` |
| `data/shards/R-S1-terms3/index.json` | `6f1ae271…` |
| `data/shards/gold-terms3-x20/manifest.jsonl` | `5a3551e2…` |
| `recordings/audio/R-S1/S1/session.log.jsonl` | `05184159…` |

## Commands

```bash
uv run python recordings/record.py --speaker S1 --kind R --start 692
uv run python recordings/build_manifest.py build --kind R --speaker S1
uv run python runs/E-032/prepare_terms.py
uv run python -m stt.data.build_mel --manifest recordings/manifests/R-S1-terms3.jsonl \
    --audio-root recordings/audio/R-S1 --out-dir data/shards/R-S1-terms3
uv run python runs/E-017/prepare_gold.py --sets R-S1-terms3 --factor 20 \
    --out-dir data/shards/gold-terms3-x20
```

## Limits

S1 is a training speaker. A gain on T-600 cannot be separated from speaker adaptation, and the
arm is a candidate only if the pre-registered gates pass; it does not authorize a release without
the new-voice readout (`runs/E-032/PLAN.md`).
