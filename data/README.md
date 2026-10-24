# `data/` — cache, manifests and the curated term list

| Path | In git? | What |
|---|---|---|
| `data/raw/<source>/` | no | Downloaded archives and extracted audio per source |
| `data/manifests/*.jsonl` | **yes** | The manifest files produced by `stt/data/sources.py` + filters, consolidated for training/eval (also the recorded sets' manifests from `recordings/`) |
| `data/en_terms.txt` | **yes** | Curated English terms (tokenizer text + fair hotwords) |
| `data/shards/` | no | Mel shards built by the mel-cache step (≈ 26 GB for 450 h) |

Large artifacts never enter git; the manifests do, because they are the reproducible input of
training and evaluation (ids, text, speaker, duration, source, split).

## Licenses (to keep in `NOTICE` and the data card)

| Source | License | Use |
|---|---|---|
| Common Voice IT 27.0 | CC0 | train / dev / test (official splits) |
| MLS Italian (OpenSLR 94) | CC-BY 4.0 | train / dev / test |
| LibriSpeech `train-clean-100` (OpenSLR 12) | CC-BY 4.0 | train (+ dev-clean) |
| FLEURS IT/EN | CC-BY 4.0 | **evaluation only** — never training, not even the FLEURS train split |
| MUSAN (noise + music) | CC-BY 4.0 | noise/hallucination set (evaluation only) |

Own recordings (`recordings/`) are the recorded mixed sets; see `recordings/data-card.md`.

## Fetching

Downloading the archives is a manual, licensed step (accepting dataset terms). Then put the
extracted tree under `data/raw/<source>/` and reuse the fetcher for anything with a URL and a
checksum:

```bash
uv run python -c "
from stt.data.sources import download
download('<url>', 'data/raw/cv_it_27/cv-corpus-27.0-it.tar.gz', sha256='<sha256>')
"
```

Then plan a split (reads the official TSVs, probes durations, normalizes text):

```bash
uv run python -c "
from stt.data.sources import plan_records, sample_by_duration, write_records
rows = plan_records('cv_it_27', 'data/raw/cv_it_27/it', 'train')
sample = sample_by_duration(rows, target_seconds=170*3600, seed=0)
write_records(sample['selected'], 'data/cv_it_train_raw.jsonl')
print(f\"{len(sample['selected'])} clips, {sample['selected_seconds']/3600:.1f} h\")
"
```

Proposed targets (`docs/03-data.md` §2): **≈ 170 h** Common Voice IT, **≈ 170 h** MLS IT,
**≈ 100 h** English (LibriSpeech `train-clean-100` or a Common Voice EN slice — decide at M2),
plus the recorded mixed sets. Dev: CV IT dev, MLS IT dev, LibriSpeech dev-clean, FLEURS IT/EN
dev. Test: FLEURS IT/EN test, CV IT official test, the recorded T set.

`ffmpeg` is required for Common Voice mp3 in some paths (`brew install ffmpeg`); FLAC reads with
`soundfile`.
