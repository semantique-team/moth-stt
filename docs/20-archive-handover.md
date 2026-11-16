# 20 — Archive handover (2026-11-16)

This is the "read me first" for anyone who opens the frozen repository. It
records what the checkout can and cannot do, what was deleted, where the
irreplaceable files are, and the public-state actions that remain.

## 1. State

- **Model 0.8.1 is published** on Hugging Face and the demo Space (2026-11-12).
- **The paper is written** (`paper/main.pdf`, LaTeX source in `paper/`); the
  arXiv submission is still a maintainer action.
- **The repository is frozen** on 2026-11-16. No new experiments, no model
  changes, no data collection in this checkout.
- **The recorded dataset v1.0.0 is prepared** in `../moth-cs-dataset` (sibling
  directory): 2,193 clips, 1.46 h, 3 speakers, CC-BY-4.0. It is **not uploaded
  yet** at the time of writing.
- **The next model** has a folder scaffold at `../moth-large`, not a git
  repository yet.

## 2. Environment

Historical training host: Apple M5 MacBook Pro, 16 GB, macOS. Python ≥ 3.12,
`uv`, `ruff`, `pytest`; `tectonic` for the paper; `ffmpeg` for audio.

```bash
cd moth-stt
uv sync --locked
uv run pytest -q          # last full run before the freeze: 1170 passed, 3 skipped
uv run ruff check .
```

Native Windows training fails on the Unix `resource` import; the CPU test suite
can collect there. Do not assume this checkout is a Windows training
environment.

## 3. What was deleted (and why)

Deleted on 2026-11-16 to reclaim ~76 GB of reproducible or re-downloadable
state:

| Path | Size | Why it is safe |
|---|---:|---|
| `data/raw/cv_it_27/` | 11 GB | Public Common Voice 27.0 IT, re-download |
| `data/raw/fleurs/` | 1.7 GB | Public FLEURS, evaluation only |
| `data/raw/librispeech/` | metadata only | audio was already absent |
| `data/raw/musan/` | 100 KB | public MUSAN; noise clips retained |
| `data/shards/` | 49 GB | Mel cache, rebuild with `stt/data/build_mel.py` |
| `runs/**/checkpoints/` | 11.3 GB | training checkpoints; only E-028/E-024 best kept in the private backup |
| `models/` | 669 MB | Whisper encoder and Parakeet, re-downloadable |
| `.venv*` | 1.7 GB | recreate with `uv sync` |
| `web/node_modules/` | 166 MB | `npm install` |

**Kept:** `data/manifests/` (220 MB of training/eval manifests),
`data/tokenizer/` (248 KB), `data/raw/room_tone.wav` (30 MB) and
`data/raw/noise/` (177 MB), `release/` (19 MB), all `runs/` JSON/logs, all
tracked docs and code. See `data/SOURCES.md` for the exact source versions.

## 4. Irreplaceable files and where they are

The private backup made on 2026-11-16 is at:

- local: `~/Backups/moth-2026-11-16/`
- iCloud: `~/Library/Mobile Documents/com~apple~CloudDocs/moth-backup-2026-11-16/`

Both were verified with a full `SHA256SUMS`. The backup contains:

- `recordings/` (all audio, manifests, scripts, session logs);
- `release/` (0.8.1 package, 19 MB);
- `tokenizer/` (the serialized SentencePiece model);
- `manifests/` (all data manifests);
- `paper/` (source, figures, PDF);
- `checkpoints/E-028-ckpt_best.pt` (the 0.8.1 acoustic checkpoint),
  `E-024-ckpt_best.pt` (0.7.0) and `E-019-whistle.safetensors` (LoRA probe);
- `moth-stt.git.bundle` (full git history).

The canonical **dataset** copy is `../moth-cs-dataset/` (audio + canonical
manifests + hashes). Do not delete either copy until the dataset is uploaded and
a second off-site copy exists.

## 5. What can be reproduced

| Task | Command | Needs |
|---|---|---|
| Score the retained hypotheses | `uv run python -m stt.eval ...` | `runs/` + `data/manifests/` |
| Rebuild paper figures | `cd paper && uv run --with matplotlib --with numpy python figures/make_figures.py` | `runs/` |
| Rebuild the INT8 package | `uv run python -m stt.export ...` | `runs/E-028/checkpoints/ckpt_best.pt` from the backup |
| Retrain | `uv run python -m stt.train --config runs/E-028/config.yaml` | external corpora + recorded audio + mel shards + a checkpoint |

The paper's Appendix C is the authority on the boundary. In particular, the
0.7.0 step conflict (26,000 in the log vs 26,442 in the release label) is
disclosed and cannot be resolved from this checkout.

## 6. Known gaps

1. **Signed consent forms** are offline, not in git. The dataset card records
   the confirmation date and option.
2. **No S2 session log is present** (T, R-S2, anchor); the data card says so.
   The 50-file addendum was accepted by the builder with 0 errors, but retake
   and skip counts for S2 cannot be reconstructed.
3. **S2_0624.wav** had ffmpeg streaming sizes in its WAV header; the canonical
   dataset copy was repaired without touching samples. Hashes are in
   `../moth-cs-dataset/releases/v1.0.0.json`.
4. **FLEURS snapshot revision was not recorded.** The filtered manifests and
   hypotheses are retained, so scores can be re-scored; bit-exact audio rebuild
   is not guaranteed.
5. **Devices were not logged for every recording session.** Do not infer them.
6. **No speaker-disjoint test.** S3 never trained but informed development.
7. **The dataset is not uploaded yet.** The package is ready in
   `../moth-cs-dataset/hf/`.
8. **No git tag / GitHub Release / DOI yet.** See §7.

## 7. Public-state actions still to do

These are deliberately left to the maintainer because they change public state.

### 7.1 Tag and release

```bash
cd moth-stt
git tag -a v0.8.1 -m "Moth 0.8.1 — 19.71 MB Italian-English code-switching ASR"
git push origin v0.8.1
gh release create v0.8.1 --title "Moth 0.8.1" --notes-file ARCHIVE.md
```

Then archive the repository read-only:

```bash
gh repo edit semantique-team/moth-stt --archive
```

### 7.2 Dataset upload

```bash
cd ../moth-cs-dataset
python scripts/verify.py
# install the client once
uv tool install "huggingface_hub[cli]"
hf auth login
# create private first, upload, inspect, then flip public
hf repo create semantique-team/moth-cs --repo-type dataset --private
hf upload semantique-team/moth-cs hf . --repo-type dataset --private
```

Check on the Hub that the card renders, the three splits load and the audio
plays. Then make the repository public and tag `v1.0.0`. Finally update
`CITATION.cff` and `releases/v1.0.0.json` with the DOI.

### 7.3 Zenodo

Deposit the dataset snapshot (or the `hf/` parquet files) and the software
snapshot (this git tag) separately; mint two DOIs. Add the dataset DOI to the
paper's artifacts paragraph before the arXiv submission, or in a v2.

### 7.4 Paper

`paper/README.md` lists the pre-submission checks. Add the dataset DOI and the
software DOI to the artifacts paragraph; do not change the measured numbers.

## 8. Next model

`../moth-large/README.md` is the scaffold. The rules:

- pin the dataset at `revision="v1.0.0"` and verify the manifest hashes;
- vendor `stt/text.py`, `features.py`, `eval.py`, `decode.py`, `release.py`
  from this checkout at commit `f3ecd1b`, with attribution;
- start a new `EXPERIMENTS.md` at E-001; use moth 0.8.1 as the baseline;
- do not reuse T as the only test set: record new speakers/sentences for a
  clean unseen-speaker evaluation.

## 9. What not to do

- Do not rewrite git history to shrink the 86 MB pack; the history is evidence.
- Do not commit audio, checkpoints, mel shards or the release package.
- Do not relicense the recorded audio without a new consent round.
- Do not describe S3 as a pristine unseen speaker.
- Do not publish the dataset before checking the consent/licence caveat in
  `../moth-cs-dataset/docs/consent.md`.
