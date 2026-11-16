# Archive record — Moth 0.8.1

**Status: frozen on 2026-11-16.** The model is released and the paper is written;
this checkout is the record of how 0.8.1 was built and measured. No new training
or model changes are planned here. A larger model will live in a separate
repository and will consume the recorded dataset, not this codebase.

## What is released

| Artifact | Where | Version |
|---|---|---|
| Model weights, browser package, demo | <https://huggingface.co/semantique-team/moth-stt> | 0.8.1 |
| Live browser demo | <https://semantique-team-moth-demo.static.hf.space> | 0.8.1 |
| Code, experiments, paper source | this repository | `v0.8.1` tag (see below) |
| Recorded dataset | [semantique-team/moth-cs](https://huggingface.co/datasets/semantique-team/moth-cs) (public, v1.0.0) | CC-BY-4.0 |
| Paper | [`paper/main.pdf`](paper/main.pdf), LaTeX in [`paper/`](paper/) | November 2026 |

The released package is **19,706,289 bytes** across seven required files; the
INT8 ONNX graph is **11,910,677 bytes**. The hashes are in
`runs/E-031/release-0.8.1.json`.

## What this checkout contains

- `stt/` — model, data pipeline, training, decoding, evaluation, export.
- `runs/E-*` — configs, metrics, hypotheses, comparison JSON, logs. This is the
  evidence for every number in the paper. Checkpoints are not versioned.
- `docs/` — design, experiment ledger, reviews, handover, paper-readiness notes.
- `paper/` — LaTeX source and generated figures.
- `web/` — ONNX Runtime Web runtime and demo tests.
- `recordings/` — protocol, scripts, data card. Audio is in the dataset
  repository, not here.
- `NOTICE` — Whisper MIT, Silero MIT and dataset attributions.

## What is not here

- Raw audio, mel shards, model checkpoints, the released ONNX package. These are
  large or re-derivable; see `data/SOURCES.md` and
  `docs/20-archive-handover.md`.
- The serialized tokenizer is in `data/tokenizer/` locally but gitignored; it is
  included in the private archive bundle, not in git.
- Signed consent forms: personal data, held offline.

## Reproducibility tiers

| Tier | What you can do from a clean clone | Needs |
|---|---|---|
| **Re-score** | Recompute WER/ETA, per-speaker tables, noise and rare-term readouts, every figure | `runs/` JSON and logs (in git) |
| **Re-export** | Rebuild the INT8 ONNX and the release package from a checkpoint | a checkpoint (private backup) + `stt/export.py` |
| **Retrain** | Re-run E-024/E-028/E-036/E-040 | external corpora, recorded audio, mel shards, checkpoints (not in git) |

The paper's Appendix C states the same boundary. Do not claim a clean clone can
retrain the model.

## Freeze checklist

- [x] Backup of recordings, release, manifests, tokenizer, paper and selected
      checkpoints, with SHA-256 verification (2026-11-16).
- [x] Root `NOTICE` and `CITATION.cff`.
- [x] Dataset v1.0.0 prepared and verified (2,193 clips, 1.46 h).
- [x] `data/SOURCES.md` and `docs/20-archive-handover.md`.
- [x] Local caches and reproducible checkpoints removed.
- [ ] Annotated git tag `v0.8.1` and GitHub Release.
- [ ] GitHub repository archived (read-only).
- [ ] Zenodo software DOI and dataset DOI.
- [x] Dataset v1.0.0 published publicly at `semantique-team/moth-cs` (2026-11-16).
- [ ] arXiv submission / v2 with the dataset DOI in the artifacts paragraph.

The unchecked items are deliberate: they change public state and were left for
the maintainer to execute. The commands are in
`docs/20-archive-handover.md`.

## Citation

Use `CITATION.cff`. Until a DOI is minted, cite the repository and the model
URL. The paper is a preprint; cite the arXiv version when it exists.

## What comes next

A larger speech-to-text model in `semantique-team/moth-large` (folder scaffolded
on 2026-11-16, not yet a git repository). It will:

- pin the dataset at `revision="v1.0.0"` and verify the manifest hashes;
- reuse `stt/text.py`, `features.py`, `eval.py`, `decode.py` and `release.py`
  from this checkout at commit `f3ecd1b`, so the evaluation semantics are
  identical;
- start a new `EXPERIMENTS.md` at E-001 and use moth 0.8.1 as the baseline;
- need new speakers and new sentences for a clean unseen-speaker test. S4 was
  never recorded; do not describe S3 as pristine.

## Contact

Open an issue on this repository or use the Semantique Team organisation on
Hugging Face: <https://huggingface.co/semantique-team>.
