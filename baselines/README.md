# Baselines (H0)

Every baseline is a **producer of hypothesis files** for `stt/eval.py`; none of them touches the
scoring code (`docs/05-evaluation.md`). Run them from the repository root.

```bash
# 1. produce hypotheses (one directory per system × variant)
uv run baselines/whisper_runner.py --model tiny --variant it --decode beam5 \
    --sets fleurs_it_test,fleurs_en_test --out baselines/runs/whisper-tiny-it-beam5

# 2. score them (writes metrics.json; --vs gives the paired difference and the tie flag)
uv run stt/eval.py --manifest data/manifests/fleurs_it_test.jsonl \
    --hyp baselines/runs/whisper-tiny-it-beam5/fleurs_it_test.jsonl \
    --out baselines/runs/whisper-tiny-it-beam5/fleurs_it_test.metrics.json
```

Each run directory gets `<set>.jsonl` (hypotheses) and `meta.json` (system, variant, clip count,
audio seconds, wall time, RTFx, errors, hotword count, size, and whether Whisper padded to 30 s).

## The matrix (from `docs/05` §7)

| System | Variants to run | Runner |
|---|---|---|
| Whisper tiny (39 M) | language auto / forced `it` × greedy / beam 5 | `whisper_runner.py` |
| Whisper base (74 M) | auto / forced `it` × greedy / beam 5 | `whisper_runner.py` |
| Cactus Whistle (16.9 MB) | auto / forced `it`; hotwords none / fair / oracle | `whistle_needle.py` (in-process, one model load) or `whistle_runner.py` (CLI template) |
| Parakeet TDT 0.6B v3 | hotwords none / fair / oracle | `parakeet_sherpa.py` (in-process) or `parakeet_runner.py` (CLI template) |

Whisper has no biasing interface, so it has no hotword variants.

## Hotword fairness (the rule)

- **fair** = English terms from the **R scripts** + the curated `data/en_terms.txt`. It is the only
  list a deployable model could know in advance.
- **oracle** = fair + the **test set's** own English terms. Always labelled
  `oracle_hotwords: true` in `meta.json` and reported as an oracle line — never compared as if it
  were deployable.
- Our own model's hotword results follow the same two-line convention.

## Whistle / Parakeet: command templates

Their CLIs change between versions, so the runners take a template instead of hard-coding flags:

- `{audio}` — appended automatically if missing;
- `{lang}` — `it` or empty (auto); when empty the whole `--language {lang}` pair is removed;
- `{keywords}` — path to a file with one hotword per line, written by the runner; when hotwords
  are off the whole `--keywords {keywords}` pair is removed.

```bash
# Whistle (adapt the flags to your needle version)
uv run baselines/whistle_runner.py --variant it --hotwords fair \
    --command "needle transcribe --language {lang} --keywords {keywords} {audio}" \
    --sets fleurs_it_test --out baselines/runs/whistle-it-fair

# Parakeet v3 via sherpa-onnx (adapt the model paths)
uv run baselines/parakeet_runner.py --hotwords none \
    --command "sherpa-onnx-offline-transducer --tokens=tokens.txt --encoder=encoder.onnx \
--decoder=decoder.onnx --joiner=joiner.onnx {audio}" \
    --size-file encoder.onnx --sets fleurs_it_test --out baselines/runs/parakeet-none
```

## Set names and manifests

`--sets` takes the manifest file stems under `data/manifests/`: `fleurs_it_test`,
`fleurs_en_test`, `fleurs_it_dev`, `fleurs_en_dev`, `cv_it_test`, `noise`. The recorded T set is
`T` with `--manifests-dir recordings/manifests` (and `--audio-root recordings/audio`). The noise
split uses `--audio-root data/raw/noise` (manifest `data/manifests/noise.jsonl`, built by
`stt/data/build_noise.py`). They are produced by the M2 data pipeline (and the recordings step for
T).

## Requirements

- Whisper: the `openai-whisper` dev dependency (already in `pyproject.toml`); model weights are
  downloaded on first use.
- Whistle: `cactus-needle` plus `soxr` (for the 48 kHz recordings) in a **separate** venv, never a
  project dependency:
  `uv venv .venv-baselines --python 3.12 && uv pip install --python .venv-baselines/bin/python cactus-needle soxr num2words soundfile`
  (`.venv-baselines/` is gitignored). Run with `PYTHONPATH=. .venv-baselines/bin/python ...`.
- Parakeet: `sherpa-onnx` (installed in the same baseline venv) plus the int8 model files; ≈ 640 MB:
  `huggingface-cli download csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 --local-dir models/parakeet-tdt-0.6b-v3-int8`
  (or the CLI-template runner `parakeet_runner.py`).
