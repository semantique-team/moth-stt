<img src="docs/assets/logo-tile.png" alt="Moth" width="76" align="left" hspace="16" vspace="4">

# moth-stt

**Moth** — a very small, open **Italian + English code-switching speech-to-text** model: a
Whisper-tiny encoder (pretrained, MIT) fine-tuned behind a **linear CTC head** over a shared
IT+EN SentencePiece BPE. **11.6 M parameters, 19.4 MB shipped**, running in a browser tab
(ONNX Runtime Web / WASM) and on any CPU.

<br clear="left" />

[**Live demo**](https://semantique-team-moth-demo.static.hf.space) ·
[**Weights**](https://huggingface.co/semantique-team/moth-stt) ·
[Design docs](docs/README.md) · [Live status](STATUS.md) · [Experiment log](EXPERIMENTS.md)

The demo runs **in the browser**: the audio never leaves the tab.

## What it does

Italian sentences with English words and phrases inside them — *"ho spostato il meeting a
domani"*, *"il deploy è andato bene, facciamo un quick sync dopo la call"*. Italian tech jargon,
acronyms, Italianised verbs (*deployato*, *pushare*, *loggare*).

| Set | n | WER | ETA |
|---|---|---|---|
| Mixed test T, all speakers | 600 | **25.20** | **54.17** |
| T, speaker S3 — **never seen in training** | 200 | **26.92** | **48.45** |
| FLEURS-IT test | 697 | **20.14** | — |
| FLEURS-EN test | 525 | **49.14** | — |
| Common Voice IT test | 14 643 | **24.37** | — |

Against Whisper-tiny on the same 600 clips (forced `it`, beam 5): WER **−25.52** and ETA **+18.12**,
both won. On FLEURS-IT it beats Whisper-tiny's best H0 variant by **7.28 points**. It also beats
Whistle on the mixed set's WER (−4.22), though Whistle stays ahead on monolingual Italian read
speech.

**Noise (0.7.0):** on a controlled SNR curve it gains **6.3 WER points at 15 dB** and **11.5 at
10 dB** over the previous checkpoint, with clean performance a tie. On a **room recorded after
training** — one the model has never heard — the gain is larger (**11.2** and **18.3**), because that
room is harsher; absolute WER there is 36.5 at 15 dB. Before 0.7.0, at 15 dB it scored worse than
Whisper-tiny does on clean audio.

**Speed:** 45× real time on 4 threads, 24× on 1 thread, with the VAD in the loop (Apple M5, WASM
SIMD; 5 s clips).

## What it does not do

No punctuation, no casing, numbers spelled out. Pure English is weak (~50% WER). Speaker coverage
is three people, and a fourth was recruited and never recorded. **Noise still hurts**: 0.7.0
was evaluated on a controlled noise pool and one unseen room, not every real environment.
English terms remain the weak spot
(~27% WER on a sentence containing one, ~14% without).

The full list, with numbers, is in the [model card](runs/E-024/MODEL_CARD.md#limitations-stated-plainly).

## Using it

```python
import soundfile as sf
from stt.onnx_runner import ReleaseMoth

runtime = ReleaseMoth("release")  # complete downloaded package; validates hashes and size
audio, sample_rate = sf.read("speech.wav", dtype="float32")
text, seconds = runtime.transcribe(audio, sample_rate=sample_rate)
```

Requires mono 16 kHz PCM in `[-1, 1]`; no implicit resampling. Sessions, filters and the
shipped binary LM load once. VAD runs once per recording; only multiple speech runs are
segmented, and any selected speech span over 30 seconds raises rather than truncating.
The release config supplies the decoder settings. No training-side pickle is required.

```bash
uv sync --locked --no-dev             # runtime-only checkout; no Torch
uv run --locked --no-dev python -m stt.onnx_runner release speech.wav
uv run --locked --no-dev python -m stt.release release
# Explicit extended asset tier, never more than 25 MB:
uv run --locked --no-dev python -m stt.release release --budget-bytes 25000000
```

The 19.44 MB figure counts the seven required model assets, including config and NOTICE.
It excludes ORT Web/WASM, page assets and Python dependencies. Use `--browser-root PATH`
with `stt.release` to report the complete browser directory separately (raw bytes, not
compressed cold HTTP transfer). `ReleaseMoth(..., budget_bytes=25000000)` explicitly enables
the extended asset tier; 20 MB remains the default.

`token_bonus` is not optional: it is what stops the beam from under-emitting. See
[`runs/E-022/RESULT.md`](runs/E-022/RESULT.md) — without it the same weights score 6.9 WER worse.

## Layout

| Path | What |
|---|---|
| `stt/` | the package: model, features, tokenizer, decoders, training, evaluation, export |
| `web/` | the browser runtime: JS log-mel, ORT session, CTC decoders, VAD, mic capture |
| `runs/E-0NN/` | one directory per experiment: config, logs, metrics, hypotheses |
| `docs/` | the design documents (`01`–`11`) and the plans |
| `recordings/` | the recorded gold/test scripts, manifests and recorder |
| `STATUS.md` | where the project is now — read this first |

## Reproducing

```bash
uv sync --locked --all-extras         # training/export and dev dependencies
uv run --locked --all-extras pytest -q
uv run --locked --all-extras ruff check .
node --experimental-vm-modules --test web/tests/unit-runtime.test.mjs

# evaluate a checkpoint on the ship-gate sets
uv run python runs/E-022/eval.py --arm frozen --int8     --checkpoint runs/E-024/checkpoints/ckpt_best.pt

# rebuild the release (verifies the export gates) and the browser package
uv run --locked --all-extras python -m stt.export --checkpoint runs/E-024/checkpoints/ckpt_best.pt --head-blocks 2 \
    --tokenizer data/tokenizer/moth_bpe.model --version 0.7.0
uv run python runs/E-010/package_browser.py --beam 12 --top-k 10 --lm-weight 0.5 --token-bonus 2.4
```

Checkpoints, raw audio, mel shards and the built `release/` are **not** in git — the mel cache
alone is 44 GB. Every number in the table above traces to a committed `runs/E-0NN/` artifact.

Runtime acceptance and limitations of the infrastructure changes are recorded in
[`runs/runtime-2026-11-10/RESULT.md`](runs/runtime-2026-11-10/RESULT.md).
These changes do not replace the published acoustic weights or establish a quality gain.
The [implementation status](docs/plans/2026-11-10-lightweight-it-en-runtime-plan.md#implementation-status)
and [mixed-dev protocol](docs/guides/mixed-dev-and-quality-experiments.md) distinguish
completed runtime work from experiments blocked on recordings and the frozen LM corpus.

## Provenance

Apache-2.0 code. The initial encoder weights come from [`openai/whisper-tiny`](https://github.com/openai/whisper)
(MIT); the no-speech guard is Silero VAD v5 (MIT). Training data: Common Voice IT (CC0), MLS
Italian (CC-BY 4.0), LibriSpeech (CC-BY 4.0), MUSAN (CC-BY 4.0), plus 50 minutes recorded for this
project. Full attributions in [`NOTICE`](NOTICE).

The name and the mark come from the family naming thread (insects); `stt` names the task, not the
parameter scale.
