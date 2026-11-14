<img src="docs/assets/logo-tile.png" alt="Moth" width="76" align="left" hspace="16" vspace="4">

# moth-stt

**Moth** — a very small, open **Italian + English code-switching speech-to-text** model: a
Whisper-tiny encoder (pretrained, MIT), **two additional transformer blocks** and a linear CTC head over a shared
IT+EN SentencePiece BPE. **11.6 M parameters, 19.7 MB required model assets (0.8.1)**, running in a browser tab
(ONNX Runtime Web / WASM) and on any CPU.

<br clear="left" />

[**Live demo**](https://semantique-team-moth-demo.static.hf.space) ·
[**Weights**](https://huggingface.co/semantique-team/moth-stt) ·
[Design docs](docs/README.md) · [Live status](STATUS.md) · [Experiment log](EXPERIMENTS.md)

For reproducible research: [artifact-audited experiment ledger](docs/12-experiment-ledger.md),
[current model and paper limitations](docs/13-current-model-and-paper-readiness.md), and
[verified operational workflows](docs/guides/repository-workflows.md). The preprint draft is
[`paper/main.pdf`](paper/main.pdf) ([LaTeX source](paper/main.tex)).

The demo runs **in the browser**: the audio never leaves the tab.

## What it does

Italian sentences with English words and phrases inside them — *"ho spostato il meeting a
domani"*, *"il deploy è andato bene, facciamo un quick sync dopo la call"*. Italian tech jargon,
acronyms, Italianised verbs (*deployato*, *pushare*, *loggare*).

| Set | n | WER | ETA |
|---|---|---|---|
| Mixed test T, all speakers | 600 | **24.08** | **55.23** |
| T, S3 — **no acoustic training audio; repeatedly evaluated** | 200 | **24.96** | **51.27** |
| FLEURS-IT test | 697 | **20.16** | — |
| FLEURS-EN test | 525 | **49.81** | — |

These are **0.8.1 INT8 ONNX reference** results with beam 12/top-k 10/LM weight 0.55/token
bonus 2.7 and the E-031 LM, not browser acoustic WER measurements. FLEURS uses filtered test
subsets. No current Common Voice test result is retained; the historical 0.7.0 fp32 result
was 24.37 WER on 14,643 clips.
S1/S2 supply 400/600 test clips and training audio; S3 shares S1's microphone. T is scripted read
speech and has repeatedly informed development, not an untouched natural code-switching test.

**Historical 0.7.0 fp32 comparisons:** against Whisper-tiny on the same 600 clips (forced `it`, beam 5): WER **−25.52** and ETA **+18.12**,
both nominal paired comparisons won on that reused aggregate. On FLEURS-IT it beats Whisper-tiny's best H0 variant by **7.28 points**. It also beats
Whistle on the mixed set's WER (−4.22), though Whistle stays ahead on monolingual Italian read
speech.

**Noise (0.7.0):** on a controlled SNR curve it gains **6.3 WER points at 15 dB** and **11.5 at
10 dB** over the previous checkpoint, with clean performance a tie. On a **room recorded after
training** — one the model has never heard — the gain is larger (**11.2** and **18.3**), because that
room is harsher; absolute WER there is 36.5 at 15 dB. Before 0.7.0, at 15 dB it scored worse than
Whisper-tiny does on clean audio in an earlier diagnostic probe; the paired E-024 control is
**33.62% at 15 dB**, not 38.4%. Do not combine those two protocols.

**Historical speed (0.7.0):** 45× real time on 4 threads, 24× on 1 thread, with the VAD in the loop (Apple M5, WASM
SIMD; 5 s clips).

## What it does not do

No punctuation, no casing, numbers spelled out. Pure English is weak (~50% WER). Speaker coverage
is three people, and a fourth was recruited and never recorded. **Noise still hurts**: 0.7.0
was evaluated on a controlled noise pool and one unseen room, not every real environment.
English terms remain the weak spot
(~27% WER on a sentence containing one, ~14% without).

The current limitations are in the [0.8.1 model card](runs/E-031/MODEL_CARD.md#limitations-stated-plainly).

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

The **19,706,289-byte (19.7 decimal MB)** figure counts the seven required 0.8.1 model assets,
including config and NOTICE. It is not the whole runtime or download footprint.
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

# Requires full git history; see workflow guide for shallow-clone recovery.
# Current 0.8.1 LM reproduction from pinned pre-E-032 git inputs:
uv run --locked --all-extras python runs/E-031/reproduce_corpus.py
# Bash equivalent: bash runs/E-031/freeze.sh
# Requires the original tokenizer. Build in a disposable checkout: writes build.json.
uv run --locked --all-extras python runs/E-031/build_lm.py
# Requires verified E-028/0.8.0 source assets in release/, not a generic 0.7 export.
uv run --locked --all-extras python runs/E-031/stage_release.py
uv run --locked --no-dev python -m stt.release runs/E-031/release
```

For the immutable HF download pin, required hashes and staging prerequisites, use the
[workflow guide](docs/guides/repository-workflows.md#current-081-reproduction). These commands
do not rerun quality gates or train new weights. For new experiments use its
[experiment workflow](docs/guides/repository-workflows.md#new-experiments). The historical Windows
`resource` import blocker was already fixed before this branch: CPU pytest can collect on
Windows, but CPU unit tests do not certify production Windows training. This documentation
task ran no full LM build, staging, quality gate or model release.

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
project. Attributions are generated in `release/NOTICE` by
[`stt/export.py`](stt/export.py), using [`stt/data/sources.py`](stt/data/sources.py);
there is no top-level NOTICE in this checkout.

The audit found unresolved checkpoint-step and S3 baseline-delta metadata conflicts in historical
records/cards. See the [paper dossier](docs/13-current-model-and-paper-readiness.md) before citing
them. The model's recorded publication is not a new verification of live hosting.

The name and the mark come from the family naming thread (insects); `stt` names the task, not the
parameter scale.
