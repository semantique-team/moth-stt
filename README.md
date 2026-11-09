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
| Mixed test T, all speakers | 600 | **25.92** | **52.64** |
| T, speaker S3 — **never seen in training** | 200 | **26.66** | **47.32** |
| FLEURS-IT test | 697 | **20.46** | — |
| FLEURS-EN test | 525 | **49.63** | — |
| Common Voice IT test | 14 643 | **24.88** | — |

Against Whisper-tiny on the same 600 clips (forced `it`, beam 5): WER **−24.80** [−28.05, −21.80]
and ETA **+16.59** [+10.61, +22.74], both won. On FLEURS-IT it beats Whisper-tiny's best H0
variant by **6.95 points** [−8.02, −5.87]. It now also beats Whistle on the mixed set's WER
(−3.50 [−5.85, −1.24]), though Whistle stays ahead on monolingual Italian read speech.

**Speed:** 47× real time on 4 threads, 24× on 1 thread, with the VAD in the loop (Apple M5, WASM
SIMD).

## What it does not do

No punctuation, no casing, numbers spelled out. Pure English is weak (~50% WER). Speaker coverage
is three people, and a fourth was recruited and never recorded. **Noise hurts a lot**: measured
+9.9 WER at 20 dB SNR against clean, so use it somewhere quiet. English terms remain the weak spot
(~27% WER on a sentence containing one, ~14% without).

The full list, with numbers, is in the [model card](runs/E-022/MODEL_CARD.md#limitations-stated-plainly).

## Using it

```python
import onnxruntime as ort
import numpy as np
from stt.features import log_mel          # the reference front-end, also ported to JS in web/
from stt.decode import beam_search, load_pieces
from stt.lm import BPETrigramLM

session = ort.InferenceSession("release/model.int8.onnx")
mel = log_mel(audio_16k_mono)             # [80, T]
log_probs = session.run(None, {"mel": mel[None]})[0][0]

text = beam_search(
    log_probs, log_probs.shape[0], load_pieces("release/vocab.json"),
    beam_width=12, top_k=10,
    lm=BPETrigramLM.load("runs/E-008/lm_bpe3_aug.pkl.gz"), lm_weight=0.5,
    token_bonus=2.4,
)
```

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
uv run pytest -q                      # 300 tests
uv run ruff check .                   # line length 100

# evaluate a checkpoint on the ship-gate sets
uv run python runs/E-022/eval.py --arm frozen --int8

# rebuild the release (verifies the export gates) and the browser package
uv run python -m stt.export --checkpoint runs/E-017/checkpoints/ckpt_best.pt --head-blocks 2 \
    --tokenizer data/tokenizer/moth_bpe.model --version 0.6.0
uv run python runs/E-010/package_browser.py --beam 12 --top-k 10 --lm-weight 0.5 --token-bonus 2.4
```

Checkpoints, raw audio, mel shards and the built `release/` are **not** in git — the mel cache
alone is 44 GB. Every number in the table above traces to a committed `runs/E-0NN/` artifact.

## Provenance

Apache-2.0 code. The initial encoder weights come from [`openai/whisper-tiny`](https://github.com/openai/whisper)
(MIT); the no-speech guard is Silero VAD v5 (MIT). Training data: Common Voice IT (CC0), MLS
Italian (CC-BY 4.0), LibriSpeech (CC-BY 4.0), MUSAN (CC-BY 4.0), plus 50 minutes recorded for this
project. Full attributions in [`NOTICE`](NOTICE).

The name and the mark come from the family naming thread (insects); `stt` names the task, not the
parameter scale.
