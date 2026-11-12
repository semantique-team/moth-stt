---
title: Moth STT — Italian + English, in your browser
emoji: 🦋
colorFrom: green
colorTo: gray
sdk: static
pinned: false
short_description: A 19 MB code-switching STT model that runs on your device
models:
  - semantique-team/moth-stt
license: apache-2.0
---

# Moth — live demo

Speech-to-text for **Italian with English words mixed in**, running **entirely in your browser**.

Nothing is uploaded: the 19.7 MB package (11.9 MB int8 ONNX, a 5.4 MB trigram LM, a 2.3 MB VAD)
is fetched once and then runs on your machine through ONNX Runtime Web / WebAssembly. Your audio
never leaves the tab, and the page keeps working if you disconnect after it has loaded.

This Space serves the same artifact that is published at
[`semantique-team/moth-stt`](https://huggingface.co/semantique-team/moth-stt), byte for byte, and
verifies every file's sha256 at load time before running it.

**0.8.1** is a decode-side update: the acoustic graph is unchanged, the trigram LM was rebuilt from
a frozen training-side corpus with all bigrams and the rare-term sentences weighted. Rare aligned
ETA rises from **28.4% to 46.9%** (23/81 → 38/81) and rare presence from **11/47 to 24/47**; the
mixed set improves from **24.71 to 24.08 WER** and **54.17 to 55.23 ETA**, while the held-out
speaker S3 is a tie at **24.96 / 51.27**. Natural-term presence is 432/817. The 0.8.0 rare-gate
exception is closed.

## What it is good at, honestly

- Italian sentences with English terms inside them: *"il deploy è andato bene, facciamo un quick
  sync dopo la call"*.
- Italian tech jargon, acronyms, Italianised verbs (*deployato*, *pushare*, *loggare*).
- It emits **nothing** on silence and steady noise (a Silero VAD gates the input).

## What it is not

- No punctuation, no casing: output is lowercase, unpunctuated, numbers spelled out.
- Pure English is weak (~50% WER); this is a code-switching model, not a bilingual one.
- **English terms remain the weak spot**, though 0.8.1 improves the rare ones: ~27% WER on a
  sentence containing a term against ~14% on a sentence without it (0.7.0 measurement). If it gets
  a term wrong, that is the known limit.
- Speaker coverage is three people. A fourth was recruited and never recorded.
- **Noise costs, but no longer destroys.** The 0.7.0 noise work is unchanged in 0.8.0; these
  figures refer to 0.7.0. Against clean audio: +2.9 WER at 20 dB SNR, +3.4 at 15 dB, +9.3 at
  10 dB. Before 0.7.0 those were +5.3, +11.4 and +22.6. On a room the model never heard it is
  11.2 points better than 0.6.0 at 15 dB and 18.3 at 10 dB, though the absolute WER there is
  higher (36.5 at 15 dB). A quiet room is fine; a loud café is not.
- Max 30 s per recording. Speak one sentence at a time — the page splits on pauses, but pauses help.

Numbers, limitations and provenance: [`semantique-team/moth-stt`](https://huggingface.co/semantique-team/moth-stt).
Code: `github.com/semantique-team/moth-stt`. Apache-2.0; the initial encoder is `openai/whisper-tiny` (MIT).
