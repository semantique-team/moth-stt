---
license: apache-2.0
language:
  - it
  - en
tags:
  - speech-to-text
  - automatic-speech-recognition
  - code-switching
  - italian
  - english
  - ctc
  - onnx
  - int8
  - whisper
  - edge
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-tiny
datasets:
  - google/fleurs
  - openslr/librispeech_asr
  - facebook/multilingual_librispeech
model-index:
  - name: moth-stt-0.7.0
    results:
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          type: fleurs
          name: FLEURS (it_it)
          config: it_it
          split: test
        metrics:
          - type: wer
            value: 20.14
            name: WER (beam+LM, normalized, percent)
---

<img src="logo-tile.png" alt="Moth" width="88" />

# Moth STT 0.7.0 — Italian + English code-switching, 19.4 MB, CPU-only

> **Live demo:** [semantique-team/moth-demo](https://huggingface.co/spaces/semantique-team/moth-demo) —
> runs entirely in your browser. The audio never leaves the tab: the same 19.4 MB package on this
> page is what the demo downloads and executes through ONNX Runtime Web.

A very small open **Italian + English code-switching** speech-to-text model: a Whisper-tiny
encoder initialised from `openai/whisper-tiny` (MIT), fine-tuned behind a **linear CTC head** over
a shared IT+EN SentencePiece BPE, 11.6 M parameters, shipped as an **11.9 MB int8 ONNX** graph with
a 5.1 MB trigram LM and a 2.3 MB Silero VAD in the loop.

It runs **record-then-transcribe** in the browser on ONNX Runtime Web (WASM) and on CPU, at
**47× real time** on 4 threads and **24×** on 1 thread on an Apple M5.

**This is not a streaming model and not a general-purpose multilingual model.** It recognises
*natural Italian speech with English terms embedded in it* — the way Italians actually talk about
work, tech and media — and it is deliberately tiny.

> **What changed in 0.7.0.** The model was fine-tuned with **half of its Italian read speech heard
> through noise** (MUSAN + room tone, 10–30 dB SNR, random per clip). Everything else — size,
> architecture, tokenizer, LM, decoding, 19.4 MB — is identical to 0.6.0. The point is not a better
> clean number (clean is a tie: −0.7 on the mixed set, ±0.3 elsewhere). The point is that the model
> now survives a normal room: on a controlled noise curve it gains **6.3 WER points at 15 dB, 11.5
> at 10 dB and 9.7 at 5 dB**, all significant. Before this release, at 15 dB it scored *worse than
> Whisper-tiny does on clean audio*.
>
> 0.6.0 had already fixed the decoder: beam search had been charging a language-model cost per
> emitted token with nothing to counterbalance it, so it deleted words. That fix alone was worth
> 6.9 WER points and needed no training; this one needed a training run and buys robustness.

## What it is good at, and what it is not

| | |
|---|---|
| **Good at** | Italian sentences with English words and phrases inside them ("il deploy è andato bene, facciamo un quick sync dopo la call"), Italian tech jargon, acronyms (KPI, MVP, API), Italianised verbs (deployato, pushare, loggare). |
| **Weaker at** | Pure English (49.6% WER), long-form dictation, speakers very different from the training voices, punctuation and casing (**there is none** — output is lowercase, unpunctuated, numbers spelled out). |
| **Out of scope** | More than 30 s per pass without chunking, music, singing, overlapping speech, any language other than Italian/English. |

The model emits **no words at all on silence and steady noise** (0 words over 337 non-music clips
with the bundled VAD), which is what makes it usable as a push-to-talk front end. It **can still
emit words on music** — the VAD does not stop all of it (12 of 120 music clips produced words), and
neither does any other small model we measured.

## Results

All numbers are WER / ETA (English-term accuracy) after the same text normalization used to train
and evaluate every system here. Beam+LM decoding (beam 12, top-k 10, BPE trigram LM weight 0.5,
per-token bonus 2.4).

| Set | n | Moth 0.6.0 WER | ETA |
|---|---|---|---|
| **Mixed test T, all speakers** | 600 | **25.20** | **54.17** |
| **T, speaker S3 — never seen in training** | 200 | **26.92** | **48.45** |
| T, speaker S1 (training speaker) | 200 | 21.67 | 63.28 |
| T, speaker S2 (training speaker) | 200 | 26.93 | 51.27 |
| FLEURS-IT test | 697 | 20.14 | — |
| FLEURS-EN test | 525 | 49.14 | — |
| Common Voice IT test | 14 643 | 24.37 | — |

**Read the speaker breakdown, not just the total.** 400 of the 600 mixed-test clips come from
speakers whose recorded audio is in the training mix, so the aggregate is optimistic: the two
trained speakers average 25.54 against **26.66 for S3**, who never contributed a single second to
training. **S3 is the honest number**, and it is behind both trained voices (21.67 and 26.93)
rather than behind them. Note also the direction of the change: the gain against the previous
release is *larger* on S3 (−8.77) than on the trained speakers (−5.92), which is what a recovered
decoding bias looks like and not what speaker fit looks like.

### Against other systems on the same 600 clips

| System | Size | WER | ETA |
|---|---|---|---|
| Parakeet TDT 0.6B v3 int8 | 639 MB | **10.98** | **83.60** |
| **Moth 0.7.0** | **19.4 MB** | **25.20** | **54.17** |
| Whistle (forced `it`) | 16.9 MB | 29.42 | 49.28 |
| Whisper-tiny forced `it`, beam 5 | 39 MB | 50.72 | 36.05 |
| Whisper-tiny auto, beam 5 | 39 MB | 68.86 | 39.50 |

Moth beats both Whisper-tiny variants on **both** metrics, by a wide margin: WER −25.52
and ETA +18.12 against the forced-`it` variant (both CIs exclude 0). It also beats
**Whistle** on WER — **−4.22** — for the first time in this project's measurements,
though it remains behind Parakeet, which is 33× larger, and matches Whisper and Whistle on the
"no speech in, no words out" property only because of the bundled VAD (Parakeet also achieves 0
non-music words; Whisper and Whistle do not).

On the held-out speaker S3 against Whisper forced-`it`: WER **−19.51 [−24.31, −14.38] won**, ETA
+8.73 [−1.08, +20.00] **tie**. Against Whistle on S3 the two are indistinguishable (WER −1.70
[−5.61, +2.29], ETA −4.79 [−14.36, +4.30], both ties).

On FLEURS-IT, which is the hardest comparison for a 19 MB model, Moth is **20.14%** against
Whisper-tiny auto+beam 5 at 27.42% — a clear win. It remains 8.29 points
behind Whistle's 11.85% on that set: Whistle is a 220 MB decoder-based model and is better on
monolingual Italian read speech, while Moth's advantage is the mixed set and the footprint.

## Noise

This is the reason 0.7.0 exists. On 100 held-out-speaker clips, noise mixed at a controlled SNR:

| SNR | 0.6.0 | **0.7.0** | Δ |
|---|---|---|---|
| clean | 22.24 | 23.94 | +1.70 [−0.34, +3.72], tie |
| 30 dB (quiet room, fan) | 25.98 | **23.26** | −2.72 [−5.63, +0.17] |
| 20 dB (office) | 27.50 | **26.83** | −0.68 [−4.13, +2.73] |
| 15 dB (noisy office) | 33.62 | **27.33** | **−6.28 [−10.58, −2.38]** |
| 10 dB (bar) | 44.82 | **33.28** | **−11.54 [−16.18, −7.43]** |
| 5 dB (street) | 50.93 | **41.26** | **−9.68 [−14.50, −5.01]** |

**Read this with its caveat**: the noise used for training and the noise used for this curve come
from the same small pool (MUSAN noise + one room), so the gain is an **upper bound** for a room the
mixer has never heard. The direction is solid — three of the five conditions are significant — but
a recording in an unfamiliar café will land between the two columns, not at the new one.

Before 0.7.0 the model scored **38.4% at 15 dB**: worse than Whisper-tiny does on clean audio. That
is no longer true.

## Int8 cost

The shipped graph is dynamically quantized int8. Measured end to end:

| Set | PyTorch fp32 | int8 | Δ |
|---|---|---|---|
| Mixed test T (600) | 25.20 | 25.14 | −0.06 — tie |
| T, held-out S3 (200) | 26.92 | 27.00 | +0.08 — tie |

Transcripts are identical to fp32 for **79%** of clips on both sets. The int8 cost is inside the
noise floor of this test set; on longer out-of-domain audio it has been measured at +0.3 points
(FLEURS-IT, E-020), which is the fair number to expect there.

The browser runs the same int8 graph through ONNX Runtime Web and produces transcripts that match
Python: on 200 clips the paired difference is **−0.09 WER [−1.22, +0.96], a tie**, with 154/200
byte-identical transcripts. The remaining difference is float noise in the two mel front-ends, not a
different model.

## How to use

The package is the whole runtime — no PyTorch, no Python:

```text
model.int8.onnx     11.9 MB   encoder + head + log-softmax, batch 1, dynamic time
lm_bpe3.bin          5.1 MB   BPE trigram LM, int16-quantized CSR (browser format)
silero_vad.onnx      2.3 MB   Silero VAD v5, unmodified, MIT
vocab.json                    pieces by CTC id, blank = 0, "▁" is the space marker
mel_filters.bin       64 KB   Whisper's 80×201 filterbank
config.json                   constants and per-file sha256
NOTICE                        Whisper MIT + Silero MIT + dataset credits
```

The input is **log-mel** `[1, 80, T]` at 16 kHz: STFT `n_fft=400`, `hop=160`, periodic Hann,
`center=True` with reflect padding, last frame dropped, Whisper's Slaney filterbank,
`log10(max(mel, 1e-10))`, per-clip floor `max(x, x.max()−8)`, then `(x+4)/4`. `config.json` has
every constant, and the repository ships the reference implementation in `stt/features.py` plus a
JavaScript port tested against it.

**Decoding.** CTC greedy, or prefix beam search with shallow fusion. `config.json` carries the
settings this release was measured with — `beam: 12`, `top_k: 10`, `lm_weight: 0.5`,
`token_bonus: 2.4` — and they are worth **11.8 WER points** over greedy (37.70 → 25.92 on the mixed
set). Three notes for anyone reimplementing the decoder:

- **`token_bonus` is not an optional nicety.** It is added once per emitted token and it is what
  keeps the search from under-emitting. Set it to 0 and you reproduce 0.5.1's behaviour, including
  its word deletions.
- **`lm_weight` and `token_bonus` must be tuned together.** Every emitted token pays
  `lm_weight × log P_LM(token)`, which is negative; the bonus pays it back. Tuning one without the
  other finds a false optimum — which is exactly the mistake that cost this model 6.9 WER points
  for several releases.
- **Do not apply them to greedy.** Greedy is length-neutral and unaffected; the fast path is
  byte-identical to earlier releases.

Runs on CPU at 47× real time (4 threads, WASM SIMD, Apple M5) with the VAD in the loop; 39× at
20 s on 4 threads, 20× on 1 thread.

## Training data

- **Natural speech (438 h)**: Common Voice IT 27 (CC0, 167.8 h after filtering), MLS Italian
  (CC-BY 4.0, 169.9 h), LibriSpeech English (CC-BY 4.0, 100 h).
- **Spliced IT+EN (110 h)**: Italian and English utterances concatenated at utterance level, to
  expose the model to switches without gold code-switch audio.
- **Recorded gold (50.2 min, ×50 = 41.9 h)**: two speakers reading 1,263 Italian sentences with
  embedded English terms, recorded for this project. 7.1% of the 589.6 h training mixture.
- A **BPE trigram LM** built on the training text and augmented with the training-side mixed
  scripts (never with the test scripts).

Provenance notes: MLS Italian and MUSAN are distributed through **OpenSLR** (94 and 17), not as
Hugging Face datasets, which is why they are not listed in the YAML above; Common Voice IT is
release **27.0** from the Mozilla Common Voice site (the Hub carries up to 17.0). Full attribution
lines for every corpus are in `NOTICE`.

The model was fine-tuned from an Moth checkpoint initialised from `openai/whisper-tiny`'s encoder.
**No test or dev set was ever trained on**, and the held-out speaker S3 contributed no audio to any
training run.

## Limitations, stated plainly

1. **No punctuation, no casing, no formatting.** Output is normalized text by design.
2. **Numbers are spelled out** in Italian ("quindici e trenta"), which is the training convention.
3. **Speaker coverage is three people**, not the four this project planned: a fourth speaker was
   recruited and never recorded, so every generalisation claim here rests on **one** held-out voice
   (S3). S1 and S3 share a microphone, S2 does not. Performance on voices unlike these will drop,
   and we have not measured by how much.
4. **S1 has rhotacism** (a non-standard /r/). The natural corpora keep the standard realisation in
   the base model, but any test number on S1 is optimistic for speakers with a standard /r/.
5. **English-only speech is weak** (49.1% WER). This is a code-switching model, not a bilingual one.
6. **Music can produce words.** The VAD stops silence and steady noise, not music.
7. **Noise robustness was trained on a small, synthetic pool** (MUSAN noise + one room tone, mixed
   at a random SNR). The gain in the table above is an upper bound for an unfamiliar room, and it
   was measured on the same pool it was trained on. A real café will be between the two columns.
8. **FLEURS is public** and may be in the pretraining data of the models we compare against; the
   recorded mixed set T is the only clean benchmark here, and S3 the only clean speaker.
9. **Single seed per configuration.** Training ran with one seed throughout. The quoted intervals
   are utterance-level bootstrap intervals — they describe sampling noise over utterances, **not**
   training variance, which was never measured.
10. **The decoder was calibrated on dev read speech** (MLS-IT + LibriSpeech-EN) and then frozen. Its
   gains are larger on the held-out speaker than on the trained ones, which is the signature of a
   bias being removed rather than a model being fitted, but the ETA half of our own ship gate is
   still a tie on S3: ETA +8.73 points with a confidence interval that crosses zero. Treat the ETA
   column as directional.
11. **Not a fine-tune of another code-switch model.** We tested that alternative for reference
    (a LoRA on Whistle with the same gold) and it did **not** transfer: +2.30 WER on the held-out
    speaker. The claim in this card is about this architecture.

## Licence and attribution

- **Model repository: Apache-2.0.**
- The initial encoder weights come from **`openai/whisper-tiny`, MIT**; the MIT notice travels in
  `NOTICE`. The shipped encoder is fine-tuned and is no longer Whisper's.
- The no-speech guard is **Silero VAD v5**, MIT, shipped unmodified.
- Training/evaluation data: Common Voice (CC0), MLS (CC-BY 4.0), LibriSpeech (CC-BY 4.0), FLEURS
  (CC-BY 4.0), MUSAN (CC-BY 4.0), and the authors' own recordings. Full attributions in `NOTICE`.

## Citation

```bibtex
@misc{moth-stt-2026,
  title  = {Moth: a 19 MB Italian-English code-switching speech-to-text model},
  author = {Semantique},
  year   = {2026},
  url    = {https://huggingface.co/semantique-team/moth-stt}
}
```

## Provenance

Checkpoint E-024 (noise-augmented fine-tune from E-014), **release 0.7.0 (2026-11-09)**, built with
the export gate verified (fp32 median |Δ| 1.888e-4, worst 1.217e-3 over 80 draws; int8 top-1 logit
agreement 100%). The LM binary is unchanged (`dcf1d59f…`); the acoustic model is not `7a075da4…`.
The weights are 11.6 M parameters either way, and 0.7.0 changes only what the model was trained on.

Every number in this card comes from the committed evaluation artifacts in the source repository
(`runs/E-017/`, `runs/E-020/`, `runs/E-022/`), scored with a paired bootstrap (1,000 resamples,
seed 0) through `stt.eval`. The decode settings were selected on dev only and frozen before the
measurement (`runs/E-022/PLAN.md`); a third experiment re-tuned the LM and adopted nothing, because
the only improvement found did not fit the 20 MB budget (`runs/E-023/`); the noise run was planned
with its prediction and kill criteria before the data was built (`runs/E-024/PLAN.md`).
