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
  - name: moth-stt-0.5.1
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
            value: 26.52
            name: WER (beam+LM, normalized, percent)
---

> **Superseded — do not publish this card.** It documents release **0.5.1**. The current release is
> **0.6.0**, whose card is [`runs/E-022/MODEL_CARD.md`](../E-022/MODEL_CARD.md): same weights and
> same int8 file, recalibrated decoder, 6.9 WER points better on the mixed test set. This file is
> kept because it is the artifact E-021 shipped and the record of what 0.5.1 claimed.

# Moth STT 0.5.1 — Italian + English code-switching, 19.4 MB, CPU-only

A very small open **Italian + English code-switching** speech-to-text model: a Whisper-tiny
encoder initialised from `openai/whisper-tiny` (MIT), fine-tuned behind a **linear CTC head** over
a shared IT+EN SentencePiece BPE, ~11.6 M parameters, shipped as an **11.9 MB int8 ONNX** graph
with a 5.1 MB trigram LM and a 2.3 MB Silero VAD in the loop.

It runs **record-then-transcribe** in the browser on ONNX Runtime Web (WASM) and on CPU, at
**52× real time** on 4 threads and **25×** on 1 thread on an Apple M5.

**This is not a streaming model and not a general-purpose multilingual model.** It recognises
*natural Italian speech with English terms embedded in it* — the way Italians actually talk about
work, tech and media — and it is deliberately tiny.

## What it is good at, and what it is not

| | |
|---|---|
| **Good at** | Italian sentences with English words and phrases inside them ("il deploy è andato bene, facciamo un quick sync dopo la call"), Italian tech jargon, acronyms (KPI, MVP, API), Italianised verbs (deployato, pushare, loggare). |
| **Weaker at** | Pure English (55.5% WER), long-form dictation, speakers very different from the training voices, punctuation and casing (**there is none** — output is lowercase, unpunctuated, numbers spelled out). |
| **Out of scope** | More than 30 s per pass without chunking, music, singing, overlapping speech, any language other than Italian/English. |

The model emits **no words at all on silence and steady noise** (0 words over 337 non-music clips
with the bundled VAD), which is what makes it usable as a push-to-talk front end. It **can still
emit words on music** — the VAD does not stop all of it, and neither does any other small model we
measured.

## Results

All numbers are WER / ETA (English-term accuracy) after the same text normalization used to train
and evaluate every system here. Beam+LM decoding (beam 5, BPE trigram LM, weight 0.2).

| Set | n | Moth 0.5.1 WER | ETA |
|---|---|---|---|
| **Mixed test T, all speakers** | 600 | **32.77** | **42.86** |
| **T, speaker S3 — never seen in training** | 200 | **35.43** | **35.21** |
| T, speaker S1 (training speaker) | 200 | 28.98 | 51.94 |
| T, speaker S2 (training speaker) | 200 | 33.88 | 41.93 |
| FLEURS-IT test | 697 | 26.52 | — |
| FLEURS-EN test | 525 | 55.48 | — |
| Common Voice IT test | 14 643 | 30.36 | — |

**Read the speaker breakdown, not just the total.** 400 of the 600 mixed-test clips come from
speakers whose recorded audio is in the training mix, so the aggregate is optimistic. **S3 is the
honest number**: that speaker never contributed a single second to training, and the model still
improves there as much as on the trained speakers (−6.13 WER [−8.68, −3.51] against the previous
checkpoint).

### Against other systems on the same 600 clips

| System | Size | WER | ETA |
|---|---|---|---|
| Parakeet TDT 0.6B v3 int8 | 639 MB | **10.98** | **83.60** |
| Whistle (forced `it`) | 16.9 MB | **29.42** | **49.28** |
| **Moth 0.5.1** | **19.4 MB** | **32.77** | 42.86 |
| Whisper-tiny forced `it`, beam 5 | 39 MB | 50.72 | 36.05 |
| Whisper-tiny auto, beam 5 | 39 MB | 68.86 | 39.50 |

Moth beats both Whisper-tiny variants on **both** metrics: WER −17.95 [−21.47, −14.88] and ETA
+6.81 [+0.85, +12.50] against the forced-`it` variant. It is **3.35 WER behind Whistle**
[1.07, 5.60] — significantly, but by about 3 points rather than the 14 of earlier checkpoints — and
it matches the best of them on the "no speech in, no words out" property: with the bundled VAD it
emits **0 non-music words over 337 clips**, which Parakeet also achieves and Whisper and Whistle do
not (on the 240-clip non-music subset before the guard was added: Whisper 54, Whistle 264,
Parakeet 0).

On the held-out speaker S3 against Whisper forced-`it`: WER −11.07 [−16.28, −5.89] **won**, ETA
−2.82 [−11.65, +6.56] **tie**.

## Int8 cost

The shipped graph is dynamically quantized int8. Measured end to end on 600 clips:

| Decoder | PyTorch fp32 | int8 | Δ |
|---|---|---|---|
| greedy | 37.70 | 38.13 | +0.43 [−0.17, +1.01] — tie |
| beam+LM | 32.77 | 32.94 | +0.17 [−0.40, +0.77] — tie |

On longer out-of-domain audio (FLEURS-IT, 7–23 s clips) the cost is small but real: +0.32
[+0.08, +0.57]. Transcripts are byte-identical to fp32 for 77% of mixed-test clips and 53% of
FLEURS clips.

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

Decoding is CTC greedy or prefix beam search with LM shallow fusion (α = 0.2). **Use the beam+LM
path for quality** — it is worth 4.9 WER points over greedy — and greedy if you need the last few
percent of speed.

Runs on CPU at 52× real time (4 threads, WASM SIMD, Apple M5) with the VAD in the loop.

## Training data

- **Natural speech (438 h)**: Common Voice IT 27 (CC0, 167.8 h after filtering), MLS Italian
  (CC-BY 4.0, 169.9 h), LibriSpeech English (CC-BY 4.0, 100 h).
- **Spliced IT+EN (110 h)**: Italian and English utterances concatenated at utterance level, to
  expose the model to switches without gold code-switch audio.
- **Recorded gold (50.2 min, ×50 = 41.9 h)**: two speakers reading 1,263 Italian sentences with
  embedded English terms, recorded for this project. 7.1% of the training mixture.
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
3. **Speaker coverage is three people.** S1 and S3 share a microphone, S2 does not. Performance on
   voices unlike these will drop, and we have not measured by how much.
4. **S1 has rhotacism** (a non-standard /r/). The natural corpora keep the standard realisation in
   the base model, but any test number on S1 is optimistic for speakers with a standard /r/.
5. **English-only speech is weak** (55.5% WER). This is a code-switching model, not a bilingual one.
6. **Music can produce words.** The VAD stops silence and steady noise, not music.
7. **FLEURS is public** and may be in the pretraining data of the models we compare against; the
   recorded mixed set T is the only clean benchmark here, and S3 the only clean speaker.
8. **Single seed per configuration.** Differences below ~1 WER point should be treated as noise
   unless a paired bootstrap says otherwise; we report CIs for exactly that reason.
9. **Not a fine-tune of another code-switch model.** We tested that alternative for reference
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

Checkpoint E-017, step 26,442. Release 0.5.1 (2026-11-07), built with the export gate verified
(fp32 median |Δ| 2.127e-4, worst 1.472e-3 over 80 draws; int8 top-1 logit agreement 100%).
Every number in this card comes from the committed evaluation artifacts in the source repository
(`runs/E-017/`, `runs/E-020/`), scored with a paired bootstrap through `stt.eval`.