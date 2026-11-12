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
  - name: moth-stt-0.8.1
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
            value: 20.16
            name: WER (beam+LM, normalized, percent)
---

<img src="logo-tile.png" alt="Moth" width="88" />

# Moth STT 0.8.1 — Italian + English code-switching, 19.7 MB, CPU-only

> **Published 2026-11-12** as a decode-only update. The acoustic graph is the published 0.8.0
> model; only the trigram LM and the version string changed.

A very small open **Italian + English code-switching** speech-to-text model: a Whisper-tiny
encoder initialised from `openai/whisper-tiny` (MIT), fine-tuned behind a **linear CTC head** over
a shared IT+EN SentencePiece BPE, 11.6 M parameters, shipped as an **11.9 MB int8 ONNX** graph with
a **5.4 MB trigram LM** and a 2.3 MB Silero VAD in the loop.

It runs **record-then-transcribe** in the browser on ONNX Runtime Web (WASM) and on CPU. This is
not a streaming model and not a general-purpose multilingual model: it recognises *natural Italian
speech with English terms embedded in it* — the way Italians talk about work, tech and media.

> **What changed in 0.8.1.** A decode-side change only: the trigram LM was rebuilt from a frozen
> training-side corpus with **all bigrams** (trigrams still count ≥2) and the term-containing
> sentences repeated ×3. The acoustic weights, the decoder settings and the runtime are unchanged.
> The pre-registered rare-term gate now **passes**: rare aligned ETA **28.4% → 46.9%** (23/81 →
> 38/81, **+18.5 [95% CI +9.6, +31.0]**), T-600 improves **24.71 → 24.08 WER** and **54.17 →
> 55.23 ETA** (both CIs exclude zero), T-S3 is a tie at **24.96 / 51.27**, FLEURS is a tie, and
> natural-term presence is **432/817** (gate ≥431). Package **19,706,289 bytes**.

## Results

All numbers are WER / ETA (English-term accuracy) after the same text normalization used to train
and evaluate every system here. Beam+LM decoding: beam 12, top-k 10, BPE trigram LM weight 0.55,
per-token bonus 2.7.

| Set | n | Moth 0.8.1 WER | ETA |
|---|---|---|---|
| **Mixed test T, all speakers** | 600 | **24.08** | **55.23** |
| **T, speaker S3 — never seen in training** | 200 | **24.96** | **51.27** |
| T, speaker S1 (training speaker) | 200 | 20.98 | 63.28 |
| T, speaker S2 (training speaker) | 200 | 26.32 | 51.56 |
| FLEURS-IT test | 697 | 20.16 | — |
| FLEURS-EN test | 525 | 49.81 | — |

**Read the speaker breakdown.** S1 and S2 contributed audio to training; **S3 never did**. S3 is
the honest speaker number and is a statistical tie with 0.8.0 (25.21 → 24.96 WER, 50.14 → 51.27
ETA). The aggregate T number is optimistic for the same reason.

Against the published 0.8.0 int8 model, with the same decoder:

| Set | 0.8.0 | **0.8.1** | Δ | 95% CI |
|---|---:|---:|---:|---|
| T-600 WER | 24.71 | **24.08** | **−0.63** | [−1.11, −0.20] |
| T-600 ETA | 54.17 | **55.23** | **+1.06** | [+0.09, +2.04] |
| T-S3 WER / ETA | 25.21 / 50.14 | 24.96 / 51.27 | ties | — |
| FLEURS-IT / EN | 20.25 / 49.81 | 20.16 / 49.81 | ties | — |

## Rare terms (the point of 0.8.1)

| Readout | 0.7.0 int8 | 0.8.0 int8 | **0.8.1 int8** |
|---|---:|---:|---:|
| Rare aligned ETA (81 occurrences) | 22.2% | 28.4% | **46.9%** |
| Rare presence (47 utterances) | 8/47 | 11/47 | **24/47** |
| Rare WER | 40.2% | 35.9% | **28.2%** |
| Natural-term presence | 441/817 | 436/817 | **432/817** |
| 16-term gold regression | 22/27 | 19/27 | 19/27 |
| T-600 aligned ETA | 565/1043 | 565/1043 | **576/1043** |

The pre-registered primary gate was rare aligned ETA ≥30%; 0.8.1 reaches **46.9%** with 13
occurrences to spare. The gain comes from the LM prior, not from new acoustic weights: the frozen
corpus contains the 50+50 training-side term sentences the 2026-10-30 LM never saw, and repeats
the 143 term-containing sentences ×3. This is build-time data weighting, not decode-time contextual
bias (E-027, rejected).

## Int8 cost

The acoustic graph `model.int8.onnx` is **byte-identical** to 0.8.0 (sha256 `7bbfad7e…`). The int8
cost measured for 0.8.0 therefore applies unchanged: T-600 and T-S3 are ties against PyTorch fp32,
and no quantization-aware training was used. The change is the 5.4 MB browser LM binary
(`3be1177b…`), which is a deterministic export of the frozen-corpus trigram table.

## Noise

The acoustic checkpoint is unchanged. Re-measured with the **same fp32 checkpoint, decoder,
100 S3 clips, pool and seed** as the 0.8.0 control, changing only the LM: clean 20.03 vs 20.37,
30 dB 21.22 vs 21.90, 20 dB 24.28 vs 25.47, 15 dB 27.67 vs 27.33, 10 dB 33.79 vs 34.63, 5 dB
40.41 vs 40.92. **Every condition is a statistical tie**; mean 5–30 dB **29.47 vs 30.05**. No
regression. This is a same-pool upper bound, one speaker, not the E-025 unseen-room curve. The
int8/browser path was not re-measured here; the graph is unchanged and acceptance is 4/4.

## How to use

The package is the whole runtime — no PyTorch, no Python:

```text
model.int8.onnx     11.9 MB   encoder + head + log-softmax, batch 1, dynamic time
lm_bpe3.bin          5.4 MB   BPE trigram LM, int16-quantized CSR (browser format)
silero_vad.onnx      2.3 MB   Silero VAD v5, unmodified, MIT
vocab.json                    pieces by CTC id, blank = 0, "▁" is the space marker
mel_filters.bin       64 KB   Whisper's 80×201 filterbank
config.json                   constants, decoder settings and per-file sha256
NOTICE                        Whisper MIT + Silero MIT + dataset credits
```

The input is **log-mel** `[1, 80, T]` at 16 kHz: STFT `n_fft=400`, `hop=160`, periodic Hann,
`center=True` with reflect padding, last frame dropped, Whisper's Slaney filterbank,
`log10(max(mel, 1e-10))`, per-clip floor `max(x, x.max()−8)`, then `(x+4)/4`. `config.json` has
every constant, and the repository ships the reference implementation in `stt/features.py` plus a
JavaScript port tested against it.

**Decoding.** CTC greedy, or prefix beam search with shallow fusion. `config.json` carries the
settings this release was measured with — `beam: 12`, `top_k: 10`, `lm_weight: 0.55`,
`token_bonus: 2.7`. The decoder was calibrated on dev read speech; `lm_weight`/`token_bonus` must
be tuned together, and do not apply them to greedy.

The real Chromium/WASM acceptance for this package passes **4/4** projects (isolated 1/4/auto
threads and non-isolated auto). Decoder-level parity between the JS and Python paths is exact on
identical logits; the single 3 s acceptance fixture differs from Python by two words with **both**
the 0.8.0 and the 0.8.1 LM, so the residual is the pre-existing browser acoustic path, not the new
LM.

## Training data

- **Natural speech (438 h)**: Common Voice IT 27 (CC0, 167.8 h after filtering), MLS Italian
  (CC-BY 4.0, 169.9 h), LibriSpeech English (CC-BY 4.0, 100 h).
- **Spliced IT+EN (110 h)**: Italian and English utterances concatenated at utterance level.
- **Recorded gold**: S1 + S2 read Italian sentences with embedded English terms, including the 50
  S1 and 50 S2 term addenda.
- **LM text**: the training transcripts plus the training-side R-S1/R-S2/pool scripts, never T/A,
  frozen with explicit evaluation exclusions. The 2026-10-30 LM predates the term addenda; 0.8.1's
  does not.

No test or dev set was ever trained on. S3 contributed no audio to any training run. T and S3 are
reused evaluation evidence, not fresh final tests.

## Limitations, stated plainly

1. **No independent natural mixed-dev.** T and S3 are reused evidence; S3 is the only held-out
   voice and shares the recording setup with S1. The rare gain is measured on T. A fresh
   independent natural mixed-dev and a new-voice rare readout remain the correct next instrument.
2. **Natural terms are close to the gate** (432/817 vs ≥431). The rare gain is bought partly with
   common-term robustness; the 16-term gold check stays 19/27.
3. **No punctuation, no casing, no formatting.** Output is normalized text by design.
4. **Numbers are spelled out** in Italian, following the training convention.
5. **Speaker coverage is three people.** S1 and S2 are training voices; S3 is the only held-out
   voice. Performance on voices unlike these will drop, and we have not measured by how much.
6. **English-only speech is weak** (49.8% WER). This is a code-switching model, not a bilingual one.
7. **Music can produce words.** The VAD stops silence and steady noise, not music.
8. **Single seed per configuration.** Intervals are utterance-level bootstrap intervals; training
   variance was not measured.
9. **Browser parity is one 3 s fixture**; the historical 200-clip browser↔Python comparison (E-022)
   was a tie, but 0.8.1 was not re-run on that sample.

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
  title  = {Moth: a 19.7 MB Italian-English code-switching speech-to-text model},
  author = {Semantique},
  year   = {2026},
  url    = {https://huggingface.co/semantique-team/moth-stt}
}
```
