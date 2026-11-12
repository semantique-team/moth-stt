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
  - name: moth-stt-0.8.0
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
            value: 20.25
            name: WER (beam+LM, normalized, percent)
---

<img src="logo-tile.png" alt="Moth" width="88" />

# Moth STT 0.8.0 — Italian + English code-switching, 19.4 MB, CPU-only

> **Live demo:** [semantique-team/moth-demo](https://huggingface.co/spaces/semantique-team/moth-demo) —
> runs entirely in your browser. The audio never leaves the tab: the same 19.4 MB package on this
> page is what the demo downloads and executes through ONNX Runtime Web.

A very small open **Italian + English code-switching** speech-to-text model: a Whisper-tiny
encoder initialised from `openai/whisper-tiny` (MIT), fine-tuned behind a **linear CTC head** over
a shared IT+EN SentencePiece BPE, 11.6 M parameters, shipped as an **11.9 MB int8 ONNX** graph with
a 5.1 MB trigram LM and a 2.3 MB Silero VAD in the loop.

It runs **record-then-transcribe** in the browser on ONNX Runtime Web (WASM) and on CPU. This is
not a streaming model and not a general-purpose multilingual model: it recognises *natural Italian
speech with English terms embedded in it* — the way Italians talk about work, tech and media.

> **What changed in 0.8.0.** Two changes: a 50-sentence term addendum recorded by a **second
> speaker** (0.7 h at ×20) inside a gentler fine-tune, and a decoder recalibration selected on dev.
> The goal was the product's hard part — **rare English terms** in Italian speech. Rare presence
> rises **8/47 → 11/47** and rare aligned ETA **22.2% → 28.4%** (+6.2 points, CI excludes zero);
> T-S3 improves **27.00 → 25.21 WER**. The mixed set is a statistical tie (**24.71 / 54.17** vs
> 25.14 / 54.65), and natural-term presence is **436/817 vs 441**. The release is published with a
> documented exception: the pre-registered primary gate was rare aligned ETA ≥30%, and 28.4%
> misses it by two aligned occurrences. The fp32 checkpoint reaches 30.9%; the int8 artifact does
> not. Read the caveats before deploying.

## Results

All numbers are WER / ETA (English-term accuracy) after the same text normalization used to train
and evaluate every system here. Beam+LM decoding: beam 12, top-k 10, BPE trigram LM weight 0.55,
per-token bonus 2.7.

| Set | n | Moth 0.8.0 WER | ETA |
|---|---|---|---|
| **Mixed test T, all speakers** | 600 | **24.71** | **54.17** |
| **T, speaker S3 — never seen in training** | 200 | **25.21** | **50.14** |
| T, speaker S1 (training speaker) | 200 | 22.27 | 61.79 |
| T, speaker S2 (training speaker) | 200 | 26.67 | 50.99 |
| FLEURS-IT test | 697 | 20.25 | — |
| FLEURS-EN test | 525 | 49.81 | — |

**Read the speaker breakdown.** S1 and S2 contributed audio to training; **S3 never did**. S3 is
the honest speaker number, and 0.8.0 improves it from 27.00 to **25.21 WER** and from 48.45 to
**50.14 ETA**. The aggregate T number is optimistic for the same reason.

## Rare terms (the point of 0.8.0)

| Readout | 0.7.0 int8 | **0.8.0 int8** |
|---|---:|---:|
| Rare presence (47 utterances) | 8/47 | **11/47** |
| Rare aligned ETA (81 occurrences) | 22.2% | **28.4%** |
| Rare WER | 40.2% | **35.9%** |
| Natural-term presence | 441/817 | 436/817 |
| 16-term gold regression | 22/27 | 19/27 |

The rare gain is real and statistically significant (+6.2 points, 95% CI [+1.1, +11.8] against
the published decoder; +7.4 against 0.7.0 with the same decoder). The costs are also real: five
fewer natural terms (still above the 431 gate) and one fewer gold term. The pre-registered rare
gate was 30%; the int8 artifact is two aligned occurrences short, and this is why the release
carries an explicit exception rather than a “gate passed” claim.

## Int8 cost

The shipped graph is dynamically quantized int8. Measured end to end with the same decoder:

| Set | PyTorch fp32 | int8 | Δ |
|---|---|---|---|
| Mixed test T (600) | 24.66 / 54.65 | 24.71 / 54.17 | tie |
| T, held-out S3 (200) | 25.55 / 49.01 | 25.21 / 50.14 | tie |

The int8 artifact does **not** give back the acoustic gains on T-S3; on T-600 the difference is
inside the noise floor. This is a better quantization outcome than the earlier E-028 export, which
used the 0.7.0 decoder and was 25.49 / 53.50 on T-600. No quantization-aware training was used; the
improvement comes from the decoder recalibration, not from new weights.

## Noise

The 0.7.0 noise work is not undone by the 0.8.0 fine-tune. Re-measured on the same E-024 pool
harness with the **same decoder** for both **fp32 checkpoints** (Python, 100 held-out-speaker S3
clips, 6 conditions): 0.8.0 is a tie at 10–30 dB (mean WER **27.33 vs 27.76**), a tie at 15 dB
(27.33 both), and **significantly better clean** (20.37 vs 23.09, **−2.72 [−4.42, −1.02]**). No
condition is a significant regression. This is a same-pool upper bound, not the E-025 unseen-room
curve; see `runs/E-028/noise-0.8.0/RESULT.md`.

## How to use

The package is the whole runtime — no PyTorch, no Python:

```text
model.int8.onnx     11.9 MB   encoder + head + log-softmax, batch 1, dynamic time
lm_bpe3.bin          5.1 MB   BPE trigram LM, int16-quantized CSR (browser format)
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
`token_bonus: 2.7`. The decoder was recalibrated on dev read speech; the `token_bonus` counteracts
the per-token LM cost, and `lm_weight`/`token_bonus` must be tuned together. Do not apply them to
greedy, which is length-neutral.

The real Chromium/WASM acceptance for this package passes **4/4** projects (isolated 1/4/auto
threads and non-isolated auto; see the repository record for the per-project JSON and timing).

## Training data

- **Natural speech (438 h)**: Common Voice IT 27 (CC0, 167.8 h after filtering), MLS Italian
  (CC-BY 4.0, 169.9 h), LibriSpeech English (CC-BY 4.0, 100 h).
- **Spliced IT+EN (110 h)**: Italian and English utterances concatenated at utterance level.
- **Recorded gold**: S1 + S2 read 1,313 Italian sentences with embedded English terms. The 0.8.0
  run adds the 50-sentence S2 addendum at ×20 (0.7 h) to the E-024 mixture, and lowers the LRs
  (head 1e-4 / encoder 1e-5).
- A **BPE trigram LM** built on the training text and augmented with the training-side mixed
  scripts (never with the test scripts).

No test or dev set was ever trained on. S3 contributed no audio to any training run. The S2
addendum is training audio; T and S3 are reused evaluation evidence, not fresh final tests.

## Limitations, stated plainly

1. **The primary rare-term gate is missed.** Rare aligned ETA is 28.4%, two aligned occurrences
   below the pre-registered 30%. The release is an explicit product decision, not a gate pass.
2. **Natural terms regress slightly** (441 → 436, still above the 431 gate) and the 16-term gold
   check drops 22/27 → 19/27. The rare gain is bought partly with common-term robustness.
3. **No punctuation, no casing, no formatting.** Output is normalized text by design.
4. **Numbers are spelled out** in Italian, following the training convention.
5. **Speaker coverage is three people.** S1 and S2 are training voices; S3 is the only held-out
   voice, and it has been observed across many experiments. No independent fourth speaker was
   recorded. Performance on voices unlike these will drop, and we have not measured by how much.
6. **English-only speech is weak** (49.8% WER). This is a code-switching model, not a bilingual one.
7. **Music can produce words.** The VAD stops silence and steady noise, not music.
8. **Single seed per configuration.** Intervals are utterance-level bootstrap intervals; training
   variance was not measured.
9. **The decoder was calibrated on dev read speech**, not on a natural mixed-dev. The rare-term
   readout is the reused T set; a fresh independent mixed-dev remains the correct future
   instrument.

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
