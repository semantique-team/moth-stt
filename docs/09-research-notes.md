# 09 — Research notes (evidence behind the decisions)

Research date: 2026-10-18 (web search + page fetches during the brainstorming session).

**Trust levels used below**

- **[V]** read on the cited page during this session.
- **[E]** from earlier searches in the same session; the figure is reported as found but the source page was not re-opened when this file was written — re-verify before quoting externally.
- **[M]** measured by us (the Mac benchmark).
- **[I]** our inference or arithmetic, not a finding.

Anything not marked is a pointer, not evidence.

## 1. Speech-to-text landscape

| Model | Size / params | License | Languages | What matters here | Trust |
|---|---|---|---|---|---|
| **Cactus Whistle** — [blog](https://cactuscompute.com/blog/whistle), [model card](https://huggingface.co/Cactus-Compute/whistle), engine [`cactus-compute/needle`](https://github.com/cactus-compute/needle) | one 16.9 MB file; 2–4-bit "Cactus Quants"; 8-layer encoder + 8-layer laddered decoder (width 512) | Apache-2.0 | en, de, fr, es, it, nl, pl | One language token per clip (detected unless named); silence/steady noise → empty transcript, but **music can still come out as text**; keyword biasing; word timestamps; 30 s per pass, windowed beyond; browser demo in a tab; front-end: 25 ms window / 10 ms hop, 80 log-mel bins band-limited 250–3500 Hz, conv stem to 80 ms frames. A LoRA fine-tune path is documented (`needle whistle finetune …`, JSONL `{audio,text,language}`) — **an earlier statement in the session that it was undocumented was wrong and was corrected.** | [V] |
| **Moonshine** ("Flavors of Moonshine", [arXiv 2509.02523](https://arxiv.org/abs/2509.02523)) | tiny 27 M | MIT | monolingual flavors; **no Italian** | Specialized monolingual tiny models beat multilingual and Whisper Small (9× larger); recipe = public data + WhisperX pseudo-labels on podcasts/radio + TTS data; 15k–94k h per language; 8×H100 for 1–3 days; RoPE, compute proportional to input length (Whisper always pads to 30 s). Browser sizes: Moonshine Tiny 26.8 MB, Whisper tiny 39.0 MB / base 73.3 MB / small 237.5 MB (uint8, transformers.js, third-party site). | [E] |
| **Parakeet TDT 0.6B v3** — [model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 0.6 B; ≈ 640 MB int8 | CC-BY-4.0 | 25 EU languages incl. it/en | FastConformer + TDT transducer (no language prefix token); Italian WER FLEURS 2.96, MLS 10.12, Common Voice 3.65 (avg ≈ 5.6%, independent benchmark); ≈ 0.04 RTF on 4 CPU threads in a tiny 8-clip test; hotwords via sherpa-onnx and NeMo phrase boosting without retraining. Offline ceiling / future teacher. | [E] |
| **FUTO asr4all** — [blog](https://futo.tech/blog/asr4all), [asr4all-s](https://huggingface.co/futo-org/asr4all-s) | 28 M / 56 M / 97 M (incl. 5.3 M PCEC) | custom, commercial use with ethical-use limits | English only | Streaming CTC over BPE-256 + blank, 25 Hz mel, DFSMN + sliding-window attention; distilled from a Granite-Speech CTC teacher with logit + hidden-state KD; PCEC = tiny model for punctuation, capitalization, error correction; ONNX/ExecuTorch exports; single-thread RTFx tables. | [V] |
| **FUTO Whisper ACFT** — [repo](https://github.com/futo-org/whisper-acft) | Whisper tiny/base/small | MIT (repo), Apache-2.0 (HF checkpoints) | multilingual and .en | Fine-tunes Whisper (self-distillation, L2 on hidden states) so a **shortened encoder context** (`audio_ctx`) does not make the *decoder* loop; encoder conv stack is length-agnostic and positions are fixed sinusoids, so slicing positions gives shorter windows. LiteRT conversions: [`litert-community/whisper-acft`](https://huggingface.co/litert-community/whisper-acft). | [V] |
| **Duyu-small** — [card](https://huggingface.co/KaanAydinli/Duyu-small) | 19.3 M active / 49.3 M stored; int8 56 MB, 4-bit 34 MB | see card | Turkish + English, code-switching, cased and punctuated | 9 Conformer blocks as 3 shared passes + product-key memory + CTC head over 2048 TR+EN pieces + 3-layer decoder; FLEURS tr 7.87%, **FLEURS en 22.88%** (the bilingual cost) vs Whisper-base 10.81% and Whistle 14.37% (language forced); int8 matches fp32 on their dev clips. Precedent for a small one-person code-switching CTC model. | [V] |
| **Metro-ASR Small** — [card](https://huggingface.co/mohammedaly22/Metro-ASR-Small) | 61.6 M | MIT | Egyptian Arabic + English code-switching | Non-autoregressive CTC Conformer + a **detachable n-gram LM head** trained from text only; 40–55× real time on a laptop CPU; code-switch WER 32.7%. | [V] |
| **Whisper tiny/base/small** — [repo](https://github.com/openai/whisper), [tiny](https://huggingface.co/openai/whisper-tiny) | 39 M / 74 M / 244 M | MIT | multilingual | Weak Italian at tiny size (a fine-tuned `whisper-tiny-it` reports 27.1% WER on Common Voice 25 after ≈ 2 h on a GB10; whisper-small-it fine-tune FLEURS 5.8% / CV 10.6%); hallucinates on silence; static language prefix; `litus-ai/whisper-small-ita` (244 M) emits `[L-EN]` code-switch tags. | [E] |
| **Distil-Whisper** ([arXiv 2311.00430](https://arxiv.org/abs/2311.00430)) | — | Apache-2.0 code | English | Pseudo-label + WER-filter vs ground truth + KD → 5.8× faster, 51% fewer params, within 1% WER; decoder depth matters (≥ 4 layers; 2 layers ≈ 5% worse). Informs approach C and the "drop the decoder" trade-off. | [E] |

## 2. Whisper encoder as a backbone for CTC

- A frozen **Whisper-base** encoder + CTC head beat frozen WavLM-base and wav2vec2-base as a Khmer backbone (validation CTC loss 0.566 vs 0.603 vs 0.784); loss only, no WER, one small project — [page](https://huggingface.co/Panhapich/pre-trained_whisper_wavLM). [V]
- A `whisper-tiny.en` CTC fine-tune exists on the Hub ([maliced/whisper-tiny-en-ctc](https://huggingface.co/maliced/whisper-tiny-en-ctc), L2-Arctic phonemes). [V]
- ACFT (above) shows shorter encoder windows work after fine-tuning. [V]
- Whisper-tiny encoder shape: width 384, 4 blocks, 6 heads, FFN 1536, ≈ 7.63 M parameters. [I] (arithmetic in `02` §4.)
- **Conclusion for this project:** supportive but weak evidence; H1/H2 are the real tests.

## 3. Code-switching research (Italian–English specifically: none found)

- No open Italian–English code-switching speech corpus was found. **CS-FLEURS** (X–English) is CC-BY-NC → evaluation-only at best; Italian coverage unverified. [E]
- Whisper code-switch adaptation saturates with ≈ 1–10 h; Whisper-small LoRA + switching tokenizer + GELU adapter → 9.4% MER on 10.6 h of Mandarin–English (ASCEND). [E]
- LoRA removes insertion/repetition loops at language switches (Hindi–English); a joint language-ID loss helped by 14–36 points in one study; TTS-synthesized code-switch audio gave up to ≈ 5 points WER (Hindi–English). [E]
- Italian and English share the Latin script → one BPE vocabulary is natural; transducers/CTC without a language prefix token may suit mixed speech. [I]

## 4. Training-free biasing

- sherpa-onnx hotwords (transducers, `modified_beam_search`, merged Feb 2026), NeMo GPU phrase boosting (CTC/RNNT/TDT/AED), Whistle's keyword list (Aho-Corasick). [E]
- Our design uses a CTC prefix beam search with a hotword trie (`02` §6). [I]

## 5. Data sources and licenses

| Dataset | License | Facts | Trust |
|---|---|---|---|
| [Common Voice IT 27.0](https://commonvoice.mozilla.org/en/datasets) | CC0 | 430 h total, 364 h validated | [E] |
| MLS Italian (OpenSLR 94) | CC-BY 4.0 | ≈ 214 h cleaned | [E] |
| VoxPopuli IT | see source | ≈ 53 h cleaned (not used in v0.1) | [E] |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | CC-BY 4.0 | evaluation only in this project | [E] |
| [LibriSpeech](https://www.openslr.org/12) | CC-BY 4.0 | `train-clean-100` ≈ 100 h | [I] from general knowledge |
| [NVIDIA Granary](https://huggingface.co/datasets/nvidia/Granary) | card lists CC-BY-3.0 | ≈ 643k h ASR over 25 languages incl. it/en; open pseudo-labeling pipeline (NeMo-speech-data-processor, hallucination filtering, ≥ 16 GB GPU) — backlog | [E] |
| NeMo ASRSet Italian | — | ≈ 487 h | [E] |

A tiny model trained from scratch needs on the order of 10⁴–10⁵ h (Moonshine paper citing Whisper), which is why this project
starts from a pretrained encoder. [E]

## 6. Browser and edge deployment

- ONNX Runtime Web with WASM multithreading needs cross-origin isolation (COOP/COEP); Safari needs `COEP: require-corp`;
  uint8/int8 quantized models are preferred on WASM; sherpa-onnx has WASM streaming demos (zipformer; en, zh/en, ja, …). [E]
- `cactus-needle` on npm runs Whistle in the browser (community-maintained, WASM CPU workers, HTTPS/localhost). [E]
- FUTO publishes single-thread RTFx tables for CPUs and phones (e.g. asr4all-s int8 on a Pixel 10 and i9-12900K). [V]

## 7. Compute

| Fact | Source | Trust |
|---|---|---|
| Kaggle: free GPU quota 30 h/week "or sometimes higher" ([docs](https://www.kaggle.com/docs/efficient-gpu-usage)); T4×2 or P100, sessions up to 12 h, 20 GB working disk, phone verification (third-party summaries: [lanmao.ai](https://lanmao.ai/o/kaggle-notebooks-free-gpu/)) | docs / third party | [V] |
| Colab Pro: 100 compute units per month, units expire after 90 days ([signup page](https://colab.research.google.com/signup)); price ≈ 9.99 USD per 100 units and a T4 burning ≈ 1.19 units/h in 2026 ([estimate](https://gpuperhour.com/blog/google-colab-alternatives)); 2024 measurements: T4 1.58, L4 3.0, A100 10.59 units/h ([varlog](https://varlog.info/colab-credit-consumption-comparison-2024/)) | docs / third party | [V] |
| GPU rental (Oct 2026): RTX 4090 ≈ 0.3–0.9 USD/h, A100 ≈ 0.3–1.8, H100 ≈ 1.3–4 | web search | [E] |
| PyTorch MPS `ctc_loss`: forward merged ([#187716](https://github.com/pytorch/pytorch/pull/187716)), backward PR [#188187](https://github.com/pytorch/pytorch/pull/188187) open at last check; **CTC forward+backward ran on MPS in torch 2.14.1 in our benchmark** | PRs / benchmark | [V], [M] |
| M5 GPU has Neural Accelerators in each core ([Apple tech talk](https://developer.apple.com/videos/play/tech-talks/111432/)); PyTorch MPS reaches only a fraction of NVIDIA throughput on LLM training ([arXiv 2501.14925](https://arxiv.org/pdf/2501.14925)) — hence the need to measure | Apple / paper | [V] |
| Our M5 benchmark | `04-training.md` §1 | [M] |

## 8. TTS research (shelved, kept for the family roadmap)

Researched before the author clarified they meant speech-to-text. No code or plan depends on it.

- Piper (16–25 M, GPL-3.0; espeak-ng GPL), Kokoro 82 M, Pocket TTS 100 M (MIT, Italian voices, training code released Aug 2026), Supertonic 3 99 M, Inflect-Nano 4 M / Vaniq-Edge 9 M (English-only). [E]
- Gap: no < 30 M Italian+English code-switching TTS. Recommended direction: a VITS-family model with its own permissive bilingual phoneme front-end; avoid GPL components in an Apache-2.0 family. [E] / [I]
- Possible later role: synthesizing code-switch training audio for the STT model (`03` §10).

## 9. Corrections made during the discussion

1. **Whistle fine-tuning is documented** (LoRA via `needle whistle finetune`); an earlier statement that it was not was wrong. It was then rejected on ownership grounds, not feasibility.
2. **MPS `ctc_loss`**: earlier notes said it was unimplemented through 2025; PRs landed in 2026 and the benchmark shows forward+backward working on 2.14.1.
3. **Whisper-base fallback size**: earlier "about 20 MB int8"; arithmetic gives ≈ 20.5–23.5 MB, i.e. at or slightly over the ceiling (`02` §7).
