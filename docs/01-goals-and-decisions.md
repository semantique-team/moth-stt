# 01 — Goals, constraints and decisions

## 1. Problem

Small open speech-to-text models handle *one language per clip*. Real Italian speech, especially about work and
technology, constantly drops English words into Italian sentences:

- "ho spostato il **meeting** a domani"
- "puoi **forwardare** la mail al **team**?"
- "il **KPI** del **sprint** è sotto target"
- "abbiamo **deployato** la **release** ieri sera"

Whisper-style models pick a language token at the start of the clip, and decoders of this size tend to
misspell or Italianize the English words, repeat themselves at the switch point, or invent text on silence.
The goal is a model that does exactly this job well, small enough to run in a browser tab.

## 2. Goals

| Goal | Meaning | Measured by (see `05-evaluation.md`) |
|---|---|---|
| **Precise** | Correct Italian, correct English words inside Italian sentences, no invented text. | WER on the mixed test set, English-term accuracy (ETA), words emitted on 1 h of silence/noise. |
| **Fast** | Much faster than real time on an ordinary laptop CPU, in the browser. | Browser RTFx on the author's M5 (bar: ≥ 10×). |
| **Small** | Browser-deployable. | Download size ≤ ≈ 20 MB (all runtime files). |
| **Open** | Weights, code, recipe and training data are open and reproducible. | Apache-2.0 code, `NOTICE`, CC0/CC-BY data or own recordings. |
| **Own** | A model the author trained, not a re-skinned third-party model. | See §6. |

## 3. Hard constraints

1. **Own model**, not a fine-tune of Cactus Whistle (decision of the author, 2026-10-18).
2. **≤ ≈ 20 MB** browser download; runs in the browser and on edge CPUs.
3. **Compute budget ≤ 10 USD** in total; free or cheap Colab/Kaggle GPUs; the author's M5 MacBook Pro (16 GB) is preferred.
4. **As little training as possible.**
5. **Italian + English code-switching** is what the model must do *best*.
6. Open licenses: the model belongs to a family of Apache-2.0 models, so training data must be CC0/CC-BY or
   the author's own recordings, and any GPL-licensed component (e.g. espeak-ng/Piper) is avoided.

## 4. What exists, and why it is not enough

Details and links in `09-research-notes.md`.

| Candidate | Status for this project |
|---|---|
| **Cactus Whistle** (16.9 MB, Apache-2.0, 7 languages incl. IT+EN, runs in a browser tab) | The strongest small model for the brief, but it is a third-party model with one language token per clip. A documented LoRA fine-tune exists; the author **rejected it as the deliverable**. Kept as the main **baseline** and architectural reference. |
| **Whisper tiny / base** (39 M / 74 M, MIT) | Weak Italian at tiny size; hallucinates on silence; static language prefix; encoder always processes 30 s of padded audio; ≥ 39 MB in the browser even quantized. Baseline, and the source of the pretrained encoder. |
| **Moonshine** (27 M tiny, MIT) | English-only; no Italian model exists. Its recipe (public data + pseudo-labels + synthetic data, 15k–94k h per language, ≈ 8×H100 for 1–3 days) costs roughly 250–2,300 USD by our arithmetic — far above the budget. |
| **Parakeet TDT 0.6B v3** (CC-BY-4.0) | Best accuracy (Italian FLEURS WER ≈ 3%, avg ≈ 5.6% across Italian sets) but ≈ 640 MB int8. Offline ceiling and future teacher; not deployable in a tab. |
| **FUTO asr4all** (28/56/97 M) | English-only; license has ethical-use limits. Architecture and distillation reference. |
| **Duyu-small** (Turkish+English, 19.3 M active / 49 M stored) | Precedent that a one-person CTC model can do code-switching; ships at 56 MB (int8) / 34 MB (4-bit); shows the bilingual cost (English FLEURS WER 22.9%). |
| **Metro-ASR Small** (Arabic+English, 61.6 M, MIT) | Another CTC code-switching precedent, with a detachable n-gram language head trained on text only (a possible follow-up here). |
| Stock transformers.js / sherpa-onnx models | No Italian+English browser model below ≈ 27 MB other than Whistle. |

## 5. Approaches considered

| | Approach | What is the author's | Size (int8) | Compute (estimate) | Risk |
|---|---|---|---|---|---|
| A | Tiny CTC model (10–15 M, Zipformer/Conformer) trained **from scratch** on ≈ 700–1,500 h of open IT/EN audio | All weights | ≈ 8–15 MB | 20–60 T4-hours | **High**: likely clearly worse than Whisper-tiny on real speech; English words barely learned. |
| **B** | **Pretrained Whisper-tiny encoder + own tokenizer, CTC head, data, training** | Architecture, vocabulary, head, training recipe, data. Encoder weights are inherited (MIT). | ≈ 8–10 MB | **Measured feasible on the M5** (≈ 5–10 h) | **Medium** — chosen. |
| C | From-scratch student **distilled from Parakeet v3** | All weights, borrowed knowledge | 10–15 MB | 40+ T4-h and much more audio | Medium-high; most engineering. |

Also considered and **rejected**:

- *LoRA fine-tune of Whistle* — works, cheap, but not an own model (D5). **Revisited 2026-11-07 as
  D28**: it runs as an *exploratory side artifact* (E-019, plan
  `docs/plans/2026-11-07-whistle-lora-probe.md`) to learn what a decoder-based code-switch model
  does with our gold; the Moth deliverable stays an own CTC model.
- *Fine-tuning Whisper tiny/base end-to-end* — too big (39–74 M) and weak at tiny size; keeps the decoder problems.
- *Moonshine-style from scratch* — budget.
- *TTS* — an earlier proposal (a small Italian+English code-switching TTS); **shelved**. It may return as a source of
  synthetic code-switch training audio (`03-data.md` §9).

### Why B

- **Acoustics are what we cannot afford to learn.** A tiny model needs on the order of 10⁴–10⁵ hours of speech
  when trained from scratch; the Whisper-tiny encoder already carries 680k hours of multilingual training.
- **The decoder is where the problems are.** A CTC head has no language prefix token, cannot loop at a
  language switch, and emits blank on silence. Italian and English share the Latin script, so one shared BPE
  covers both.
- **Precedents (hypothesis-level evidence).** A frozen Whisper-base encoder with a CTC head beat WavLM-base and
  wav2vec2-base as a Khmer backbone (validation CTC loss only); FUTO's ACFT shows the Whisper encoder tolerates
  short audio windows; Whisper-style CTC variants exist on the Hub. None of this is proof for our task — H1 and
  H2 are the test.
- **It is cheap to find out.** H1 (frozen encoder + linear head) takes under an hour on the Mac.

## 6. What "own model" means here

The author accepted this definition on 2026-10-18:

- **Own:** the architecture composition (encoder + CTC head, no decoder), the tokenizer, the head, the training
  recipe, the data selection and recordings, the evaluation suite, the export and the browser runtime.
- **Inherited:** the initial *encoder weights* from `openai/whisper-tiny` (MIT license). They are fine-tuned during
  H2/H3, so the shipped encoder is no longer Whisper's.
- **Consequence:** the MIT copyright and permission notice must travel with the weights (`NOTICE`); the model card
  states the derivation plainly.

## 7. Success tiers

| Tier | Bar |
|---|---|
| **Ship v0.1 (minimum)** | ≤ 20 MB total runtime download. Beats Whisper-tiny on mixed-set WER **and** ETA under the same `normalize()`. Zero words emitted on the 1-hour silence/noise set (music reported separately). ≥ 10× real time in browser WASM on the author's M5. int8 within the noise interval of fp32. |
| **Stretch** | Within the noise interval of Whistle on the mixed set. |

Absolute WER numbers are deliberately not promised. A rough, **unmeasured** expectation for Italian read speech is
WER in the teens to low twenties; H0 replaces this guess with measurements.

## 8. Non-goals for v0.1

Streaming; voice-activity detection; punctuation and casing; languages other than Italian and English;
pseudo-labeling with Parakeet; WebGPU; the Whistle fine-tune; TTS; speaker diarization; word timestamps;
training on proprietary data. Each is listed in the follow-up backlog (`08-risks-and-open-questions.md`).

## 9. Budget

| Resource | Cost | Role |
|---|---|---|
| **M5 MacBook Pro, 16 GB** | 0 USD | Primary training and evaluation machine (measured feasible, `04-training.md`). |
| Kaggle notebooks | 0 USD — [30 GPU hours/week](https://www.kaggle.com/docs/efficient-gpu-usage); T4×2 or P100; sessions up to 12 h; phone verification required | Fallback for sweeps or the Whisper-base variant. |
| Colab Pro | ≈ 10 USD for 100 compute units; a T4 burns ≈ 1.1–1.6 units/h → roughly 60–85 T4-hours; A100 ≈ 5–10 units/h → 10–18 h (third-party measurements; units fluctuate and expire after 90 days) | Only if the Mac runs prove too slow. |
| Moonshine-style rental | ≈ 250–2,300 USD (8×H100, 1–3 days) | **Out of budget.** |

Any paid spend is written into the `EXPERIMENTS.md` entry that incurred it, with a running ledger; stop at 8 USD.
