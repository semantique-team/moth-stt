# 08 — Risks, open questions, follow-ups

## 1. Risk register

| # | Risk | Impact | Mitigation | How we find out |
|---|---|---|---|---|
| R1 | The Whisper encoder degrades on unpadded (truncated) input | Lower accuracy | Fine-tune on truncated clips (CTC needs no decoder compatibility); fall back to padded 30 s windows | H1 padded vs truncated |
| R2 | Whisper-tiny encoder is too weak for Italian quality | Misses the Whisper-tiny/Whistle bars | Extra head blocks; BPE size; more epochs; **Whisper-base fallback** (≈ 20.5–23.5 MB, needs relaxing the ceiling) | H2 gate |
| R3 | CTC conditional independence → spelling and word-boundary errors on rare English words | Low ETA | Hotwords; mixed data; tokenizer term list; follow-up n-gram LM head | ETA per category |
| R4 | Mixed data is scarce; overfits to the two training speakers | Good on S1/S2, poor on S3 | S3 held out; splicing/synthesis arms; early stopping; report seen vs unseen speaker | H3 per-speaker table |
| R5 | Acronyms ("KPI") are intrinsically hard (letter names differ across languages) | Very low acronym accuracy | Hotwords; reported separately; may be excluded from the headline bar | Per-category WER |
| R6 | int8 quantization loses accuracy | Quality drop vs fp32 | Static calibration; fp32 LayerNorm path; fp16 weights (≈ 16–17 MB) | Export gate |
| R7 | Browser speed below 10× real time | Misses the bar | Early random-weight export (H4); threads/SIMD; stride-2 pooling to 25 Hz; WebGPU later | M3 smoke test |
| R8 | Train/serve skew (JS mel, browser resampler) | Browser accuracy differs from Python | Mel parity tests; e2e on recorded clips through the browser path | Parity + Playwright |
| R9 | MPS quirks: non-deterministic CTC backward, memory growth in long runs, throttling, silent numeric issues | Lost runs, irreproducibility | Checkpoint/resume every 20 min; parity guard at start; pinned torch; cloud fallback | Training logs |
| R10 | Licensing slips (dataset terms, Whisper MIT notice, LLM-generated text terms, speaker consent) | Cannot publish | `NOTICE`, data card, per-speaker consent, provider terms checked before release | M8 checklist |
| R11 | Test set too small / noisy | Wrong conclusions from 1–2 point gaps | Bootstrap CIs; "tie" rule; optional anchor set; grow the set if gates are ambiguous | Wide CIs |
| R12 | Budget creep | Over 10 USD | Mac-first; cost ledger in every entry; stop at 8 USD | Ledger |
| R13 | Cannot match Whistle | Stretch missed | Bar is Whisper-tiny; Whistle stays a baseline; report honestly | H0 vs final |
| R14 | CTC emits text on music/non-speech | False transcripts in apps | Silero VAD guard (D25): 0 non-music words on the 240 noise/silence clips; music reported separately, 89% gated | Noise set (music arm) |
| R15 | Italian dominance: English words come out Italianized ("miting") | Core use case fails | English term list in tokenizer text; EN data share; mixed data; ETA gate | ETA, per-term errors |
| R16 | Facts in `09-research-notes.md` that came from third-party summaries prove wrong | Mis-set expectations | Each claim marked verified/unverified; H0 measures what matters | H0 |

## 2. Open questions (none block the implementation plan)

1. ~~**Model name** and the new repository's name/org.~~ **Decided (2026-10-19):** model **moth**, repository **`semantique-team/moth-stt`** on GitHub (private for now, like `ant-4b`).
2. **Who records** (S2, S3) and under what consent; **dataset release license** (CC-BY 4.0 proposed) or private.
3. **Source of English-accent audio** (Common Voice EN Italian-accent slice — existence and size unverified; else LibriSpeech).
4. **BPE size** (512/1024/2048): ablate after H2 or fix at 1024?
5. **LLM-generated sentences**: provider terms for publishing them; or hand-written/hybrid.
6. **Target browsers**: Chrome/Edge only, or Safari too (needs COEP `require-corp`, SIMD support)?
7. **Phone/low-end device** speed target, beyond the M5 bar.
8. **Hotword UX** in the demo (free text list vs file).
9. **Fallback acceptance:** if H2 forces the Whisper-base variant, is a ≈ 25 MB ceiling acceptable?
10. **MUSAN (or other noise set) license** — verify before use.

## 3. Follow-up backlog (explicitly out of v0.1)

| Item | Why later |
|---|---|
| Streaming (re-decode growing buffer; or a streaming student distilled from this model) | Needed for live dictation; not for v0.1 (the Silero VAD guard is already shipped, D25) |
| Punctuation/casing/error-correction post-processor (FUTO PCEC-style, ≈ 5 M params) | Product polish |
| Detachable n-gram language head trained from text only (Metro-ASR idea) | Cheap accuracy for rare English words |
| Pseudo-labeling Italian audio with Parakeet v3 / NVIDIA Granary pipeline | Data scale-up for v0.2 |
| TTS-synthesized code-switch audio (shelved TTS plan) | Only if it helps the mixed set |
| Word-level splicing | Only if utterance splicing is not enough |
| sherpa-onnx integration; native edge packaging | Distribution |
| WebGPU backend | Only if WASM speed is short |
| Quantization-aware fine-tuning, 4-bit weights | Size/accuracy trade |
| More languages in the family; Whisper-base variant as a larger sibling | Family roadmap |

## 4. Stop / pivot criteria

- **Pivot to Whisper-base encoder** if H2 cannot beat Whisper-tiny on Italian FLEURS/CV dev after the escalation list.
- **Revisit approach C (distillation)** if both tiny and base encoders fail the mixed-set bar.
- **Revisit the Whistle fine-tune** only if the author reopens decision D5.
- **Stop and re-plan** if browser RTFx stays under 5× after pooling, or if int8 and fp16 both fail the export gate.
- **Pause spending** at 8 USD.
