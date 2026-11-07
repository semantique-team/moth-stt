# 06 — Export and browser runtime

## 1. Scope of v0.1

- **Record-then-transcribe**: capture audio, then transcribe it. Clips ≤ 30 s run in one pass; longer audio is split.
- **No streaming, no WebGPU, no punctuation.** (Backlog: pseudo-streaming by re-decoding a growing buffer,
  WebGPU.) The **no-speech guard** is **Silero VAD v5** (MIT, 2.33 MB) on the 16 kHz PCM **before** the
  features: a clip is decoded only when more than **10%** of its 32 ms windows are speech
  (`stt/vad.py`, `web/vad.js`, D25 and the E-011 addendum). It gates 100% of digital silence and 100% of the
  MUSAN noise clips, which the earlier mel-energy guard could not; that guard survives only as
  `--no-speech-guard` for reproducing the E-011/E-016 artifacts. Music is reported, not gated.
- The same ONNX graph runs on native `onnxruntime` for edge devices; a sherpa-onnx integration is a follow-up.

## 2. Pipeline in the browser

```mermaid
flowchart TD
    A["Microphone or file"] --> B["AudioWorklet PCM capture / decodeAudioData"]
    B --> C["Resample to 16 kHz mono (OfflineAudioContext)"]
    C --> D["Chunk at silences if longer than 30 s"]
    D --> V{"Silero VAD: speech fraction > 0.10?"}
    V -->|no| Z["Empty transcript, guarded: true"]
    V -->|yes| E["JS log-mel (400/160/80, Whisper constants)"]
    E --> F["onnxruntime-web WASM: int8 encoder + head"]
    F --> G["CTC decode in JS: greedy or hotword beam"]
    G --> H["Join chunks, show text + timings"]
```

| Stage | Specification |
|---|---|
| Capture | `getUserMedia` (mono) → `AudioWorklet` collects float PCM at the context's native rate; or a dropped file via `decodeAudioData`. |
| Resample | `OfflineAudioContext(1, n·16000/sr, 16000)` to 16 kHz mono. Any difference from the training resampler is covered by an end-to-end test on recorded clips. |
| Chunking | If longer than 30 s: frame energy in 25 ms windows with an adaptive threshold; cut at the quietest point within the last 5 s before 30 s; minimum chunk 1 s; transcripts joined with a space. |
| No-speech guard | Silero VAD v5 (`silero_vad.onnx`, MIT) on the 16 kHz PCM: 512-sample windows with 64 samples of context (576-sample input), `state [2,1,128]`, `sr` int64 scalar; a window is speech at P > 0.5, a clip is decoded when the speech-window fraction is > 0.10 (`stt/vad.py`, `web/vad.js`). |
| Features | Direct, exact implementation of `02` §2: reflect-pad 200, Hann (periodic) window, **true 400-point real DFT** via precomputed 201×400 cos/sin matrices (≈ 16 M MACs per audio-second — negligible; a zero-padded 512-point FFT would shift the frequency grid and is **not** allowed), power, multiply by the shipped `mel_filters.bin`, log10, per-clip max−8 clamp, `(x+4)/4`; drop the last frame. |
| Inference | `onnxruntime-web`, `executionProviders: ['wasm']`, SIMD on, `numThreads = min(4, hardwareConcurrency)` when cross-origin isolated, else 1; graph optimization `all`; dynamic input length. |
| Decode | Greedy in ~20 lines; hotword beam search in ~100 lines of JS, a port of the Python reference. `▁` → space; no SentencePiece runtime needed. |
| Output | Lowercase text without punctuation (v0.1), per-chunk text, timings (feature, inference, decode), RTFx. |

**Cross-origin isolation:** WASM threads need `SharedArrayBuffer` → `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` (Safari requires `require-corp`). The demo ships a tiny static server that sets
them and falls back to single-thread mode without them.

## 3. Artifact

```text
release/
  model.int8.onnx      ≈ 8–10 MB   encoder + head + log-softmax
  lm_bpe3.bin          ≈ 5.1 MB    BPE trigram LM (browser binary, E-008/E-010)
  silero_vad.onnx      2.33 MB     Silero VAD v5, MIT, unmodified (D25)
  vocab.json           pieces by CTC id, blank = 0, "▁" space marker
  mel_filters.bin      80×201 float32, 64 KB (Whisper's filterbank, MIT)
  config.json          constants and hashes (below)
  NOTICE               Whisper + Silero MIT notices, dataset credits
  README.md            model card
```

```json
{
  "name": "moth-stt", "version": "0.4.0", "sample_rate": 16000,
  "features": {"n_fft": 400, "hop": 160, "n_mels": 80, "window": "hann_periodic", "center": true,
               "pad_mode": "reflect", "drop_last_frame": true, "floor_range": 8.0, "scale": "(x+4)/4",
               "mel_filters": "mel_filters.bin"},
  "model": {"file": "model.int8.onnx", "input": "mel", "output": "log_probs", "frame_rate_hz": 50, "max_seconds": 30},
  "tokenizer": {"vocab": "vocab.json", "blank_id": 0, "space_marker": "▁"},
  "decode": {"default": "beam", "beam": 5, "top_k": 8, "lm": "lm_bpe3.bin", "lm_weight": 0.2, "hotword_bonus": 2.0},
  "vad": {"file": "silero_vad.onnx", "window": 512, "context": 64, "threshold": 0.10},
  "files": {"model.int8.onnx": "sha256:…", "vocab.json": "sha256:…", "mel_filters.bin": "sha256:…",
            "lm_bpe3.bin": "sha256:…", "silero_vad.onnx": "sha256:…"}
}
```

The loader verifies every hash before use. Distribution: a Hugging Face model repository under Apache-2.0; the demo caches
files with the browser Cache API after the first download (the model downloads once).

## 4. Export

1. **Export fp32:** `torch.onnx.export` of the inference module (batch 1, dynamic time axis; opset ≥ 17), including the
   final `log_softmax`. Sinusoidal positions are **generated in-graph** from the dynamic length (`Range`/`Sin`/`Cos`) so that a
   1500×384 constant (2.3 MB fp32) does not enter the file; if in-graph generation is awkward, store the table as fp16 (≈ 1.1 MB).
2. **Validate:** PyTorch fp32 vs ONNX fp32 logits, max-abs diff ≤ 1e-3, on fixed clips of 1, 5, 10, 29 s.
   **Re-defined by E-021 (2026-11-07):** a single random draw is not a check — the typical gap is
   ~1.8e-4 and the tail crosses 1e-3 about once in 80 draws, always on the 29 s clip. The gate is
   now a **seeded sweep** (20 draws per length): the **median** must be ≤ 1e-3 and the **worst**
   draw ≤ 1e-2. Real speech measures 2.2–3.1e-4; a single draw over 1e-3 no longer fails the
   build. Evidence: `runs/E-021/RESULT.md`.
3. **Quantize:** `onnxruntime.quantization.quantize_dynamic` (weights int8, per-channel for MatMul). Convolutions may remain
   fp32 if dynamic quantization does not support them in the graph (≈ +1.6 MB). If int8 is outside the noise interval of fp32:
   (a) static quantization with ≈ 200 dev clips for calibration, (b) keep the LayerNorm-adjacent ops in fp32,
   (c) fp16 weights (≈ 16–17 MB; still under the ceiling).
4. **Package:** write `config.json` (including the `vad` block), hashes and `NOTICE`; copy
   `stt/assets/silero_vad.onnx` verbatim (2.33 MB, MIT) and list its sha256 in `files`. The browser
   `package_browser.py` then adds the LM binary and the beam settings.

## 5. Tests

| Test | What it proves |
|---|---|
| `normalize` property + fixture tests | one text rule |
| **Mel parity**: Python reference vs JS on fixture clips (1 s, 5 s, 29 s, silence, clipping) | no train/serve skew; tolerance 1e-3 max-abs after normalization |
| **Padding equivalence**: per-clip max clamp on padded vs unpadded clip | the front-end claim in `02` §2 |
| **Encoder parity**: own module vs `transformers.WhisperEncoder`, 30 s padded, fp32 | faithful reproduction (≤ 1e-4) |
| **Batch parity**: batched+masked vs single-clip logits | masks are correct |
| **Export parity**: PyTorch fp32 vs ONNX fp32 vs ONNX int8 logits and greedy transcripts | export does not change the model |
| **VAD guard**: Python and JS use the same 512+64 contract and thresholds; synthetic tone, digital silence and noise are gated, a committed FLEURS speech fixture is not | the guard does not reject real speech and is not bypassed in the browser |
| **Decoder parity**: Python vs JS greedy and beam on fixture log-probs | the port is faithful |
| **Browser e2e**: headless Chromium (Playwright) with a fake audio device (`--use-fake-device-for-media-stream --use-file-for-fake-audio-capture=clip.wav`) or a file input, ≈ 20 clips | transcript in the browser equals Python's; resampler difference is acceptable |
| **Speed benchmark** | browser RTFx feeds the ≥ 10× bar; also logs load time and memory |

## 6. Error handling and user-facing behavior

| Situation | Behavior |
|---|---|
| Microphone permission denied / no device | Clear message; file upload still works |
| No WASM SIMD / unsupported browser | Message naming the requirement; no crash |
| Not cross-origin isolated | Single-thread mode, with a visible note |
| Model download fails or hash mismatch | Error with retry; cache entry discarded |
| Silence / noise | VAD says no speech: empty transcript (no placeholder text), `guarded: true`, no model call |
| Audio > 30 s | Chunked automatically |
| Hotword list empty or malformed | Ignored (greedy) with a note |

## 7. Expected performance (to be measured, not assumed)

Forward cost ≈ 0.8 GFLOP per audio second. If WASM int8 sustains ≈ 10 GFLOP/s effective single-threaded, that is ≈ 12×
real time; threads and SIMD should improve on that. The ≥ 10× bar is therefore plausible but **unproven** — H4 measures it
with a random-weight export early so that a miss can still change the architecture (stride-2 pooling to 25 Hz halves the cost).
