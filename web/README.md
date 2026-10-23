# `web/` — browser runtime (Task 3.5 smoke, M7 demo)

Plain-JS port of the runtime path from `docs/06` §2, no bundler:

| File | What |
|---|---|
| `mel.js` | Whisper log-mel, direct 400-point DFT (201×400 cos/sin matrices), exact `docs/02` §2 constants |
| `decode.js` | greedy + beam CTC decode (twins of `stt/decode.py`) |
| `lm.js` | browser reader for the compact BPE trigram LM (`stt/lm.py`) |
| `mic.js` / `mic-worklet.js` | record-then-transcribe mic: AudioWorklet PCM → 16 kHz mono |
| `ort.js` | `release/` loader (verifies every `sha256:` line), `onnxruntime-web` WASM session, timings + RTFx |
| `index.html` | demo page: file or microphone → transcript, timings, RTFx |
| `server.py` | static server with COOP/COEP (WASM threads need cross-origin isolation) |
| `tests/smoke.spec.ts` | Playwright: WAV → browser path → RTFx |
| `tests/decoder.spec.ts` | Playwright: JS beam/LM parity fixtures |
| `tests/mic.spec.ts` | Playwright: fake microphone → 16 kHz PCM |

## Run the smoke

```bash
cd moth-stt
uv run python -m stt.export --random --vocab-size 1024 --out release/   # random-weight int8
cd web
npm install
npx playwright install chromium      # first time only
npm test                              # writes test-results/rtfx-smoke.json
```

Manual page: `npm run serve` → <http://127.0.0.1:8765/web/index.html> (the server serves the
repository root so `/release/…` and `/node_modules/…` resolve).

Notes: the benchmark follows `docs/05` §3 (warm-ups then medians). With random weights the
transcript is meaningless — the smoke measures that the graph runs and the speed bar is
reachable. Mel parity against `stt/features.py` and the record-then-transcribe demo are M7.
