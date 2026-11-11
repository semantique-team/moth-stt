# `web/` — browser runtime (Task 3.5 smoke, M7 demo)

Plain-JS port of the runtime path from `docs/06` §2, no bundler:

| File | What |
|---|---|
| `mel.js` | Whisper log-mel, direct 400-point DFT (201×400 cos/sin matrices), exact `docs/02` §2 constants |
| `vad.js` | Silero VAD v5 no-speech guard, 512+64-sample windows, speech fraction > 0.10 (twin of `stt/vad.py`) |
| `decode.js` | greedy + beam CTC decode (twins of `stt/decode.py`) |
| `lm.js` | browser reader for the compact BPE trigram LM (`stt/lm.py`) |
| `mic.js` / `mic-worklet.js` | record-then-transcribe mic: AudioWorklet PCM → 16 kHz mono |
| `ort.js` | `release/` loader (verifies every `sha256:` line), `onnxruntime-web` WASM session, timings + RTFx |
| `index.html` | demo page: file or microphone → transcript, timings, RTFx |
| `server.py` | static server with COOP/COEP (WASM threads need cross-origin isolation) |
| `tests/smoke.spec.ts` | Playwright: WAV → browser path → RTFx (speech fixture; the tone is gated) |
| `tests/vad.spec.ts` | Playwright: tone fixture gated, speech fixture decoded (D25) |
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

## Real WASM Acceptance

From `web/`, use the existing lockfile (no new dependencies):

```powershell
npm ci
npx playwright install chromium
npm run test:acceptance
```

On this Windows host, keep downloads outside the repository. Verify the temporary parent exists
first; use your configured trusted CA with `NODE_EXTRA_CA_CERTS` if corporate TLS requires it
(do not disable certificate verification):

```powershell
Test-Path -LiteralPath "$env:TEMP\opencode"
$env:npm_config_cache = "$env:TEMP\opencode\npm-cache"
$env:PLAYWRIGHT_BROWSERS_PATH = "$env:TEMP\opencode\playwright-browsers"
npm ci
npx playwright install chromium
npm run test:acceptance
```

`playwright.acceptance.config.ts` is independent of the old smoke config: Python from `PYTHON`
(an executable path, not shell arguments), otherwise `python`. It starts repo-root-only servers
on `127.0.0.1:8767` (`server.py`, COOP/COEP) and `:8768` (stdlib, no isolation), refuses server
reuse, and uses no SharedArrayBuffer-enabling flags. Projects cover isolated auto/1/4 threads
and non-isolated auto fallback. Each uses a fresh browser context and a minimal routed
`/acceptance.html`, not the demo startup.

The original real release defaults to sibling `../moth-stt` relative to the repository root;
override with absolute `MOTH_RELEASE_DIR`. Node verifies every configured asset SHA256 before
an exact filename allowlist fulfills same-origin requests. No release binaries are copied.
The required public speech fixture is `tests/fixtures/speech.wav` (FLEURS, CC-BY 4.0).
Optional audio defaults to sibling `../moth-stt-experiments/evaluation/manifest.json`; set
`MOTH_ACCEPTANCE_AUDIO` to an absolute manifest JSON path to override it. The first IT, EN and
spliced entries are read in place, hashed and checked against manifest hashes where provided
(the existing splices have no manifest hash). An absent default records reduced
coverage; an explicit missing manifest, missing group or bad hash fails. No audio is copied.

Acceptance asserts real isolation and runner/benchmark thread selection, nonempty real speech,
silence/tone gating, invalid PCM, overlap rejection and recovery, and a 31-second speech input
rejected with segmentation disabled (the production limit is per segment, not per recording).
Contextual calls require a verified single-piece whole output word and the release start
marker; missing positive coverage fails. Temporary scorer instrumentation preserves original
scores and asserts delta/finish use, exact bonus/prefix/terminal settings and positive completion
scores, with no-options, empty-context and zero-bonus controls. Methods are restored before
timing. This synthetic glossary is API coverage, not evidence of natural code-switching or
recognition improvement.

Each project attaches JSON and writes portable measurements under
`runs/browser-acceptance-2026-11-10/`. Reports include source/asset/audio hashes, Node/browser/ORT
versions, cold raw asset responses and Resource Timing, and end-to-end same-audio timings
with/without bias (one warmup, three repeats, no hardware speed threshold). Route fulfillment
disables cache and its byte counts are **not HTTP-compressed transfer claims**. Optional
`usedJSHeapSize` snapshots are **not RSS, peak memory or total WASM memory**. No recordings,
traces, screenshots, environment dumps or private absolute paths are retained. The dated
`RESULT.md` describes the actual verified run; these checks are not a required heavyweight CI job.
