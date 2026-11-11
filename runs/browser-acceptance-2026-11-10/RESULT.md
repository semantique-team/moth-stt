# Browser Acceptance: 2026-11-10

**PASS: 4/4 Chromium projects, 41.1 seconds**, final reviewer-follow-up run. Real release 0.7.0 executed
through JS mel, Silero VAD, ONNX Runtime Web WASM and beam/LM decoding. No runtime edits,
new dependencies, lockfile changes, training, audio copies, binaries or commits.

## Environment

- Windows x64, Node v20.20.2, Python 3.12.10, Playwright 1.64.1, Chromium 156.0.8078.4.
- ORT Web/common 1.30.0; browser reported hardwareConcurrency 8 and SIMD enabled.
- Base code commit: `7b198bae12f5a58912bacf188430f0d802af9a39`. Acceptance files were
  uncommitted; exact runtime, acceptance source and lockfile hashes are in each JSON.
- Original sibling release read in place. All five configured asset hashes matched in
  Node before routing; the browser loader independently verified its configured hashes.
- Model SHA256: `7a075da40eb37de6988947efc717824db8b144769628db078e48dcc35b5ab0af`.

## Measurements

Same public 3-second Italian fixture; each benchmark has one warmup and three measured
repeats. Bias/no-bias timings wrap the entire `transcribe` call, including input checks.
The runner benchmark's RTFx uses its internal stage-total timing instead. Scorer instrumentation
is restored before timing; the bias timing uses bonus 1.75 and prefix bonus 0.

| Project | Isolated / SAB | Threads | Cold runner load ms | Runner RTFx | No-bias wall median ms | Bias wall median ms |
|---|---|---:|---:|---:|---:|---:|
| isolated-auto | true / true | 4 | 761.00 | 27.77 | 111.25 | 101.83 |
| isolated-1 | true / true | 1 | 685.01 | 18.67 | 182.61 | 178.63 |
| isolated-4 | true / true | 4 | 698.90 | 27.75 | 104.89 | 100.87 |
| nonisolated-auto | false / false | 1 | 658.40 | 17.78 | 167.40 | 176.00 |

These are descriptive shared-host timings, not speed gates or proof of bias overhead or
speedup. Later measurements varied materially even within a project. No flags forced
SharedArrayBuffer. The two servers bound only loopback, served only the development
repository root, used ports 8767/8768 and refused reuse.

## Verified Behavior

- Real speech produced nonempty beam transcripts and acoustic output frames in all projects.
- First IT, EN and spliced IT/EN clips from the existing sibling evaluation manifest also
  produced nonempty acoustic results in every project. IT/EN hashes matched their manifest;
  spliced bytes were hashed and recorded, but the manifest provides no expected splice hash.
- Three-second digital silence and synthetic 440 Hz tone returned empty text, zero acoustic
  frames and `guarded: true`.
- Wrong PCM type, 559-sample input, NaN and Infinity were rejected with meaningful errors.
- Overlapping calls were rejected; the first call completed and the next call recovered
  with the same transcript. Invalid-input calls also preceded successful inference.
- Tiled 31-second speech with segmentation explicitly disabled was rejected as
  `segment must contain 560..480000 samples (max 30 s); got 496000`; subsequent inference
  recovered. This tests the per-segment ceiling, not a blanket recording-duration limit.
- Empty contextual sequences and zero-bonus contextual calls matched the baseline text.
- Real bias calls used CTC ID 3, verified from the loaded vocabulary as the single piece
  for the whole output word `a`, including the configured word-start marker. Missing a
  verified word now fails acceptance rather than falling back to empty-context success.
- Temporary wrappers on the browser-imported `ContextualBias.prototype.delta` and `finish`
  called the original methods and returned their scores unchanged during real WASM runner
  transcriptions. Both methods were restored in `finally` before timing. No acoustic outputs
  or scorer outputs were mocked.
- Every project observed exactly 17,880 delta calls and 12 finish calls on its positive-bias
  transcription, with 300 original positive completion scores. Assertions verified the
  complete observed settings: bonus 1.75, prefix bonus 0, terminal sequence `3`. A nondefault
  bonus distinguishes forwarding from accidentally relying on the scorer's default bonus.
- No-options controls observed zero scorer calls. Empty-context and zero-bonus controls
  each observed delta/finish calls but zero positive scores, with exact expected terminal
  sets and bonus values. Positive-bias results also required nonempty text and acoustic frames.
  This catches ignored contextual options without requiring a changed transcript. It remains
  synthetic glossary API coverage, not a natural glossary or recognition gain.
- No browser page or console errors. Actual `.wasm` resource loading was observed;
  loaded ORT module/WASM local hashes and byte lengths are included in each report.

## Assets And Memory

Each project fetched exactly one response for each of the six allowlisted release files:
**19,437,500 raw body bytes**, including config. Model, vocab, mel filters, LM and VAD were
read directly from the original release, never copied into this repository. A fresh browser
context and Playwright routing disabled HTTP caching; browser Resource Timing values are
retained separately. They do **not** establish HTTP-compressed production download sizes.
ORT WASM itself was 14,239,897 bytes, additional to release assets.

`performance.memory.usedJSHeapSize` was available but reported the same coarse 10,000,000
bytes before load, after load and after calls in all four projects. These snapshots are not
an informative footprint estimate, peak usage, RSS or total WASM memory.

## Failures And Fixes

1. Initial `npm ci` encountered the corporate TLS chain error. The already-configured local
   CA was passed via process-local `NODE_EXTRA_CA_CERTS`; TLS verification stayed enabled.
2. The failed install left read-only generated dependency directories on this Windows
   workspace. Clearing their read-only attributes allowed `npm ci` to complete (21 packages).
3. Initial acceptance failed before inference because the harness assumed every audio
   manifest entry had `sha256`. Existing spliced entries do not. The harness now verifies
   hashes where supplied and records computed hashes with explicit verification coverage.
4. Corrected suite passed 4/4 in 43.3 seconds. After adding ORT loaded-asset fingerprints
   and a SIMD assertion, the initial full run passed 4/4 in 49.7 seconds.
5. Reviewer identified that a string-only positive-bias assertion could pass if the runner
   ignored contextual options. Added real scorer-use/settings/completion assertions and
   controls, removed the empty fallback, and refreshed all four reports and source hashes.
   The rerun initially encountered Windows read-only generated test-result directories;
   clearing those attributes allowed the unchanged command to run. Final suite: 4/4 in
   41.1 seconds, with all new scorer assertions passing.

Post-run checks: all report source hashes matched current files; privacy checks found no
private absolute paths or secret-like fields; `git diff --check` passed.

No concrete production-runtime bug was found or changed.

## Reproduce

From `web/` in PowerShell, with the original sibling release available:

```powershell
Test-Path -LiteralPath "$env:TEMP\opencode"
$env:npm_config_cache = "$env:TEMP\opencode\npm-cache"
$env:PLAYWRIGHT_BROWSERS_PATH = "$env:TEMP\opencode\playwright-browsers"
# If required, set NODE_EXTRA_CA_CERTS to your existing trusted corporate CA file.
npm ci --no-audit --no-fund
npx playwright install chromium
npm run test:acceptance
```

`PYTHON` optionally selects an executable path; otherwise `python` is used.
`MOTH_RELEASE_DIR` optionally overrides the original sibling release directory.
`MOTH_ACCEPTANCE_AUDIO` optionally overrides the existing manifest path, not a WAV path.
Absent default optional audio records reduced coverage; explicit invalid input fails.
See `web/README.md` for defaults and accounting details.

## Limits

This is browser runtime acceptance, not WER/ETA evaluation or Python/browser numerical parity.
The audio is reused public FLEURS material; splicing is correlated and is not natural
code-switching. User recordings remain for later. No full Python suite, training, microphone
or other-browser tests were run. No heavyweight CI requirement was added. Reports contain
portable hashes, transcripts and measurements only, with no raw audio, private absolute paths,
environment dumps, screenshots, traces or browser binaries.
