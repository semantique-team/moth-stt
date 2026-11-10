# Runtime Acceptance: 2026-11-10

Acceptance checks: **PASS**. Original release 0.7.0, unchanged assets; no training or tuning.

Version, settings and all seven file hashes/sizes match the committed `runs/E-024/release-0.7.0.json` record; pin commit and record hash in `report.json`.

| Path | Clips | WER | Total runtime s | RTF | Realtime x |
|---|---:|---:|---:|---:|---:|
| vad | 104 | 40.4817% | 52.499 | 0.0402 | 24.87 |
| no_vad | 104 | 39.8213% | 35.919 | 0.0275 | 36.35 |

Audio: 1305.60s. Python asset load: 0.280s. No warmup.

Identical-logit JS/Python: beam 10/10, greedy 10/10 exact raw transcripts. Round-robin first IT/EN/spliced clips; real binary LM, canonical Python mel, no VAD.

Existing 459 paired synthetic decoder fixtures and benchmark self-checks: PASS (`validation-tests.txt`).

Rerun used repository `.venv/Scripts/python.exe` with `PYTHONPATH` cleared and Torch 2.14.1+cpu available. Final timing run did not overlap the repository test suite.

Post-run checks: Ruff passed; targeted benchmark/runtime/release/LM tests: **139 passed, 2 skipped**. Repository Windows suite excluding `tests/test_train.py`: **446 passed, 3 skipped, 5 failed**. Training tests were excluded because `stt.train` imports Unix-only `resource`. The five unrelated failures are POSIX path assumptions and unquoted shell executable paths in `tests/test_baselines_common.py` and `tests/test_sources.py`; no fixes outside this directory were attempted. See `validation-tests.txt`.

VAD synthetic controls: 0 words / 4 clips. No-VAD controls: 5 words.

Model SHA256: `7a075da40eb37de6988947efc717824db8b144769628db078e48dcc35b5ab0af`.
Normalizer SHA256: `d74abd1ed067e9f65436c502b919121df5911936a9aa09d24f47cadc67783d25`.

## Stage Timing

| Path | Stage | Seconds |
|---|---|---:|
| vad | vad | 8.843 |
| vad | segmentation | 0.006 |
| vad | mel | 13.339 |
| vad | acoustic | 9.058 |
| vad | decoder | 21.037 |
| no_vad | mel | 9.832 |
| no_vad | acoustic | 7.502 |
| no_vad | decoder | 18.579 |

Segmented speech clips: 56; gated clips: 4.

VAD minus no-VAD WER: +0.660 percentage points, paired 95% CI [-0.456, +1.852]. No quality improvement demonstrated.

## Reproduce

From the repository root in PowerShell:

```powershell
uv sync --locked --all-extras
uv run --locked --all-extras python runs/runtime-2026-11-10/benchmark.py --release ../moth-stt --manifest ../moth-stt-experiments/evaluation/manifest.json
```

Requires Python 3.12+, Node, NumPy, ONNX Runtime, soundfile, num2words and pytest. uv sync installs locked dependencies. The sibling manifest/audio and original release are prerequisites; the benchmark does not download assets. Hypotheses and official normalized reference JSONL are retained; no audio or logits.

Official `stt.eval.main` scores normalized raw references for mono-language clips and the existing reference for spliced clips. All `en_spans` are empty: ETA is null. See `metrics.vad.json` for paired bootstrap versus the fresh baseline, `report.json` for per-stage/per-clip timing, source/asset/audio hashes and environment.

## Limitations

- Reused known public FLEURS sample; not an independent quality test.
- Spliced clips reuse constituent IT/EN audio; samples are correlated.
- No natural code-switching set or validated en_spans; ETA is null, no natural ETA claim.
- Mixed numeric text uses official Italian fallback, not language-aware span normalization.
- Four existing synthetic silence/white-noise controls only; no real-world nonspeech claim.
- JS decoder only on native ONNX float32 outputs; no browser/WASM acoustic or JS mel parity.
- Serial VAD then baseline, no warmup, default ORT threads, shared host; timings not isolated.
- RTF excludes asset loading, file IO/hash, scoring, Node IPC and validation tests.
- Per-utterance bootstrap is descriptive, not independent-speaker inference.
