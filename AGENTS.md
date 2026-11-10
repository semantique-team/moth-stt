# AGENTS.md — moth-stt

Instructions for AI agents working on this repository.

## Git policy (strict)

- **Never run `git commit`, `git push`, or anything that rewrites history (tags, force-push, rebase) unless the
  user explicitly asks in the current conversation.**
- At the end of a task, show `git status` / `git diff` and propose commit messages; then stop and wait.
- Commit one logical change at a time, conventional commits, only after approval.

## What this repository is

**Moth** (`moth-stt`) — a very small, open **Italian + English code-switching speech-to-text** model:
pretrained Whisper-tiny encoder (MIT) + two additional transformer blocks + a linear CTC head over
a shared IT+EN SentencePiece BPE. The recorded 0.7.0 release has 11,575,169 acoustic parameters,
an 11.91 MB int8 ONNX graph and 19,440,760 bytes of required assets including LM and VAD.
It is record-then-transcribe in the browser (ONNX Runtime Web / WASM) and on CPU, not streaming.

- Documentation index: [`docs/README.md`](docs/README.md). Original design is historical.
- Verified workflows: [`docs/guides/repository-workflows.md`](docs/guides/repository-workflows.md).
- Experiment evidence: [`docs/12-experiment-ledger.md`](docs/12-experiment-ledger.md).
- Model and paper caveats: [`docs/13-current-model-and-paper-readiness.md`](docs/13-current-model-and-paper-readiness.md).
- Current state (read first when resuming): [`STATUS.md`](STATUS.md) — done / in progress / next tasks.
- Implementation plan: [`docs/plans/2026-10-19-implementation-plan.md`](docs/plans/2026-10-19-implementation-plan.md)
  — historical, not a current task checklist. Use STATUS and executed evidence for current work;
  resolve disagreements against code and retained artifacts, documenting rather than guessing.

## Conventions

- Python ≥ 3.12 managed with `uv`; `ruff` (line-length 100) and `pytest` must pass before proposing a commit.
- Every experiment gets an `EXPERIMENTS.md` entry and a `runs/<id>/` directory (config, metrics and logs are
  committed; checkpoints, mel shards and raw audio never enter git).
- All text (training, tokenizer, scoring) passes through `stt.text.normalize` — the one text rule.
- Test and dev sets must never be directly trained on; review `T` by hand before recording.
  T/S3 have repeatedly informed development: do not call them untouched final tests. Anchor A
  reuses T sentences and is not independent dev. S3 must not contribute training audio.
- Apache-2.0; keep Whisper MIT and dataset attributions in generated `release/NOTICE`.
- No secrets in the repository, ever.

## Rules of the road

- Don't add dependencies without a reason in the plan or an explicit request.
- Keep changes small and within the plan's task boundaries; update the plan when reality diverges.
- Historical training host: M5 MacBook Pro 16 GB; on macOS use power and `caffeinate -i` for long
  runs, checkpoints every ~20 min. Native Windows training currently fails on Unix `resource`;
  use the platform notes in the workflows guide rather than assuming this checkout is the Mac.
