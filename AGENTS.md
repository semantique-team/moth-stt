# AGENTS.md — moth-stt

Instructions for AI agents working on this repository.

## Git policy (strict)

- **Never run `git commit`, `git push`, or anything that rewrites history (tags, force-push, rebase) unless the
  user explicitly asks in the current conversation.**
- At the end of a task, show `git status` / `git diff` and propose commit messages; then stop and wait.
- Commit one logical change at a time, conventional commits, only after approval.

## What this repository is

**Moth** (`moth-stt`) — a very small, open **Italian + English code-switching speech-to-text** model:
pretrained Whisper-tiny encoder (MIT) + a linear CTC head over a shared IT+EN SentencePiece BPE, ≈ 8 M
parameters, ≈ 8–10 MB int8 ONNX, record-then-transcribe in the browser (ONNX Runtime Web / WASM) and on CPU.

- Design docs: [`docs/README.md`](docs/README.md), then `docs/01` … `docs/09`.
- Current state (read first when resuming): [`STATUS.md`](STATUS.md) — done / in progress / next tasks.
- Implementation plan: [`docs/plans/2026-10-19-implementation-plan.md`](docs/plans/2026-10-19-implementation-plan.md)
  — follow it task by task (`- [ ]` checkboxes); the design docs win over guesses.

## Conventions

- Python ≥ 3.12 managed with `uv`; `ruff` (line-length 100) and `pytest` must pass before proposing a commit.
- Every experiment gets an `EXPERIMENTS.md` entry and a `runs/<id>/` directory (config, metrics and logs are
  committed; checkpoints, mel shards and raw audio never enter git).
- All text (training, tokenizer, scoring) passes through `stt.text.normalize` — the one text rule.
- Test and dev sets are never trained on; the recorded mixed test set `T` is reviewed by hand before recording.
- Apache-2.0; keep the Whisper MIT notice and dataset attributions in `NOTICE`.
- No secrets in the repository, ever.

## Rules of the road

- Don't add dependencies without a reason in the plan or an explicit request.
- Keep changes small and within the plan's task boundaries; update the plan when reality diverges.
- On this machine (M5 MacBook Pro 16 GB): long runs on power with `caffeinate -i`, checkpoints every ~20 min.
