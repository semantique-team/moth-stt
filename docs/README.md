# moth-stt — IT+EN code-switching speech-to-text (design)

## Current Documentation (Audited 2026-11-10)

| Need | Source |
|---|---|
| Current checkout/model state and blockers | [STATUS.md](../STATUS.md), current verified section |
| How to install, record, prepare data, train, evaluate, export or publish | [Repository workflows](guides/repository-workflows.md) |
| Every experiment, numbers, benchmark populations, adoption/rejection | [12 - Experiment ledger](12-experiment-ledger.md), E-000-E-025 and extra audits |
| Exact model, release assets, fp32/int8/baseline tables, paper caveats | [13 - Current model and paper readiness](13-current-model-and-paper-readiness.md) |
| Executed decisions and corrected scientific interpretations | [11 - Evidence and decisions](11-evidence-and-decisions.md) |
| Architecture audit, observed risks and 20/25 MB budget | [14 - Architecture audit](14-architecture-audit-2026-11-10.md) |
| Resume after compaction: full session state and next experiment | [15 - Session handoff](15-session-handoff-2026-11-11.md) |
| Audit tooling: safe tempo, InterCTC, finalist eval, diagnostics, contextual bias, LM builder | [Audit improvements guide](guides/audit-improvements-no-training.md) |
| Portable readiness check for the other PC | [Other-PC preflight](guides/other-pc-preflight.md) |
| Real Chromium/WASM acceptance | [Browser acceptance report](../runs/browser-acceptance-2026-11-10/RESULT.md) |
| Independent mixed-dev and future quality experiment safeguards | [Mixed-dev protocol](guides/mixed-dev-and-quality-experiments.md) |

The audit baseline is `main` **`cd1ce1b`**. Published weights are **0.8.0 / E-028 + the E-030 decoder** (released 2026-11-12); **0.7.0 / E-024** remains the historical control. Main also contains newer runtime infrastructure. Original experiment
JSON and cards are retained as evidence, including conflicts explicitly flagged in docs 12-14.
These documents do not certify live hosting, recover missing weights/audio, or replace new
experiments needed for stronger paper claims.

## Historical Design

> **Status:** docs 01-09 began as the **design snapshot of 2026-10-18**; some sections were later
> amended, but unqualified design numbers and proposals are not current implementation evidence.
> The project has since been implemented and run: the live state is [`STATUS.md`](../STATUS.md), the
> experiment log is [`EXPERIMENTS.md`](../EXPERIMENTS.md), and the executed decision log, negative
> results, self-corrections and threats to validity are in
> [`11-evidence-and-decisions.md`](11-evidence-and-decisions.md).
> They were drafted inside `ant-4b/NewModel/` and now live in `moth-stt/docs/` (moved 2026-10-19).
> **Name:** **moth** — decided 2026-10-19, together with the repository name `moth-stt` (see `08-risks-and-open-questions.md`).

## Summary

A very small, open, locally runnable **speech-to-text model for Italian speech with English words mixed in**
("ho spostato il *meeting* a domani", "mandami il *feedback* sul *deploy*"). It is part of a family of small open
models, next to ANT-4B (the System-1 decision model in this repository), but it is a different kind of model
and gets its own repository.

The model is the **pretrained Whisper-tiny encoder (MIT) plus a new CTC head** over one shared Italian+English
BPE vocabulary — the Whisper decoder is dropped. It has about 8 M parameters, ships as an int8 ONNX file of
roughly 8–10 MB (hard ceiling: about 20 MB), and runs in the browser (WASM) and on any CPU through
`onnxruntime`. It is fine-tuned on open human-transcribed Italian and English speech plus a small set of
recorded Italian–English mixed sentences, **on the author's M5 MacBook Pro (16 GB) with PyTorch/MPS**.
Kaggle/Colab are the fallback. The total paid-compute budget is **10 USD** and is expected to stay unspent.

## Key facts

| Item | Decision |
|---|---|
| Task | Speech-to-text. Italian with English words mixed in (code-switching) is the core use case. |
| Model | Whisper-tiny encoder (pretrained, MIT) + linear CTC head over a shared IT+EN SentencePiece BPE (V = 1024 to start). No autoregressive decoder. |
| Size | ≈ 8.0 M parameters; ≈ 8–10 MB at int8; hard ceiling ≈ 20 MB download. |
| Output (v0.1) | Lowercase, no punctuation. |
| Runtime | Browser (ONNX Runtime Web, WASM) and native `onnxruntime`. v0.1 is record-then-transcribe (clips ≤ 30 s). |
| Training | Fine-tuning on an M5 MacBook Pro 16 GB (PyTorch MPS, bf16 autocast). Fallback: Kaggle (30 free GPU h/week) or Colab. Budget ≤ 10 USD. |
| Data | Open human-transcribed IT (Common Voice, MLS) and EN (LibriSpeech or Common Voice), plus recorded IT+EN mixed sentences. No teacher/pseudo-labels in v0.1. |
| Evaluation | Own mixed test set (≈ 600 utterances, ≥ 3 speakers, one held out), FLEURS IT/EN, Common Voice IT test, a 1-hour silence/noise set. Bars are relative to measured baselines. |
| License | Code Apache-2.0; `NOTICE` for Whisper (MIT) and dataset attributions. |

## Documents

Read the current guides above first. The following table indexes both historical and live material.

| File | Contents |
|---|---|
| [`01-goals-and-decisions.md`](01-goals-and-decisions.md) | Problem, constraints, what exists and why it is not enough, approaches considered, why this one, success tiers, non-goals, budget. |
| [`02-architecture.md`](02-architecture.md) | The model: layers, shapes, parameter arithmetic, tokenizer, CTC, decoding, size and compute budgets, variants and fallbacks. |
| [`03-data.md`](03-data.md) | Every dataset, license, size, filter, split; mixed-sentence generation; the recording protocol; the test, dev and noise sets. |
| [`04-training.md`](04-training.md) | Mac benchmark results, training environment, recipes for H1–H3, hyperparameters, checkpointing, cloud fallback, time estimates. |
| [`05-evaluation.md`](05-evaluation.md) | Text normalization, metrics, manifests, baselines (H0), statistics, gates, reporting. |
| [`06-export-and-web-runtime.md`](06-export-and-web-runtime.md) | ONNX export, int8, artifact format, browser pipeline, JS feature extractor, tests. |
| [`07-repo-layout-and-roadmap.md`](07-repo-layout-and-roadmap.md) | New-repo layout with unit contracts, tooling, milestones M0–M8, `EXPERIMENTS.md` template. |
| [`08-risks-and-open-questions.md`](08-risks-and-open-questions.md) | Risk register, open questions, follow-up backlog, stop/pivot criteria. |
| [`09-research-notes.md`](09-research-notes.md) | The evidence behind the decisions, with links and a verified/unverified marker per claim. |
| [`10-optimization-review.md`](10-optimization-review.md) | Architecture/code optimization review: loader, attention, CTC head, eval, export, capacity variants, recent STT papers, and what can safely run while H2 trains. |
| [`11-evidence-and-decisions.md`](11-evidence-and-decisions.md) | What the project actually decided, rejected and corrected: executed decision log, negative results, self-corrections, threats to validity, reproduction index. |
| [`12-experiment-ledger.md`](12-experiment-ledger.md) | Artifact-audited E-000-E-025 ledger, benchmark inventory, release measurements, discrepancies and acceptance/rejection rationale. |
| [`13-current-model-and-paper-readiness.md`](13-current-model-and-paper-readiness.md) | Current architecture, exact parameter/asset counts, lineage, benchmark precision/splits, reproducibility limits and paper deliverables. |
| [`14-architecture-audit-2026-11-10.md`](14-architecture-audit-2026-11-10.md) | Detailed architecture audit: observed risks, parameter/asset budgets, prior experiments, and conditional improvement paths within 20-25 MB. Read-only findings, not new measured gains. |
| [`15-session-handoff-2026-11-11.md`](15-session-handoff-2026-11-11.md) | Session handoff: PR #3, S2 data, E-026/E-027, E-028 fp32/INT8 results, quantization sweep, decisions D36-D37 and the E-029 QAT plan; same-day update records the negative QAT outcome. |
| [`guides/audit-improvements-no-training.md`](guides/audit-improvements-no-training.md) | Local implementation follow-up: safe tempo, optional training-only InterCTC, guarded finalist selection, CTC diagnostics, contextual bias and frozen-corpus LM tooling. No new trained model. |
| [`guides/other-pc-preflight.md`](guides/other-pc-preflight.md) | Portable no-training readiness report: required dependencies, missing assets, hashes, disk and recording-provenance checklist. |
| [`../runs/browser-acceptance-2026-11-10/RESULT.md`](../runs/browser-acceptance-2026-11-10/RESULT.md) | Real Chromium/WASM acceptance in isolated 1/4-thread and nonisolated modes, with contextual scorer instrumentation and explicit measurement limits. |
| [`guides/repository-workflows.md`](guides/repository-workflows.md) | Source-checked operational commands, platform restrictions and protocol for new experiments. |
| [`plans/2026-10-19-implementation-plan.md`](plans/2026-10-19-implementation-plan.md) | Original implementation plan: ordered tasks M0–M8 with tests, commands, gates and commit points (historical; not maintained). |
| [`plans/2026-11-10-lightweight-it-en-runtime-plan.md`](plans/2026-11-10-lightweight-it-en-runtime-plan.md) | Runtime follow-up merged in main via PR #2; consult its implementation-status table. Mixed-dev recordings and quality experiments remain blocked/not executed; native acceptance is not browser acceptance. |
| [`plans/2026-10-24-post-E014-plan.md`](plans/2026-10-24-post-E014-plan.md) | Historical post-E-014 plan; E-016/E-017 completed, original distillation not run, D34 keeps ETA as a gate, first E-015 gold addendum rejected. Not a current task list. |
| [`plans/2026-10-30-vad-guard.md`](plans/2026-10-30-vad-guard.md) | Silero VAD as the no-speech guard (D25, release 0.4.0): implemented, with the measurement table and the invocation trap. |
| [`plans/2026-11-07-whistle-lora-probe.md`](plans/2026-11-07-whistle-lora-probe.md) | **E-019 completed and rejected at L1:** historical exploratory Whistle LoRA protocol, never Moth; later arms not run after kill. One recipe, not architecture-causality proof. |
| [`../STATUS.md`](../STATUS.md) | Live state: what is done, what is in progress, what is next. |
| [`../EXPERIMENTS.md`](../EXPERIMENTS.md) | Historical research log through E-025 and the later E-015 addendum; use doc 12 for corrected evidence and unexecuted IDs. |

## Decision log

| # | Decision | Why |
|---|---|---|
| D1 | Build a speech-to-text model (not another decision model, not TTS). | The family needs a new *kind* of model; STT was the intended one. A TTS plan was researched and **shelved**, not deleted. |
| D2 | Core job: Italian speech with English words mixed in. | This is where Whisper-class small models fail (language token per clip, hallucination at switches). |
| D3 | As little training as possible, as small as possible. | Constraint from the author. |
| D4 | Must run in the browser / on edge: ≈ 20 MB ceiling. | Constraint from the author. |
| D5 | Own model — **not** a Cactus Whistle LoRA fine-tune. | Whistle is kept as baseline and architectural reference only. |
| D6 | Compute budget ≤ 10 USD; Mac/Colab/Kaggle. | Constraint from the author; rules out the from-scratch Moonshine recipe (≈ 8×H100 for 1–3 days). |
| D7 | Approach B: pretrained Whisper-tiny encoder + own CTC head/tokenizer/data/training. | Reuses acoustic knowledge we cannot afford to learn; drops the decoder that causes hallucinations and language-token problems. |
| D8 | v0.1 uses human-transcribed audio only (no Parakeet pseudo-labeling). | Keeps the plan simple and cheap. |
| D9 | Train on the author's M5 MacBook Pro 16 GB. | Measured feasible (see `04-training.md`). |
| D10 | Evaluation bars are relative to measured baselines (Whisper-tiny, Whistle, Parakeet). | Absolute WER targets would be guesses before H0. |
| D11 | v0.1 scope: record-then-transcribe; no streaming, no punctuation, no WebGPU. | YAGNI. The no-speech guard was out of scope too, then added as an energy check (D23) and replaced by Silero VAD (D25) — see `11` §2. |
| D12 | The project lives in a new repository, not in `ant-4b`. | Different model, dependencies and licenses. |
| D13 | Model name **moth**; repository **`moth-stt`** (decided 2026-10-19). | Family naming thread with ANT (insects); the suffix names the task (speech-to-text), not the parameter scale. |

## Glossary

- **CTC** — connectionist temporal classification: the head emits a token or *blank* per encoder frame; a loss and a collapse rule turn frame outputs into text. No decoder loop.
- **BPE** — byte-pair-encoding subword vocabulary (SentencePiece).
- **WER** — word error rate, computed on normalized text (`05-evaluation.md`).
- **ETA** — English-term accuracy: share of annotated English-origin reference words reproduced exactly.
- **RTFx** — real-time factor, audio seconds processed per wall-clock second (higher is faster).
- **H0–H3** — the experiment ladder: baselines, frozen-encoder probe, full fine-tune, code-switch data.
- **S1/S2/S3** — the three recording speakers; S3 is held out from all training data.

## Next step

The design phase is closed. Use the verified section of [`STATUS.md`](../STATUS.md) for current
state, [the workflow guide](guides/repository-workflows.md) for execution, and
[paper readiness](13-current-model-and-paper-readiness.md#7-paper-readiness) for the evidence
still needed. The 2026-11-10 audit does not authorize or execute a new training experiment.
