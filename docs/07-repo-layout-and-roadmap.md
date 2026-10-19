# 07 — Repository layout and roadmap

The project lives in its own **new repository**, `semantique-team/moth-stt`, not in `ant-4b`: it is a different kind of model, with different dependencies
and licenses (Whisper MIT, dataset attributions). It follows the same conventions as `ant-4b`: `EXPERIMENTS.md`,
`runs/`, `tests/`, Apache-2.0.

## 1. Layout

```text
moth-stt/
  README.md   LICENSE (Apache-2.0)   NOTICE (Whisper MIT + dataset credits)
  EXPERIMENTS.md              hypothesis → config → metrics → conclusion (same template as ant-4b)
  pyproject.toml              uv-managed; torch 2.14.*, see 04 §2
  docs/                       these design documents (moved from ant-4b/NewModel/)
  stt/
    text.py                   normalize(): the one text rule
    features.py               reference log-mel (the JS port is tested against it)
    tokenizer.py              SentencePiece BPE train/load, vocab export, CTC id mapping
    data/                     fetch + cache, filters, manifests, mel shards, mixed-sentence tools
    model.py                  Whisper-tiny encoder + CTC head, variable-length forward, key-padding mask
    convert_whisper.py        one-off: HF whisper-tiny weights → own module (+ parity test)
    train.py                  loop, --device flag, checkpoint/resume, dev evaluation
    decode.py                 greedy + hotword beam (Python reference)
    eval.py                   WER, per-category/speaker, ETA, bootstrap CI, noise-set test
    export.py                 ONNX export + int8 quantization + release/ packaging
  baselines/                  whisper, whistle, parakeet runners that emit hypothesis files
  recordings/                 protocol, script generator/dedupe tools, data card (audio lives on HF, not git)
  web/                        JS: mel, ort-web runner, CTC decode, demo page + static server, Playwright tests
  tests/                      parity and property tests (normalize, mel, encoder, batch, export, decode)
  runs/                       config + metrics.json (+ logs) per experiment; checkpoints are gitignored
```

## 2. Unit contracts

Each unit has one purpose, a narrow interface, and can be tested without the others.

| Unit | Does | Interface | Depends on |
|---|---|---|---|
| `text.normalize` | Defines text equality | `normalize(str, lang_hint) -> str` | `num2words` |
| `features` | Reference log-mel | `log_mel(float32[N]) -> float32[80, N//160]` | numpy, `mel_filters` |
| `tokenizer` | BPE train/encode/decode; CTC ids | `train(texts, V)`, `encode`, `pieces` | `sentencepiece`, `text` |
| `data` | Manifests and mel shards from raw sources | `build(source) -> manifest.jsonl + shards` | `text`, `features` |
| `model` | Encoder + head; variable length; masks | `forward(mel, lengths) -> log_probs, out_lengths` | torch |
| `train` | Optimization loop with resume | CLI: `--config runs/<id>/config.yaml --device` | `model`, `data`, `tokenizer`, `eval` (dev) |
| `decode` | Greedy + hotword beam | `greedy`, `beam` | `tokenizer` |
| `eval` | Scoring and statistics | CLI: manifest + hyp → `metrics.json` | `text` |
| `export` | ONNX/int8/release folder | CLI: checkpoint → `release/` | `model`, `tokenizer` |
| `baselines/*` | Produce hypothesis files | `transcribe(wav) -> str` | third-party runtimes |
| `web/*` | Browser inference | `transcribe(pcm, {hotwords}) -> {text, timings}` | `onnxruntime-web`, `release/` |

## 3. Tooling and conventions

- `uv` for environments; `ruff` for lint/format; `pytest` for tests; tests that need no GPU or large data run in CI
  (GitHub Actions): normalize, mel parity (fixture clips committed, ≤ a few hundred KB), decode parity, packaging checks.
- Configs are YAML under `runs/<id>/config.yaml`; every run records git SHA, torch version, device, seed.
- Experiment IDs: `E-000` environment/feasibility, `E-001+` in order; H-labels from `04` appear in titles.
- Large artifacts (mel shards, checkpoints, audio) never enter git.
- Conventional commits, as in `ant-4b`.

## 4. `EXPERIMENTS.md` template

Same fields as `ant-4b/EXPERIMENTS.md`, plus compute and cost:

```markdown
## E-0NN — <title> (H<k>)

- **Date**:
- **Hypothesis**:
- **Model configuration**: (encoder, head, vocab size, variant; link to `runs/E-0NN/config.yaml`; git SHA)
- **Dataset**: (manifests, hours per source, mixture weights, filters)
- **Training parameters**: (optimizer, LR, schedule, batch seconds, epochs, augmentation, precision)
- **Hardware**: (e.g. Apple M5, 16 GB, MPS bf16; torch 2.14.1)
- **Compute and cost**: (wall-clock; paid units/dollars, running ledger vs 10 USD)
- **Metrics**: (table from `05` §8, CIs, link to `metrics.json`)
- **Result**:
- **Conclusion**: (hypothesis confirmed/rejected; gate decision; next experiment)
```

First entry to write: **E-000 — Mac feasibility benchmark (2026-10-18)** from `04-training.md` §1.

## 5. Roadmap (input for the implementation plan, not the plan itself)

| Milestone | Deliverable | Notes |
|---|---|---|
| **M0** | Repo skeleton, `text.normalize` + tests, `eval.py` + tests (WER, ETA, bootstrap), manifest schema | No GPU, no data |
| **M1** | Recorded mixed test set (T, 3 speakers) + H0 baselines table | Recording can start **now**, in parallel; interim H0 on FLEURS only |
| **M2** | Data pipeline: Common Voice IT, MLS IT, LibriSpeech/EN slice → manifests + mel shards; tokenizer; build report | Needs `ffmpeg`; ≈ 26 GB cache |
| **M3** | `model.py` + Whisper conversion + parity tests; random-weight ONNX int8 export and browser RTFx smoke test (**H4, early**) | De-risks speed |
| **M4** | H1 probe (padded vs truncated) | < 1 h compute |
| **M5** | H2 full fine-tune; gate vs Whisper-tiny; escalation ladder if needed | ≈ 5–10 h compute |
| **M6** | H3 code-switch data arms (gold, spliced, optional synthetic) | ≈ 1–3 h each |
| **M7** | Final export (int8), JS runtime + demo, parity + Playwright e2e, browser RTFx | — |
| **M8** | Release: Hugging Face model card, NOTICE, data card, `EXPERIMENTS.md` complete, README | Dataset license first (model and repo name decided 2026-10-19: `moth-stt`) |

Dependencies: M2 → M4 → M5 → M6 → M7; M0 → M1; M3 only needs M0. The first meaningful result is H0 (M1); the first
model result is H1 (M4).
