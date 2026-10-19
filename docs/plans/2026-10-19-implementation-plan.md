# Implementation Plan — moth-stt v0.1

> **For agentic workers:** implement task by task, in order (respect the dependency notes). Steps use checkbox
> (`- [ ]`) syntax. Each task ends with a commit; run the listed tests before committing. When a step says
> *record E-0NN*, add the `EXPERIMENTS.md` entry in the same commit (template: `docs/07` §4). Never train or tune
> on test data; every number reported comes from `eval.py` output committed under `runs/`.
>
> **Status sync (2026-10-19):** the checkboxes below were not maintained while M0–M3 landed — [`STATUS.md`](../../STATUS.md)
> is the live snapshot and the boxes are not reliable for those phases. Done at that date: **M0, M1 (1.1–1.3),
> M2 code (2.1–2.6) + manifests (E-002) + the real mel cache (2.5: 291.9 h in `data/shards/`), M3 (3.1–3.5, incl.
> E-003)**; 2.3 is done too (pool of 3,137 sentences + the BPE tokenizer built by `stt/data/build_tokenizer.py`,
> stats in the build report), so **4.2 (H1) is ready to run**; 1.4 is partial (Whisper tiny only) and still waits
> on the recordings, the Whistle/Parakeet CLIs and Common Voice IT; 4.1 is done (`stt/train.py` + tests +
> `runs/E-004/config.yaml`).

**Goal:** ship **moth v0.1** — a very small, open **Italian + English code-switching speech-to-text** model:
pretrained Whisper-tiny encoder (MIT) + a linear CTC head over a shared IT+EN SentencePiece BPE (V = 1024),
≈ 8.0 M parameters, ≈ 8–10 MB int8 ONNX, record-then-transcribe in a browser tab (ONNX Runtime Web / WASM) and
on any CPU, fine-tuned on the author's M5 MacBook Pro within a ≤ 10 USD budget (expected: 0).

**Definition of done (ship tier — `docs/05` §5; absolute numbers are relative to the measured H0 baselines):**

| Gate | Condition |
|---|---|
| Size | ≤ 20 MB total runtime download (model + vocab + config + mel filters) |
| Mixed-set quality | WER and ETA better than Whisper-tiny's **best** H0 variant (auto or forced `it`): the 95% paired-bootstrap CI of (ours − Whisper-tiny) lies entirely below 0 for WER and entirely above 0 for ETA |
| Silence/noise | 0 words on the 1 h non-music noise/silence set (music reported separately) |
| Speed | browser RTFx ≥ 10× on the author's M5 (WASM, threads + SIMD) |
| int8 | within the noise interval of fp32 on dev and the mixed test |
| Stretch | mixed-set WER within the noise interval of Cactus Whistle |

**Tech stack:** Python ≥ 3.12 via `uv`; `torch==2.14.*` (MPS, bf16 autocast, fp32 CTC — the benchmark pins this);
`sentencepiece`; `soundfile`; `num2words`; `onnx` + `onnxruntime`; `pytest`; `ruff`. Dev-only extras:
`transformers` (weight-mapping parity test) and `openai-whisper` (mel reference test). Web: `onnxruntime-web`,
plain JS, a tiny static server, Playwright.

**Design docs:** [`docs/README.md`](../README.md) and `01`…`09`. This plan converts them into ordered tasks;
when a step is ambiguous, the design doc wins — and the fix is written back into this plan.

**Repo conventions (mirror `ant-4b`):** `EXPERIMENTS.md` entry per experiment; `runs/<id>/config.yaml` +
`metrics.json` + logs are committed, checkpoints and mel shards never; conventional commits; Apache-2.0; CI
(GitHub Actions) runs every test that needs no GPU and no large data.

**Environment prerequisites (check first):** M5 MacBook Pro 16 GB on power; `uv` installed; `brew install ffmpeg`;
≈ 30 GB free disk for raw data cache + ≈ 26 GB mel shards (the machine has ≈ 108 GB free); for long runs use
`caffeinate -i` and close heavy apps.

## Milestones

| Phase | Milestone | Deliverable | Design refs |
|---|---|---|---|
| M0 | Skeleton, `text`, `eval` | repo + tests green in CI | 05 §1–4, 07 §1–3 |
| M1 | Recordings + H0 | mixed test set T, baseline table | 03 §6–8, 05 §6–7 |
| M2 | Data pipeline | manifests, tokenizer, mel shards, build report | 03 §2–5 |
| M3 | Model + export + H4 early | parity-tested model, int8 ONNX, browser RTFx smoke | 02, 06 |
| M4 | H1 probe | frozen-encoder result, truncated vs padded decision | 04 §5 |
| M5 | H2 fine-tune | full model, gate vs Whisper-tiny | 04 §5 |
| M6 | H3 code-switch | data arms, per-category/speaker tables | 04 §5 |
| M7 | Runtime + demo | demo page, e2e, RTFx ≥ 10 | 06 |
| M8 | Release | HF model card, NOTICE, filled checklist | 07 §5 |

Dependencies: M0 → M1, M2, M3; M2 → M4 → M5 → M6 → M7. **M1 recording can start on day 1, in parallel with
everything else.**

---

## Phase M0 — Bootstrap, the one text rule, the scorer

### Task 0.1 — Repo skeleton and tooling

**Files:** create `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`, `stt/__init__.py`, `tests/test_smoke.py`, `runs/.gitkeep`.

- [ ] **Create `pyproject.toml`** (`uv` managed): name `moth-stt`, requires-python `>=3.12`; dependencies
      `torch==2.14.*`, `numpy`, `sentencepiece`, `soundfile`, `num2words`, `onnx`, `onnxruntime`; dev group
      `pytest`, `ruff`, `transformers`, `openai-whisper`.
- [ ] **Create `.gitignore`:** `__pycache__/`, `.venv/`, `data/`, `runs/**/ckpt*.pt`, `release/`, `*.onnx`,
      `web/node_modules/`, `web/test-results/`, `.DS_Store`.
- [ ] **Sync and verify:** `uv sync` → `uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"`
      prints `2.14.x True`.
- [ ] **Configure ruff** (defaults, line-length 100) and pytest (`testpaths = ["tests"]`) in `pyproject.toml`.
- [ ] **CI:** `.github/workflows/ci.yml` on push/PR: `uv sync`, `uv run ruff check .`, `uv run pytest -q`.
      `tests/test_smoke.py` does `import stt` so the workflow is green from the first commit.
- [ ] **Commit** `chore: bootstrap repo (uv, ruff, pytest, CI)`.

### Task 0.2 — `stt/text.py` — normalize(), the one rule

**Files:** create `stt/text.py`; test `tests/test_text.py`. **Spec:** `docs/05` §1.

- [ ] **Write the failing tests.** Fixture table must include at least: `"L'APPUNTAMENTO È ALLE 15:30"`,
      `"un’idea"`, `"mail—risposta/slack"`, `"l' ho"`, `"città perché ù"`, `"über ça ñ"`, `"emoji 😀 ok"`,
      a decimal (`"3,5"`), plus properties: idempotent on every fixture, no digits and no punctuation in output.
- [ ] **Run** `uv run pytest tests/test_text.py -q` → FAIL (`ImportError`).
- [ ] **Implement** `normalize(text: str, lang_hint: str = "it") -> str`: NFKC → lowercase; typographic
      apostrophes (`’ ‘ ´`) → `'` and whitespace after apostrophe removed; digits → words via `num2words`
      (`lang_hint`; Italian default — document the English-context limitation from `05` §1); hyphens/dashes/slashes
      → space; keep `a–z`, `à è é ì í ò ó ù ú`, `'`, space; fold other Latin diacritics to ASCII base (NFKD +
      strip combining marks); drop everything else; collapse whitespace; strip.
- [ ] **Run** tests → PASS; `uv run ruff check .`.
- [ ] **Commit** `feat(text): add the one normalization rule`.

### Task 0.3 — `stt/eval.py` — WER, ETA, bootstrap

**Files:** create `stt/eval.py`; test `tests/test_eval.py`. **Spec:** `docs/05` §2–4.

- [ ] **Write the failing tests:** WER hand values (perfect, 1 sub, deletion, insertion, corpus-level sum);
      ETA on a fake manifest with `en_spans` (exact-match-aligned numerator); bootstrap CI seeded → deterministic,
      and CI of a zero difference contains 0; breakdown by `category` and `speaker`.
- [ ] **Implement** the scorer: Levenshtein alignment over words (pure Python, no dependency); corpus WER =
      (S+D+I)/ref words; per-category, per-speaker-group (S1+S2 vs S3); insertion rate per 100 ref words;
      ETA = share of `en_spans` reference words aligned as exact matches (list per-term errors);
      noise-set report (total words, non-empty clips, music separately); paired bootstrap over utterances
      (1,000 resamples, seeded) incl. CI of the difference between two systems; `metrics.json` with manifest hash,
      hypothesis hash, normalizer version, git SHA, config path, all metrics and CIs.
- [ ] **CLI:** `eval.py --manifest M --hyp H [--breakdown category,speaker] [--bootstrap 1000] [--vs other_metrics.json]`
      writes `metrics.json` and prints the `docs/05` §8 table.
- [ ] **Run** → PASS; **Commit** `feat(eval): add WER, ETA and bootstrap scorer`.

### Task 0.4 — Manifest schema + validator

**Files:** create `stt/manifest.py`; test `tests/test_manifest.py`; fixture `tests/fixtures/manifest_sample.jsonl` (5 lines). **Spec:** `docs/03` §4.

- [ ] **Write the failing tests:** load a valid sample; reject missing keys, wrong `lang`, `category`/`en_spans`
      on non-mixed rows; `norm_hash(text)` is stable; duplicate-id detection.
- [ ] **Implement** `load_manifest(path) -> list[dict]` with validation and `norm_hash = sha1(normalize(text))`.
- [ ] **Run** → PASS; **Commit** `feat(manifest): add schema loader and validator`.

### Task 0.5 — `EXPERIMENTS.md` with E-000

**Files:** create `EXPERIMENTS.md` (template from `docs/07` §4).

- [ ] **Write E-000 — Mac feasibility benchmark (2026-10-18)** from `docs/04` §1: torch 2.14.1, MPS, shape-exact
      random model, 32×10 s per step; numbers table (bf16 MPS 394 ms/step → ≈ 812 audio-h/h, peak 4.4 GiB;
      CTC forward+backward natively on MPS); conclusion: M2-H2 feasible on the Mac; pin `torch==2.14.*`.
- [ ] **Commit** `docs(experiments): add E-000 Mac feasibility benchmark`.

---

## Phase M1 — Recordings, mixed test set, baselines (H0)

> Can start immediately, in parallel with M2/M3. M1 is the first measurable result (H0).

### Task 1.1 — Mixed-sentence scripts (`recordings/generate_scripts.py`)

**Files:** create `recordings/generate_scripts.py`, `recordings/README.md`; test `tests/test_scripts.py`. **Spec:** `docs/03` §6.

- [ ] **Write the failing tests:** category quotas (T = 600: single_word 150, phrase 100, brand 100, acronym 60,
      italianized 80, switch 60, control_it 30, control_en 20); T stratified so each of S1/S2/S3 gets 200;
      dedupe (norm-hash equality + word-3-gram Jaccard > 0.8) across T / R / pool; provenance fields
      (generator, date, reviewed).
- [ ] **Implement** generation with the chosen LLM provider (JSONL in/out, resumable), the dedupe/quotas logic,
      `--kind T|R|pool` and a review-report mode that lists unreviewed sentences. R = 600–1,000 per speaker for
      S1/S2; pool = 3–5k for tokenizer text. Every sentence carries `category` and `en_spans` ([start,end) into
      `norm.split()`).
- [ ] **Generate the real scripts**, review every T sentence by hand (naturalness, pronounceability), log
      provenance; store under `recordings/scripts/`.
- [ ] **Run** tests → PASS; **Commit** `feat(recordings): add mixed-sentence script generator and scripts`.

### Task 1.2 — Recording protocol, manifests, consent

**Files:** create `recordings/README.md` (protocol), `recordings/build_manifest.py`, `recordings/consent-template.md`,
`recordings/data-card.md`; test `tests/test_recordings.py`. **Spec:** `docs/03` §7.

- [ ] **Write the failing tests** (generate tiny WAVs with `soundfile` in the test): naming `S1_0001.wav`
      enforced; duration/charset checks; manifest rows carry `speaker`, `category`, `en_spans`, `split`.
- [ ] **Implement** the manifest builder: per-speaker folders in, T/R/A JSONL manifests out; light silence trim
      **report only** (never edit audio by default); retake log parsed.
- [ ] **Record:** T first (600 sentences; S1/S2/S3 = 200 each incl. the anchor A = 60 sentences read by all
      three), then R for S1/S2. Protocol details from `docs/03` §7: one speaker through a laptop microphone
      capture path, one through a phone; native sample rate is fine (the pipeline resamples); WAV 16-bit mono;
      conversational speed, English words pronounced as in conversation, no "fixing" of Italianized
      pronunciations. **Before publishing anything:** written consent per speaker (CC-BY 4.0 proposed, private
      otherwise) recorded in the data card.
- [ ] **Run** tests → PASS; **Commit** `feat(recordings): add protocol, manifest builder and T/R scripts state`
      (audio itself stays out of git; see `recordings/data-card.md` for where it lives).

### Task 1.3 — Baseline runners (`baselines/`)

**Files:** create `baselines/common.py`, `baselines/whisper_runner.py`, `baselines/whistle_runner.py`,
`baselines/parakeet_runner.py`; test `tests/test_baselines_common.py`. **Spec:** `docs/05` §7.

- [ ] **Implement `common.py`:** `transcribe(wav) -> str` contract, hypothesis-file writer (`{"id", "hyp", "latency_s"}`),
      `--sets fleurs_dev,fleurs_test,cv_test,mixed,noise`, per-system size-on-disk and Mac-CPU RTFx recording.
- [ ] **Whisper runner:** tiny and base; language auto vs forced `it`; greedy and beam 5; record whether/when
      30 s padding is used. **Whistle runner:** via the `needle` CLI; auto vs forced `it`; ± keyword list.
      **Parakeet runner:** TDT 0.6B v3 int8 offline (sherpa-onnx or ONNX export); ± hotwords.
- [ ] **Hotword fairness rule:** the fair variant uses only R-script terms + the curated English-terms list;
      any variant with test-set terms is labelled **oracle** in every table.
- [ ] **Run** on the available sets (interim: FLEURS IT/EN first; mixed T when recorded). Hypothesis files into
      `baselines/runs/<system>/`.
- [ ] **Commit** `feat(baselines): add whisper, whistle and parakeet hypothesis runners`.

### Task 1.4 — E-001, the H0 table

- [ ] **Fill the `docs/05` §8 table** with measured numbers: mixed WER + CI, per category, ETA, FLEURS-IT/EN,
      CV-IT, noise words, RTFx, size. Every relative bar in `docs/04`/`docs/05` is calibrated **now** against
      real measurements (tie rule: CI includes 0).
- [ ] **Commit** `docs(experiments): add E-001 H0 baselines`.

---

## Phase M2 — Data pipeline (tokenizer, mel cache)

### Task 2.1 — Source fetchers + license checks

**Files:** create `stt/data/sources.py`, `data/README.md` (cache layout + licenses). **Spec:** `docs/03` §2.

- [ ] **Install:** `brew install ffmpeg` (Common Voice mp3); FLAC via `soundfile`.
- [ ] **Write tests** for split selection and the ~170 h sampling rule (deterministic seed) — no network in tests.
- [ ] **Implement fetchers** with checksums, official splits only: **Common Voice IT 27.0** (CC0; validated set;
      train/dev/test official), **MLS IT** (CC-BY; dev/test official), **LibriSpeech `train-clean-100`** or a
      **Common Voice EN** slice (CC0) — *verify first whether an Italian-accented English slice exists and is
      usable; record the decision*; **FLEURS IT/EN** dev + test **evaluation-only, never train** (also not the
      FLEURS train split); noise candidate (MUSAN or similar) — *verify the license before downloading*.
- [ ] **Commit** `feat(data): add source fetchers and license notes`.

### Task 2.2 — Filters + build report

**Files:** create `stt/data/filters.py`; test `tests/test_filters.py`. **Spec:** `docs/03` §3.

- [ ] **Write failing tests** per filter with synthetic cases: duration 1–30 s (warn > 20 s), charset after
      `normalize()`, digits dropped from training, speaking rate outside 4–30 chars/s (initial guess; set the
      bounds from the real histogram and record it), CTC feasibility (`len(tokens) ≤ T_enc` + repeat room),
      sentence-overlap hash against every dev/test norm text, corrupt/all-zero/clipping > 1%.
- [ ] **Implement** filters + build report counting drops by reason.
- [ ] **Run**; **Commit** `feat(data): add filters and build report`.

### Task 2.3 — Tokenizer

**Files:** create `stt/tokenizer.py`, `data/en_terms.txt` (curated 1–2k English work/tech terms); test
`tests/test_tokenizer.py`. **Spec:** `docs/02` §5, `docs/03` §6.

- [ ] **Create `data/en_terms.txt`** (spots, apps, brands, acronyms, tech verbs as Italians write them).
- [ ] **Write failing tests:** SentencePiece BPE `V=1024`, `model_type=bpe`, `character_coverage=1.0`, no
      BOS/EOS/PAD, `unk_id=0`; encode/decode round-trip on fixtures; `unk` never produced for charset-valid text;
      CTC mapping `blank=0`, `ctc_id = sp_id + 1`, `V_ctc = V_sp + 1`; exported `vocab.json` pieces by CTC id;
      determinism given the corpus.
- [ ] **Implement + train** on: training transcripts (IT-weighted ≈ 70/30 by text), the mixed pool, and the
      English terms multiple times (up-weight). Test sentences never enter the tokenizer text; term-level
      overlap is allowed. Emit stats into the build report: tokens/word, tokens/sec, share of English terms
      tokenizing to ≤ 3 pieces.
- [ ] **Run**; **Commit** `feat(tokenizer): add shared IT+EN BPE tokenizer and export`.

### Task 2.4 — `stt/features.py` + mel parity

**Files:** create `stt/features.py`, `stt/assets/mel_filters.bin` (80×201 float32, ≈ 64 KB, from `openai-whisper`'s
`mel_filters.npz`, MIT); test `tests/test_features.py`, fixtures `tests/fixtures/clips/*.wav` (≤ 300 KB total).
**Spec:** `docs/02` §2.

- [ ] **Write failing tests:** shape `[80, N//160]`; parity vs `whisper.log_mel_spectrogram` on the fixture clips
      (max-abs ≤ 1e-4 after accounting for the per-clip clamp); padded-vs-unpadded per-clip max equivalence.
- [ ] **Implement** exactly per the `02` §2 table: 400/160 periodic Hann, center with 200-sample reflect pad,
      drop the last STFT frame, power spectrum, Whisper Slaney filterbank, `log10(max(mel, 1e-10))`, per-clip
      `max(x, x.max() − 8)`, `(x + 4) / 4`.
- [ ] **Run**; **Commit** `feat(features): add reference log-mel with Whisper parity tests`.

### Task 2.5 — Mel shards + index

**Files:** create `stt/data/build_mel.py`; test `tests/test_shards.py`. **Spec:** `docs/03` §5.

- [ ] **Implement:** manifest → decode (`soundfile`/`ffmpeg`) → 16 kHz mono → `features.log_mel` → float16 shard
      `.npy` (1–2 GB) + index JSON (`id`, offset, frames, duration); `--workers`; **delete raw audio after the
      shard passes a spot check**. Plan on 57.6 MB per audio hour (≈ 26 GB for 450 h).
- [ ] **Test:** build a tiny shard from the fixtures, read back via mmap, values match `features.log_mel`.
- [ ] **Build the real cache** for the full H2 mixture (≈ 440 h) and store the build report (hours per source
      after filters).
- [ ] **Commit** `feat(data): add mel shard builder` (+ build report JSON).

### Task 2.6 — Dataset + augmentation

**Files:** create `stt/data/dataset.py`; test `tests/test_dataset.py`. **Spec:** `docs/04` §3–4.

- [ ] **Write failing tests:** duration bucketing (≤ 320 audio-seconds and ≤ 64 clips per batch); key-padding
      mask correctness (masked batch == per-clip loop within tolerance); SpecAugment bounds (freq: 2 masks ≤ 15
      bins; time: ≤ 5 masks, width ≤ 40 frames, ≤ 20% total); mel tempo stretch factor ∈ [0.9, 1.1] by
      interpolation; seed determinism; dev/eval loader never augments.
- [ ] **Implement** the mmap loader + batch sampler + augmentations, with a `light` flag (H1: frequency masks only).
- [ ] **Run**; **Commit** `feat(data): add mmap dataset, bucketing and mel augmentation`.

### Task 2.7 — E-002, dataset build

- [ ] **Write E-002**: hours per source after filters (expect ≈ 340 h IT + ≈ 100 h EN), mixture weights fixed
      for H2 (IT 0.7 / EN 0.3 by audio time), tokenizer stats, shard sizes, any decision recorded in 2.1
      (EN source, noise license).
- [ ] **Commit** `docs(experiments): add E-002 dataset build`.

---

## Phase M3 — Model, conversion, export, early browser smoke (H4)

### Task 3.1 — `stt/model.py`

**Files:** create `stt/model.py`; test `tests/test_model.py`. **Spec:** `docs/02` §3–4, §8.

- [ ] **Write failing tests:** parameter count == `8_027_009` for V_sp = 1024 (V_ctc = 1025; `conv1` 92,544; `conv2` 442,752;
      4 blocks × 1,774,080 = 7,096,320; final LN 768; head 394,625); output length `T_enc = ceil(T_mel/2)`;
      batched+masked logits equal single-clip logits within tolerance; no `transformers` import at runtime.
- [ ] **Implement** the ~100-line plain-PyTorch encoder: conv1 (k3, GELU), conv2 (k3, stride 2, GELU), sinusoidal
      positions sliced to `T_enc` (generated, not stored), 4 pre-LN blocks (width 384, 6 heads × 64, q/v/out
      bias, **no k bias**, q/k scaled by 64^−0.25, bidirectional), MLP 384→1536→384, final LN; CTC head
      `Linear(384, V_ctc)`; `forward(mel, mel_lengths) -> (log_probs, out_lengths)` with the key-padding mask for
      training and batch-1 inference path.
- [ ] **Run**; **Commit** `feat(model): add Whisper-tiny encoder + CTC head`.

### Task 3.2 — `stt/convert_whisper.py` + parity

**Files:** create `stt/convert_whisper.py`; test `tests/test_convert.py`. **Spec:** `docs/02` §3.

- [ ] **Implement** the one-off mapping `openai/whisper-tiny` (MIT) → own module; save
      `models/whisper_tiny_encoder.pt` (checksums recorded; the file itself is gitignored/heavy, re-fetched by script).
- [ ] **Write the parity test** (dev extra `transformers`): own encoder vs `transformers.WhisperEncoder` on a
      30 s padded input, fp32, max-abs ≤ 1e-4 — offline on random weights (CI). *Amended 2026-10-19 after
      measuring the released checkpoint:* the strict 1e-4 holds only there; on real weights the reference is the
      eager path with the checkpoint's position table substituted, gate 1e-3 (measured 5.2e-4, relative 2.8e-5)
      because `transformers` itself moves 2.2e-3 between its own sdpa/eager kernels and the checkpoint's
      positions differ from the formula by 3.03e-4 (see `docs/02` §3 and `stt/convert_whisper.py` docstring).
- [ ] **Run**; **Commit** `feat(convert): add whisper-tiny weight mapping with parity test`.

### Task 3.3 — `stt/decode.py` — greedy

**Files:** create `stt/decode.py`; test `tests/test_decode.py`. **Spec:** `docs/02` §6.

- [ ] **Write failing tests** on hand-built log-probs: repeats collapse, blanks drop, `▁` → space, empty input →
      empty string.
- [ ] **Implement** `greedy(log_probs, length) -> str` (argmax → collapse → join pieces).
- [ ] **Run**; **Commit** `feat(decode): add greedy CTC decode`.

### Task 3.4 — `stt/export.py` — ONNX + int8

**Files:** create `stt/export.py`; test `tests/test_export.py`. **Spec:** `docs/06` §3–4.

- [ ] **Write the failing tests:** ONNX fp32 vs PyTorch logits ≤ 1e-3 max-abs on 1/5/10/29 s clips; the
      sinusoid table is generated in-graph (`Range`/`Sin`/`Cos`) or shipped as fp16 (≤ 1.1 MB) — never 2.3 MB
      fp32; `release/` writer emits `model.int8.onnx`, `vocab.json`, `mel_filters.bin`, `config.json` (with
      sha256 lines), `NOTICE`.
- [ ] **Implement** export (opset ≥ 17, batch 1, dynamic time) and `onnxruntime.quantization.quantize_dynamic`
      (int8 weights, per-channel MatMul; convs may stay fp32, +≈ 1.6 MB). Fallbacks when int8 misses the fp32
      noise interval: static quantization with ≈ 200 dev clips → fp32 LayerNorm path → fp16 weights (≈ 16–17 MB).
- [ ] **Run**; **Commit** `feat(export): add ONNX export and int8 quantization`.

### Task 3.5 — `web/` smoke + H4 early RTFx → E-003

**Files:** create `web/index.html`, `web/mel.js`, `web/decode.js` (greedy), `web/ort.js`, `web/server.py`
(COOP/COEP headers), `web/tests/smoke.spec.ts` (Playwright). **Spec:** `docs/06` §2, §5, §7.

- [ ] **Implement** the JS mel port (direct 400-point real DFT with precomputed 201×400 cos/sin matrices; **no
      512-point FFT**), the WASM runner (`executionProviders: ['wasm']`, SIMD, `numThreads = min(4, hw)` when
      isolated), and a file-input smoke page that prints the transcript, timings and RTFx.
- [ ] **Playwright smoke test** on a fixture clip through a **random-weight** int8 export → transcript parity
      expectations are trivial but the graph runs; **measure browser RTFx now** (H4 early).
- [ ] **If RTFx < 10×:** escalate (threads/SIMD → stride-2 pooling to 25 Hz variant per `02` §7) **before** H2;
      record the decision.
- [ ] **Commit** `feat(web): add browser smoke runtime` + **record E-003** (export + early RTFx).

---

## Phase M4 — H1 probe (frozen encoder)

### Task 4.1 — `stt/train.py` core

**Files:** create `stt/train.py`, `runs/E-004/config.yaml`; test `tests/test_train.py`. **Spec:** `docs/04` §2–3, §6.

- [ ] **Write failing tests** (tiny synthetic shard, CPU): one optimizer step runs; resume from a checkpoint
      reproduces the loader position; JSONL log line schema; non-finite guard triggers.
- [ ] **Implement** the loop: `--config runs/<id>/config.yaml --device {mps,cuda,cpu}`; device-agnostic; bf16
      autocast on encoder/head matmuls, fp32 `log_softmax` + `CTCLoss(blank=0, zero_infinity=True)`; AdamW;
      warmup + cosine; grad clip 1.0; checkpoint every 20 min (model, optimizer, scheduler, scaler, loader
      position, RNG) keeping last 3 + best; JSON log per 50 steps; abort > 20 non-finite steps; skip-and-count
      `inf` batches; **start-of-run parity guard**: same small batch, CTC loss on MPS vs CPU within 1e-3 fp32.
- [ ] **Run**; **Commit** `feat(train): add training loop with resume and guards`.

### Task 4.2 — H1 run (truncated vs padded)

**Files:** `runs/E-004/config.yaml` (+ outputs). **Spec:** `docs/04` §5-H1.

- [ ] **Config:** encoder frozen, `Linear(384, V_ctc)` trained; ≈ 100 h Italian (Common Voice + MLS); 3 epochs;
      peak LR 1e-3; light SpecAugment; two arms: `input: truncated` and `input: padded30` (CTC sees only real
      frames). Compute < 1 h per arm (padded ≈ 2×).
- [ ] **Run both arms**; evaluate dev WER (greedy) with CIs + CTC loss curves; log throughput and memory.
- [ ] **Decide** (H1 gate): head learns (dev WER clearly below the blank baseline and still falling) and
      truncated is not worse than padded beyond the noise interval. If truncated fails → H2 trains on padded
      windows and the decision is recorded.
- [ ] **Commit** `runs: E-004 H1 probe` + **record E-004**.

---

## Phase M5 — H2 full fine-tune

### Task 5.1–5.3 — Run, track, evaluate

**Files:** `runs/E-005/config.yaml` (+ outputs). **Spec:** `docs/04` §5-H2.

- [ ] **Config (starting points, tune on dev only):** init = pretrained encoder + H1 head (or freeze encoder
      for the first 1,000 steps); mixture IT 0.7 / EN 0.3; AdamW betas (0.9, 0.98), eps 1e-8, wd 0.01; peak LR
      head 1e-3, encoder 1e-4; warm-up 1,500 steps → cosine to 5%; ≈ 320 audio-seconds per batch (≈ 5,000
      steps/epoch at 440 h); 8 epochs with early stop on dev; grad clip 1.0; bf16 autocast, fp32 CTC; average
      the last 3–5 checkpoints.
- [ ] **Run** (≈ 5–10 h on the M5; `caffeinate -i` on power; watch memory — lower the audio-seconds budget first
      if pressured). Dev WER every 2,000 steps, IT and EN sets separately; selection metric = macro-average.
- [ ] **Final evaluation:** FLEURS IT/EN test, Common Voice IT test, mixed test T, 1 h noise set (hard bar: zero
      non-music words), quick browser RTFx with a current export; paired-bootstrap differences vs the H0 numbers.
- [ ] **Gate:** beats Whisper-tiny on Italian FLEURS **and** Common Voice dev. If not, run the escalation ladder
      in order, one `EXPERIMENTS.md` entry per attempt:
      1. more epochs / LR tuning;
      2. extra head blocks (+2 blocks, 11.6 M — `02` §7);
      3. BPE size 512 / 2048 ablation;
      4. **Whisper-base fallback** (≈ 20.5–23.5 MB — explicit decision to relax the ceiling to ≈ 25 MB).
- [ ] **Commit** `runs: E-005 H2 full fine-tune` + **record E-005** (+ escalation entries).

---

## Phase M6 — H3 code-switch data

**Spec:** `docs/04` §5-H3, `docs/03` §6–7, §10.

- [ ] **Task 6.1 — H3a (gold):** continue from the H2 checkpoint; 1–2 epochs; LR 5e-5 (encoder) / 3e-4 (head);
      mixture by audio time IT 0.5 / EN 0.2 / gold 0.3 (S1+S2 script R). Measure mixed WER per category and
      per speaker group (S1+S2 seen vs S3 held out), ETA, FLEURS regression. **Commit** + **record E-006**.
- [ ] **Task 6.2 — H3b (utterance splicing):** implement `stt/data/splice.py` (concatenate one IT + one EN clip,
      transcript = concatenation; loudness match; tests on synthetic WAVs); mixture IT 0.4 / EN 0.15 / gold 0.2
      / spliced 0.25. Same measures. **Commit** + **record E-007**.
- [ ] **Task 6.3 — H3c (only if H3b falls short):** word-level splicing or TTS-synthesized audio (`docs/03` §10).
- [ ] **Task 6.4 — Gate:** keep a data source only if it improves mixed WER or ETA beyond the noise interval
      without hurting FLEURS IT/EN. Freeze the v0.1 candidate: `best_avg.pt` + the chosen mixture. **Commit** +
      **record** the decision.

---

## Phase M7 — Final export, web runtime, demo

### Task 7.1 — Final int8 export + gates

**Spec:** `docs/06` §3–4, §5.

- [ ] **Export** the frozen candidate; verify export gate: ONNX fp32 vs PyTorch ≤ 1e-3; int8 greedy transcripts
      within the noise interval of fp32 on dev and the mixed test; total download ≤ 20 MB; every hash in
      `config.json` verified by the loader. **Measure the int8 gate on the target platform(s):** x86_64
      onnxruntime int8 kernels deviate markedly more than ARM64 (measured 2026-10-19, see the E-003 addendum)
      — a Mac-only pass is not a pass.
- [ ] **Commit** `feat(export): freeze v0.1 release folder` + metrics.

### Task 7.2 — JS ports (mel parity, decode, hotword beam)

**Spec:** `docs/02` §6, `docs/06` §5.

- [ ] **Mel parity:** Python reference vs JS on fixture clips (1 s, 5 s, 29 s, silence, clipping), ≤ 1e-3 max-abs
      after normalization.
- [ ] **Decode ports:** greedy + **hotword beam search** (CTC prefix beam over a trie of hotword token sequences,
      backoff of partial bonuses; defaults beam 8, bonus 2.0, tuned on dev). Python is the reference; JS validated
      on fixture log-probs.
- [ ] **Commit** `feat(web): add mel and CTC decode ports with parity tests`.

### Task 7.3 — Demo page

**Spec:** `docs/06` §2, §6.

- [ ] **Implement** record-then-transcribe: `getUserMedia` + AudioWorklet capture; `OfflineAudioContext` resample
      to 16 kHz mono; chunking > 30 s (25 ms energy windows, cut at the quietest point within the last 5 s, min
      1 s); file input alternative; hotword list input; timings (feature/inference/decode) + RTFx display; error
      UX table (`06` §6: no mic, no SIMD, not isolated, hash mismatch, silence → empty, > 30 s, bad hotwords).
- [ ] **Playwright e2e:** fake audio device (`--use-fake-device-for-media-stream
      --use-file-for-fake-audio-capture=clip.wav`) or file input, ≈ 20 clips **through the browser path** —
      transcripts equal to Python's on the same clips.
- [ ] **Commit** `feat(web): add record-then-transcribe demo with e2e tests`.

### Task 7.4 — Browser RTFx benchmark + noise via browser

- [ ] **Benchmark protocol** (`docs/05` §3): 5/10/20 s clips, 3 warm-ups then median of 10 runs, Chrome and
      Safari on the M5, 1 and 4 threads; record load time and memory. **Bar: ≥ 10×.**
- [ ] Re-run the noise set through the browser path (the demo must emit nothing on non-music silence/noise).
- [ ] **Commit** `docs: record browser RTFx and noise results` + **record E-00N**.

---

## Phase M8 — Release

### Task 8.1 — HF model repo, model card, NOTICE, data card

- [ ] **Publish** the `release/` folder to the Hugging Face model repository (Apache-2.0): `model.int8.onnx`,
      `vocab.json`, `mel_filters.bin`, `config.json`, `NOTICE`.
- [ ] **NOTICE:** Whisper copyright/permission notice verbatim (MIT) + dataset attributions (Common Voice, MLS,
      LibriSpeech, FLEURS, noise set) + the “derived from `openai/whisper-tiny`, encoder fine-tuned” disclosure.
- [ ] **Model card + data card:** derivation, training data summary, per-speaker consent decision, LLM-generated
      text terms checked against provider terms, evaluation table with the H0-relative results.
- [ ] **Commit** `docs: add model card and NOTICE`.

### Task 8.2 — Repo finish line

- [ ] Update `README.md` (drop “design phase”, add measured results and the demo link); complete
      `EXPERIMENTS.md` (all entries, cost ledger — expected 0 USD; the stop-at-8-USD rule respected); tag `v0.1.0`.
- [ ] **Commit** `docs: finalize v0.1 release`.

### Task 8.3 — Acceptance checklist

- [ ] Fill the Definition-of-done table above with **measured** numbers and links to `runs/*/metrics.json`;
      check the stop/pivot criteria (`docs/08` §4) and write the honest verdict (shipped / stretch missed /
      pivot taken). **Commit** `docs: fill v0.1 acceptance checklist`.

---

## Cross-cutting rules

- **Commits:** one per task, conventional commits; `ruff` + `pytest` green before every commit; CI must stay green.
- **Reproducibility:** every run records git SHA, torch version, device, seed in `runs/<id>/config.yaml`; MPS CTC
  backward is non-deterministic — say so in the entry (repeatable in distribution, not bit-exact).
- **Data discipline:** test/dev data never trained on; every T sentence is human-reviewed before recording; audio
  and checkpoints never enter git.
- **Budget:** ledger in every entry; stop and review at 8 USD (expected: no paid compute at all).
- **Open questions to resolve at their moment** (`docs/08` §2): Italian-accented EN slice + MUSAN license (M2);
  speakers and consent (M1); BPE size (after H2, only if time); target browsers (M7); LLM text terms (M8);
  Whisper-base fallback acceptance (only if the ladder reaches step 4).
- **Stop/pivot criteria** (`docs/08` §4): re-plan if browser RTFx < 5× after pooling or if both int8 and fp16
  fail the export gate; revisit approach C only if tiny and base encoders both fail the mixed-set bar.

**First action:** Task 0.1.
