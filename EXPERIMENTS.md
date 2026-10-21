# Research log

Every experiment records: hypothesis, model configuration, dataset, training parameters,
hardware, compute and cost, metrics, result, conclusion (template:
`docs/07-repo-layout-and-roadmap.md` §4). Architecture changes are never made without an entry here.

Compute ledger: 0 USD spent of the 10 USD budget; stop and review at 8 USD.

---

## E-000 — Mac feasibility benchmark (2026-10-18)

- **Date**: 2026-10-18
- **Hypothesis**: the M5 MacBook Pro 16 GB can fine-tune the shippable ≈ 8.03 M model
  (Whisper-tiny encoder + CTC head) with PyTorch/MPS within the project's time budget.
- **Model configuration**: a throw-away stand-in with the model's exact shape and random weights
  (Whisper-tiny encoder shape + CTC head, V = 1024, 8.03 M parameters, built from
  `nn.TransformerEncoderLayer`); full fine-tuning step = forward + CTC loss + backward + AdamW,
  on 32 synthetic clips of 10 s per step (mel `[32, 80, 1000]`, 500 encoder frames, 50 targets each).
- **Dataset**: synthetic clips only.
- **Training parameters**: AdamW, single step per setting; fp32 vs bf16 autocast; torch 2.14.1;
  device `mps` and CPU compared.
- **Hardware**: Apple M5, 16 GiB unified memory, macOS; scratch virtualenv (not this repository).
- **Compute and cost**: ≈ 0 (local machine).
- **Metrics**:

| Setting | CTC loss on | ms/step | Audio-seconds/s | Audio-hours/hour | Peak GPU memory |
|---|---|---|---|---|---|
| fp32 | MPS | 578 | 554 | ≈ 554 | 4.3 GiB |
| bf16 autocast | MPS | **394** | **812** | ≈ **812** | 4.4 GiB |
| fp32 | CPU | 577 | 554 | ≈ 554 | 5.4 GiB |
| bf16 autocast | CPU | 432 | 741 | ≈ 741 | 5.4 GiB |

- **Findings**: CTC loss forward **and** backward ran natively on MPS in torch 2.14.1 (no CPU
  fallback, `PYTORCH_ENABLE_MPS_FALLBACK` unset); bf16 autocast ≈ 1.4× faster than fp32 and
  error-free; peak memory ≈ 4.4 GiB at 320 audio-seconds per step; one epoch over 100 h of audio
  ≈ 7–11 minutes. Not yet verified (checked in the first real runs): real Whisper weights, the real
  data loader, MPS↔CPU CTC parity on a small batch, sustained thermal throttling, throughput with
  key-padding masks and real length distributions.
- **Result**: H2 (≈ 440 h × 8 epochs ≈ 3,500 audio-hours) should take ≈ 5–10 h on this machine;
  H1 under 1 h.
- **Conclusion**: hypothesis confirmed. Torch version pinned to `==2.14.*`; device `mps` with bf16
  autocast is the default for training.

---

## Planned experiments

| ID | Milestone | What | Design ref |
|---|---|---|---|
| E-003 | M3 | H4 early: random-weight int8 ONNX export, browser RTFx smoke (de-risks the ≥ 10× bar) | `docs/06`, `docs/04` §5-H4 |
| E-004 | M4 | H1 probe: frozen encoder + linear head, truncated vs padded 30 s | `docs/04` §5-H1 |
| E-005 | M5 | H2 full fine-tune on the IT+EN mix; gate vs Whisper-tiny (escalation ladder if it fails) | `docs/04` §5-H2 |
| E-006 | M6 | H3a: gold mixed recordings (S1+S2 script R) | `docs/04` §5-H3 |
| E-007 | M6 | H3b: gold + utterance-level splicing | `docs/04` §5-H3 |

---

## E-001 — H0 baselines: Whisper tiny (first pass)

- **Date**: 2026-10-19
- **Hypothesis**: Whisper tiny is a weak Italian baseline and collapses on English when the language is forced to Italian; auto language detection matches forced Italian on FLEURS-IT and wins on FLEURS-EN. The per-language best tiny variant becomes the concrete H2 bar.
- **Model configuration**: `openai-whisper` tiny (39 M), greedy (beam 1), fp32 on CPU, `padded_30s: true`, no hotword interface. Variants: language **auto** vs forced **it**.
- **Dataset**: `fleurs_it_test` (697 utterances, 16,035 reference words) and `fleurs_en_test` (525 utterances, 11,438 reference words), evaluation-only, scored under our `normalize()`.
- **Procedure**: `baselines/whisper_runner.py` → hypothesis files → `stt/eval.py` (corpus WER, seeded paired bootstrap with 1,000 resamples). Paired differences are auto − forced-`it` on the same utterances.
- **Hardware**: Apple M5, CPU only.
- **Compute and cost**: ≈ 10 minutes wall-clock; 0 USD (ledger 0 of 10).
- **Metrics**:

| Variant | FLEURS-IT WER [95% CI] | FLEURS-EN WER [95% CI] | ins/100w IT | ins/100w EN | RTFx IT | RTFx EN |
|---|---|---|---|---|---|---|
| tiny, auto, greedy | 0.3212 [0.3096, 0.3338] | **0.1386** [0.1283, 0.1504] | 5.16 | 2.77 | 53.3 | 41.8 |
| tiny, forced `it`, greedy | **0.3209** [0.3091, 0.3333] | 0.8190 [0.7863, 0.8490] | 5.13 | 2.43 | 65.5 | 8.7 |

Paired auto − forced-`it`: FLEURS-IT **+0.03 pp** (tie — CI includes 0); FLEURS-EN **−68.1 pp** (not a tie).

- **Findings**: forcing Italian on Italian audio changes nothing measurable (auto detects Italian anyway); forcing it on English breaks the model (81.9 % WER) — exactly the failure mode the one-language CTC vocabulary is meant to avoid. The forced-Italian English run is also ~7× slower (RTFx 8.7): the decoder hallucinates longer output. Greedy decoding is very fast overall (RTFx 41.8–65.5 on CPU).
- **Result**: the H2 gate now has concrete numbers: beat **32.09 %** WER on FLEURS-IT and **13.86 %** on FLEURS-EN under our normalization.
- **Conclusion**: hypothesis confirmed. Remaining H0 rows: Whisper base, beam 5, Cactus Whistle, Parakeet v3, CV-IT test, the mixed set T (once recorded) and the noise set; beam 5 and base will be filled before the H2 gate.

---

## E-002 — Dataset build (M2)

- **Date**: 2026-10-19
- **Hypothesis**: the open sources downloaded at M2 (MLS IT, LibriSpeech train-clean-100 + dev-clean, FLEURS IT/EN dev+test) yield ≈ 170 h Italian + ≈ 100 h English of clean training audio and complete dev/test manifests within the `docs/03` §3 drop rules.
- **Model configuration**: n/a (data only).
- **Dataset**: MLS Italian (OpenSLR 94, CC-BY 4.0), LibriSpeech train-clean-100 and dev-clean (OpenSLR 12, CC-BY 4.0), FLEURS `it_it`/`en_us` dev+test (CC-BY 4.0, evaluation only). Common Voice IT 27 (CC0) still to download — its ≈ 170 h Italian slot is open.
- **Procedure**: `stt/data/build_manifests.py` — dev/test sets first (they define the overlap hashes), then train sets sampled deterministically to the target hours (`sample_by_duration`, seed 0) and filtered (duration 1–30 s, digits, charset, speaking rate 4–30 chars/s, `norm_hash` overlap). Audio-health checks (corrupt/silent/clipping) are deferred to the mel shard build, which decodes every file anyway.
- **Hardware**: Apple M5, CPU only.
- **Compute and cost**: ≈ 4 minutes wall-clock; 0 USD (ledger: 0 of 10 USD).
- **Metrics**:

| Set | Kept / rows | Hours | Dropped (reason: n) |
|---|---|---|---|
| `mls_it_train` (sampled from 247.4 h available) | 40,953 / 40,982 | **169.89** | speaking_rate: 29 |
| `ls_en_train` (sampled from 100.6 h available) | 28,370 / 28,372 | **100.00** | speaking_rate: 2 |
| `mls_it_dev` / `mls_it_test` | 1,248 / 1,262 | 5.18 / 5.27 | — |
| `ls_en_dev` | 2,691 / 2,703 | 5.31 | duration: 9, speaking_rate: 3 |
| `fleurs_it_dev` / `fleurs_it_test` | 334 / 697 | 1.25 / 2.74 | digits: 51 / 162, duration: 6 / 6 |
| `fleurs_en_dev` / `fleurs_en_test` | 343 / 525 | 0.88 / 1.39 | digits: 51 / 121, charset: 1 |

- **Findings**: FLEURS ships **headerless** TSVs with a non-unique `id` column (the id names the sentence, not the recording — several recordings share it); the manifest id is therefore the filename stem (verified unique). FLEURS transcripts contain digits far more often than expected (≈ 30–35 % of rows dropped as digits); FLEURS is evaluation-only and its numbers are written as digits, while the recorded test scripts spell them as words — metric comparability is handled by `normalize()`. MLS IT offers more than the estimated 214 h (247.4 h survive the text filters). **Zero** train rows dropped for overlap with dev/test.
- **Result**: 269.9 h of training audio (169.9 h IT + 100.0 h EN) plus dev/test manifests; H0/H1/H2 are unblocked. CV IT 27 can add ≈ 170 h Italian later without reworking anything.
- **Mel cache (Task 2.5, 2026-10-19)**: `stt/data/build_mel.py` over all 9 manifests → **291.90 h** cached as float16 shards in `data/shards/<set>/` (18 `.npy` files, **16 GB**): `mls_it_train` 169.89 h / 7 shards / 9.1 GB, `ls_en_train` 100.00 h / 4 shards / 5.4 GB, dev/test sets (mls dev+test 10.45 h, ls dev 5.31 h, FLEURS 6.26 h) ≈ 1.2 GB. **0 decode errors** — the deferred audio-health checks (corrupt/silent/clipping) found nothing to drop. Spot check: 5 random clips per set re-computed from raw audio and compared bit-exact to the mmap read → all 9 sets pass. Wall-clock ≈ 90 s on 8 workers (M5, numpy FFT); peak parent memory bounded by the batched consumption added to `build_shards`. **Raw training audio deleted the same day after a second, explicit spot check (5 clips per set, bit-exact):** MLS IT (15 GB) and LibriSpeech (6.6 GB), 72,524 files — both re-downloadable via `data/raw/fetch.sh`. FLEURS (1.7 GB) is kept until the remaining H0 baselines (task 1.4: Whisper base/beam 5, Whistle, Parakeet) have run from it.
- **Conclusion**: hypothesis confirmed, with one open scope note (CV IT pending). Proceed to E-001 (H0 baselines).

> **Addendum (2026-10-21) — Common Voice IT 27.0 joins the mix.** Archive `cv-corpus-27.0-2026-09-11`
> (9.8 GB, extracted to `data/raw/cv_it_27/it`, 11 GB). CV IT train has **256.7 h** available (not
> 364 h: that figure includes non-train splits/other), sampled deterministically to 170 h:
>
> | Set | Kept / sampled | Hours | Dropped |
> |---|---|---|---|
> | `cv_it_train` | 113,118 / 115,076 | **167.79** | speaking_rate 1,216, overlap 593, charset 146, empty 2, duration 1 |
> | `cv_it_dev` | 14,697 / 15,185 | 24.86 | speaking_rate 460, charset 27, duration 1 |
> | `cv_it_test` | 14,643 / 15,183 | 25.81 | speaking_rate 518, charset 22 |
>
> The 593 train overlaps are the second-line guard at work: the incremental build seeds the train
> filter with the existing dev/test manifests **and the T script hashes**, so no training row can
> share a normalized sentence with an evaluation set. The official splits are speaker-disjoint
> (0 client ids shared between train/dev/test). `read_cv_tsv` now uses the **clip stem** as id: the
> old `cv-<client_id>` collided for every clip read by the same speaker.
>
> Mel shards: 167.31 h train (7 files, 9.0 GB), 24.83 h dev, 25.78 h test (1.3 + 1.4 GB), **0 decode
> errors**, spot check pass — 217.92 h / 11.7 GB added, cache now **509.8 h / 27 GB**. The manifest
> build needed an incremental path (`--only`, `existing_eval_hashes`, merged report) because the
> MLS/LibriSpeech raw audio is deleted; building every set again would require re-downloading 22 GB.
>
> H2 data is now **437.7 h** (337.7 IT + 100 EN); the "70/30 by audio time" label in `docs/04` is
> 77/23 in the design's own absolute budget (≈340 + 100 h), and H2 uses all of it.

---

## E-003 — H4 early: int8 export + browser RTFx (2026-10-19)

> **Addendum (2026-10-19, CI):** the int8 numbers above are macOS ARM64. Controlled cross-platform
> check (same seeded weights, byte-identical fp32 graph; macOS arm64 vs Linux x86_64 via Docker,
> onnxruntime 1.31.0 both): the x86_64 int8 kernels deviate ≈40× more on random-weight tiny models
> (max |Δ log_probs| 0.85 vs 0.02; top1 0.28–0.36 vs ≥0.98) — a kernel-behavior difference, not a
> file or code difference. Unit tests must therefore bound int8 noise per platform (see
> `tests/test_export.py` constants); **Task 7.1's real int8 gate must be measured on the target
> platform**, not only on the Mac.

- **Date**: 2026-10-19
- **Hypothesis**: with a random-weight int8 ONNX export the browser WASM path (WAV → JS log-mel →
  `onnxruntime-web` → greedy decode) runs end to end, and RTFx on the M5 is ≥ 10× (`docs/04` H4,
  `docs/06` §7) — measured **before** fine-tuning, so a miss could still change the architecture.
- **Model configuration**: random weights in the default Moth shape (Whisper-tiny encoder + linear
  CTC head, V_sp = 1024, V_ctc = 1025, 8,027,009 params); `stt/export.py` (opset 17, batch 1,
  dynamic time, sinusoids generated in-graph) + `quantize_dynamic` (QInt8, per-channel). Artifacts:
  `model.int8.onnx` 8.23 MB, `vocab.json` 0.02 MB, `mel_filters.bin` 0.06 MB → **8.32 MB total**
  (< 20 MB ceiling). Export gates: fp32 vs PyTorch max |Δ| 3.8e-6 (gate 1e-3); int8 vs PyTorch
  max |Δ| 2.9e-2 with 99.7 % top-1 agreement.
- **Dataset**: none — the synthetic 5 s fixture `web/tests/fixtures/clip.wav`, tiled to 10/20 s.
- **Training parameters**: n/a (random weights).
- **Hardware**: Apple M5, 16 GB; Chromium 156.0.8078.4 headless (Playwright); `onnxruntime-web`
  1.30.0 WASM, SIMD, cross-origin isolated (COOP/COEP); torch 2.14.1 export; macOS.
- **Compute and cost**: ≈ 2 minutes wall-clock (export + smoke + table); 0 USD (ledger: 0 of 10).
- **Metrics** (`runs/E-003/rtfx.json`; 3 warm-ups + median of 10 runs):

| Clip | Threads | Feature | Inference | Decode | Total | RTFx |
|---|---|---|---|---|---|---|
| 5 s | 4 | 39.3 ms | 28.2 ms | 0.16 ms | 67.6 ms | **74.0×** |
| 10 s | 4 | 78.5 ms | 61.4 ms | 0.32 ms | 140.8 ms | **71.0×** |
| 20 s | 4 | 161.7 ms | 146.4 ms | 0.63 ms | 309.3 ms | **64.7×** |
| 5 s | 1 | 36.6 ms | 95.1 ms | 0.15 ms | 131.9 ms | 37.9× |
| 10 s | 1 | 75.0 ms | 221.5 ms | 0.32 ms | 302.6 ms | 33.1× |
| 20 s | 1 | 145.3 ms | 465.2 ms | 0.58 ms | 611.1 ms | 32.7× |

  Also: every `sha256:` line of `config.json` verified in the page; model load 135 ms; manual mel
  parity of `web/mel.js` vs `stt/features.py` on the fixture max |Δ| 1.1e-6 (the committed parity
  test is Task 7.2).
- **Findings**: the direct 400-point DFT costs ≈ 40 ms per 5 audio-seconds (≈ 120× real time alone)
  and is the larger half of the browser time; int8 inference scales linearly with clip length;
  4 threads give ≈ 2.4× over single-thread and even the non-isolated fallback (1 thread) stays
  ≈ 33×. Headless Chromium only — Safari and recorded clips come with the M7 benchmark (`docs/05` §3).
- **Result**: browser RTFx ≈ 65–74× (4 threads) and ≈ 33–38× (1 thread) on the random-weight int8
  export; both far above the 10× bar.
- **Conclusion**: hypothesis confirmed; **no pooling escalation needed** (`docs/06` §7). Proceed to
  H1 (E-004) after the H0 baselines (E-001).

---

## E-004 — H1 probe: frozen encoder, truncated vs padded (2026-10-19)

- **Date**: 2026-10-19
- **Hypothesis**: a linear CTC head on the **frozen** Whisper-tiny encoder learns Italian from MLS
  (dev WER clearly below the blank baseline, still falling) and truncating clips to their real
  length is not worse than padding every clip to 30 s (`docs/04` §5-H1).
- **Model configuration**: pretrained `openai/whisper-tiny` encoder (4 blocks, width 384, 6 heads,
  frozen) + `Linear(384, 1025)` head, V_sp = 1024 (BPE task 2.3), V_ctc = 1025; **394,625 trainable**
  parameters of 8,027,009. `stt/train.py`; bf16 autocast, fp32 CTC (blank 0), AdamW, warm-up 500 →
  cosine to 5%, grad clip 1.0, light SpecAugment, MPS↔CPU CTC parity guard at start (Δ 3.8e-6).
- **Dataset**: `mls_it_train` (MLS IT, 169.9 h, 40,953 clips), dev `mls_it_dev` (5.18 h, 1,248
  clips, 40,407 reference words). Test and FLEURS never touched.
- **Training parameters**: three arms, same seed 0, 3 epochs unless noted.

| Arm | Input | Batch | Steps | Epochs | Wall time |
|---|---|---|---|---|---|
| A | `truncated` | 320 s, 64 clips | 5,871 (1,957/epoch) | 3 | 23 min |
| B | `padded30` | 96 s, 8 clips | 20,970 (7,055/epoch) | 3 | 111 min |
| C (control) | `truncated` | 96 s, 8 clips | 6,990 (6,990/epoch) | 1 | 11 min |

  B and C use smaller batches on purpose: padded 30 s windows cost ~4× the attention memory per
  clip, so the original 320 s / 64 clip batch needed ~17 GB (measured: 16.9 GB RSS and ~15 audio-s/s
  with heavy swapping) and was stopped. C exists only to separate *padding* from *more steps*.
- **Hardware**: Apple M5, 16 GB, AC power, `caffeinate -i`; torch 2.14.1, MPS, bf16; macOS.
- **Compute and cost**: 145 min total wall-clock on the local machine; **0 USD** (ledger: 0 of 10).
- **Metrics** (`runs/E-004/*/dev.metrics.json`, greedy, bootstrap 1,000):

| Arm | Dev WER | 95% CI | Ins/100 w | Audio-s/s | Peak mem |
|---|---|---|---|---|---|
| A truncated, 3 ep | 0.7577 | [0.7522, 0.7630] | 2.05 | ≈590 | 4.3 GB |
| B padded30, 3 ep | **0.7079** | [0.7019, 0.7141] | **1.00** | ≈310 | 5.2 GB |
| C truncated, 1 ep | 0.7727 | [0.7675, 0.7780] | 1.81 | ≈1,200 | 1.1 GB |

  Paired bootstrap (same 1,248 utterances): **B − A = −4.98 points, CI [−5.32, −4.64]** (not a tie).
- **Findings**: the head learns from both inputs (blank baseline = 100% deletions; A/B/C are 24–29
  points better) but the curves flatten late: B was 0.7079 at 18,000 and 19,500 steps, A still
  improved by ~1.3 points over its last 2,871 steps. **At matched steps** (≈7,000, same 96 s
  batches, same audio per step) C is 0.7727 while B interpolates to ≈0.741 between its 6,000- and
  7,500-step evals: on paper **padding is worth ≈3.2 points on its own**; the remaining ≈1.8 of
  B's 4.98 come from its 3× more optimizer steps. *This reading is wrong — C's LR schedule was
  also different; the addendum below gives the corrected control.* Truncation cuts peak memory 4×
  (1.1 vs 4.2–5.2 GB) and doubles throughput at equal batches.
- **Result**: H1 gate passed on the learning check. The input check is **not** failed — see the
  schedule-corrected control in the addendum below.
- **Conclusion (superseded)**: the first reading was **H2 trains on `padded30`** (per the plan's H1
  rule), accepting the ~2× compute and the small batches. The addendum retracts it: at a matched
  schedule the two inputs are equivalent, so **H2 uses `truncated`**.

> **Addendum (2026-10-21) — schedule-corrected control.** Arm C differed from B in the LR schedule,
> not only in the input: C's cosine spanned 1 epoch (6,990 steps) while B's spanned 3 (20,970), so
> at step 6,000 B was still at ~0.84 of peak LR and C was at ~0.10. The "matched steps" reading was
> therefore confounded. A new arm **C2** (`runs/E-004/arm-truncated-96-sched3/`) uses the same
> truncated 96 s / 8 clip batches as C but the 3-epoch cosine, stopped at 6,990 steps (`--max-steps`):
>
> | Step | B padded30 | C2 truncated, same schedule |
> |---|---|---|
> | 1,500 | 0.8203 | 0.8200 |
> | 3,000 | 0.7971 | 0.7970 |
> | 4,500 | 0.7769 | 0.7782 |
> | 6,000 | 0.7642 | 0.7652 |
> | 6,990 | ≈0.7406 (interpolated) | **0.7375** [0.7319, 0.7431] |
>
> The curves are the same within ±0.001 (the dev CI is ±0.006); if anything truncated is a hair
> better at the end. This is what the architecture predicts: the encoder already masks padding and
> uses generated sinusoids, so padding to 30 s and padding to the batch maximum compute the same
> per-clip function — only the tensor is larger (and the conv sees a boundary difference at the last
> frame). **The 3.2-point "padding effect" was the schedule, not the input.** Decision revised:
> **H2 trains on `truncated`** (4× less memory than padded at equal clips, ~2.5× less compute, and
> short Common Voice clips pack batches instead of wasting a 30 s tensor each). C2's artifacts:
> `dev-hyp.jsonl`, `dev.metrics.json`, `train.log.jsonl`; wall time ≈12 min.
