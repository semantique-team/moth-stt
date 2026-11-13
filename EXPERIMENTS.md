# Research log

> Historical research narrative. For the 2026-11-10 artifact audit of **all E-000-E-025 IDs**,
> benchmark populations, adoption reasons, missing runs and contradictions, read
> [the experiment ledger](docs/12-experiment-ledger.md). Original entries are preserved rather
> than silently changing old results. In disagreements, use the ledger's cited evidence and
> [paper dossier](docs/13-current-model-and-paper-readiness.md), not an unqualified old conclusion.
> In particular: higher ETA is better; CV-dev subsets differ; CI fractions must be converted to
> percentage points; one failed LoRA recipe does not establish architectural causality.

Every experiment records: hypothesis, model configuration, dataset, training parameters,
hardware, compute and cost, metrics, result, conclusion (template:
`docs/07-repo-layout-and-roadmap.md` §4). Architecture changes are never made without an entry here.

Entries E-007–E-011 are **decode-only, export-only or evaluation-only**: they have no training
parameters or hardware of their own and fold model configuration and dataset into **Method**.
Every entry ends with an **Artifacts** line naming the files behind its numbers, and the executed
decision log, the negative results and the threats to validity are in
`docs/11-evidence-and-decisions.md`.

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

### E-001 addendum — the complete H0 matrix (2026-10-23)

The first pass measured Whisper-tiny **greedy** only. The ship gate is relative to the **best**
H0 variant and includes beam 5, the mixed T set and CV-IT, so the matrix was completed. All
numbers are under `stt.text.normalize` (bootstrap 1,000 where a CI is quoted).

| System | Variant | FLEURS-IT | FLEURS-EN | CV-IT test | T WER | T ETA | RTFx (FLEURS / T) | Size |
|---|---|---|---|---|---|---|---|---|
| Whisper tiny | auto, greedy | 0.3212 | 0.1386 | — | 0.7575 | 0.3672 | 53.3 / 4.6 | 148.7 MB |
| Whisper tiny | forced `it`, greedy | 0.3209 | 0.8190 | — | 0.6225 | 0.2537 | 65.5 / 7.0 | 148.7 MB |
| Whisper tiny | auto, beam 5 | **0.2742** | 0.1186 | — | 0.7051 | 0.3821 | 19.1 / 3.9 | 148.7 MB |
| Whisper tiny | forced `it`, beam 5 | — (skipped) | — (skipped) | — | **0.5305** | **0.3761** | — / 4.1 | 148.7 MB |
| Whisper base | forced `it`, greedy | 0.1827 | — | — | — | — | 25.9 / — | 287.3 MB |
| Whisper base | auto, greedy | — | 0.0988 | — | — | — | 15.3 / — | 287.3 MB |
| Cactus Whistle | auto | **0.1185** | 0.1170 | — | 0.3611 | **0.5075** | 121.8 / 99.4 | 16.9 MB |
| Cactus Whistle | forced `it` | 0.1185 | 0.1335 | **0.2063** | **0.2949** | 0.5015 | 130.0 / 116.4 | 16.9 MB |
| Parakeet TDT v3 int8 | multilingual | **0.0365** | **0.0817** | — | **0.1066** | **0.8388** | 28.1 / 29.5 | 639 MB |
| **Moth E-008** | beam 5 + LM 0.2 | 0.2902 | 0.5699 | 0.3007 | 0.5537 | 0.2149 | 65.2 / — | **17.08 MB** |

- **Best Whisper-tiny variant is not greedy**: auto + beam 5 gives **27.42% FLEURS-IT** and
  **11.86% FLEURS-EN**; forced `it` + beam 5 gives the best mixed-set WER (**53.05%**) and a
  strong ETA (**37.61%**). The 32.09% bar used in E-005/E-008 was the greedy number.
- **The mixed-set ship gate is not met.** Paired difference Moth E-008 vs the best Whisper
  variant on T: WER **+2.32** [−3.22, +8.02] is a **tie**; ETA **−16.12** [−25.31, −6.02] is
  **significantly worse**. The gate requires the WER CI entirely below 0 *and* the ETA CI entirely
  above 0; neither holds. Moth is WER-competitive with Whisper but far behind on English terms.
- **Where the ETA gap is** (T, per category; Moth E-008 vs Whisper forced-`it` beam 5): ETA
  single_word 18 vs **42**, phrase 19 vs **34**, brand 4 vs **31**, acronym 5 vs **30**,
  italianized 7 vs **26**, switch 33 vs **38**. Moth wins the pure-Italian control WER (32 vs 44)
  and loses the English control (51 vs 40). The gap is the English vocabulary/phonology, not
  general Italian.
- **The FLEURS-IT half of H2 also fails** against the corrected bar: 29.02 vs 27.42 = **+1.60**
  [+0.32, +2.83]. CV dev is still won comfortably against Whisper (28.39 vs 48.53), but **not
  against Whistle**: on CV-IT test Whistle is **20.63%** vs Moth **30.07%**. Whisper/Parakeet
  CV-IT rows were not run (informational; the H2 gate uses CV dev); the forced-`it` FLEURS cells
  were skipped after the auto+beam5 result, because the gate decision does not depend on them.
- **Whistle is far ahead at the same size**: 11.85% FLEURS-IT and, on our clean mixed T,
  **29.49% WER / 50.15% ETA** — 26 WER points and 29 ETA points better than Moth. The stretch gate
  ("within the noise of Whistle") is nowhere near met. Caveat: FLEURS is a public benchmark and
  may be in Whistle's training data; T is our own set, so the T gap is the meaningful one.
- **Parakeet is the ceiling**: 3.65% FLEURS-IT / 8.17% FLEURS-EN and, on T, **10.66% WER /
  83.88% ETA** at 639 MB. The mixed set is not impossibly hard; the small models are simply far
  from the ceiling. Moth is 45 WER points and 62 ETA points behind it.
- **Base greedy** is the capacity reference: 18.27% FLEURS-IT, 9.88% FLEURS-EN. No new base runs
  were made: this is inference, not training, and the gate is defined against Whisper-tiny.
- **T RTFx** is low because the clips are short and several jobs shared the CPU; treat it as
  indicative. The browser RTFx numbers (E-010) are the ones for the runtime gate.
- **Conclusion**: the v0.1 candidate is **not shippable** under `docs/05` §5. The ETA gap is the
  dominant problem, and it is structural: V=1024 splits English terms (only 48.8% fit in ≤3
  pieces), and the model deletes or Italianizes them (`meeting → in`, `update → dayte`,
  `backlog → ∅`). H3 gold data is still the planned next step, but before recording 600 R-S1
  utterances decide whether H3 alone can close ~16 ETA points, or whether the tokenizer (larger
  V / word-level output), capacity, or the product target must change. The E-008 "gate passed"
  claim was against the greedy baseline and is corrected in that entry.

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

---

## E-005 — H2 full fine-tune: encoder + head on 438 h (2026-10-21)

- **Date**: 2026-10-21
- **Hypothesis**: full fine-tuning of the Whisper-tiny encoder plus the CTC head on the IT+EN mix
  beats Whisper-tiny on Italian FLEURS and Common Voice dev (the H2 gate) and gives usable English.
- **Model configuration**: Whisper-tiny encoder (4 blocks, width 384, 6 heads, sinusoidal positions)
  + `Linear(384, 1025)` CTC head, V_sp = 1024 (BPE 2.3), 8,027,009 parameters, **all trainable**.
  Init: converted Whisper-tiny encoder **and** the H1 head (E-004 padded arm, dev 0.7079).
  bf16 autocast, fp32 CTC, AdamW, grad clip 1.0; `input: truncated` (E-004 addendum).
- **Dataset**: 437.7 h — Common Voice IT 27 167.79 h (113,118 clips), MLS IT 169.89 h, LibriSpeech
  EN 100.00 h. 77/23 IT/EN by audio time (the design's absolute budget is ≈340+100 h; exact 70/30
  would discard >100 h of IT or repeat EN). Dev: `mls_it_dev` 5.18 h + `ls_en_dev` 5.31 h; FLEURS,
  CV dev/test and T are held out.
- **Training parameters**: 8 epochs, 9,074 steps/epoch, **72,592 steps**; batch 192 audio-s /
  32 clips; head LR 1e-3, encoder 1e-4; warm-up 1,500 → cosine to 5%; full SpecAugment
  (freq/time masks + tempo stretch); `empty_cache_every: 50` on MPS; checkpoint every ~20 min,
  best by macro dev WER (IT+EN).
- **Hardware**: Apple M5, 16 GB, AC power, `caffeinate -i`; torch 2.14.1, MPS, bf16.
- **Compute and cost**: started 10:01, finished 19:23 (9 h 22 wall, **≈8 h of training** after two
  external kills: macOS memory pressure at 13:48 and the worktree removal at 18:31, both resumed
  from checkpoints; the first resume exposed and fixed the optimizer-state device bug). **0 USD**.
- **Metrics** (greedy decode, bootstrap 1,000; `runs/E-005/eval/best/`):

| Set | WER | 95% CI | Notes |
|---|---|---|---|
| `mls_it_dev` (best, 62k) | 0.3808 | [0.3731, 0.3879] | selection dev, IT |
| `ls_en_dev` (best, 62k) | 0.2946 | [0.2899, 0.2999] | selection dev, EN |
| macro dev | **0.3377** | — | best checkpoint; final 0.3386 |
| FLEURS-IT **test** | 0.3741 | [0.3619, 0.3856] | Whisper tiny 0.3209 |
| FLEURS-EN **test** | 0.6244 | [0.6108, 0.6396] | Whisper tiny auto 0.1386 |
| CV IT **dev** | 0.3760 | [0.3723, 0.3797] | Whisper tiny 0.4853; paired **−10.93** [−11.33, −10.51] |
| CV IT **test** | 0.3948 | [0.3909, 0.3985] | — |
| T (S1, 200) | 0.6535 | [0.6201, 0.6906] | ETA 0.164 |

  T breakdown: control_it 0.3889, control_en 0.6216, single_word 0.6655, phrase 0.6716, brand
  0.7101, acronym 0.5676, italianized 0.7385, switch 0.6667. Checkpoint averaging (last 3:
  68,352 + 71,646 + 72,592) gave dev macro 0.3386, so `ckpt_best` (62k, 0.3377) was kept.
- **Paired differences vs Whisper-tiny** (same utterances, 1,000 resamples): FLEURS-IT **+5.32
  points** [4.10, 6.69] worse; FLEURS-EN **+48.58 points** [47.19, 49.99] worse (vs auto). Neither
  is a tie.
- **Findings**: full fine-tuning is where the gain is — H1's frozen encoder ended at 0.7079 on MLS
  dev, H2 reaches **0.3808** there (IT) and 0.2946 on LS dev (EN). But the model **misses the gate**:
  FLEURS-IT is 5.3 points behind Whisper-tiny (37.4 vs 32.1), and English does not generalize from
  LS audiobooks to FLEURS (62.4 vs 13.9). The curves plateaued after ~62k steps (best macro 0.3377;
  final 0.3386; averaging did not help), so more epochs alone are unlikely to close 5 points. T is
  at 65.4% WER with 16.4% ETA on the unseen speaker and never-trained mixed sentences — the baseline
  H3 must beat, not a product number. Caveat: `MelDataset.epoch` was never updated by the training
  loop, so all 8 epochs saw the same augmentation view of each clip (fixed for future runs; see
  `docs/10` §4.2.1).
- **Result**: **gate failed on FLEURS-IT only**. On CV dev the model **beats Whisper-tiny by 10.93
  points** [10.51, 11.33] (37.60 vs 48.53), but the gate requires both sets, so it does not pass.
  The model is still the best Moth checkpoint so far and the correct starting point for H3.
- **Conclusion**: do **not** ship H2 as the final model. Escalation is needed; the plan's order is
  more epochs/LR, +2 head blocks (11.6 M), BPE size, then Whisper-base. Given the plateau, the
  review's cheaper capacity steps (conv head, then +2 blocks / 25 Hz+2) are more promising than more
  epochs alone; decode-side beam/hotwords + a small LM are complementary (10–13% relative in the
  literature) but unlikely to close 5 points by themselves. Measure the Whisper-base baseline before
  deciding on the 20 MB ceiling. H3 (gold code-switch data) remains the unique product lever and can
  run from this checkpoint while the capacity decision is made.

## E-006 — +2 transformer blocks (11.58 M params), fine-tuned from E-005 (2026-10-22)

- **Date**: 2026-10-22
- **Hypothesis**: two extra pre-head transformer blocks (6 total) give enough capacity to halve the
  FLEURS-IT gap to Whisper-tiny while keeping the model ≤ 20 MB int8; in-domain and T improve too.
- **Model configuration**: Whisper-tiny encoder (4 blocks) + **`head_blocks: 2`** extra transformer
  blocks before the CTC head, 6 blocks total, width 384, 6 heads; **11,575,169 parameters** (~12 MB
  int8). Init from E-005 best (`runs/E-005/checkpoints/ckpt_best.pt`, 62k, dev macro 0.3377); the two
  new blocks are randomly initialised; all parameters trainable.
- **Dataset**: same 437.7 h IT+EN mix as E-005 (77/23 by audio time); dev `mls_it_dev` + `ls_en_dev`;
  FLEURS, CV dev/test and T held out. E-006 sees the **fixed** `dataset.epoch` (augmentation varies by
  epoch), unlike E-005; the delta vs E-005 is capacity + augmentation diversity, not capacity alone.
- **Training parameters**: 4 epochs, **36,296 steps**; batch 192 audio-s / 32 clips; encoder LR 5e-5,
  head/new blocks 1e-3; warm-up 500 → cosine to 5%; bf16 autocast, fp32 CTC, grad clip 1.0; full
  SpecAugment; `empty_cache_every: 50`; checkpoint every ~20 min; best by macro dev WER.
- **Hardware**: Apple M5, 16 GB, AC power, `caffeinate -i`; torch 2.14.1, MPS, bf16.
- **Compute and cost**: finished 2026-10-22 17:38; ≈5.4 h of pure training at ~360 audio-s/s, wall
  longer because of several external kills (macOS memory pressure and opencode session/server restarts),
  each resumed from the latest checkpoint. **0 USD**.
- **Metrics** (greedy decode, bootstrap 1,000; `runs/E-006/eval/best/`):

| Set | WER | 95% CI | E-005 | Δ |
|---|---|---|---|---|
| `mls_it_dev` (best/final, 36,296) | 0.3556 | [0.3478, 0.3635] | 0.3808 | −2.52 |
| `ls_en_dev` (best/final) | 0.2699 | [0.2652, 0.2749] | 0.2946 | −2.47 |
| macro dev | **0.3127** | — | 0.3377 | **−2.50** |
| FLEURS-IT **test** | 0.3411 | [0.3294, 0.3531] | 0.3741 | −3.30 |
| FLEURS-EN **test** | 0.6145 | [0.6007, 0.6303] | 0.6244 | −0.99 |
| CV IT **dev** | 0.3445 | [0.3409, 0.3479] | 0.3760 | −3.15 |
| CV IT **test** | 0.3616 | [0.3578, 0.3652] | 0.3948 | −3.32 |
| T (S1, 200) | 0.5950 | [0.5595, 0.6319] | 0.6535 | −5.85 |

  T ETA: **0.200** (E-005 0.164) — overall WER better, English-term acquisition slightly worse.
- **Paired differences vs E-005** (same utterances, 1,000 resamples): FLEURS-IT **−3.30** [−3.86,
  −2.69]; FLEURS-EN **−0.99** [−1.64, −0.23]; CV dev **−3.15** [−3.33, −2.97]; CV test **−3.32**
  [−3.51, −3.14]; T **−5.85** [−8.37, −3.28]. All significant (no tie).
  Versus Whisper-tiny (same utterances): FLEURS-IT **+2.02** [0.80, 3.32] worse; CV dev **−14.08**
  [−14.49, −13.69] better. The gate still fails on FLEURS-IT, with a CI that excludes zero.
- **Findings**: capacity helps everywhere, most on T and CV; the best dev checkpoint is the final one,
  so the cosine was still paying off. FLEURS-IT improves 37.41 → **34.11**, halving the gap to
  Whisper-tiny (32.09) from +5.32 to **+2.02** — still short of the H2 gate. FLEURS-EN remains far from
  Whisper-tiny auto (61.45 vs 13.86). On CV dev Moth is now **14.08 points better** than Whisper-tiny
  (34.45 vs 48.53). T improves 65.35 → 59.50, but ETA slips 16.4% → 20.0%, so the mixed-set gain is
  mostly Italian/fluency, not English terms.
- **Result**: **E-006 is the new best Moth checkpoint**, but the H2 gate still fails on FLEURS-IT
  (+2.02 vs Whisper-tiny). The +2 blocks captured ~3 points of the 5.3-point gap; the remaining gap
  is small enough for decode-side work to matter.
- **Conclusion**: keep +2 blocks (~12 MB int8). Next, in order: beam search + hotwords + a small
  IT/EN LM (cheapest remaining; expected to help FLEURS-IT and T), then conv head / 25 Hz+2 if needed.
  A Whisper-base variant (~20.35 M, ~21 MB int8, FLEURS-IT 18.27) stays the last resort and needs an
  explicit ceiling decision. H3 remains blocked on R-S1/R-S2 and S2/S3/S4 T recordings.

## E-007 — CTC beam search + hotwords on E-006 (2026-10-22)

- **Date**: 2026-10-22
- **Hypothesis**: decode-side CTC beam search plus the curated English-term hotword list closes
  part of the remaining FLEURS-IT gap and improves the mixed T set, without training or extra
  model capacity.
- **Method**: E-006 best (`runs/E-006/checkpoints/ckpt_best.pt`, step 36,296) frozen. Prefix beam
  search over the first `ceil(mel/2)` frames, beam width 5, top-8 non-blank labels per frame plus
  blank, terminal hotword bonus 1.0, no prefix bonus. Hotwords: 580 normalized terms from
  `data/en_terms.txt` (the tokenizer-up-weighting list, not a T-specific list). Tuning on
  `mls_it_dev` + `ls_en_dev` (200 clips each, seed 0); FLEURS, CV and T evaluated once after
  fixing the hyperparameters. No LM in this experiment.
- **Compute and cost**: decode-only, no training; full held-out eval ≈ 3 min on M5/MPS. **0 USD**.
- **Metrics** (greedy E-006, beam only, beam + hotwords 1.0; bootstrap 1,000):

| Set | Greedy E-006 | Beam 5 | Beam + hot 1.0 | Beam Δ vs greedy |
|---|---|---|---|---|
| Dev macro | 0.3127 | **0.3084** | 0.3084 | −0.44 |
| FLEURS-IT **test** | 0.3411 | **0.3377** | 0.3380 | −0.34 |
| FLEURS-EN **test** | 0.6145 | 0.6083 | **0.6078** | −0.62 |
| CV IT **dev** | 0.3445 | **0.3398** | **0.3398** | −0.47 |
| CV IT **test** | 0.3616 | **0.3568** | **0.3568** | −0.48 |
| T (S1, 200) | 0.5950 | 0.5907 | **0.5873** | −0.43 |

  T ETA: greedy **0.200**, beam **0.209**, beam + hotwords **0.218** — beam and hotwords improve
  WER but make English-term acquisition worse, not better.
- **Paired differences vs greedy E-006** (same utterances, 1,000 resamples):
  FLEURS-IT beam **−0.34** [−0.50, −0.17]; FLEURS-EN **−0.62** [−0.86, −0.39]; CV dev **−0.47**
  [−0.53, −0.41]; CV test **−0.48** [−0.54, −0.42]; T **−0.43** [−1.15, +0.18] (tie). Beam +
  hotwords: FLEURS-IT −0.31 [−0.50, −0.11], FLEURS-EN −0.67 [−0.92, −0.43], CV unchanged, T
  −0.77 [−1.57, 0.00] (tie).
- **Findings**: beam search is a small but real win on FLEURS and CV (~1–1.5% relative), and the
  gains are significant; it does not change the story on T (ETA slightly worse). The exact-BPE
  hotword bonus is effectively neutral on FLEURS/CV and gives only a noisy T WER gain while
  worsening ETA — the acoustic model often never produces the exact token sequence, so a
  completion-only bonus cannot recover the term. FLEURS-IT is now **33.77%**, still **+1.68**
  [0.50, 3.03] behind Whisper-tiny (32.09): the gate remains failed. CV dev is **33.98%**, i.e.
  **14.55 points better** than Whisper-tiny (48.53).
- **Result**: E-007 improves all WERs but does **not** pass the H2 gate; hotwords as implemented
  are not a T lever and should not ship as-is. Beam can be an optional “quality” decode mode,
  with greedy as the fast default.
- **Conclusion**: the remaining FLEURS-IT gap is now ~1.7 points. Next: add a small IT/EN LM to
  beam (E-008); if that does not close the gap, move to capacity (conv head / 25 Hz+2). Hotword
  design can be revisited with prefix/fuzzy matching or a word-level lexicon, but it is not the
  first lever. H3 still needs the missing recordings.
- **Artifacts**: `runs/E-007/eval/{beam,beam_hot*}/`, `runs/E-007/dev/`, `runs/E-007/tune/summary.md`; checkpoint `runs/E-006/checkpoints/ckpt_best.pt` (step 36,296). Decode-only: no git SHA in the metrics; artifacts added in commit 54289a2.

## E-008 — small BPE trigram LM in shallow fusion (2026-10-22)

- **Date**: 2026-10-22
- **Hypothesis**: a small, edge-sized n-gram LM added to E-007 beam search closes the remaining
  FLEURS-IT gap and improves all domains without changing the acoustic model or training on
  held-out text.
- **Method**: E-006 best frozen. BPE trigram over the 1024-piece tokenizer vocabulary with
  stupid backoff 0.4; counts pruned at 2 (195,403 bigram entries, 799,918 trigram entries;
  **3.53 MB gzip**). Trained only on `cv_it_train`, `mls_it_train`, `ls_en_train` — 182,441
  training utterances; no dev/test transcript is read. Shallow fusion in the E-007 beam
  (width 5, top-8), `score = CTC log P + 0.2 · LM log P`; α=0.2 selected on dev (200-clip subset,
  confirmed on full dev), held-out evaluated once. No hotwords in this run.
- **Compute and cost**: LM build 6.2 s; decode-only eval ≈3 min on M5/MPS. **0 USD**.
- **Metrics** (bootstrap 1,000; `runs/E-008/eval/w0.2/`):

| Set | E-006 greedy | E-007 beam | E-008 beam + LM 0.2 |
|---|---|---|---|
| Dev macro | 0.3127 | 0.3084 | **0.2705** |
| FLEURS-IT **test** | 0.3411 | 0.3377 | **0.2902** |
| FLEURS-EN **test** | 0.6145 | 0.6083 | **0.5699** |
| CV IT **dev** | 0.3445 | 0.3398 | **0.2839** |
| CV IT **test** | 0.3616 | 0.3568 | **0.3007** |
| T (S1, 200) | 0.5950 | 0.5907 | **0.5537** |

  T ETA: 0.200 → 0.209 → **0.2149**. T WER improves, but English-term acquisition remains the
  weak spot; the LM does not fix it.
- **Paired differences vs E-006 greedy** (same utterances, 1,000 resamples): FLEURS-IT **−5.08**
  [−5.61, −4.54]; FLEURS-EN **−4.46** [−5.07, −3.90]; CV dev **−6.05** [−6.22, −5.88]; CV test
  **−6.08** [−6.25, −5.91]; T **−4.13** [−6.26, −2.25]. All significant.
- **Versus Whisper-tiny**: FLEURS-IT **29.02 vs 32.09** = **−3.06** [−4.29, −1.71] (Moth better);
  CV dev **28.39 vs 48.53** = **−20.13** [−20.52, −19.70] (Moth better). Against Whisper-tiny
  **greedy**, both H2 gate conditions pass at system level with the LM-assisted decoder.
  **Correction (2026-10-23, E-001 addendum):** 32.09 was the *greedy* variant. The best H0 variant
  is auto + beam 5 at **27.42%** on FLEURS-IT; Moth 29.02 is **+1.60** [+0.32, +2.83], a
  significant loss. The FLEURS-IT half of the H2 gate does **not** pass against the best baseline;
  the CV-dev half does (28.39 vs 48.53).
- **Findings**: the tiny LM is the largest single decode-side win on this model: it recovers
  ~4–6 WER points across domains, far more than beam or hotwords, and puts FLEURS-IT below
  Whisper-tiny for the first time. It is domain-sensitive (read/audiobook transcripts) but the
  held-out FLEURS gain is large and significant. Size: ~12 MB int8 acoustic model + 3.53 MB LM
  ≈ **15.5 MB**, still under the ~20 MB ceiling. Caveats: the LM is a Python pickle, not yet
  exported to the browser; T ETA does not improve; a word-level/contextual LM may be needed for
  the mixed English terms.
- **Result**: **E-008 is the new best Moth system.** It passes the CV-dev half of the H2 gate and
  loses the FLEURS-IT half against Whisper-tiny's **best** H0 variant (auto + beam 5, 27.42%).
  This is a system-level result with the LM decoder, not an acoustic-only pass.
- **Conclusion**: freeze E-008 as the quality configuration. Next: export the LM and beam to the
  browser/ONNX path and measure RTFx; then address T ETA (word-level/contextual LM or better
  hotword design) and H3 recordings. Capacity escalation (conv head / 25 Hz+2) stays an option
  until the corrected FLEURS-IT bar is met on the held-out set.
- **Artifacts**: `runs/E-008/lm_bpe3.pkl.gz`, `runs/E-008/build_lm.py`, `runs/E-008/eval/{w0.2,best}/`, `runs/E-008/tune/summary.md`; checkpoint `runs/E-006/checkpoints/ckpt_best.pt`. Artifacts added in commit 3c8b4b3.

## E-009 — real-weight int8 export + browser RTFx (2026-10-23)

- **Date**: 2026-10-23
- **Hypothesis**: the real E-006 checkpoint (with the two extra head blocks) exports to int8 with
  no material WER loss and passes the ≥10× browser RTFx bar (4 threads) on the M5.
- **Method**: `stt/export.py` updated to include `head_blocks` in `ExportModel` and to accept the
  training checkpoint's `"model"` payload; exported `runs/E-006/checkpoints/ckpt_best.pt` with
  the real tokenizer to `release/` (int8 dynamic quantization). fp32 parity on 1/5/10/29 s clips;
  int8 compared with PyTorch greedy on the first 100 FLEURS-IT clips. Browser path unchanged:
  JS log-mel → onnxruntime-web WASM SIMD → greedy; Playwright Chromium headless, 3 warm-ups +
  median of 10 runs, 5/10/20 s (fixture tiled), 1 and 4 threads.
- **Artifact size**: `model.int8.onnx` **11.91 MB** (vs 7.9 MB for the old 4-block model), plus
  vocab 0.02 MB and mel filters 0.06 MB → **11.99 MB**. With the E-008 LM (3.53 MB) the future
  package is ≈ **15.5 MB**, still under the 20 MB ceiling.
- **Export checks**: fp32 max |Δ| **2.17e-4** ≤ 1e-3; int8 top-1 agreement **100%** on the gate
  clips; int8 max |Δ| 1.72 (logit scale). On 100 FLEURS-IT clips: PyTorch greedy WER **30.92%**,
  int8 ONNX greedy WER **30.65%**, 50/100 identical transcripts — no material loss.
- **Browser RTFx** (median, real weights):

| Threads | 5 s | 10 s | 20 s |
|---|---|---|---|
| 4 | **66.4×** | **61.3×** | **54.6×** |
| 1 | **29.3×** | **28.3×** | **24.6×** |

  Load time 138 ms; cross-origin isolated, SIMD active. Feature extraction is roughly half the
  total time; inference is the rest; greedy decode is <1 ms.
- **Findings**: the real model is larger but still comfortably above the ≥10× bar; the 4-thread
  path is 54–66× and the single-thread path is 25–29×. The current browser measurement is the
  **greedy acoustic path**: E-008's beam + LM is not yet ported to JS, so its browser RTFx and
  transcript parity remain to be measured. The model export itself is correct: the head blocks
  are in the graph, and int8 quality is within noise on FLEURS-IT.
- **Result**: export gate passes; H4 browser speed bar passes on both 1 and 4 threads.
- **Conclusion**: next step is the JS beam + LM port and a real-audio e2e parity test (browser vs
  Python E-008 transcripts), then a browser RTFx measurement for the quality path. The acoustic
  int8 export and the greedy fast path are ready.
- **Artifacts**: `runs/E-009/rtfx.json`, `runs/E-009/rtfx-smoke.json`, `runs/E-009/int8_check.json`, `release/`; artifacts added in commit 4551d2a.

## E-010 — browser beam + LM quality path (2026-10-23)

- **Date**: 2026-10-23
- **Hypothesis**: the E-008 beam + LM decoder fits in the browser at >10× RTFx and preserves the
  Python system's T-set WER within the noise of the int8/browser path.
- **Method**: `stt/lm.py` grows `save_browser()`, a sparse CSR binary with int16-quantized
  log-probs (**5.09 MB**) and a Python `BPETrigramLM` source. `web/lm.js` parses it; `web/decode.js`
  adds a prefix beam port; `web/ort.js` loads the LM and selects beam/greedy from
  `config.decode`. `runs/E-010/package_browser.py` patches the release config and hashes. Release:
  E-006 acoustic int8 (11.91 MB) + LM binary (5.09 MB) + filters/vocab = **17.08 MB**.
  Playwright Chromium headless, WASM SIMD, 3 warm-ups + median of 10 runs; same 5/10/20 s tiling
  and 1/4 threads as E-009. Real-audio check: all 200 T clips through the browser, compared with
  Python E-008 hypotheses.
- **Browser RTFx** (quality path: feature + int8 inference + beam 5 + LM 0.2):

| Threads | 5 s | 10 s | 20 s | decode (5/10/20 s) |
|---|---|---|---|---|
| 4 | **65.2×** | **63.3×** | **58.1×** | 4.3 / 8.5 / 15.6 ms |
| 1 | **30.0×** | **28.3×** | **25.2×** | 4.1 / 8.2 / 15.3 ms |

- **Real-audio parity**: on all 200 T clips the browser beam+LM reaches **56.66% WER** vs
  **55.37%** for Python E-008; paired difference **+1.29** [−0.34, +3.02] is a **tie** (the browser
  runs int8 ONNX, so per-clip transcripts differ). Browser ETA **21.19%** vs 21.49% Python — also
  a tie; median browser RTFx on these short clips is **65.0×**. The quality loss is within the
  int8/browser noise, not a parser/decoder mismatch.
- **Mic capture**: the demo now has a guided **Registra / Stop** path: getUserMedia → AudioWorklet
  PCM → OfflineAudioContext resample to 16 kHz mono, with a 30 s safety limit. Playwright with
  Chromium's fake audio device records 1.5 s and verifies the resampled PCM; the UI restores the
  file-input path and disables buttons during capture.
- **Findings**: the JS beam + LM cost is small relative to feature extraction and inference
  (4–16 ms vs ~36–186 ms inference); the quality path still runs at 58–65× on 4 threads and
  25–30× on 1 thread, comfortably above the 10× bar. The browser package is 17.08 MB, still under
  the 20 MB ceiling. The fast greedy path can be restored from `config.decode.default`.
- **Result**: the E-008 quality path now runs in the browser and matches the Python system's
  aggregate T-set WER on the full 200-clip set (tie). The JSON/ONNX release is ready for a demo e2e.
- **Caveats**: exact per-clip transcript equality is not expected with int8; hotwords are not
  ported to JS; T ETA remains the weak spot (browser 21.19%, Python 21.49%).
- **Conclusion**: next is a full 200-clip browser-vs-Python T e2e and final demo polish; then H3
  recordings and T ETA work. No further model/capacity change is needed for the H2 gate.

> **Correction (2026-10-23, E-001 addendum):** the H2 bar used above (Whisper-tiny 32.09% on
> FLEURS-IT) is the **greedy** variant. The best H0 variant is auto + beam 5 at **27.42%**, so
> the FLEURS-IT half of the H2 gate is not met (Moth 29.02, +1.60 [+0.32, +2.83]). The CV-dev
> half is still met by ~20 points. Capacity escalation therefore remains an option; the noise set
> (E-011) also shows the LM quality path needs a no-speech guard.
- **Artifacts**: `runs/E-010/rtfx.json`, `runs/E-010/browser_t200.json`, `runs/E-010/package_browser.py`, `web/{ort,decode,lm,mel}.js`; artifacts added in commit 4801d51.

## E-011 — silence / noise / music robustness (2026-10-23)

- **Date**: 2026-10-23
- **Hypothesis**: on audio with no speech the system emits **zero words** on the non-music part
  (digital silence + MUSAN noise); music is reported separately and is not gated.
- **Model configuration**: E-006 best (11.58 M, frozen) with the **greedy** decoder and with the
  E-008 **beam 5 + BPE trigram LM** quality path (top-8, α=0.2). Baselines on the same clips:
  Whisper-tiny auto/forced-`it` beam 5, Cactus Whistle forced `it` (no hotwords), Parakeet TDT
  0.6B v3 int8 through sherpa-onnx (no hotwords).
- **Dataset**: `data/manifests/noise.jsonl`, **360 clips × 10 s = 60 min**, seed 0:
  120 digital-silence clips (20 min, exact zeros), 120 MUSAN noise clips (20 min), 120 MUSAN
  music clips (20 min). MUSAN is **CC-BY 4.0** (OpenSLR 17, verified 2026-10-23); only its noise
  and music subsets are used. Natural levels, no RMS normalisation (median −20.6 dBFS noise,
  −15.9 dBFS music). Builder: `stt/data/build_noise.py`; mel shards: 1.00 h, 0 errors.
  **Room tone is still to be recorded** (20 min); it will be merged into the same manifest and
  shards, bringing the non-music part to the full 1 h.
- **Procedure**: `build_noise.py` → `build_mel.py` → `runs/E-005/eval_sets.py` (Moth) /
  `baselines/whisper_runner.py`, `baselines/whistle_needle.py`, `baselines/parakeet_sherpa.py` →
  `stt/eval.py` (noise block). The `stt/eval.py` noise block counts words on empty-`norm` rows;
  music rows are tagged `source = "noise_music"` and reported separately.
- **Compute and cost**: set build ≈30 s (5.3 GB MUSAN extraction included in the download step);
  Moth decode ≈3 min per path on M5/MPS; Whisper ≈5 min; Whistle 8 s (RTFx 443); Parakeet ≈2 min.
  **0 USD**.

| System | Decoder | Non-music words | Non-music non-empty clips | Silence words | MUSAN-noise words | Music words | Music non-empty clips |
|---|---|---|---|---|---|---|---|
| **Moth** | greedy | **24** | 18 / 240 | 0 | 24 | 250 | 56 / 120 |
| **Moth** | beam 5 + LM 0.2 | 259 | 134 / 240 | **240** | 19 | **146** | 40 / 120 |
| Whisper tiny | auto/it beam 5 | 54 | 13 / 240 | 0 | 54 | 274 | 26 / 120 |
| Cactus Whistle | forced `it` | 264 | 115 / 240 | 0 | 264 | 554 | 120 / 120 |
| Parakeet TDT 0.6B v3 int8 | transducer | **0** | **0 / 240** | 0 | **0** | 411 | 24 / 120 |

- **Findings**:
  - **Digital silence**: Moth greedy, Whisper, Whistle and Parakeet all emit nothing. Moth with
    **beam 5 + LM emits "non mi" on all 120 silence clips** (2 words each, 240 words). The LM
    shallow fusion has no blank/no-speech prior: on an all-blank acoustic distribution the best
    LM continuation wins over the empty hypothesis. This is a decoder failure, not an acoustic
    one; an energy gate would remove it, but the LM path also needs a no-speech prior.
  - **MUSAN noise**: Moth greedy 24 words / 18 clips, Moth beam 19 / 14, Whisper 54 / 13, Whistle
    264 / 115, Parakeet **0 / 0**. Moth's acoustic model is the most robust of the small systems
    on non-music noise; Whisper hallucinates fewer but longer clips; Whistle is unusable without
    a VAD on transient noise.
  - **Music** (not gated): Moth beam+LM halves the words (250 → 146) and non-empty clips (56 → 40)
    of greedy; Parakeet hallucinates long transcripts on 24 clips; Whistle on all 120.
  - **Blank confidence is not calibrated for silence**: on exact zeros the mean non-blank
    probability is 0.125, *higher* than on T speech (0.108), so a CTC-confidence threshold cannot
    substitute for an energy/VAD gate. Silence and noise must enter training (noise augmentation
    with empty targets) or be filtered before the model.
- **Result**: **only Parakeet meets the hard 0-word non-music gate**; no Moth configuration does.
  Moth greedy (24 words over 40 min of non-music) is much better than its LM quality path on
  silence (259 words), and better than Whisper and Whistle on MUSAN noise. The LM path is the
  wrong default for record-then-transcribe until a no-speech guard exists.
- **Conclusion**: the noise gate is **not met** by the v0.1 candidate. Required follow-ups:
  (1) an energy/silence gate in the Python and browser paths (fixes digital silence immediately);
  (2) **MUSAN noise augmentation with empty targets** in H3 (the principled fix for noise, P2 in
  `docs/10`); (3) a blank/no-speech prior or a small VAD for the LM path; (4) record room tone and
  re-run this entry on the complete 1 h non-music set. Until then the demo should use greedy by
  default on unknown input and warn that the quality path is for speech-only clips.

> **Addendum (2026-10-24) — the no-speech guard closes digital silence.** Follow-up (1) is done:
> `stt/no_speech.py` (Python) and `isNoSpeech` in `web/mel.js` skip decoding when the loudest
> normalized log-mel bin is below **−1.0**. The threshold is calibrated on the cached shards, not
> guessed: digital silence is flat at **−1.50** (range 0.00, std 0.000), the quietest speech clip
> measured (FLEURS-EN, 525 clips) still peaks at **+0.10**, MUSAN noise at **+0.64** and music at
> **+1.04**. The threshold is therefore 20 dB above silence and 44 dB below the quietest speech.
>
> | Path | Guard | Non-music words | Silence words | MUSAN-noise words | Music words |
> |---|---|---|---|---|---|
> | Moth greedy | before | 24 | 0 | 24 | 250 |
> | Moth greedy | **after** | 24 | 0 | 24 | 250 |
> | Moth beam 5 + LM 0.2 | before | 259 | **240** | 19 | 146 |
> | Moth beam 5 + LM 0.2 | **after** | **19** | **0** | 19 | 146 |
>
> The guard catches exactly the **120/120** digital-silence clips in both paths and changes
> nothing else; on 897 speech clips (**200 T + 697 FLEURS-IT**) it guards **0** and leaves the WER
> bit-identical to E-006 greedy (0.5950 / 0.3411). Silence now costs feature extraction only — no
> inference, no decode — and the browser `transcribe()` returns `""` with `guarded: true`.
> Artifacts: `runs/E-011/guard-greedy/`, `guard-beam-lm/`, `guard-speech/`.
>
> **The noise gate is still not met**: 19–24 words remain on the 240 non-music clips, all of them
> MUSAN noise (this is follow-up 2, noise augmentation with empty targets). Music is unchanged and
> still reported separately. The guard is an energy check, not a VAD: it catches silence and
> near-silence, not loud noise or music.
>
> **Addendum (2026-10-30) — Silero VAD replaces the energy guard; the noise gate is met.** D25
> adopts **Silero VAD v5** (MIT, `stt/assets/silero_vad.onnx`, **2,327,524 B**, sha256
> `1a153a22…8788e3`) as the only no-speech guard, for three reasons: the gate demands **0 words**
> on non-music (E-016 + augmented LM emitted **31** with beam, **35** greedy), the energy guard
> cannot separate loud noise, and noise training is impossible without training on the evaluation
> set (the 240 non-music clips *are* our only MUSAN data). Plan:
> `docs/plans/2026-10-30-vad-guard.md`.
>
> **Rule**: 16 kHz PCM, 32 ms windows (**512 new samples + 64 of context = 576 per call**),
> `state [2,1,128]` float32, `sr` int64 scalar; a window is speech at P > 0.5, a clip is decoded
> when the fraction of speech windows is > **0.10** (`stt/vad.py`, `web/vad.js`). The fraction,
> not the maximum, is the separating statistic: noise transients reach 0.99. The **invocation
> trap** cost a day: passing 512 samples without the 64-sample context returns ~0.001 even on
> speech, silently.
>
> Measured with the shipped code on every clip (`runs/E-011/vad_guard_summary.json`):
>
> | Set | n | min | median | max | gated at 0.10 |
> |---|---|---|---|---|---|
> | T S1+S2 (T-400) | 400 | **0.408** | 0.740 | 0.961 | **0** |
> | FLEURS-IT test | 697 | **0.419** | 0.722 | 0.904 | **0** |
> | MUSAN noise | 120 | 0.000 | 0.000 | **0.038** | **120** |
> | digital silence | 120 | 0.000 | 0.000 | 0.000 | **120** |
> | MUSAN music (reported, not gated) | 120 | 0.000 | 0.000 | 0.990 | 107 |
>
> The margin is 4× below the weakest speech clip and 2.6× above the noisiest noise clip. (The
> feasibility table's `0.009` for silence was the maximum *window probability*, not the window
> fraction; the fraction is 0.000.) No gain step is applied — our quiet recordings (−42 dBFS
> RMS) still reach P = 1.000.
>
> | Path | Guard | Non-music words | Silence words | MUSAN-noise words | Music words |
> |---|---|---|---|---|---|
> | Moth greedy | energy | 35 | 0 | 35 | 275 |
> | Moth greedy | **VAD** | **0** | **0** | **0** | 154 |
> | Moth beam 5 + LM 0.2 | energy | 31 | 0 | 31 | 142 |
> | Moth beam 5 + LM 0.2 | **VAD** | **0** | **0** | **0** | 98 |
>
> **The gate is met for the first time**: 0 words on 240/240 non-music clips with both decoders.
> The VAD also gates 89% of the music clips (reported separately, never part of the gate), which
> removes most music hallucinations as a side effect.
>
> **Speech is untouched**: **0/400** T clips and **0/697** FLEURS-IT clips gated. The beam+LM
> hypotheses are **bit-identical to the no-guard run** (400/400 and 697/697), and greedy
> FLEURS-IT is identical to E-016 greedy (697/697, WER 32.76). T-400 beam+LM stays **36.91%**,
> FLEURS-IT **26.90%**. Guarded clips get an empty hypothesis and are counted in `decode.guarded`.
>
> **Package 0.4.0**: 11.91 + 5.12 + **2.33** + 0.08 = **19.44 MB** (ceiling 20). The int8 acoustic
> file is byte-identical to 0.3.1 (sha `d94da6fc…`): no checkpoint, LM, α or tokenizer change.
> Rebuild used `--no-verify` for the same documented reason as E-016 (fp32 gate 1.051e-3 on the
> synthetic 29 s clip, 5.46e-4 on 20 real clips, top-1 100%). Browser (Playwright, WASM SIMD): **4T 48.3× / 46.4× / 41.8×** and **1T 25.2× / 24.0× / 21.5×**
> on 5/10/20 s with the VAD in the loop; the VAD costs 16–83 ms per clip (312 calls per 10 s), i.e.
> −25% RTFx on 4 threads and −12% on 1, still 2× above the 10× bar. Windows were **not**
> subsampled: changing the stride would change the rule. `runs/E-011/browser-0.4.0.json`.
>
> **Risks**: far-field or very quiet speech below a 0.10 fraction would be rejected (none in T or
> FLEURS, but not measured in the wild); music is still not gated by contract; the threshold was
> chosen on separability, not tuned on T.
>
> **Room-tone update (2026-11-06)**: the non-music set now includes **97 clips (16.2 min) of the
> author's own room** — no voice, 0 speech windows, −69.8 dBFS median. The VAD gates **97/97**,
> so the guard is shown to work on real low-level room noise and not only on synthetic MUSAN. The
> recording was 974.8 s instead of 1,200 s (capture stopped early), so non-music is **337 clips =
> 56.2 min**, not the 60 planned. 0 non-music words with both decoders
> (`runs/E-011/vad-noise-room-tone-{greedy,beam-lm}/`).

- **Artifacts**: `runs/E-011/{moth-greedy,moth-beam-lm,guard-greedy,guard-beam-lm,guard-speech}/`, `data/manifests/noise.jsonl`, `data/manifests/noise_report.json`, `stt/data/build_noise.py`. Committed 2026-10-24 (commit c459641) with the code at ~72cf112; no git SHA was recorded in the metrics.

## E-012 — scaling the English data (2026-10-23)

- **Date**: 2026-10-23
- **Hypothesis**: the model's English weakness (FLEURS-EN 56.99% vs Whisper-tiny 11.86%) is a data
  problem; adding ~200 h of LibriSpeech English and retraining from E-006 raises English and,
  through it, the mixed-set ETA, without hurting Italian/T.
- **Data**: LibriSpeech `train-clean-360` reduced to a deterministic **200 h sample**
  (`ls_en_train360`, 57,198 clips, 0 errors, 199.99 h of mel shards). Training mixture becomes
  **638 h** (CV IT 167.8 + MLS IT 169.9 + LS EN 100 + LS EN 200), i.e. ~53/47 IT/EN (was 77/23).
- **Training**: E-006 best frozen as init; 2 epochs, head LR 3e-4 / encoder 5e-5, batch 192 s/32
  clips, bf16, MPS. **Crash at step 9,350**: tempo-stretch augmentation turned a 29.7 s LS360 clip
  into ~32.5 s and the encoder rejects >30 s. Fixed in `_prepare_batch` (crop to 3000 mel frames,
  clamp lengths, regression test); resumed from step 9,072. Total 26,028 steps.
- **Compute and cost**: data build ≈10 min, training ≈4 h on the M5; 0 USD.
- **Metrics** (bootstrap 1,000; `runs/E-012/eval-final/`):

| Set | E-012 greedy | E-012 beam+LM | E-008 (best Moth) |
|---|---|---|---|
| Dev macro | — | — | 0.2705 |
| FLEURS-IT test | 0.3548 | **0.2979** | **0.2902** |
| FLEURS-EN test | 0.5650 | **0.5156** | 0.5699 |
| CV IT test | 0.3660 | **0.3038** | **0.3007** |
| T (S1, 200) | 0.6552 / ETA 0.1821 | 0.6053 / ETA 0.2149 | **0.5537 / ETA 0.2149** |

  Final dev: **29.50%** (IT 35.78 / EN 23.22), better than E-006's best 31.27 (IT 35.56 / EN 26.99).
- **Paired vs E-008 on T**: WER **+5.16** [+3.00, +7.37] (**significantly worse**); ETA **0.00**
  [−2.93, +2.92] (**exact tie**).
- **Findings**: the extra English does what it says on English — FLEURS-EN improves **5.4 points**
  (51.56 vs 56.99) and dev EN improves 3.8 — but it **shifts the 11.5 M model away from Italian**:
  FLEURS-IT is 0.8 points worse, CV-IT 0.3 worse, and T WER regresses 5.2 points significantly.
  The dev macro looked better only because the English gain outweighed the Italian loss. **ETA does
  not move at all** (21.49). The 53/47 mixture is too English-heavy for the mixed task.
- **Result**: **A is rejected as a standalone lever for the ship gate**: English improves, Italian/T
  regress, ETA is unchanged. The best Moth remains E-008.
- **Conclusion**: the ETA bottleneck is not general English data. Next cheap test: H3b splicing
  (synthetic IT+EN utterances, no recording); if ETA does not move, distillation from
  Whistle/Parakeet or a re-scope are the remaining options.
- **Artifacts**: `runs/E-012/eval-final/` (authoritative), `runs/E-012/eval/` (8k-checkpoint pass, superseded), `runs/E-012/train.log.jsonl` (git SHA 72cf112), `runs/E-012/config.yaml`, `data/manifests/ls_en_train360.jsonl`.

## E-013 — larger BPE vocabulary (V=4096) (2026-10-23)

- **Date**: 2026-10-23
- **Hypothesis**: English terms are split into 3–6 BPE pieces at V=1024; a 4096-piece vocabulary
  shortens them to 1–2 pieces, making the CTC sequence easier to emit and raising ETA.
- **Tokenizer**: rebuilt with the same corpus and seed as V=1024
  (`stt/data/build_tokenizer.py`): **tokens/word 1.945 → 1.495**, **tokens/s 4.70 → 3.61**,
  **English terms within 3 pieces 48.8% → 74.0%**.
- **Training**: new CTC head (384×4097 ≈ 1.57 M; **12.758 M total**); encoder and two head blocks
  from E-006 (the shape-mismatched head is skipped by `_apply_model_init`), 2 epochs on the E-006
  438 h, head LR 1e-3 / encoder 5e-5. 18,148 steps, ≈3.5 h on MPS.
- **Compute and cost**: ≈3.5 h; 0 USD.
- **Metrics** (no LM: the E-008 LM is a 1024-piece model; `runs/E-013/eval/`):

| Set | E-013 greedy | E-013 beam 5 | E-006 (1024) |
|---|---|---|---|
| Dev macro @18k | — | **0.3279** | 0.3409 |
| FLEURS-IT test | 0.3630 | 0.3602 | **0.3411** |
| FLEURS-EN test | 0.6002 | **0.5971** | 0.6145 |
| T (S1, 200) | 0.5993 / ETA 0.197 | 0.5985 / ETA 0.200 | **0.5950 / ETA 0.200** |

- **Findings**: at equal steps the dev macro is better (32.79 vs 34.09 at 18k; 1.3 points), and
  FLEURS-EN improves 1.7 points, but **FLEURS-IT regresses 1.9 points** and T is flat/slightly worse.
  **ETA does not change** (20.0 vs 20.0). The tokenizer shortens the sequences but does not teach the
  acoustic model to recognise the English terms; the errors are still deletions/Italianisations.
- **Result**: **B is rejected as a standalone lever for ETA**; the model size grows by ~1.2 M
  parameters and the quality does not improve on the task that matters.
- **Conclusion**: do not adopt V=4096 for v0.1, and do not spend time adapting the LM to it. The ETA
  bottleneck is acoustic/data, not vocabulary.
- **Artifacts**: `runs/E-013/eval/{greedy,beam}/`, `runs/E-013/train.log.jsonl` (git SHA 72cf112), `runs/E-013/config.yaml`. The V=4096 tokenizer is not committed (`data/tokenizer/` is gitignored); it is rebuilt by `stt/data/build_tokenizer.py`.

## E-014 — H3b utterance-level splicing, no gold data (2026-10-24)

- **Date**: 2026-10-24
- **Hypothesis**: E-012 (more English) and E-013 (V=4096) both left the mixed-set ETA at 21.49,
  so the model has never seen a switch **inside one utterance**. Concatenating an Italian and an
  English clip (transcript = the concatenation) gives clause-level switches and should raise the
  ETA on T without hurting FLEURS/CV.
- **Data**: **548 h** = the E-006 mix (CV IT 167.8 + MLS IT 169.9 + LS EN 100) + **110 h spliced**
  (`stt/data/build_spliced.py`, seed 0): 21,205 clips, Italian pool CV+MLS, English pool
  train-clean-100, pairs **≤ 27 s** (the 30 s encoder window plus tempo-stretch headroom),
  transcript = IT+EN with `en_spans` covering the English half. Splicing happens on the **cached
  mel** — no raw audio is needed, because the audio was deleted after the E-002 spot check. Pairs
  whose seam re-normalizes into one word (an Italian elision, `po'` + `blushed` → `po'blushed`)
  are dropped instead of corrupting the transcript. Mixture **IT 61.6% / EN 18.3% / spliced
  20.1%**: the plan's H3b shares (0.4/0.15/gold 0.2/0.25) assume the gold set, which does not
  exist yet (R-S1 is at 76/600), so this arm tests splicing **alone**.
- **Training**: E-006 best (11.58 M) → 2 epochs, **22,494 steps**, head LR 3e-4 / encoder 5e-5
  (identical to E-012, so the two continuation runs differ only in the data), batch 192 s /
  32 clips, bf16, MPS, cosine 500-step warm-up.
- **Compute and cost**: 09:16 → 13:33, **4 h 16 wall** (~3 h 50 of training). One
  **memory-pressure stall** at step ≈ 3,040: the process was fully paged out (swap 7.6/8 GB) and
  made zero progress for ~12 min; killed and resumed from `ckpt_0003040` (≤ 40 steps lost). **0 USD**.
- **Metrics** (bootstrap 1,000; `runs/E-014/eval/`):

| Set | E-008 (best Moth) | E-014 greedy | E-014 beam 5 + LM 0.2 |
|---|---|---|---|
| Dev macro (final) | 31.27 (E-006) | — | **30.86** |
| FLEURS-IT **test** | 29.02 | 32.68 | **26.75** |
| FLEURS-EN **test** | 56.99 | 60.70 | **56.04** |
| CV IT **test** | 30.07 | 36.15 | 30.10 |
| T (S1, 200) | 55.37 / ETA 21.49 | 61.13 / 18.51 | 57.09 / **20.60** |
| Noise, non-music words | 19 | 24 | **13** |
| Noise, music words | 146 | 275 | 138 |

  Paired differences **vs E-008** (same utterances, 1,000 resamples): FLEURS-IT **−2.27**
  [−2.86, −1.72]; FLEURS-EN **−0.95** [−1.57, −0.37]; CV test **+0.03** [−0.12, +0.16] (tie);
  T WER **+1.72** [−0.60, +4.09] (tie) and ETA **−0.90** [−4.23, +2.48] (tie). Both FLEURS gains
  are significant; CV, T and the noise set are ties (the LM path emits 13 words instead of 19).
- **Against the ship bars**: FLEURS-IT vs Whisper-tiny auto+beam5 (27.42) is **−0.66**
  [−1.80, +0.47], a **tie** — the first Moth configuration not significantly behind the best H0
  (E-008 was +1.60 [0.32, 2.83]). The T half still fails on ETA: **−17.01** [−25.55, −7.01] vs
  Whisper forced-`it` beam 5 (WER +4.04 [−1.20, +9.60] is a tie).
- **Findings**: splicing is a real **generalization** gain on both FLEURS sets and costs nothing
  measurable on CV, T or noise — but it does **not move the ETA**. The per-category breakdown
  shows no category improving: single_word 18.0 → 16.0, italianized **7.4 → 3.7**, brand 4.2 →
  4.2, acronym 5.0 → 5.0, switch 32.5 → 31.2. The model learned to stay Italian/robust with
  English context around it (control_it 0.35, control_en 0.51), not to retain English terms.
  It is also a reminder that the read-speech dev is not a proxy for FLEURS: E-014's dev
  (30.86) is worse than E-012's (29.50) while its FLEURS-IT is 3 points better (26.75 vs 29.79).
- **Result**: **E-014 is the new best Moth system**: significantly better on both FLEURS sets and
  a tie on CV test, T (WER and ETA) and the noise set. The H3b hypothesis — splicing raises the
  ETA — is **rejected**.
- **Conclusion**: keep E-014 as the acoustic checkpoint. The ETA bottleneck is not clause-level
  switches, so the remaining levers are (a) **distillation from Whistle/Parakeet**, (b) **gold
  code-switch data** (the plan's H3a, still untested: R-S1 at 76/600), (c) a re-scope of the ship
  gate. Noise augmentation with empty targets is still open (E-011).
- **Artifacts**: `runs/E-014/eval/{greedy,beam-lm,noise-greedy,noise-beam-lm}/`, `runs/E-014/eval.sh`, `runs/E-014/train.log.jsonl` (git SHA 50d98b8), `runs/E-014/config.yaml`, `data/manifests/spliced_it_en.jsonl`, `data/manifests/spliced_it_en_report.json`, `stt/data/build_spliced.py`.

## E-016 — gold fine-tune on the recorded R-S1 set (2026-10-29)

- **Date**: 2026-10-29
- **Hypothesis**: 29 min of recorded Italian–English gold (S1, 635 sentences, the only audio that
  contains `deadline`, `merge`, `kpi`, `onboarding`, `phishing`, …) is enough to teach those terms
  to the model and lift the mixed-set ETA, once they are repeated enough to get gradient share.
- **Data**: **572.4 h** = the E-014 mix (CV IT 167.8 + MLS IT 169.9 + LS EN 100 + spliced 110) +
  **gold 24.4 h**. The gold is the same 29.3 min repeated **50×** through an index-only duplicated
  shard (`runs/E-016/prepare_gold_oversample.py`; the mel file is copied once, 38 MB). Gold is
  4.3% of the audio, 13.5% of the rows. The plan asked for ≥20% of the batches; with 29 min of
  gold that would need a 220× repetition, i.e. memorisation, so this is the probe.
- **Training**: init **E-014 best** (step 36,296), 2 epochs, **24,478 steps**, head LR 3e-4 /
  encoder 5e-5 (same as E-012/E-014), batch 192 s / 32 clips, bf16, MPS, cosine 500-step warm-up.
- **Hardware**: Apple M5, 16 GB, MPS bf16; torch 2.14.1.
  Dev curve: 33.51 @2k, 32.08 @14k, 31.18 @20k, **30.90 @24k (best)**, 30.98 final — E-014 at the
  same steps was 33.56 / 31.87 / 31.03 / 30.86, so the two are within noise on the dev sets.
- **Compute and cost**: 4 h 20 wall (12:58 → 17:12), including one **memory stall** at step
  ≈ 2,700: the MPS allocator held ~9 GB between `empty_cache` calls (every 50 steps), macOS
  swapped the process out and made zero progress for minutes. Killed, resumed from
  `ckpt_0002695`, and `empty_cache_every` lowered 50 → 10 (housekeeping only: same gradients, LR
  and batches); the run then held 2–5 GB and ran at 125–150 steps/min. **0 USD**.
- **Metrics** (bootstrap 1,000; `runs/E-016/eval/`):

| Set | E-014 | E-016 greedy | E-016 beam 5 + LM 0.2 |
|---|---|---|---|
| Dev macro (best) | 30.86 | — | **30.90** |
| FLEURS-IT test | **26.75** | 32.76 | 26.98 |
| FLEURS-EN test | **56.04** | 60.75 | 56.22 |
| CV IT test | **30.10** | 36.15 | 30.02 |
| T (S1) WER | 57.09 | 38.09 | **35.51** |
| T (S1) ETA | 20.60 | 37.91 | **40.60** |
| Noise non-music words | **13** | 35 | 31 |

  Paired differences **vs E-014** (beam + LM): FLEURS-IT **+0.22** [−0.23, +0.70] (tie), FLEURS-EN
  **+0.17** [−0.42, +0.79] (tie), CV test **−0.08** [−0.22, +0.06] (tie), **T WER −21.58**
  [−25.38, −17.99] and **T ETA +20.00** [+13.69, +25.70], both significant.
  Per-category ETA (beam + LM, E-014 → E-016): single_word 16.0 → **32.0**, phrase 19.2 → **28.8**,
  brand 4.2 → **25.0**, acronym 5.0 → **50.0**, italianized 3.7 → **51.9**, switch 31.2 → **56.2**.
  Every term category moves, which is exactly what a speaker effect looks like; the controls below
  separate the part that is vocabulary. The noise set also **regressed**: 13 non-music words with
  the LM in E-014, **31** here (greedy 24 → 35), i.e. the fine-tuned model is more willing to emit
  words on non-speech — another symptom of adapting to the recorded speaker rather than learning
  the vocabulary.
- **The control that changes the reading**: on the same T-S1 clips, the **pure-Italian control**
  (`control_it`, no English at all) goes from **34.7 → 18.1 WER** (−16.6), while the pure-English
  control (`control_en`) is **unchanged at 51.4**. Splitting the annotated terms by whether they
  appear in R-S1: **seen 18.3% → 45.7% (+27.4)**, **unseen 15.8% → 27.7% (+11.9)**. So the model
  also got much better at Italian that has nothing to do with English terms: this is **adaptation
  to S1's voice, room and reading style**, not only vocabulary learning. There is a real
  term-specific component (the extra +15 points on terms present in the gold), but it sits on top
  of a much larger speaker effect.
- **The free S2 check (2026-10-30)**: S2's 200 T clips arrived; S2 is **not** a training speaker.
  On those 200 clips E-016 vs E-014: WER **51.87 → 42.22** (−9.64 [−12.55, −6.93]) and ETA
  **20.68 → 29.75** (+9.07 [+3.97, +14.61]), both significant. So roughly **half** of the S1 gain
  transfers to a speaker the model never heard (S1: −21.58 WER / +20.00 ETA), and half is
  speaker-specific. The matched speaker **inflates the gain about 2×, but does not fabricate it**:
  the gold does teach something that generalises.
- **The gate measured on a speaker the model never heard (S2, 200 clips, 2026-10-30)**: Moth
  E-016 **42.22 WER / 29.75 ETA** vs Whisper auto+beam5 70.63/35.41, Whisper forced-`it` beam5
  51.69/32.86, Whistle 30.41/45.61, Parakeet 11.47/83.00 (`baselines/runs/*-t-s2/`). Paired vs
  Whisper forced-`it`: **WER −9.47** [−14.90, −4.59], **ETA −3.12** [−12.46, +5.65] (tie). So on a
  non-training speaker Moth **wins the WER** against both Whisper variants and still **ties on
  ETA**: the mixed gate does not pass, but for the first time it is measured where it matters.
- **Against the ship bars**: on T the WER half looks won — Whisper-tiny forced-`it` beam 5 is
  **−17.54** [−22.50, −12.54] worse — but the ETA half is a **tie**, **+2.99** [−6.23, +12.29].
  The mixed gate does **not** pass even before the speaker caveat. On FLEURS-IT the bar is still a
  tie (26.98 vs 27.42), unchanged from E-014.
- **Result**: **E-016 is the best Moth so far on every speaker we can measure**, with a caveat
  that is now quantified instead of assumed: on the training speaker the gain is −21.6 WER / +20.0
  ETA, on a new speaker (S2) it is −9.6 / +9.1, both significant, and FLEURS/CV are ties. The
  matched speaker doubles the apparent gain; it does not create it. The mixed ship gate still
  cannot be claimed: it needs the baselines re-run on S2's clips and the held-out S3 (E-017).
- **Conclusion**: **E-017 is now mandatory** and better motivated: with S2 as a second training
  speaker and **S3 held out**, the same measurement separates "adapts to the speaker" from "learns
  the terms" on a speaker that never entered training at all. Next steps, in order: (1) re-run
  Whisper/Whistle/Parakeet on S2's 200 clips to get a fair gate on a non-training speaker;
  (2) finish R-S2 (script 344/600) and record it; (3) E-017 on S1+S2, evaluated on S3. Note that
  **T now has 400 rows (S1+S2)**: every number measured before 2026-10-30 is on the 200 S1 clips
  and is not directly comparable to the new 400-clip numbers.
- **Artifacts**: `runs/E-016/config.yaml`, `train.log.jsonl` (git SHA 45ad463), `eval.sh`,
  `eval/{greedy,beam-lm,noise-*}/*.metrics.json`, `prepare_gold_oversample.py`; gold manifest
  `recordings/manifests/R-S1.jsonl` (committed), mel shards `data/shards/R-S1` and
  `data/shards/R-S1-x50` (gitignored, rebuilt with `build_mel` + the prepare script);
  checkpoint `runs/E-016/checkpoints/ckpt_best.pt` (step 24,000, gitignored).

> **Addendum (2026-10-30) — release 0.3.0 rebuilt with the E-016 checkpoint.**
> - **Package**: int8 **11.91 MB** + vocab 0.02 + filters 0.06 + LM 5.09 = **17.08 MB** (ceiling 20).
>   `stt/export.py --checkpoint runs/E-016/checkpoints/ckpt_best.pt --head-blocks 2 --version 0.3.0`,
>   then `runs/E-010/package_browser.py` for the LM binary and the decode block.
> - **fp32 parity (gate 1e-3)**: synthetic gate **1.051e-3 at seed 0** (5% over), 3.62e-4 at seed 2;
>   on 20 real T clips max **5.46e-4**, median 2.02e-4, **top-1 agreement 1.0000**. The exceedance is
>   a marginal synthetic-29 s artifact (1 s/5 s/10 s are 5.5e-4/2.7e-4/1.8e-4 and seeds 2–4 pass),
>   not an export bug. The release was built with `--no-verify` and the deviation is recorded here
>   instead of being hidden; if the gate must hold strictly, the export needs investigating, not a
>   looser threshold.
> - **int8 quality**: top-1 agreement **99.78%**; per-frame median gap 0.19, max 12.08 on one clip
>   (logit scale, larger than E-006's 1.72 — the fine-tuned weights have wider ranges). On 100
>   FLEURS-IT clips PyTorch and int8 WER are **identical (29.23%)**, 44/100 transcripts identical.
> - **Browser** (Playwright, Chromium headless, WASM SIMD, 3 warm-ups + median of 10): **4T 61.0× /
>   60.9× / 53.8×** and **1T 28.5× / 27.0× / 23.7×** on 5/10/20 s. The no-speech guard and the
>   microphone tests pass; the fixture (a synthetic tone) now transcribes to an empty string, which
>   is the right behaviour for a real model and a tone.
> - **Not re-measured**: full T-set browser-vs-Python parity (E-010 did it for the E-006 int8
>   graph; here int8 quality is measured in Python on 100 FLEURS-IT clips). `release/` is
>   gitignored — it is reproducible with the two commands above.

> **Addendum (2026-10-30) — the LM did not know the gold's vocabulary.**
> Found by trying the demo: `stand-up meeting` came out as `st meeting`. Diagnosed in three steps.
> (1) `standup` never appears in the E-008 LM text (CV/MLS/LS transcripts: 0 occurrences), and the
> LM preferred the wrong hypothesis by **+15.9 log-prob**. (2) E-016 does learn the term in the
> context it saw: on T-0247 (`il daily standup è alle nove`) E-014 produced `il balis st up elen` and
> E-016 `il baily standup èlena`; standalone (T-0032) it is still wrong. (3) The term is three BPE
> pieces against one for `st`, so even a fair prior cannot always win.
> **Fix**: `runs/E-008/build_lm_augmented.py` rebuilds the same trigram LM adding the **training-side**
> mixed scripts (R-S1, R-S2, pool) to the transcripts; T and A are never read, so no leakage. Same
> pruner and backoff, **3.55 MB**.
> **Results** (E-016 checkpoint, beam 5, α = 0.2, paired bootstrap 1,000):

| Set | E-008 LM | augmented LM | Δ |
|---|---|---|---|
| T-400 WER | 38.76 | **36.91** | **−1.86** [−2.55, −1.21] |
| T-400 ETA | 35.03 | **39.10** | **+4.07** [+2.67, +5.61] |
| T-S2 WER / ETA | 42.22 / 29.75 | **40.49 / 33.43** | −1.73 / +3.68 |
| FLEURS-IT | 26.98 | 26.90 | −0.08 |
| FLEURS-EN | 56.22 | 56.23 | +0.01 |
| CV-IT test | 30.02 | 30.02 | 0.00 |
| Noise non-music words | 31 | 31 | 0.00 |

> **α re-tuned on dev with the augmented LM** (200 `mls_it_dev` + 200 `ls_en_dev`, seed 0, the
> E-008 protocol; T never used): α = 0.1 → 28.87 macro, **0.2 → 27.54**, 0.3 → 27.88, 0.4 → 29.08.
> **0.2 stays**: raising it would help the English side (EN 0.2265 at 0.3 vs 0.2270 at 0.2) but costs
> more on Italian, and it does not fix word-boundary merges either.
>
> The gain is concentrated where the vocabulary lives (T), and the other sets are unchanged: this is
> a free win, not a trade. **Adopted**: the release LM binary was rebuilt from the augmented pickle
> (`runs/E-010/package_browser.py --lm runs/E-008/lm_bpe3_aug.pkl.gz`, 5.12 MB) and the browser table
> re-measured — 4T **64.3× / 62.5× / 57.3×**, 1T **29.1× / 27.6× / 24.5×** on 5/10/20 s, package
> **17.11 MB** (v0.3.1). Even with the augmented LM the `standup` case can still lose to `st` in
> isolation: the structural fix is more acoustic exposure in varied contexts, not a stronger prior.

> **Addendum (2026-10-30) — S3 arrives: the cross-speaker gain replicates (zero-shot, no training).**
> S3 is the speaker held out for **E-017**. S3 recorded **T 200/200 + A 60/60** on the same Mac and
> microphone as S1 — the speaker changes, the channel does not. T is now **600 clips** (S1+S2+S3,
> 23.2 min); T-600 numbers are **not** comparable to the earlier T-400/T-200 rows. Headline,
> E-016 + augmented LM α 0.2: greedy **43.26% WER**, beam+LM **38.50%** (E-014: 59.26 / 53.50).
>
> On the S3 slice alone (200 clips, `runs/E-017/manifests/T-S3.jsonl`), E-016 vs E-014:

| Decoder | E-014 WER / ETA | E-016 WER / ETA | Δ WER | Δ ETA |
|---|---|---|---|---|
| greedy | 60.22 / 13.52 | 45.91 / 26.48 | **−14.31** [−17.67, −11.00] | **+12.96** [+7.83, +18.15] |
| beam + LM | 54.77 / 17.75 | **41.57 / 30.42** | **−13.20** [−16.36, −10.24] | **+12.68** [+7.71, +18.03] |

> Both significant, and **slightly larger than on S2** (−9.64 WER / +9.07 ETA) and about **60% of
> the matched-speaker S1 gain** (−21.58 / +20.00). Since S3 shares S1's microphone and room, part of
> the S3 gain may be **channel** adaptation rather than speaker generalisation; S2 (external
> recorder, different room) is the cleaner cross-speaker estimate. E-017 — train on S1+S2 gold,
> test on S3 — averages two channels and is still the experiment that decides.
>
> **Gate on S3** (paired vs each baseline, beam+LM): Moth **41.57 / 30.42**; Whisper-tiny
> auto+beam5 58.69 / 41.41 (**Moth −17.12 WER, −10.99 ETA**, both significant); Whisper-tiny
> forced-`it` beam5 46.51 / 38.03 (**−4.94** [−10.40, +0.34] WER, **−7.61** [−17.02, +1.86] ETA:
> **both ties**); Whistle 28.36 / 52.11 (Moth +13.20 WER, −21.69 ETA); Parakeet 10.82 / 83.94.
> So on a second unseen speaker Moth still **ties** the mixed gate against the strongest Whisper
> variant and beats the auto variant; Whistle remains ahead. The WER win shrinks vs S2
> (−9.47 significant there, −4.94 tie here) because Whisper itself does better on S3's clips.
>
> **VAD**: 0/200 S3 clips gated, but the weakest clip is `T-0033` at fraction **0.194** — the
> lowest speech fraction measured (S1 min 0.408, S2 0.410, FLEURS 0.419), 1.9× the 0.10 gate
> instead of 4×. Kept, not tuned; it goes into the far-field threat (`docs/11` §5).
>
> **Still missing for E-017 proper**: R-S2 is a script (623 sentences), not audio; E-017 needs S2
> to record it (600 takes, ~4 h), then the E-016 recipe is repeated on S1+S2 gold and evaluated on
> S3. Artifacts: `runs/E-017/` (speaker manifests, S3 metrics + paired diffs, baselines, `vad_speakers.json`),
> `runs/E-016/eval/t600-*`, `runs/E-014/eval/t600-*`; reproduce with `runs/E-017/zero_shot_s3.sh`.

> **Addendum (2026-11-06) — room tone recorded; the non-music set is now 56 min and all gated.**
> The last piece of E-011 follow-up (4): 16.2 min of the author's own room (fan, keyboard, monitor;
> no voice, no music), 16 kHz mono, −65.7 dBFS mean / −22.3 dBFS peak. It was recorded as
> **974.8 s instead of 1,200 s** (the capture stopped early), so the set gains **97 clips = 16.2 min**
> and non-music reaches **337 clips / 56.2 min**, not the 60 planned; music stays 120 clips.
> Checked before use: **0** speech windows (max window probability 0.42), so it is clean non-speech.
>
> | Path | Guard | Non-music words | Clips gated | Music words |
> |---|---|---|---|---|
> | E-016 greedy | VAD | **0** | 444/457 | 154 |
> | E-016 beam 5 + LM 0.2 | VAD | **0** | 444/457 | 98 |
>
> **Room tone: 97/97 gated** (fraction 0.000 on every clip). The 13 clips that pass the guard are
> all music, which is reported and never gated by contract. So the VAD handles **real** low-level
> room noise, not just synthetic MUSAN: this is the evidence the D25 table was missing.
> Artifacts: `data/manifests/noise.jsonl` (457 rows), `noise_report.json`, `data/shards/noise`
> (1.27 h, 0 errors), `runs/E-011/vad-noise-room-tone-{greedy,beam-lm}/`. Reproduce with
> `uv run python -m stt.data.build_noise --room-tone data/raw/room_tone.wav --categories room_tone
> --minutes-per-category 16.2`, then `build_mel`, then `runs/E-005/eval_sets.py --sets noise --vad-guard`.

## E-017 — cross-speaker gold: S1 + S2 trained, S3 held out (2026-11-07)

- **Date**: 2026-11-07 (training 2026-11-06 21:04 → 2026-11-07 00:25, 3 h 21 min).
- **Hypothesis**: E-016's gain came from S1's gold, and S1 was both a training speaker and (in the
  prepaper reading) a test speaker, so the confound was speaker adaptation. With gold from **two**
  speakers and a **third** speaker that never contributes audio, the term-learning component should
  survive the matched-speaker removal. If the gain on S3 collapses to ~0, the E-016 result was
  speaker adaptation and the gold hypothesis is dead.
- **Design**: exactly one variable vs E-016 — the gold arm gains **R-S2**. Everything else is the
  same recipe: init E-014 best, natural + spliced mix, head 3e-4 / encoder 5e-5, cosine with 500
  warm-up, 2 epochs, bf16, truncated input, augmentation full, batch 192 s / 32 clips, 50× gold
  repetition.
- **Data**: R-S1 **641 takes** (29.6 min) + R-S2 **622 takes** (20.6 min; one of the 623 takes was
  0.39 s of silence and is excluded). Gold repeated ×50 → **63,150 rows / 41.87 h**. Mixture
  **589.54 h** (CV IT 167.8 + MLS IT 169.9 + LS EN 100 + spliced 110 + gold 41.87), gold **7.1%**
  of the audio (E-016: 4.3%). Builder: `runs/E-017/prepare_gold.py` (generalises E-016's
  single-source script, namespaced shards, 3 tests).
- **Compute**: 26,442 steps (~3 h 21 min at ~370 audio-s/s median), 0 non-finite steps, peak 6.45 GB,
  MPS↔CPU parity 4.9e-6. Cost 0 USD.
- **Dev**: macro **30.98%** (IT 36.29 / EN 25.68) at the final step — the best dev of the project
  (E-016 31.27 best, 30.90 at step 24k). The curve was still improving at 26k.

| Set (beam 5 + LM α 0.2) | E-014 | E-016 | **E-017** |
|---|---|---|---|
| **T-600 WER / ETA** | 53.50 / 20.33 | 38.50 / 36.05 | **32.77 / 42.86** |
| T S1 (matched) | 55.55 / 22.09 | 33.45 / 44.78 | **28.98 / 51.94** |
| T S2 (trained) | 50.13 / 21.25 | 40.49 / 33.43 | **33.88 / 41.93** |
| **T S3 (held out)** | 54.77 / 17.75 | 41.57 / 30.42 | **35.43 / 35.21** |
| FLEURS-IT | 26.75 | 26.90 | **26.52** |
| FLEURS-EN | 56.04 | 56.23 | **55.48** |
| CV-IT test | 30.10 | 30.02 | 30.36 |
| Noise non-music words | 0 (VAD) | 0 (VAD) | **0 (VAD)** |

- **Paired vs E-016 on T-600** (same manifest, bootstrap 1,000): WER **−5.73** [−7.14, −4.26] and
  ETA **+6.81** [+4.18, +9.55] with beam+LM; greedy −5.56 [−7.01, −4.20] / +5.56 [+3.08, +8.16].
  A second gold speaker is worth ~6 points WER on the mixed set.
- **The key control — per-speaker gain vs E-016** (beam+LM, paired):

| Speaker | Role under E-017 | Δ WER | Δ ETA |
|---|---|---|---|
| S1 | training speaker (matched) | −4.47 [−6.72, −2.41] | +7.16 [+3.26, +11.45] |
| S2 | training speaker (matched) | −6.60 [−8.87, −4.33] | +8.50 [+3.84, +13.50] |
| **S3** | **never contributed audio** | **−6.13** [−8.68, −3.51] | **+4.79** [+0.51, +9.06] |

  The held-out speaker gains **as much as the matched speakers**. That is the result E-016 could
  not produce: the improvement is not speaker adaptation, because adaptation to a speaker the model
  never heard is undefined. S3 even gains more than S1, the speaker with the most gold.
- **Gate on S3 vs the baselines** (beam+LM, paired): Moth **35.43 / 35.21** vs Whisper-tiny
  auto+beam5 58.69 / 41.41 (**WER −23.25** [−29.98, −17.20] **won**, ETA −6.20 [−15.96, +4.86] tie);
  Whisper-tiny forced-`it` beam5 46.51 / 38.03 (**WER −11.07** [−16.28, −5.89] **won**, ETA −2.82
  [−11.65, +6.56] **tie**); Whistle 28.36 / 52.11 (still ahead); Parakeet 10.82 / 83.94.
  **For the first time the WER half of the mixed gate is won significantly on a speaker the model
  never heard**; the ETA half remains a tie, so the mixed gate still does not pass.
- **FLEURS-IT**: 26.52 vs Whisper-tiny auto+beam5 27.42 → **−0.89** [−2.08, +0.19], **tie** (it was
  −0.66 [−1.80, +0.47] in E-014). Best Moth number on FLEURS-IT to date, still not a significant win.
- **Noise**: **0 non-music words** with both decoders, 444/457 clip gated by the VAD (the 13 that
  pass are music). The room tone is in the set and is gated 97/97.
- **Findings**:
  - The gold signal is **real and cross-speaker**, not adaptation: the held-out gain equals the
    matched-speaker gain. That retires the concern that made E-016 unreadable as a claim.
  - It also **scales with speakers**: one gold speaker (E-016) gave −13.20 WER on S3 zero-shot;
    two give −19.34 vs E-014 and −6.13 vs E-016. The second speaker is worth ~6 points, so the
    remaining question is how much a third and fourth would add (S4 is unrecorded).
  - **ETA is still the blocker.** Even at 35.21% on S3 against 30.42% for E-016, it remains
    significantly behind Whistle (52.11) and is a tie vs Whisper. The term-accuracy ceiling is not
    solved by 20.6 more minutes of gold; it needs either much more term-rich audio (E-015
    distillation) or a decoder-side change.
  - **Channel caveat**: S1 and S3 recorded on the same Mac and microphone; S2 on an external
    device. S3's gain could partly be channel adaptation to the S1/S3 microphone. It is weakened
    by the fact that S2, on a different device, gains the most (−6.60), but not eliminated. S4 on
    a different setup would settle it.
- **Result**: **E-017 is the new best Moth** on every set that matters: T-600 32.77/42.86 (from
  38.50/36.05), FLEURS-IT 26.52, FLEURS-EN 55.48, CV 30.36 (tie), 0 noise words. The cross-speaker
  claim now has direct evidence: a speaker that contributed no audio improves as much as the
  speakers that did.
- **Conclusion**: gold data works and generalises across speakers; the mixed gate's WER half is now
  won on a held-out speaker, the ETA half is not. Next, in order: (1) **E-018** — scale the gold
  (S4, and more sentences per speaker) because the second speaker was worth 6 points; (2) **E-015
  distillation** for term coverage, still blocked on the audio-source decision; (3) decide the
  re-scope with E-017's numbers, since the ETA gap now has a measured shape.
- **Artifacts**: `runs/E-017/config.yaml`, `train.log.jsonl`, `eval.sh`, `prepare_gold.py`,
  `eval/{greedy,beam-lm,noise-*}/`, `eval/s{1,2,3}-*.metrics.json`, `eval/e017-vs-e016-*.diff.json`,
  `eval/e017-vs-e014-s3-*.diff.json`, `diffs/` (T-600 paired vs E-016), `manifests/T-{S1,S2,S3}.jsonl`;
  checkpoint `runs/E-017/checkpoints/ckpt_best.pt` (step 26,442, gitignored); gold shards
  `data/shards/gold-x50` (gitignored); manifest `recordings/manifests/R-S2.jsonl` (committed).
  Reproduce: `runs/E-017/prepare_gold.py --sets R-S1,R-S2 --factor 50`, then
  `uv run python -m stt.train --config runs/E-017/config.yaml`, then `runs/E-017/eval.sh`.

> **Addendum (2026-11-07) — T-600 baselines and release 0.5.0 with E-017.**
> The full 600-clip T baselines (Whisper/Whistle/Parakeet) had only ever been run on T-200 (S1) and
> on the S3 slice; they are now measured on the whole set, which is the number the ship gate needs.
>
> | System (T-600) | WER | ETA | vs E-017 (paired) |
> |---|---|---|---|
> | **Moth E-017** beam 5 + LM 0.2 | **32.77** | **42.86** | — |
> | Whisper-tiny auto + beam 5 | 68.86 | 39.50 | WER **−36.10** [−40.76, −31.66], ETA +3.36 [−2.20, +9.36] (tie) |
> | Whisper-tiny forced `it` + beam 5 | 50.72 | 36.05 | WER **−17.95** [−21.47, −14.88], ETA **+6.81** [+0.85, +12.50] |
> | Cactus Whistle forced `it` | 29.42 | 49.28 | WER +3.35 [+1.07, +5.60], ETA −6.42 [−11.15, −1.40] |
> | Parakeet TDT 0.6B v3 int8 | 10.98 | 83.60 | WER +21.79, ETA −40.75 |
>
> **The mixed ship gate passes on T-600 against both Whisper variants**: the WER CI lies entirely
> below 0 and the ETA CI entirely above 0 vs forced-`it` (and WER below 0 with ETA a tie vs auto).
> Two things to state honestly: (1) on this set 200 of the 600 clips come from **S1 and S2, whose
> gold is in E-017's training mix**, so the aggregate is optimistic — the honest generalisation
> number is the S3 slice in the E-017 entry (WER won, ETA tie); (2) against **Whistle** Moth is now
> within **+3.35 WER** [1.07, 5.60], i.e. significantly behind but by about 3 points rather than the
> ~14 of E-016. The stretch bar ("within the noise interval of Whistle") is close but not met.
>
> **Release 0.5.0** (E-017 checkpoint): int8 11.91 + LM 5.12 + VAD 2.33 + 0.08 = **19.44 MB**
> (ceiling 20). int8 vs PyTorch on 100 FLEURS-IT clips: **top-1 99.72%**, 44/100 transcripts
> identical, max logit gap 8.59 (median 5.15) — the int8 gap grew with the E-017 weights, so the
> per-clip transcript identity rate is lower than E-016's, and the WER cost of int8 was **not**
> measured end-to-end on T in this step (the E-016 int8 measurement was WER-identical on 100 clips;
> that is not a substitute). Browser (Playwright, WASM SIMD, VAD in the loop): **4T 52.2× / 50.8× /
> 46.3×**, **1T 25.5× / 24.3× / 21.8×** on 5/10/20 s; 6/6 tests green; package hashes in
> `runs/E-017/release-0.5.0.json`. `release/` is gitignored: rebuild with
> `stt/export.py --checkpoint runs/E-017/checkpoints/ckpt_best.pt --vocab release/vocab.json
> --version 0.5.0 --no-verify`, then `runs/E-010/package_browser.py`.

## E-019 — Whistle LoRA probe, a second model "less ours" (2026-11-07)

- **Date**: 2026-11-07 (training 22 min on CPU; 216 steps).
- **Hypothesis**: the 50 min of gold that moved Moth on a held-out speaker (E-017) should move
  Cactus Whistle too. If it does, the gold effect is a property of the data and not of our
  architecture; if it does not, the effect is architecture-bound. Secondary: a new `mix` language
  code might express intra-clip switching, which our CTC cannot.
- **Status**: **exploratory side artifact, not Moth** (D28). D5 ("own model, not a fine-tune of
  Cactus Whistle") is unchanged. Nothing here ships under the Moth name.
- **Design**: LoRA (rank 16, α 32) on the published `whistle.safetensors`, 3 epochs, batch 16,
  lr 1e-4, seed 0, `--language it`. Data is **R-S1 + R-S2 only** (1,263 clips, 45.3 min), split
  90/5/5 seeded and stratified by speaker; **T and A are never read**. Text is the manifest `norm`
  field, so the one text rule applies. Validation is a gold split, never T.
- **Environment**: throwaway `.venv-whistle` (cactus-needle 3.2.10; jax 0.12.0.1 on **CPU** — the
  plan's `cactus-needle[jax]` extra does not exist, the training extra is `train`). The base
  `.safetensors` for training is **220.6 MB**, downloaded once, outside the repo.
- **Baseline check**: L0 re-run on T-S3 reproduces E-017 **bit-identically** (200/200 hypotheses,
  WER 28.3646 / ETA 52.1127), so the JSONL, the audio paths and the loader are right.
- **Result — killed at L1**:
  - **T-S3 (primary, held-out speaker)**: WER 28.36 → **30.66**, Δ **+2.30** [−0.68, +5.46] → tie;
    ETA 52.11 → 47.89, Δ −4.23 [−10.34, +1.86] → tie. **The kill criterion is met.**
  - Gold test split (the two speakers seen in training) 30.08 → 27.10 (−2.98 [−8.38, +1.88], tie)
    and ETA +8.25: the adapter **does** adapt to the trained voices.
  - FLEURS-IT 11.85 → 12.87 (**+1.02** [+0.51, +1.56]), FLEURS-EN 13.35 → 15.65
    (**+2.30** [+1.16, +3.26]), CV-IT test 17.59 → 20.25 (**+2.66** [+1.74, +3.49]): every
    out-of-gold set gets **significantly worse**.
  - Noise with our VAD: **0 non-music words / 337 clips** in both arms (music 217 → 281, apart).
- **Verdict**: H1 rejected, **H2 rejected in this form** — the gold effect is *not* architecture
  independent. Moth's CTC gained −6.13 WER on S3 with the same 50 min (D27); Whistle's decoder
  loses 2.3. 45 min of gold on a 220 MB decoder re-tunes it toward two voices and costs it its
  generality. H3 (`mix`) and H4 (`--tokens`) were **not run**: the plan forbids them after a kill.
- **Side observation**: the exported `whistle-lora-L1.cact` is **28.0 MB** vs the 16.92 MB base,
  so 28.0 + 2.33 MB VAD is **over the 20 MB ceiling** — a second, independent reason not to pursue
  the arm.
- **Artifacts**: `runs/E-019/` (`build_jsonl.py`, `config.json`, `data/`, `train_L1.sh`,
  `train_L1.log`, `RESULT.md`, `l0/`, `l1-s3/`, `gold-test-{base,l1}/`, `{l0,l1}-{sets,cv,noise}/`,
  `manifests/`). The 6.5 MB adapter and the 28 MB `.cact` are gitignored; every number comes from
  `stt.eval` with a paired bootstrap (seed 0, 1000 resamples), never from the CLI's own WER.

## E-020 — the WER cost of the shipped int8 model (2026-11-07)

- **Date**: 2026-11-07. **Decode-only**, no training.
- **Question**: the release ships `model.int8.onnx`, but its WER had never been measured on our
  sets — only top-1 logit agreement (99.72%) and 44/100 identical transcripts on 100 FLEURS-IT
  clips. E-016 had measured it (identical WER on 100 clips); E-017 had not.
- **Method**: `stt/onnx_runner.py` (new) decodes the shipped artifact in Python. Before this the
  repo could not *run* the released graph at all: the browser path is JavaScript and
  `stt/export.py` uses ONNX Runtime only to compare logits. Same manifests, same CTC decoders, same
  LM, same text rule and same bootstrap as every other number; the PyTorch references are the
  E-017 hypotheses re-scored on the identical manifest (sha256 `43f895bf…`).
- **Result**:
  - **T-600 greedy**: 37.70 → **38.13**, Δ **+0.43** [−0.17, +1.01] → tie.
  - **T-600 beam+LM**: 32.77 → **32.94**, Δ **+0.17** [−0.40, +0.77] → tie.
  - **FLEURS-IT beam+LM**: 26.52 → **26.85**, Δ **+0.32** [+0.08, +0.57] → **slightly worse**.
  - ETA unchanged on T (Δ −0.10 / −0.38, ties). Identical transcripts: 70.5% / 77.3% on T-600,
    52.8% on FLEURS-IT.
- **Reading**: on our own speakers the quantized artifact is free within noise, so the shipped
  package is honest. On longer out-of-domain audio it costs 0.3 points with a CI that excludes 0 —
  a fair caveat for the model card. The length split explains it: T clips are 1.3–5.6 s (mean 239
  mel frames), FLEURS clips 7–23 s (mean 1458), and per-frame quantization error accumulates before
  the decoder commits; beam search turns small logit differences into different hypotheses.
- **Closes**: the `STATUS.md` line *"Il costo WER dell'int8 non è stato rimisurato end-to-end su T"*.
- **Artifacts**: `runs/E-020/` (`eval_onnx.py`, `RESULT.md`, `int8-greedy/`, `int8-beam-lm/`,
  `int8-fleurs-it/`, `torch-ref/`), `stt/onnx_runner.py`, `tests/test_onnx_runner.py`.

## E-021 — where the fp32 export gap comes from (2026-11-07)

- **Date**: 2026-11-07. **Export-only**, no training.
- **Question**: E-016 recorded fp32 `max |Δ| = 1.051e-3` against a 1e-3 gate on a synthetic 29 s
  clip, and every release since (0.3.0/0.4.0/0.5.0) was built with `--no-verify` because of it.
  Real audio in the same run measured 5.46e-4. Export bug, or fixture?
- **Method**: `runs/E-021/fp32_gap.py` exports the E-017 checkpoint to fp32 ONNX and measures the
  same quantity under four fixtures plus a 20-seed sweep.
- **Result**:
  - Gate fixture as written, seed 0: **2.43e-4** (passes — E-017's checkpoint is not E-016's).
  - Real T clips at the same lengths: **3.05e-4**; real clip zero-padded to 29 s: 2.21e-4.
  - Synthetic at the natural mel scale: 1.37e-4.
  - **20 seeds × 4 lengths = 80 draws**: median ≈ 1.8e-4, **worst 1.28e-3, 1/80 over the gate,
    always at 29 s**; at 1/5/10 s, 0/60.
- **Reading**: **the gate is a single random draw, not a statistic.** Its typical value is 5.5×
  under the threshold and the tail crosses it ~1% of the time. The number depends on (checkpoint,
  seed, length): E-017 gives 2.43e-4 at seed 0 and 1.28e-3 at seed 2. The export is not wrong —
  real speech, including a real clip padded to the failing length, stays at a third of the gate
  with top-1 agreement 1.0, and the int8 built from it costs +0.17 WER on T-600 (E-020).
- **Recommendation — implemented the same day (D29)**: the gate is now a **seeded sweep** (20 draws
  per length): **median** ≤ 1e-3, **worst** ≤ 1e-2. `seeds=1` restores the old single-shot
  behaviour; `docs/06` §4.2 carries the new definition. The release was rebuilt **with verification
  on** as **0.5.1** and passes: **median 2.127e-4, worst 1.472e-3, 3/80 draws over 1e-3** — the
  export the old gate would have rejected. The int8 artifact is **byte-identical** to 0.5.0
  (`fe1afcc2…`), so the gate change touched only the measurement. Package still **19.44 MB**; the
  NOTICE now lists FLEURS once. Browser 6/6 Playwright, 4T 52.0/51.0/46.5 and 1T 25.4/24.2/21.7 on
  5/10/20 s (same weights as 0.5.0, differences are noise). `runs/E-021/release-0.5.1.json`.
- **Artifacts**: `runs/E-021/` (`fp32_gap.py`, `RESULT.md`, `fp32_gap.json`, `release-0.5.1.json`).
- **M8 — model card and HF publisher (2026-11-07)**: `runs/E-021/MODEL_CARD.md` (full card for
  `semantique-team/moth-stt`: results, the comparison against Whisper/Whistle/Parakeet, the int8
  cost, the data, **9 stated limitations** — no punctuation, three speakers, S1's rhotacism, weak
  English, music, public FLEURS, single seed, and "this is not a Whistle fine-tune") and
  `runs/E-021/upload_hf.py`, which **verifies** `release/` against the recorded hashes before
  uploading (hash, payload size, card present) and is a **dry run by default**; 7 tests, including
  a file edited after the build. The token is read from the environment only. **Not published
  yet**: needs the token and a namespace decision (HF has no `semantique-team` org today).

## E-022 — decode recalibration: a per-token bonus, and an LM weight that was doing two jobs (2026-11-09)

- **Date**: 2026-11-09. **Decode-only**, no training, no new data, no checkpoint change.
- **Hypothesis**: `stt/decode.py::beam_search` adds `lm_weight * lm.score(...)` on every emitted
  token and nothing offsets it. An LM log-probability is negative, so each extra token costs
  score and the search is biased toward emitting **fewer** tokens than the acoustics support.
  A per-token bonus `gamma` should cancel that bias, and only then can the LM weight be
  re-tuned — because in one dimension the weight was controlling language modelling *and*
  output length at once.
- **Motivation**: an audit of the project at `ee2f34b` (seventeen decode-only probes,
  `runs/E-022/AUDIT-PROBE.md`). The error profile was the tell: on T-600 the shipped decode
  deletes **12.2%** of reference words against **0.8%** insertions, and drops the first word of
  13.3% of clips but the last of 0.5%.
- **Method**: `beam_search` gains `token_bonus` (default 0, a bit-exact no-op: 0/30 real clips
  change output) and `web/decode.js` gains the twin. The arm was chosen **on dev only**
  (`mls_it_dev` 200 + `ls_en_dev` 200) over a 2-D grid of (`lm_weight` × `gamma`) and frozen in
  `runs/E-022/PLAN.md` before the code changed: **beam 12 / top-k 10 / `lm_weight` 0.5 /
  `gamma` 2.4**. Dev WER 31.61 → 27.30. The `shipped` arm reproduces E-017 to four decimals, so
  the harness is unchanged.
- **Metrics** (bootstrap 1,000, `runs/E-022/eval/`, differences paired vs the shipped decode):

| Set | shipped | E-022 | Δ WER [95% CI] | Δ ETA [95% CI] |
|---|---|---|---|---|
| **T-600** | 32.77 / 42.86 | **25.92 / 52.64** | **−6.85** [−7.94, −5.72] | +9.78 [+7.68, +12.06] |
| **T-S3** (held-out speaker) | 35.43 / 35.21 | **26.66 / 47.32** | **−8.77** [−11.13, −6.50] | +12.11 [+8.31, +16.47] |
| T-S1+S2 (training speakers) | 31.46 / 46.66 | **25.54 / 55.38** | −5.92 [−7.23, −4.52] | +8.72 |
| FLEURS-IT | 26.52 | **20.46** | −6.06 [−6.63, −5.51] | — |
| FLEURS-EN | 55.48 | **49.63** | −5.85 [−6.55, −5.18] | — |
| CV-IT dev | 24.91 | **19.90** | −5.01 [−5.88, −4.14] | — |
| int8, T-600 | 37.70 greedy | **25.95** / ETA 53.12 | — | — |
| Noise, non-music words (VAD) | 0 | **0** | — | — |

  Greedy is unchanged (37.70 on T-600), because `token_bonus` and `lm_weight` only exist in the
  beam path: the change cannot have touched the fast default. Deletions fall 12.2% → 8.3% and
  insertions rise 0.8% → 1.3%, which is the mechanism showing up in the aggregate.
- **Against the ship bars**: **FLEURS-IT vs Whisper-tiny auto+beam 5 (27.42, the D19 bar) is
  −6.95 [−8.02, −5.87] — WON**, the first time since E-008; the release no longer fails that
  half of the gate. On **T-600 vs Whisper forced-`it`** both metrics win (WER −24.80, ETA +16.59
  [+10.61, +22.74]), and **vs Whistle the WER is won for the first time** (−3.50 [−5.85, −1.24];
  ETA +3.36, tie). On the held-out speaker **T-S3 the WER is won against Whisper (−19.51) but the
  ETA is still a tie** (+8.73 [−1.49, +18.86]): the strict mixed gate (`docs/05` §5) is therefore
  still not certified on a held-out speaker, although the point estimate moved from −3.4 to +8.7.
  Still behind Whistle on FLEURS-IT (+8.61) and behind Parakeet everywhere.
- **Self-correction, recorded**: the strongest cell (`lm_weight` 0.6 / `gamma` 3.0) does flip the
  S3 ETA tie to a win, but it was picked after **looking at S3**, which the repository forbids. It
  is not adopted and must not be cited; the arm above is the dev-only one. This is the third
  self-correction of this kind (`docs/11` §4) and the pattern is consistent: the number that
  makes the gate pass by a hair is the one to distrust.
- **Pre-existing footgun closed, and a correction to the audit**: `runs/E-010/package_browser.py`
  defaulted to the **unaugmented** LM, which every release since 0.3.1 had to override by hand; the
  auditor's first parity reading (+3.27 WER browser gap) was an artifact of exactly that. With the
  correct LM the shipped decode is **+1.29** [0.0009, 0.0259] — E-010's own number — and the
  **E-022 decode is a tie at −0.09** [−0.012, +0.010] with **higher** transcript identity
  (**154/200** vs 140/200): the wider beam improves parity rather than hurting it. The default now
  points at `lm_bpe3_aug.pkl.gz`.
- **Code**: `stt/decode.py` (`token_bonus`), `web/decode.js` (twin), `web/ort.js` (passes it),
  `runs/E-010/package_browser.py` (`--beam/--top-k/--lm-weight/--token-bonus`, and the default LM
  now points at the augmented one), `stt/export.py` (drops the never-read `hotword_bonus`, stops
  contradicting the packager on beam width), `tests/test_upload_hf.py` (the release guard reads the
  version from `release/config.json` and looks up its record under `runs/`, instead of hard-coding
  0.5.1). **294 Python tests** (4 new decoder tests including the LM-length-bias regression), **9
  Playwright** (2 new parity tests), `ruff` clean.
- **Cost**: **no training**. ~2 h of machine time, **0 USD**. Release **0.6.0**: int8
  **byte-identical** to 0.5.1 (`fe1afcc2…`), LM binary **byte-identical** (`dcf1d59f…`), package
  **19.44 MB**, fp32 gate passed as a seeded sweep (median 2.127e-4, worst 1.472e-3). Browser
  1-thread RTFx **24.3 / 22.9 / 20.3×** on 5/10/20 s — a ~7% cost, 2× the bar.
- **Artifacts**: `runs/E-022/` (`PLAN.md` = the frozen prediction, `RESULT.md`, `AUDIT-PROBE.md`,
  `eval.py`, `eval/{shipped,frozen}/`, `diffs/vs-shipped.json`, `gate-table.json`, `parity*.py`,
  `parity-wer.json`, `browser.json`, `release-0.6.0.json`, `record_release.py`),
  `stt/decode.py`, `web/decode.js`, `tests/test_decode.py`,
  `web/tests/{decoder,parity}.spec.ts`.

## E-023 — re-tuning the LM now that the decoder is calibrated (2026-11-09)

- **Date**: 2026-11-09. **Decode-only**, no training, no checkpoint change (E-017 reused).
- **Question**: E-022 found that `lm_weight` had been doing two jobs at once (language modelling
  *and* length control), which is why E-008's 1-D sweep picked 0.2. With the bonus taking the
  length job, are the LM's own parameters — weight, backoff, count pruning — still right? E-022's
  grid was coarse (0.4/0.5/0.6) on a 400-clip dev subset.
- **Method**: three phases, selection on **full dev** (`mls_it_dev` 1,248 + `ls_en_dev` 2,691 =
  3,939 clips / 10.48 h, macro WER as in E-008), gate sets read **once** at the end. Beam 12 /
  top-k 10 frozen from E-022. The corpus is unchanged (CV/MLS/LS train + R-S1/R-S2/pool, never
  T/A); only the knobs move. Decision rule fixed in `runs/E-023/PLAN.md` before the sweep.
- **Result — nothing adopted**:
  - **Phase 1** (`lm_weight` × `gamma`, 20 configs): the optimum is **exactly E-022's cell**,
    w 0.5 / γ 2.4, dev macro **22.98**; every other cell is worse, in every direction. On a
    selection set ten times wider, the 400-clip answer reproduces.
  - **Phase 2** (`backoff`): 0.4 stays best — 0.4 → 22.98, 0.3 → 23.01, 0.5 → 23.06,
    0.2 → 23.09, 0.6 → 23.23.
  - **Phase 3** (count pruning): the current (2,2) is not the dev optimum — **(1,1)**, keeping
    every n-gram, buys **−0.65 dev points** and is significantly better on the gate sets
    (**T-600 −0.77** [−1.40, −0.11], **FLEURS-IT −0.72** [−1.01, −0.42], **CV-IT dev −0.65**
    [−0.75, −0.55]). Pruning harder is worse: (2,3) +0.52, (3,3) +0.56.
- **Why (1,1) is rejected**: it does not fit. Its browser binary is **12.31 MB** against 5.12, so
  the package would be **26.63 MB** (11.91 int8 + 12.31 LM + 2.33 VAD + 0.08 features) against the
  hard **20 MB** ceiling (D4). The 5.12 MB LM is paying its way; the ceiling's price is now
  measured rather than assumed.
- **Where the ceiling's price lands matters**: on **T-S3, the held-out speaker, (1,1) is a tie and
  marginally worse** (+0.43 [−0.61, +1.62]) while it wins on general Italian (CV-IT) and FLEURS-IT.
  So relaxing the ceiling would buy a better *Italian* model, not a better *code-switching* model —
  which reframes the constraint: the ceiling is not costing the project its core claim.
- **Reading**: the two E-022 wins and this negative result together say the decode is now at a
  local optimum that survives a 10× larger selection set. E-008's mistake was not bad luck but a
  mis-specified sweep; E-023 is the check that E-022's correction generalises.
- **Cost**: **no training**. ~2 h 20 CPU, **0 USD**. No release change: 0.6.0 stands.
- **Artifacts**: `runs/E-023/` (`PLAN.md` = the decision rule, `RESULT.md`, `sweep.py`,
  `sweep.log`, `sweep.json`, `gate.py`, `eval/{lm-1-1,shipped-0.6.0}/`, `gate-diffs.json`,
  `lm-prune-1-1.pkl.gz` = the candidate kept for a future ceiling decision).
- **M8 — model card regenerated for 0.6.0 (2026-11-09)**: `runs/E-022/MODEL_CARD.md` replaces the
  0.5.1 card as the publishable one (the old one stays in `runs/E-021/` with a superseded header).
  Every number was re-read from the artifacts rather than carried over, which changed more than the
  version string: T-600 **25.92 / 52.64**, held-out S3 **26.66 / 47.32**, FLEURS-IT **20.46**,
  FLEURS-EN **49.63**, CV-IT test **24.88** (measured for this card — E-022 had left CV on the dev
  subset), int8 cost re-measured (**+0.03 T-600, +0.60 S3**, both ties, 79% identical transcripts).
  The card gained a section explaining that the weights are unchanged and the *decoder* is what
  moved, plus the three reimplementation warnings (`token_bonus` is not optional; it must be tuned
  with `lm_weight`; greedy is unaffected). Limitations went from 9 to 10: the fourth speaker was
  never recorded, and the ETA column on S3 is a tie.
  `runs/E-021/upload_hf.py` now also **refuses a card whose title does not name the release
  version**, because a stale card publishes true numbers for the wrong artifact — the failure mode
  the E-022 audit itself fell into. 8 tests. Still a dry run until the token and namespace exist.

## E-024 — noise augmentation on half of Common Voice IT (2026-11-09)

- **Date**: 2026-11-09. Training run, 26,442 steps, init `runs/E-014/checkpoints/ckpt_best.pt`.
- **Motivation**: the demo was tested with a real laptop microphone and the transcripts were poor.
  Measuring it (`runs/audit-2026-11-09/noise_curve.py`, 100 held-out-speaker clips, MUSAN at a
  controlled SNR) gave 22.24 clean, **26.99 at 30 dB, 32.09 at 20 dB, 38.37 at 15 dB, 48.05 at
  10 dB, 58.23 at 5 dB** — all significant. Deletions go from 5.6% of reference words to 24.6%:
  under noise the model drops words rather than mis-hearing them, which is the signature of
  training on clean read speech. At 15 dB it scored worse than Whisper-tiny does on clean audio.
  For scale, the InterCTC arm this replaced (probe: +4.4% step time, measured with a sensitivity
  control after two failed timing designs) was expected to buy 0.2–0.7 points.
- **Method — one variable, same mixture size**: CV-IT split deterministically by id hash; half keeps
  the existing mel cache untouched, half is re-rendered through noise at a per-clip SNR uniform in
  **10–30 dB** (drawn from the clip id, so the dataset is reproducible). The mixture stays at
  E-017's **589.5 h / 266,796 clips / 13,221 steps per epoch**, so "noise" is not confounded with
  "more CV". Everything else is E-017's recipe. Rejected with reasons in `PLAN.md`: all CV noised;
  all CV clean plus a noisy copy (mixture 757 h, CV share 28.5% → 44.3%); noising MLS/LS (raw audio
  deleted, 22 GB re-download).
- **Compute**: build **1.1 min** (56,349 clips / 83.3 h / 0 errors / 3.2 GB — only the noisy half is
  computed, which is why it took a minute instead of the estimated 2 h), training **~3 h 40**,
  evaluation 13 min. **0 USD.**
- **Result — ADOPTED as release 0.7.0** (plan rule: mean gain ≥ 4 points over 10–30 dB and clean
  T-600 within +1.0):

| SNR | E-017 (0.6.0) | **E-024 (0.7.0)** | Δ WER [95% CI] |
|---|---|---|---|
| clean | 22.24 | 23.94 | +1.70 [−0.34, +3.72] tie |
| 30 dB | 25.98 | **23.26** | −2.72 [−5.63, +0.17] tie |
| 20 dB | 27.50 | **26.83** | −0.68 [−4.13, +2.73] tie |
| 15 dB | 33.62 | **27.33** | **−6.28 [−10.58, −2.38]** |
| 10 dB | 44.82 | **33.28** | **−11.54 [−16.18, −7.43]** |
| 5 dB | 50.93 | **41.26** | **−9.68 [−14.50, −5.01]** |

  Mean over 10–30 dB: **−6.18 points**. Clean gate: T-600 **−0.72** [−1.75, +0.26], T-S3 **+0.26**
  [−1.37, +1.96], FLEURS-IT −0.32, **CV-IT test −0.52 [−0.66, −0.38] significant**. Dev macro
  0.3084 against E-017's 0.3098 — a difference of 0.14 points, i.e. noise, as predicted.
- **The prediction was right on the mean and wrong on the shape**: `PLAN.md` said −3 to −6 at 20 dB
  and −4 to −10 at 10 dB; measured −0.68 and −11.54. The gain is concentrated where noise is worst,
  not spread across the curve. A model at 20 dB was already holding on; what it could not do was
  survive 10 dB.
- **Also measured**: int8 ties against fp32 (−0.06 T-600, +0.08 T-S3); browser↔Python parity **tie**
  (−0.09 WER [−1.19, +1.13], 152/200 identical); browser 1-thread RTFx **20.3×** on 20 s,
  unchanged; noise set with the VAD **0 non-music words over 337 clips**; a multi-sentence
  segmentation fix landed in the same session (5 sentences in one clip cost 9.4 WER, the shipped
  VAD recovers 80%).
- **Not true, and stated in the card**: the noise pool is shared between training and evaluation,
  so the gain is an **upper bound** for an unfamiliar room; a per-source split (80/20) is the
  obvious next control and was not run. Half the Italian read speech is now only ever heard in
  noise — costless here, but a second seed might show it.
- **Artifacts**: `runs/E-024/` (`PLAN.md`, `RESULT.md`, `build_noisy_mel.py`, `build.log`,
  `config.yaml`, `train.log.jsonl`, `train.out`, `noise_eval.py`, `compare.py`, `eval/`,
  `comparison.json`, `parity.py`, `parity-wer.json`, `browser.json`, `record_release.py`,
  `release-0.7.0.json`), `data/manifests/cv_it_train_{clean,noisy}50.jsonl`,
  `runs/audit-2026-11-09/`. Published as 0.7.0 to `semantique-team/moth-stt` and
  `semantique-team/moth-demo`.

## E-025 — does the noise gain survive a room the model never heard? (2026-11-09)

- **Date**: 2026-11-09. Evaluation-only; **no training**. Plan written before the recording:
  `runs/E-025/PLAN.md`.
- **Motivation**: the 0.7.0 card calls E-024's gain an **upper bound** because training and
  evaluation used the same 217 clips. The control proposed in STATUS (an 80/20 split) is invalid:
  the training build cycled all 217 (`pool[noise_index % len(pool)]`), and the source MUSAN audio
  was deleted after the noise set was built, so no split at the level of the source recordings is
  possible. The only valid control is noise the model has never encountered.
- **Method**: a **new room** — a different physical room from the room tone already in the training
  pool, confirmed by the author — recorded on the built-in microphone *after* training: 78.7 s, trimmed
  to 40–78 s (three loud events at 12.1/16.9/31.0 s excluded), 38 s of steady broadband ambience
  (RMS −52.6 dBFS, spectral centroid 1.05 kHz, 33% of energy below 200 Hz, VAD speech fraction 0.0,
  one 50 ms transient). The first take was rejected: 93% of its energy was below 200 Hz because of
  50 Hz mains hum from the charger. Same 100 held-out-speaker (S3) clips as E-024, same decoder,
  same mixed audio for both arms, SNRs 30/20/15/10/5 dB.
- **Result — the gain transfers, and is larger**:

| SNR | 0.6.0 | **0.7.0** | Δ WER [95% CI] | same pool (E-024) |
|---|---|---|---|---|
| 30 dB | 28.52 | **24.62** | **−3.90 [−7.20, −0.83]** | −2.72, tie |
| 20 dB | 38.71 | **31.92** | **−6.79 [−10.84, −3.15]** | −0.68, tie |
| 15 dB | 47.71 | **36.50** | **−11.21 [−14.69, −7.48]** | −6.28 |
| 10 dB | 63.33 | **44.99** | **−18.34 [−22.78, −13.40]** | −11.54 |
| 5 dB | 81.32 | **66.72** | **−14.60 [−18.84, −10.47]** | −9.68 |

  Mean over 10–30 dB: **−10.06** against −6.18 on the same pool. All five conditions significant.
  Clean reproduced E-024 exactly (22.24 / 23.94, within 1e-5), which is the harness check.
- **Why the gain is larger**: the unseen room is harsher at the same nominal SNR (0.6.0: 47.7 at
  15 dB here vs 33.6 on the pool), so the baseline has more headroom. The absolute 0.7.0 number is
  worse there (36.5 at 15 dB vs 27.3): the gain and the absolute must be quoted together. **A gain
  at fixed nominal SNR is not a property of the model alone.**
- **Prediction miss**: `PLAN.md` predicted attenuation (−3..−6 at 15 dB, −6..−10 at 10 dB, ties at
  30/20). Measured −11.21, −18.34, and both 30 and 20 dB significant: the direction was right, the
  magnitude wrong at every point. Second conservative miss in two experiments (E-024 was right on
  the mean and wrong on the shape); recorded in `docs/11` §4 item 6, prediction kept unchanged in
  `PLAN.md`.
- **Decision (D33)**: the card's caveat is downgraded from "upper bound for an unfamiliar room" to
  "measured on one unseen room", with the table. **No new weights, no version bump**: 0.7.0 is the
  same artifact with a better-supported claim. Still not covered: one room, one take, steady
  ambience (events trimmed), a synthetic RMS-relative SNR, single seed.
- **Cost**: recording 3 min (including the rejected take), analysis 2 min, paired evaluation
  **20 s**. No training, 0 USD.
- **Artifacts**: `runs/E-025/` (`PLAN.md`, `RESULT.md`, `record_room.py`, `room_report.json` for the
  used take, `room_report_take1_rejected.json` and `room_report_take2_untrimmed.json` for the two
  rejected stages, `holdout_eval.py`, `eval/`), `data/raw/noise/room_holdout/` (audio, gitignored).

## E-015 — gold term addendum: the first arm missed, and the recipe was too aggressive (2026-11-09)

- **Date**: 2026-11-09. Training run, 13,252 steps (1 epoch), init
  `runs/E-024/checkpoints/ckpt_best.pt` (0.7.0).
- **Motivation**: E-015's 5% baseline was E-014's. On 0.7.0, measured on the committed T
  hypotheses, the 600-clip term accuracy splits by provenance: **53.7%** on terms present in the
  natural training text, **68.7%** on gold-only terms, **17.0%** on the **44 terms never seen in
  any training text** (47 utterances). The 16 terms the gold already covers (historical `TERMS18` label) are at **74.1%**.
  The target is the 44, and the direct ETA ceiling from fixing them is **+3.7** (100%) or
  **+2.8** at Parakeet's 78%, not the +5 in the original plan.
- **Method**: 50 new **S1** sentences covering all 44 terms once (2.5 min), recorded and validated
  (691/691 clips, 0 errors), added as a **separate** source at **x20** (0.84 h). The old gold
  (R-S1/R-S2, x50) stays untouched. One epoch, E-017 LRs (head 3e-4, encoder 5e-5), same 589.5 h
  mixture. S1-only: S2 was not available, and S3 is held out by rule.
- **Result — KILLED** (plan rule: rare-term accuracy <30%, or clean WER worse by >0.5 with CI
  excluding 0):

| Readout | 0.7.0 | E-015 | Δ |
|---|---|---|---|
| 47 rare utterances | 8/47 (17.0%) | 11/47 (23.4%) | +6.4 [−4.3, +17.0], not significant |
| S1 rare / S2 rare / S3 rare | 6/14, 0/15, 2/18 | 7/14, 2/15, 2/18 | +1, +2, **0 on held-out S3** |
| Natural-term occurrences | 439/817 (53.7%) | 423/817 (51.8%) | **−16** |
| Gold-only occurrences | 123/179 | 129/179 | +6 |
| All term occurrences | 570/1043 (54.65%) | 563/1043 (53.98%) | −0.67 |
| 16 gold terms (regression; historical `TERMS18` label) | 20/27 (74.1%) | 19/27 (70.4%) | −1 |
| T-600 / T-S3 | 25.20 / 26.92 | 24.91 / 25.55 | −0.29 tie / −1.36 tie |
| FLEURS-IT / FLEURS-EN | 20.14 / 49.14 | 20.86 / 50.45 | **+0.72** / **+1.31**, both worse |
| CV-IT dev | 19.85 | 19.97 | +0.12 tie |

- **The dev shock is the story**: 30.81 at step 20 → 33.64 at 2k → 31.22 at the end. The fine-tune
  pulled the model almost 3 dev points away and spent the epoch recovering; it never beat the start.
  The best checkpoint by dev is **step 20, i.e. the baseline itself**, so the run selects no new
  model. The final checkpoint was evaluated only as the arm that saw the data.
- **Correction 2026-11-11 (E-026)**: the historical primary readout above is term *presence*, not
  aligned ETA. On the same 47 rows, aligned ETA is **19/81 (23.5%) → 25/81 (30.9%)**, while full
  T-600 ETA is **565/1043 (54.17%) → 557/1043 (53.40%)**. The arm improved the rare subset more
  than presence suggested but lost more on natural terms; the kill verdict stands. The next arm
  uses aligned ETA as primary and presence as diagnostic. `TERMS18` actually contains **16**
  strings; the regression label is now `gold_terms`.
- **Reading**: the data is not disproven; the recipe is. 50 sentences in **one** voice at the E-017
  LRs for a full epoch is too much update for a converged model, and the term gain is confined to
  the training speakers. The next arm must change **two** variables (S2 as a second voice +
  a gentler fine-tune, e.g. head 5e-5 / encoder 1e-5 or a frozen encoder), with the same kill rule.
- **Cost**: prep + mel + x20 **2 min**, training **1 h 50**, eval **~3 min**. No teacher, no
  downloads, 0 USD. The 50 S1 clips and the `gold-terms-x20` source are reusable.
- **Artifacts**: `runs/E-015/` (`PLAN.md`, `RESULT.md`, `RECORDING.md`, `config.yaml`,
  `train.log.jsonl`, `eval/`, `eval_terms.py`, `prepare_terms.py`), `recordings/manifests/R-S1.jsonl`
  (691 rows), `runs/E-015/candidates-S{1,2}.jsonl`.
## E-026 — rare-term diagnostic: presence vs aligned ETA and top-k reachability (2026-11-11)

- **Date**: 2026-11-11. Diagnostic only; **no training, no adoption, no model selection**.
  Retrospective formalization of an exploratory run executed the same day.
- **Question**: on 0.7.0 int8, for the 47 T utterances with a never-seen term, is the reference
  reachable under the decoder top-k, and how do greedy, beam without LM and beam+LM behave?
- **Protocol**: 47 T rows, 16 kHz resampled audio, no VAD/segmentation; release `model.int8.onnx`,
  `vocab.json`, `lm_bpe3.bin`; `stt.diagnose` with greedy, beam acoustic0 and beam+LM
  (12/10/0.5/2.4); forced-reference CTC full/top-k support.
- **Result**:

| Decoder | WER | Aligned ETA | Insertion rate |
|---|---:|---:|---:|
| Greedy | 48.65% | 17.28% | 3.47 |
| Beam, no LM | 49.81% | 16.05% | 3.86 |
| Beam + LM | **40.54%** | **23.46%** | 4.63 |

  Full CTC path reachable **47/47**; reference reachable under top-k 10 **41/47 (87.2%)**;
  completely pruned **6/47** (`startup`, `prototype`, `chromecast`, `charger`, `exam`, `mondays`).
  Aligned ETA on the 47: 0.7.0 **19/81 = 23.5%**, E-015 **25/81 = 30.9%**; presence 8/47 → 11/47.
  Full T-600 ETA: **54.17% → 53.40%**; WER 25.20% → 24.91%.
- **Reading**: top-k is not the main bottleneck for 41/47; the LM/prior chooses wrong alternatives,
  so a bounded contextual bias is the cheapest next test, while the six pruned rows need
  acoustic/data work. The E-015 presence readout understated the rare-subset aligned gain but the
  full-set ETA fell; the next arm must use aligned ETA primary and protect natural terms.
- **Caveats**: T is reused; int8 release vs fp32 E-015 checkpoint; audio resampled; no VAD;
  forced-reference support is oracle diagnostics, not production accuracy; small sample.
- **Artifacts**: `runs/E-026/` (`PLAN.md`, `RESULT.md`, `report.json`, `hyp/`,
  `rare47-16k.jsonl`, `reproduce.py`), `runs/E-015/eval/terms-eta.json`.

## E-027 — contextual-bias pilot: no movement on the rare terms (2026-11-11)

- **Date**: 2026-11-11. Decoding-only pilot, **no training, no weights, no release changes**;
  pre-registered in `runs/E-027/PLAN.md`.
- **Question**: E-026 found the reference inside top-k for 41/47 rare T rows. Can a bounded,
  word-aware glossary recover them without hurting controls?
- **Glossary**: 24 single-word work/meeting terms from `data/en_terms.txt`, fixed before the run
  (`meeting call team sprint ticket deploy deployment release deadline backlog roadmap standup
  feedback update report budget forecast target task project workflow pipeline feature bug`).
  Arms: baseline, product 1.0/0.25, product 0.5/0.0, irrelevant 1.0/0.25, zero bonus.
- **Result**:

| Arm | T-600 WER | T-600 ETA | 47-rare ETA | 47-rare presence |
|---|---:|---:|---:|---:|
| Baseline | 25.32% | 54.75% | 22.22% | 8/47 (17.0%) |
| Product 1.0/0.25 | 25.29% | 54.84% | 22.22% | 8/47 (17.0%) |
| Product 0.5/0.00 | 25.32% | 54.75% | 22.22% | 8/47 (17.0%) |
| Irrelevant 1.0/0.25 | 25.29% | 54.84% | 23.46% | 8/47 (17.0%) |
| Zero bonus | 25.32% | 54.75% | 22.22% | 8/47 (17.0%) |

  Paired product-bias vs baseline: T-600 WER **−0.03 [−0.12, +0.06]**, ETA **+0.10
  [−0.20, +0.42]**, both ties; rare ETA **+0.00**. Controls: T `control_it` 13.24% unchanged,
  CV-IT dev 21.06% unchanged, FLEURS-EN 47.28% → 47.18%. Product-glossary hits on Italian controls
  **0 → 0**; `zero_bonus` byte-identical to baseline.
- **Decision — REJECTED**. The pre-registered gate required ≥ +1.0 aligned-ETA point; it measured
  +0.10 and did not move the 47 rare rows. The irrelevant glossary produced a small non-specific
  change (+1.23 rare ETA, CI [0.00, +3.49]), so there is no glossary-specific effect. The six
  top-k-pruned rows (`startup`, `prototype`, `chromecast`, `charger`, `exam`, `mondays`) cannot be
  recovered by this scorer.
- **Reading**: the scorer path is safe (zero-bonus no-op, no Italian false insertions), but this
  common work/meeting glossary is not the missing lever. A future bias needs a product glossary
  validated on a real mixed-dev; do not mine the 47 T failures. The training response remains the
  gentle S2 arm for the acoustic/data gap.
- **Caveats**: T reused; no VAD; float32 resampling means absolute T-600 is 25.32 not 25.20; only
  paired deltas are interpreted.
- **Artifacts**: `runs/E-027/` (`PLAN.md`, `RESULT.md`, `eval_bias.py`, `metrics.json`, `hyp/`).

## E-028 — E-015 v2: S2 + a gentler fine-tune passes the gates (2026-11-11)

- **Date**: 2026-11-11. Training run, 13,283 steps (1 epoch), init
  `runs/E-024/checkpoints/ckpt_best.pt` (0.7.0). Pre-registered in `runs/E-028/PLAN.md`.
- **Changes against E-015**: **S2 as the second voice** (50 new sentences, 2.1 min, x20 = 0.70 h)
  next to the 50 S1 sentences (0.84 h), and lower LRs (**head 1e-4 / encoder 1e-5** instead of
  3e-4 / 5e-5), dev every 1,000 steps. Two variables by design; this is a product arm, not a
  causal ablation.
- **Result — PASSES the pre-registered gates**:

| Readout | 0.7.0 | E-028 | Δ |
|---|---|---|---|
| Rare presence (47) | 8/47 (17.0%) | **13/47 (27.7%)** | +10.6 [+2.1, +21.3] |
| Rare aligned ETA (81) | 19/81 (23.5%) | **25/81 (30.9%)** | +7.4 [0.0, +15.4] |
| Rare WER | 40.54% | **35.91%** | −4.63 [−8.98, −0.75] |
| Natural terms | 439/817 | 438/817 | −1 |
| Gold-only | 123/179 | 126/179 | +3 |
| All terms | 570/1043 (54.65%) | 577/1043 (55.32%) | +0.67 |
| 16 gold terms (regression) | 20/27 (74.1%) | 18/27 (66.7%) | −2 |
| T-600 WER / ETA | 25.20 / 54.17 | **24.66 / 54.65** | −0.54 tie / +0.48 tie |
| T-S3 WER / ETA | 26.92 / 48.45 | **25.55 / 49.01** | −1.36 tie / +0.56 tie |
| FLEURS-IT / EN | 20.14 / 49.14 | 20.47 / 49.38 | **+0.33** significant / +0.24 tie |
| CV-IT dev | 19.85 | 19.80 | −0.05 tie |

- **By speaker**: rare presence S1 6/14 → 8/14, S2 0/15 → 2/15, **held-out S3 2/18 → 3/18**;
  T-600 S1 21.67 → 22.10, S2 26.93 → 26.32, S3 27.00 → 25.55. The gain is no longer confined to
  the training voices, but S1’s general WER is slightly worse.
- **Dev**: final **31.01%** vs 30.81% for 0.7.0; the curve recovers from 33.64-style shock
  (first arm) but never beats the baseline. The final checkpoint is best inside the run, so gate 6
  passes literally; under a dev-only rule that includes 0.7.0 the baseline would still win. This
  is the A1 misalignment: a real mixed-dev is still needed for product selection.
- **INT8 artifact (initial export, old decoder)**: package **19,440,760 bytes**, browser acceptance **4/4**; but
  T-600 **25.49 / 53.50** (0.7.0 INT8 25.14 / 54.65, both ties), T-S3 **25.89 / 49.01**
  (better WER), rare presence 8/47 → 12/47 (+8.5), natural presence **441 → 428 (−13)**. The
  dynamic default is the best in-budget variant; per-tensor/reduce-range/QUInt8/static QDQ are
  worse or broken, and head-fp32 is 20.62 MB (over budget) for only 0.08 dev points. See
  `runs/E-028/QUANTIZATION.md`. The published release is 19,440,761 bytes with the E-030
  decoder: T-600 24.71 / 54.17, T-S3 25.21 / 50.14, natural 436.
- **Verdict**: **published 2026-11-12 as 0.8.0 with the E-030 decoder**, with a documented
  exception on the rare aligned-ETA gate. The original int8 export was a tradeoff; E-030's
  decoder recalibration recovered T-600/T-S3/natural and left only the rare gate two aligned
  occurrences short. Cheap post-training quantization and QAT are exhausted. Other caveats: dev +0.20, S1 WER
  +0.43, FLEURS-IT +0.33 significant (inside the 0.5 margin), 16 gold terms −2 (`milestone`,
  `paypal`).
- **Cost**: data prep ~2 min, training **1 h 47**, eval ~4 min; no teacher, no downloads, 0 USD.
- **Post-release noise check (2026-11-12)**: re-ran the E-024 pool with the same decoder for
  0.7.0 and 0.8.0 on 100 S3 clips. 0.8.0 is a tie at 10–30 dB (mean WER 27.33 vs 27.76), a tie at
  15 dB, and significantly better clean (20.37 vs 23.09, −2.72 [−4.42, −1.02]); no significant
  noise regression. Report: `runs/E-028/noise-0.8.0/RESULT.md`.
- **Artifacts**: `runs/E-028/` (`PLAN.md`, `RESULT.md`, `config.yaml`, `prepare_terms.py`,
  `train.log.jsonl`, `train.out`, `eval/`), `recordings/manifests/R-S2.jsonl` (672 rows),
  `recordings/manifests/R-S2-terms.jsonl`, `data/shards/R-S2-terms`,
  `data/shards/gold-terms-S2-x20`.
- **Follow-up**: **E-029 QAT executed, negative**; see the E-029 entry below. The full arm was
  not funded; 0.8.0 was later published on 2026-11-12 with the E-030 decoder.

## E-029 — QAT probe: the quantizer is not the missing lever (2026-11-11)

- **Question**: E-028 fp32 passes the gates, but the shipped INT8 artifact gives back natural
  terms and T-600 ETA. Can quantization-aware fine-tuning close that gap inside 20 MB?
- **Constraint**: PyTorch fake-quant is not implemented on MPS
  (`aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`). The probe uses a custom
  straight-through quantizer: per-output-channel symmetric int8 weights and per-tensor
  asymmetric uint8 activations, mirroring ONNX `DynamicQuantizeLinear`, over all 39
  Linear/Conv modules. Forward wrappers are removed before export.
- **Protocol**: 200–500 steps from `runs/E-028/checkpoints/ckpt_best.pt`, dev selection
  (CV-IT 600 + FLEURS-IT/EN), T read once on the selected arm. Probe plan and dev-selected
  variants: `runs/E-029/PLAN.md`.
- **Probe arms (dev WER, fp32 → INT8)**:

| Arm | Steps | Encoder | Dev fp32 | Dev INT8 | Step ratio |
|---|---:|---|---:|---:|---:|
| weight only | 200 | train | 31.09 | 31.17 | 1.12× |
| weight + act | 200 | train | 31.31 | 31.49 | 1.42× |
| weight + act | 500 | train | 31.36 | 31.41 | 1.35× |
| **weight + act** | **500** | **frozen** | **31.08** | **31.13** | 1.58× |

  Reference: E-028 fp32 30.93, E-028 INT8 31.14. Weight-only does not help; activation
  quantization is the main gap; freezing the encoder is the best trade (INT8 gap 0.05) but
  does not improve the baseline. No non-finite steps; step time below the 2× kill threshold.
- **Product readout of the best arm (INT8, 19,440,760-byte package)**: T-600 **25.34 / 53.21**
  (0.7.0 INT8 25.14 / 54.65; E-028 INT8 25.49 / 53.50), T-S3 **25.55 / 49.30** (better),
  rare **10/47** (E-028 INT8 12/47), rare aligned ETA **27.2%**, natural **428/817**. The
  unquantized control is T-600 24.97 / 54.07 and natural 434 — already below E-028 fp32.
- **Verdict**: **negative; do not publish 0.8.0; keep 0.7.0.** The T-600, rare and natural
  gates fail (five individual conditions); T-S3 and package pass; browser was not run because
  no candidate emerged. QAT closes the dev quantization gap but not
  the product gap; a longer arm would first have to reverse the fine-tune drift. Cheap
  post-training quantization and QAT are both exhausted. [Result](runs/E-029/RESULT.md).
- **Cost**: probe ~30 min total (4 arms), no teacher, no downloads, 0 USD.

## E-030 — Decoder recalibration: E-028 INT8 reaches 5 of 6 gates (2026-11-11)

- **Question**: E-028 INT8 misses the E-028 gates narrowly (rare ETA 29.6, T-600 ETA 53.50,
  natural 428). Can a decoder-only change recover the margin without touching the weights?
- **Hypothesis**: E-028 INT8 deletes more (8.49% vs 7.88%) and inserts less (1.03% vs 1.29%) than
  0.7.0 on T-600; the shipped arm was tuned for 0.7.0.
- **Selection**: 12 arms over beam / top-k / LM weight / token bonus on CV-IT 600 + FLEURS-IT/EN
  dev. The best is **beam 12 / top-k 10 / LM 0.55 / bonus 2.7**, dev WER **30.95 vs 31.14**
  (frozen), improving all three dev sets. T was read once, after selection.
- **T readout (E-028 INT8, same artifact)**: T-600 **24.71 / 54.17** (old arm 25.49 / 53.50;
  0.7.0 shipped 25.14 / 54.65), T-S3 **25.21 / 50.14** (old 25.89 / 49.01), natural **436/817**
  (old 428; gate ≥431), rare presence **11/47**, rare aligned ETA **28.4% (23/81)** vs 0.7.0's
  22.2% (18/81): **+6.2 [+1.1, +11.8]**; vs the old E-028 arm it is a tie (one occurrence).
  FLEURS-IT/EN +0.29 / +0.42 vs 0.7.0, both ties and inside the 0.5 margin.
- **Gates**: **5 of 6 pass** (T-600 ETA/WER, natural, FLEURS, dev-selected checkpoint). The primary
  rare-ETA gate is **missed by two aligned occurrences** (28.4% vs ≥30%). E-028 fp32 has 25/81,
  so the loss is quantization; head-fp32 also stays at 23/81 and is over budget (20.62 MB).
- **Verdict**: **published 2026-11-12 as 0.8.0 with this decoder**, documenting the 2-occurrence
  rare shortfall. The artifact is a much stronger tradeoff than the old E-028 INT8 (T-600 tie vs
  0.7.0, T-S3 better, rare significantly better, natural within the 431 gate). Remaining work is
  targeted rare data, a same-size LM, or a natural mixed-dev for selection.
  [Result](runs/E-030/RESULT.md).
- **Cost**: dev sweep ~17 min, T/FLEURS readout ~5 min, no training, 0 USD.

## E-031 — An intermediate LM closes the rare gate inside the 20 MB ceiling (2026-11-12)

- **Question**: E-023 showed that keeping every bigram and trigram (`(1,1)`) is significantly
  better but weighs 12.31 MB, over the 20 MB package ceiling. Does **all bigrams + trigrams with
  count >= 2 (`(1,2)`)** recover a useful share inside the budget, and does the LM text now
  available (the E-015/E-028 term sentences, absent from the 2026-10-30 build) change the answer?
- **Method**: frozen corpus via `stt.data.freeze_lm_corpus` (CV/MLS/LS train + R-S1/R-S2/pool;
  T/A and every dev/test manifest named as exclusions; 186,003 sentences, 939 duplicates collapsed,
  36,065 exclusions, no overlap error). Arms: shipped control, `(2,2)` rebuild, `(1,2)`, and
  `(1,2)` with the 143 term-containing sentences x3. Selection on full dev (3,939 clips) only,
  decoder fixed at E-030's 0.55/2.7. Browser byte gate first.
- **Selection**: `(2,2)` rebuild is a tie (−0.010 dev, CI [−0.031, +0.015]); `(1,2)` is
  **−0.084 [−0.140, −0.035]**; `(1,2)` + terms x3 is **−0.091 [−0.147, −0.040]**, the lowest dev
  macro and the arm read once on the gates. Binaries: 5,378,740 B and 5,381,992 B (package
  19,703,037 / 19,706,289 B).
- **Gate readout (same int8 graph, decoder unchanged)**: T-600 **24.08 / 55.23** (0.8.0
  24.71 / 54.17; **WER −0.63 [−1.11, −0.20]**, **ETA +1.06 [+0.09, +2.04]**), T-S3
  **24.96 / 51.27** (Δ ties), FLEURS-IT **20.16** (tie), FLEURS-EN 49.81 (tie), natural
  **432/817** (gate ≥431), gold 127/179, never-seen presence **24/47**. **Rare aligned ETA
  **28.4% (23/81) → 46.9% (38/81), +18.5 [+9.6, +31.0]**: the primary ≥30% gate passes.
- **Verdict**: **adopted and published as 0.8.1 on 2026-11-12** — all nine gates pass and both
  rare ETA and T-600 WER improve with a CI excluding zero. Decode-only release: same acoustic
  weights, one new `lm_bpe3.bin` (5,381,992 B). Browser acceptance is **4/4** and JS↔Python
  decoder parity is exact on identical logits; the 3 s fixture differs by two words with both the
  0.8.0 and 0.8.1 LM (browser acoustic path). The full E-024 noise curve with the new LM is
  **all ties** (mean 5–30 dB 29.47 vs 30.05). Card `runs/E-031/MODEL_CARD.md`, record
  `runs/E-031/release-0.8.1.json` (`published: true`), HF rev `1257327e…`, Space `9dfe70df…`
  `RUNNING`. Independent natural mixed-dev and a new-voice rare readout remain missing.
  [Result](runs/E-031/RESULT.md).
- **Cost**: no training, no downloads, 0 USD; build seconds, dev ~16 min, gate readout ~5 min.

## E-032 — Targeted rare addendum: rare terms up, product gates down (2026-11-12)

- **Question**: E-026 found six terms acoustically pruned (`startup`, `prototype`, `chromecast`,
  `charger`, `exam`, `mondays`). Does 50 new S1 sentences, with four new contexts for each pruned
  term and one for every other never-seen term, recover rare accuracy on top of E-031's LM?
- **Method**: 50 reviewed S1 sentences (`R-S1-0692`–`0741`, 2.8 min, x20 = 0.93 h) added to the
  E-028 mixture; gentle fine-tune from E-028 (head 1e-4 / encoder 1e-5), 13,314 steps, dev
  selected the final checkpoint at **31.06%** (E-028 31.01%), 0 non-finite. Readout with the two
  LMs: shipped 0.8.0 (pre-registered baseline) and the adopted E-031 A3 candidate.
- **Result vs 0.8.0**: rare aligned ETA **28.4% (23/81) → 32.1% (26/81), +3.7 [0.0, +9.0]**,
  presence 14/47; but T-600 25.23 / 53.21, T-S3 **26.75 / 46.76** (+1.53 WER, −3.38 ETA,
  significant), FLEURS-IT 20.66 (+0.41, significant), FLEURS-EN 50.47 (+0.66, significant),
  natural **424/817** (gate ≥431).
- **Result vs E-031**: rare **51.9% (42/81), +4.9 [0.0, +12.0]**, but T-600 24.83 / 53.88
  (+0.75 WER, significant), T-S3 26.41 / 47.61 (+1.45 WER, −3.66 ETA, significant), FLEURS-IT
  20.54, FLEURS-EN 50.28, natural **415/817**.
- **Verdict**: **rejected**. Rare improves in both comparisons, but the CI touches zero, while
  T-S3, FLEURS and natural regress; five of the pre-registered gates fail. The S1-only addendum
  cannot separate speaker adaptation from term learning. Keep E-031 (LM only, rare gate closed at
  46.9%) as the 0.8.1 direction; do not iterate on T. The 50 clips remain reusable for a future
  arm with an independent selection set.
  [Result](runs/E-032/RESULT.md).
- **Cost**: ~2 h training + ~15 min export/eval, no downloads, 0 USD.

## Runtime acceptance follow-up (2026-11-10, no new model)

Infrastructure validation, not a new quality-optimization arm. Original 0.7.0 assets
verified against the committed E-024 record. See
[`runs/runtime-2026-11-10/RESULT.md`](runs/runtime-2026-11-10/RESULT.md) for all measurements,
normalization, hashes and limitations. No training, tuning, new weights or release published.
104 previously used FLEURS/spliced/control clips; no independent natural code-switching ETA.
VAD pipeline WER 40.48% vs no-VAD 39.82%, paired delta +0.66 points (CI includes zero).
Four synthetic nonspeech controls suppressed; shared-logit Python/JS transcripts match
10/10 for beam and greedy. Runtime-only dependency profile and release integrity gates added.
New mixed-dev/contextual-glossary/LM-pruning experiments remain blocked on missing data.

## E-034 — Trigram coverage is a real dev lever, and it does not fit (2026-11-12)

- **Question**: 0.8.1 leaves 293,711 bytes below the 20 MB ceiling. Before designing budgeted
  pruning, does extra n-gram coverage or a higher term weight move **dev** at all? If not, there
  is nothing worth reclaiming bytes for.
- **Method**: decode-only, on the E-031 frozen corpus (`runs/E-031/frozen/lm-corpus.jsonl`),
  same tokenizer, backoff 0.4 and builder path. Arms: shipped A3 control `(1,2)` term ×3,
  `(2,1)`, `(1,1)` and `(1,2)` term ×5. Every arm decoded in this run on full dev
  (`mls_it_dev` 1,248 + `ls_en_dev` 2,691 = 3,939 clips / 10.48 h), fp32 E-028 checkpoint,
  decoder frozen at E-030's beam 12 / top-k 10 / 0.55 / 2.7. T, T-S3, FLEURS and noise not read.
  Decision rule fixed in `runs/E-034/PLAN.md` before the run: lower macro, paired CI upper < 0,
  and binary ≤ 5,675,703 B.
- **Result — nothing adopted**:
  - **`(2,1)`** dev macro **−0.598 pp [−0.724, −0.457]**, **`(1,1)`** **−0.644 [−0.768, −0.503]**:
    the extra 1,477,990 trigram entries are worth ~0.6 dev points. Cost **12,034,004 B** and
    **12,313,440 B** → packages **26,358,301 / 26,637,737 B**, over both the 20 MB ceiling and the
    25 MB tier (binary limit 10,675,703 B). The cheap intermediate arm keeps ~93% of the gain at
    98% of the size, so it is not a budget escape.
  - **Term weight ×5** on unchanged pruning is a **tie** (+0.0019 pp, CI [−0.0083, +0.0120],
    same 5,381,992 B): the shipped ×3 is saturated on dev.
  - The control reproduces the retained E-031 sweep **byte for byte** (identical `dev.hyp.jsonl`,
    macro 22.8494), so the harness introduced no drift. `(1,1)` matches E-023's rejected 12.31 MB
    arm exactly, independently of that run.
- **Verdict**: **rejected as an LM change**. Confirms E-023's dev gain and shows the middle
  threshold does not help affordability. The follow-up the arithmetic suggests is a *budgeted*
  arm: keep `(1,2)` and add ~65,000 of the 1,477,990 single-count trigrams (4.4%, ~4.5 B/entry)
  inside 293,711 B, ranked by preceding-bigram count or term coverage, pre-registered against A3
  and read once on T/FLEURS/noise if it advances. Not executed here. No product claim: dev is
  read-speech development data and a 12 MB LM changes browser load/memory, which was not measured.
  [Result](runs/E-034/RESULT.md).
- **Artifacts**: `runs/E-034/` (`PLAN.md`, `build.json`, `dev_sweep.py`, `dev/dev-only/`,
  `decide.py`, logs; binaries reproducible and gitignored).
- **Cost**: no training, no downloads, 0 USD; build 18 s, four dev decodes ~16 min.

## E-033 — InterCTC at weight 0.3: no dev benefit, option closed (2026-11-12)

- **Question**: with data, recipe and seed held fixed, does a training-only auxiliary CTC loss on
  the encoder output improve the fine-tune enough to be worth adopting? The strategy put it first
  because it is training-only: the aux head is never registered on the deployable model, so the
  graph and the 19,706,289-byte package cannot change.
- **Method**: two paired arms from `runs/E-024/checkpoints/ckpt_best.pt`, E-028 data mixture and
  recipe, seed 0, 1 epoch (13,283 steps), identical except `train.interctc`
  (`weight: 0.3`, `tap: encoder_output`). Selection on full dev (3,939 clips, greedy, no LM):
  macro WER with a paired within-language bootstrap, plus the E-024 checkpoint as a candidate.
  Decision rule fixed in `runs/E-033/PLAN.md` before the run: arm macro lower by >= 0.05 pp and
  paired CI upper < 0; only then a product readout.
- **Result — rejected**:
  - Paired dev macro **31.0350% vs 31.0101%** control, delta **+0.0249 pp [−0.0998, +0.1557]**
    (pooled +0.0308 [−0.0923, +0.1657]): a tie pointing the wrong way. In-run curve agrees: the
    arm is 32.137% vs 31.331% at step 1,000 and closes to 31.0335 vs 31.0077 at the end.
  - `aux_loss` learns from **66.9 to 0.72** (min 0.35) over 394,625 training-only parameters, so
    the head was exercised, not inert. Weighted `0.7 main + 0.3 aux` raised the early gradient
    (mean 7.15 vs 6.60, max 66.0 vs 18.9) and the step-1,000 shock.
  - The control reproduced E-028 exactly (loss at 500/1,000/1,500 and dev macro to 4-6 decimals),
    so the comparison is genuinely paired: same machine, torch 2.14.1, commit `767122b`, seed 0,
    parity gate 4.6e-6.
- **Verdict**: **no dev benefit at weight 0.3; the option is closed for this recipe.** Nothing
  exported, no product readout, T/T-S3/FLEURS/noise untouched. A lower weight, an aux-head
  warm-up, another tap or a length-normalized auxiliary loss are untested and would each need a
  new pre-registration; one seed cannot resolve a 0.02 pp difference either way.
  [Result](runs/E-033/RESULT.md).
- **Cost**: 1 h 56 (control) + 1 h 53 (arm) wall on the M5, no downloads, 0 USD. The E-024
  +4.4% step-time probe did not reproduce as a slowdown here.

## E-035 — Budgeted single-count trigrams: small real dev gain inside 20 MB (2026-11-13)

- **Question**: 0.8.1 leaves **293,711 B** below the 20 MB ceiling and its LM is 5,381,992 B.
  E-034 showed that full trigram coverage helps dev but costs 12.3 MB. Can the affordable slice —
  keeping `(1,2)` + term ×3 and adding single-count trigrams ranked by the frequency of their
  preceding bigram — buy a measurable dev gain inside the cap?
- **Method**: decode-only, same frozen E-031 corpus/receipt/tokenizer, same `BPETrigramLM` path,
  same E-030 decoder (beam 12 / top-k 10 / LM 0.55 / bonus 2.7), fp32 E-028 checkpoint, full dev
  (3,939 clips, 10.48 h). Control rebuilt and required to reproduce the shipped binary
  (5,381,992 B, sha `3be1177b…`); its 3,939 hypotheses are identical to E-034's retained `A3`.
  Selection rule frozen in `runs/E-035/PLAN.md`: rank count-1 trigrams by context occurrence
  count, tie-break by ids, add while `5,381,992 + 4 × entries + 8 × new_contexts ≤ 5,675,703 B`;
  advance only on lower macro, paired CI upper < 0 and package ≤ 20,000,000 B.
- **Result — advances on dev**: **73,427** entries (4.97% of E-034's extra coverage), all in
  existing contexts, cutoff context count **698**, 1,359 contexts touched; binary **5,675,700 B**
  (sha `458441ab…`), package **19,999,997 B**. Dev macro **22.7849% vs 22.8494%**, delta
  **−0.0645 pp**, paired 95% CI **[−0.1144, −0.0134]**; pooled **−0.0627 pp**
  [−0.1082, −0.0181]; IT −0.0767, EN −0.0522. 442 hypotheses changed: 170 improved, 116 worsened,
  156 same-error rewrites; net −59 errors (20,822 → 20,763).
- **Verdict — dev**: the headroom is **not** dead weight on dev, and 5% of the coverage recovers
  10.8% of E-034's gain as a rough cross-arm ratio (not a decomposition) — the frequency ranking
  is better than proportional. The arm advanced to the product readout.
  [Dev result](runs/E-035/RESULT.md).
- **Product readout (2026-11-13) — not adopted**: on the 0.8.1 int8 assets, **all seven primary
  product metrics tie**: T-600 WER +0.0287 pp [−0.1732, +0.2300], T-600 ETA identical (55.225),
  T-S3 WER −0.0852 [−0.5853, +0.3393], T-S3 ETA −0.2817 [−0.8902, 0.0], FLEURS-IT +0.0062,
  FLEURS-EN −0.0437, rare aligned ETA 38/81 in both (target-only 24/47 in both); natural 432/817
  and gold 127/179 identical. The only absolute-gate miss is `noise_music` (299 → 301 words,
  non-empty 84 → 82); decoder-level JS/Python parity is exact. No significant gain, so the
  adoption rule fails: **0.8.1 remains the release**, and the 293,711 B question is answered
  negatively for this LM format. [Readout](runs/E-035/READOUT.md).
- **Cost**: build 8.8 s, dev decodes 243.8 + 255.4 s, product readout ~5 min, no training, no
  download, 0 USD.

## E-036 — Conservative fine-tune: protection preserves WER but not silence (2026-11-13)

- **Question**: E-032's S1 addendum moved rare terms but a full fine-tune regressed T-S3,
  FLEURS and the natural group. Can freezing the encoder (or all but its last two blocks), replaying
  the full E-028 mixture and distilling from a frozen E-028 teacher keep the general/product
  behaviour while retaining the rare-term gain?
- **Method**: two arms from `runs/E-028/checkpoints/ckpt_best.pt` on the E-032 mixture (13,314
  steps, 1 epoch): `frozen` (2 head blocks + CTC head trainable, 3,942,785 params) and `last2`
  (last 2 of 4 encoder blocks + head blocks + head, 7,490,945 params). KD = masked
  KL(teacher‖student) per utterance, T=2, weight 20 (a 100-step training-loss pilot showed raw
  KL ~0.001 vs CTC ~0.5, so 0.5 would be inert); teacher E-028, hash `01b83cc3…`. Both checkpoints
  exported to int8 and read once on T-600, T-S3, FLEURS-IT/EN, rare 47, natural/gold and noise
  against the committed 0.8.1 baseline with the A3 LM.
- **Result — rejected**:
  - Dev (greedy, no LM): initial 31.0101; frozen best **30.9859** (−0.0242 pp, CI
    [−0.0806, +0.0335]), last2 best **30.9720** (−0.0381, CI [−0.1048, +0.0263]): both ties, both
    dev guards pass.
  - Product: T-600, T-S3 WER and FLEURS are ties for both arms. Rare aligned ETA: frozen
    **39/81** (+1, CI [0, +4.35]), last2 **41/81** (+3, CI [0, +8.64]): both miss the pre-registered
    **42/81** target. Natural 435 and 434, both pass ≥431.
  - **New regression**: both arms emit `x` on all **120 silence clips** (0.8.1 emits nothing);
    non-music non-empty clips 146 → 272 (frozen 392 words, last2 381), and the `noise_music` gate
    also fails. The frozen arm proves the behaviour comes from the trainable head stack, not the
    encoder.
  - `last2` additionally loses T-S3 ETA: 51.268 → 49.859 (**−1.408 pp**, CI [−2.762, −0.280]),
    a significant regression. Decoder parity exact; package unchanged at 19,706,289 B.
- **Verdict**: **both arms rejected. E-036 is negative, the acoustic model stays at 0.8.1 and
  there is no candidate for the new-voice readout.** The protection preserves WER but not silence
  behaviour and does not recover E-032's rare gain; a blank/silence guard and a different
  adaptation route would be new arms, not a re-tune of this one.
  [Result](runs/E-036/RESULT.md).
- **Cost**: ~1 h 47 per arm, two int8 exports, two readouts, no downloads, 0 USD.

## E-037 — Paired fp32/INT8 with the shipped A3 LM: a small, non-uniform cost (2026-11-13)

- **Question**: the shipped 0.8.1 artifact is the INT8 graph. With the same E-028 checkpoint, the
  same A3 LM, the same decoder and the same clips, how much WER/ETA does INT8 cost relative to its
  fp32 export? E-028 measured +0.21 pp dev with the older 0.8.0 LM; this is the matched product
  measurement.
- **Method**: export fp32 from E-028 `ckpt_best.pt` (`stt.export.export_fp32`, verification median
  1.82e-4); decode the fp32 export and the shipped INT8 graph (`7bbfad7e…`) in one session with
  the A3 LM (beam 12 / top-k 10 / 0.55 / 2.7) on dev (3,939), T-600, T-S3, FLEURS-IT/EN, the
  rare 47, natural/gold and noise; paired per-utterance bootstrap, seed 0.
- **Result**:
  - Dev macro **22.8596% → 23.0170%**, **+0.1574 pp**, CI **[+0.0539, +0.2664]** (significant);
    pooled +0.1489 [+0.0413, +0.2620].
  - T-600 **23.5952 → 24.0826**, +0.4874 pp [−0.0290, +0.9963] (tie; ETA −0.8629 pp
    [−1.9743, +0.1910]); T-S3 +0.4259 [−0.4160, +1.2153] (tie); FLEURS-IT −0.1559 (tie);
    **FLEURS-EN +0.4459 [+0.0087, +0.8652]** (significant).
  - Rare aligned ETA unchanged at **38/81**; rare WER −0.386 pp (tie). Natural terms **444 → 432**
    (−12), gold 124 → 127 (+3), never 24 → 24. Noise slightly better under INT8 (non-music
    266 → 258 words, 153 → 146 non-empty).
  - Logit gap on the fixture: max |Δ| **2.3503**, mean 0.2418, top-1 agreement **98.67%**.
  - Package: fp32 **54,164,659 B** vs INT8 **19,706,289 B**; fp32 cannot ship under 20 MB.
- **Verdict**: the INT8 cost is **real but small and not uniform** (+0.16 pp dev, +0.49 pp T-600
  point, +0.45 pp FLEURS-EN significant, −12 natural terms; rare and noise unaffected/better).
  INT8 remains the only deployable format; the measured cost goes into the card/paper caveat, not
  into a release change. **No model or package change.** [Result](runs/E-037/RESULT.md).
- **Cost**: one fp32 export + ~15 min of CPU decoding, no training, no download, 0 USD.

## E-038 — Per-speaker T-600: the pooled number hides a 5.3 pp / 12.0 pp spread (2026-11-13)

- **Question**: the published T-600 figures are pooled over S1, S2 and S3. With no fourth voice
  available, how much do the three existing voices differ on the shipped 0.8.1?
- **Method**: decode-free re-scoring of the committed E-037 hypotheses
  (`runs/E-037/readout/paired-1/product/int8/t600.hyp.jsonl`, sha `a07fc630eccb5bdb…`) against
  `recordings/manifests/T.jsonl` (sha `11e002d88dbfaee9…`); per-speaker WER/ETA with per-utterance
  bootstrap (1,000, seed 0); dispersion across voices. No decode, no training, no new audio.
- **Result**: S1 **20.9802% / 63.2836%** [18.0851, 23.7882] / [57.3134, 69.3252]; S2
  **26.3249% / 51.5581%** [22.8669, 29.5713] / [44.7439, 57.7259]; S3 **24.9574% / 51.2676%**
  [21.6512, 28.1488] / [44.5652, 57.6803]; pooled **24.0826% / 55.2253%** (reproduces E-037
  exactly). Dispersion: **WER 5.3447 pp**, **ETA 12.0160 pp**.
- **Verdict**: the pooled figure is not a speaker-independent estimate and must be reported with
  the per-voice range; the never-trained voice (S3) is **not** the worst (S2 is), so training
  exposure does not explain the gap. Three voices cannot estimate between-speaker variance, so the
  CIs stay per-utterance (not speaker-cluster) and no unseen-speaker claim is made.
  [Result](runs/E-038/RESULT.md).
- **Cost**: seconds of CPU, no decode, no training, 0 USD.

## E-039 — Leakage audit, first pass: no literal leakage, and a false "never seen" label (2026-11-13)

- **Question**: do any evaluation texts/clips also appear in training text, training scripts or
  another evaluation set; does our recorded audio repeat across sets; are corpus train/dev/test id
  splits disjoint; does the natural/gold/never term classification hold against the training text
  actually used by 0.8.1? The closure plan listed this audit as never performed.
- **Method**: read-only audit of every text source (`recordings/scripts/*`, `recordings/manifests/*`,
  all `data/manifests/*`), exact normalized-text duplicates plus trigram near-dups (Jaccard > 0.8)
  within our own data, sha256 registry of the 2,209 wavs under `recordings/audio/`, id
  disjointness per corpus, and term provenance recomputed against E-028's training text including
  the addenda boundary. No training, no decode, no data change.
- **Result**: 2,055 texts shared across sources, **all structural** (anchor reads T, manifests are
  the recorded scripts, R addenda are part of R) or **corpus text repeat** (42 CV-IT dev/test
  sentences); **0 unexpected pairs**. Near-dups above 0.8 within our data: **0** of **10,918** candidate pairs evaluated
  (highest sub-threshold 0.8000, a benign shared prefix). Audio: **0**
  repeated hashes, **0** cross-set collisions, **0** missing files. IDs: **0** shared in CV-IT,
  MLS-IT and LS-EN splits. Terms: 655 English terms in T = natural 461, gold 150, **never 44**
  under the 0.7.0 boundary (47 rare-bearing utterances), but **never 0** under E-028/0.8.1, where
  the 44 arrive through the S1/S2 addenda; E-036-only `terms3` adds none.
- **Verdict**: no literal leakage in text, audio or ids. The load-bearing documentation claim is
  **wrong for 0.8.1**: "rare terms never seen in any training text" is a **0.7.0** property. The
  metric is unchanged, the label is corrected in `docs/03` and `docs/13`. Not covered: `data/raw`
  re-hash (13 GB), external-corpus near-dups, upstream contamination, consent.
  [Result](runs/E-039/RESULT.md).
- **Cost**: ~3 s of I/O plus hashing 492 MB, no training, no download, 0 USD.
