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
- **Conclusion**: hypothesis confirmed, with one open scope note (CV IT pending). Proceed to E-001 (H0 baselines).
