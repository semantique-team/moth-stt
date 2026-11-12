# 15 - Session handoff, 2026-11-11

**Purpose.** This is the resume document for the session that merged PR #3, recorded S2, ran E-028
and exhausted the cheap INT8 quantization routes. It is deliberately detailed so a fresh agent can
continue without reading the whole conversation.

> **Update, same day:** E-029 QAT was executed after this handoff and is **negative**. The probe
> closes the dev quantization gap but fails the T-600, rare and natural gates; the full arm was
> not funded. **E-030** then recalibrated the decoder on the same E-028 INT8 artifact: T-600
> 24.71 / 54.17, T-S3 25.21 / 50.14, natural 436, rare ETA 28.4%. That is **5 of 6 gates**; the
> primary rare gate is two occurrences short. 0.8.0 is **not** published and 0.7.0 stays online
> pending a product decision. See [`runs/E-029/RESULT.md`](../runs/E-029/RESULT.md),
> [`runs/E-030/RESULT.md`](../runs/E-030/RESULT.md) and the live `STATUS.md`.

## TL;DR

- **Published model: 0.7.0 / E-024, unchanged.** The Hugging Face model and the static demo still
  serve the 0.7.0 package; nothing was uploaded in this session.
- **Repo HEAD:** `9bd552c` on `main` (pushed). Earlier session commits: `f0441c0` (S2 data),
  `2ce66ba` (E-028), `eacba0b` (E-026/E-027), `7b4396c` (PR #3 merge).
- **S2 is recorded and integrated:** 50/50 new clips, R-S2 manifest 672 rows, R-S2-terms 50 clips
  (2.1 min, x20 = 0.70 h).
- **E-028 fp32 is a real candidate:** rare presence 17.0% → 27.7%, rare aligned ETA 23.5% → 30.9%,
  T-600 24.66 / 54.65, T-S3 25.55 / 49.01, natural terms 439 → 438.
- **E-028 INT8 is not a clean ship:** T-600 25.49 / 53.50 against 0.7.0 INT8 25.14 / 54.65 (ties),
  rare presence 8/47 → 12/47, natural 441 → 428. The post-training quantization sweep is exhausted
  (default is best in budget; head-fp32 over budget; static QDQ broken).
- **Decision:** 0.8.0 is **not published**. The next experiment is **E-029 QAT**
  (`runs/E-029/PLAN.md`), using a custom MPS-compatible straight-through quantization because
  PyTorch's fake-quant op is not implemented on MPS.

## Repository state and verification

- HEAD `9bd552c`, working tree clean except two untracked logs that are not ours:
  `runs/E-016/eval-aug-lm.log`, `runs/E-016/eval-aug-sets.log`.
- `ruff check .` clean; `pytest -q` **881 passed, 1 skipped**; Playwright 11/11; real
  Chromium/WASM acceptance on the E-028 INT8 release **4/4**.
- No secrets, no audio, no checkpoints, no `.onnx` variants in git. `runs/E-028/release/`,
  `runs/E-028/checkpoints/`, `runs/E-028/quant/*.onnx` are gitignored.

## What happened, in order

1. **PR #3 merged** (`7b4396c`): audit + tooling. The new audit was renumbered to
   `docs/14-architecture-audit-2026-11-10.md` because `main` already had docs 12 (ledger) and 13
   (model/paper readiness). Conflicts in `STATUS.md` and `docs/README.md` resolved.
2. **E-026 diagnostic** (`eacba0b`): 47 rare T utterances on 0.7.0 INT8. Full CTC path reachable
   47/47; top-k reachable 41/47; completely pruned 6/47 (`startup`, `prototype`, `chromecast`,
   `charger`, `exam`, `mondays`). Beam+LM 40.54% WER / 23.46% aligned ETA vs greedy 48.65% /
   17.28%. Aligned ETA on the 47: 19/81 = 23.5% for 0.7.0, 25/81 = 30.9% for E-015; full T-600 ETA
   54.17% → 53.40%. Result and reproduction: `runs/E-026/`.
3. **E-027 contextual bias** (`eacba0b`): pre-registered 24-term work/meeting glossary; **rejected**.
   Rare ETA 22.22% → 22.22%, T-600 ETA +0.10 (CI includes zero), no Italian false insertions; the
   irrelevant glossary also moved slightly. `runs/E-027/`.
4. **S2 recordings** (`f0441c0`): `~/Downloads/R-S2-addendum.zip` SHA-256
   `43f14f4986ebfc48af4549e229af842a5037b66475fa7435a01bbd3538395fb9`; 50 WAVs for cues 0624-0673,
   48 kHz mono, 1.94-3.32 s, no clipping/silence. Builder: 672/673 clips; the only error is the
   pre-existing 0.39 s `S2_0001.wav`, which remains excluded. No session log was returned, so
   retakes are unknown. `R-S2-terms.jsonl` is 50 clips, 2.1 min, x20 = 0.70 h. Mel
   `data/shards/R-S2-terms` and gold `data/shards/gold-terms-S2-x20` built.
5. **E-028 training** (`2ce66ba`): S2 + lower LRs (head 1e-4 / encoder 1e-5), 13,283 steps,
   1 h 47, 0 non-finite. It passes the pre-registered fp32 gates. See `runs/E-028/RESULT.md`.
6. **E-028 INT8 export and gates** (`9bd552c`): package **19,440,760 bytes** (same as 0.7.0),
   model hash `7bbfad7e288640538c5c0147f81738bfd63021727b8760b1f3cbbcc4f939764d`, browser 4/4.
   INT8 quality tradeoff below. Quantization sweep: `runs/E-028/QUANTIZATION.md`.

## E-028 fp32 numbers

| Readout | 0.7.0 | E-028 | Δ |
|---|---:|---:|---:|
| Rare presence (47) | 8/47 = 17.0% | **13/47 = 27.7%** | +10.6 [+2.1, +21.3] |
| Rare aligned ETA (81) | 19/81 = 23.5% | **25/81 = 30.9%** | +7.4 [0.0, +15.4] |
| Rare WER | 40.54% | **35.91%** | −4.63 [−8.98, −0.75] |
| Natural terms | 439/817 | 438/817 | −1 |
| Gold-only | 123/179 | 126/179 | +3 |
| All terms | 570/1043 (54.65%) | 577/1043 (55.32%) | +0.67 |
| 16 gold terms | 20/27 (74.1%) | 18/27 (66.7%) | −2 |
| T-600 WER / ETA | 25.20 / 54.17 | **24.66 / 54.65** | −0.54 tie / +0.48 tie |
| T-S3 WER / ETA | 26.92 / 48.45 | **25.55 / 49.01** | −1.36 tie / +0.56 tie |
| FLEURS-IT / EN | 20.14 / 49.14 | 20.47 / 49.38 | **+0.33** sign. / +0.24 tie |
| CV-IT dev | 19.85 | 19.80 | −0.05 tie |

By speaker on T-600: S1 21.67 / 63.28 → 22.10 / 62.39; S2 26.93 / 51.27 → 26.32 / 52.97;
S3 27.00 / 48.45 → 25.55 / 49.01. Rare by speaker: S1 6/14 → 8/14, S2 0/15 → 2/15,
**S3 2/18 → 3/18**. Five positive rare flips, no negative: `startup`, `screenshot`, `popup`,
`exam`, `checkout`. Gold regressions: `milestone` (“mileston”), `paypal` (“papal”).

Dev: final **31.01%** vs 30.81% for 0.7.0 (+0.20); the curve recovered from 31.53 at step 3,000,
but never beat the baseline. The final checkpoint is the best inside the run, but under a dev-only
selection that includes 0.7.0 the baseline would still win. This is the A1 misalignment: a real
mixed-dev is still missing.

## E-028 INT8 (the artifact that ships)

Package 19,440,760 bytes; browser 4/4; `model.int8.onnx` hash `7bbfad7e…`.

| Set | 0.7.0 INT8 | E-028 INT8 | Δ |
|---|---:|---:|---:|
| T-600 WER / ETA | 25.14 / 54.65 | 25.49 / 53.50 | +0.34 tie / −1.15 tie |
| T-S3 WER / ETA | 27.00 / 48.45 | 25.89 / 49.01 | −1.11 tie / +0.56 tie |
| Rare presence | 8/47 | 12/47 | +8.5 [+2.1, +17.0] |
| Rare aligned ETA | 18/81 = 22.2% | 24/81 = 29.6% | +7.4 |
| Natural presence | 441/817 | 428/817 | −13 |

## Quantization sweep (selected on dev, not T)

Dev = CV-IT dev 600 + FLEURS-IT dev 334 + FLEURS-EN dev 343.

| Variant | Dev WER | Package | Verdict |
|---|---:|---:|---|
| fp32 reference | 30.93 | 53.90 MB | not deployable |
| **INT8 default** | **31.14** | **19.44 MB** | **best in budget** |
| INT8 head fp32 | 31.06 | 20.62 MB | over budget, +0.08 only |
| INT8 per-tensor | 31.26 | 19.33 MB | worse |
| INT8 QUInt8 | 31.30 | 19.45 MB | worse |
| INT8 reduce_range | 31.56 | 19.44 MB | worse |
| Static QDQ | **91.44** | 19.52 MB | broken |

No cheap post-training variant closes the gap. Head-fp32 would require pruning ~0.62 MB of LM
for a 0.08 dev gain; not worth it. **0.8.0 is not published.**

## Decisions to carry forward

- **D36**: do not publish E-028 INT8 as 0.8.0; keep 0.7.0. The fp32 arm is a candidate, but the
  shipped INT8 artifact gives back natural terms and T-600 ETA.
- **D37**: fund **E-029 QAT** as the next experiment. PyTorch fake-quant is unsupported on MPS
  (`aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`), so E-029 needs a custom
  straight-through quantization path; see `runs/E-029/PLAN.md`.

## Reproduction commands

```bash
# S2 data (already done; audio stays out of git)
uv run python runs/E-028/prepare_terms.py --set R-S2
uv run python -m stt.data.build_mel \
    --manifest recordings/manifests/R-S2-terms.jsonl \
    --audio-root recordings/audio/R-S2 --out-dir data/shards/R-S2-terms
uv run python runs/E-017/prepare_gold.py --sets R-S2-terms --factor 20 \
    --out-dir data/shards/gold-terms-S2-x20

# Training and fp32 gate
uv run python -m stt.train --config runs/E-028/config.yaml
uv run python runs/E-022/eval.py --arm frozen \
    --checkpoint runs/E-028/checkpoints/ckpt_best.pt \
    --sets t600,t-s3,fleurs_it_test,fleurs_en_test,cv_it_dev \
    --out runs/E-028/eval/gate
uv run python runs/E-015/eval_terms.py \
    --hyp runs/E-028/eval/gate/t600.hyp.jsonl \
    --baseline runs/E-024/eval/gate/t600.hyp.jsonl \
    --out runs/E-028/eval/terms.json

# INT8 artifact + gate
uv run python -m stt.export --out runs/E-028/release \
    --checkpoint runs/E-028/checkpoints/ckpt_best.pt \
    --vocab release/vocab.json --head-blocks 2 --version 0.8.0
uv run python runs/E-028/package_release.py
uv run python runs/E-022/eval.py --arm frozen \
    --checkpoint runs/E-028/checkpoints/ckpt_best.pt --int8 \
    --release runs/E-028/release --sets t600,t-s3 --out runs/E-028/eval/int8
```

The `release/` and `runs/E-028/release/` directories are gitignored; the E-028 record is
`runs/E-028/release-0.8.0.json`.

## Artifacts and where to look

- `runs/E-028/RESULT.md` — full result, gates and caveats.
- `runs/E-028/QUANTIZATION.md` — INT8 sweep and release tradeoff.
- `runs/E-028/PLAN.md` — pre-registered E-028 protocol.
- `runs/E-028/release-0.8.0.json` — hashes, package size, fp32/int8 metrics, browser acceptance.
- `runs/E-026/RESULT.md` — diagnostic; `runs/E-027/RESULT.md` — rejected bias pilot.
- `runs/E-029/PLAN.md` — next QAT experiment.
- `docs/11` §3.8-§3.10, D35-D37 — decisions and corrected interpretations.
- `STATUS.md`, `EXPERIMENTS.md` — live state and experiment log.

## Open risks / what not to do

- Do not publish 0.8.0 without a product decision; 0.7.0 is the current published model.
- Do not use T or S3 for checkpoint/quantization selection; T is reused evidence.
- S3 must never contribute training audio.
- The audio ZIP is in `~/Downloads`; it is not in git. The S2 session log was not returned.
- The release package is 19.44 MB; the 20 MB ceiling is a hard constraint. fp32 is ~53.9 MB.
- QAT is not a guaranteed win; the E-029 probe must be killable before a full run.
