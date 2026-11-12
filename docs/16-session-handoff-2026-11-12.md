# 16 - Session handoff, 2026-11-12: 0.8.0 published

**Purpose.** Resume document for the session that turned E-028 into the published **0.8.0**
release, after the negative QAT probe (E-029) and the decoder recalibration (E-030). Read this
first, then `STATUS.md` and the linked experiment reports.

## TL;DR

- **Published model: 0.8.0**, checkpoint **E-028** + decoder **E-030**. Same architecture and
  package size as 0.7.0; only the checkpoint, decoder settings and card changed.
- **Model:** https://huggingface.co/semantique-team/moth-stt
- **Demo:** https://huggingface.co/spaces/semantique-team/moth-demo (static Space, `RUNNING`)
- **Package:** **19,440,761 bytes**, 7 files. Runtime payload 19,436,178 bytes; config+NOTICE
  4,583 bytes. Model hash `7bbfad7e288640538c5c0147f81738bfd63021727b8760b1f3cbbcc4f939764d`.
- **Decoder:** beam 12 / top-k 10 / LM 0.55 / token bonus 2.7.
- **HF revisions:** release upload `e5339bc01aec284974685269310a6f735d8e9ea8`; current model card
  revision `c5cca27bca17ce7e1d7a7a95340961fa3b13d6d6`. **Space:** `70d930b6167671b6541e2a5c725151165d3a3ff3`.
- **Published metrics:** T-600 **24.71 / 54.17**, T-S3 **25.21 / 50.14**, FLEURS-IT **20.25**,
  FLEURS-EN **49.81**, natural **436/817**, rare presence **11/47**, rare aligned ETA **28.4%**.
- **The one exception:** the pre-registered primary gate was rare aligned ETA ≥30%; the int8
  artifact is **two aligned occurrences short** (23/81 vs 25/81). The release is published with
  that documented exception, not as a gate pass. E-028 fp32 has 25/81.
- **Next:** independent natural mixed-dev and a new-speaker rare addendum; then a gentle fine-tune
  if the data justify it. Do not tune on T/S3.

## Repository state

> **Follow-up later on 2026-11-12.** After this handoff, **E-031** rebuilt the LM inside the
> budget and **published it as 0.8.1**: same int8 graph, rare aligned ETA **28.4% → 46.9%**,
> T-600 **24.08 / 55.23**, package 19,706,289 B, browser acceptance 4/4, noise ties. **E-032**
> (a 50-sentence S1 acoustic addendum) was **rejected**: rare up, T-S3/FLEURS/natural down.
> Independent natural mixed-dev is still missing and blocked on a new speaker. Read
> [`docs/17-session-handoff-2026-11-12-e031.md`](17-session-handoff-2026-11-12-e031.md) for the
> current state; this document remains the 0.8.0 session record.

- HEAD at handoff: **`3d399a4`** (`test(E-028): re-measure the 0.8.0 noise curve`) plus the commit
  that adds this document. Working tree clean except two unrelated logs:
  `runs/E-016/eval-aug-lm.log`, `runs/E-016/eval-aug-sets.log`.
- Recent history:
  - `0e118c7` — release 0.8.0, model card, Space/HF publication, docs.
  - `90958d7` — release record + browser acceptance report reconciliation.
  - `3d399a4` — 0.8.0 noise check.
  - `3f5a471` — E-030 decoder experiment.
  - `ea6606a` — E-029 QAT negative.
- `ruff check .` clean; `pytest -q` **881 passed, 1 skipped**. `git diff --check` clean.
- No weights, audio, checkpoints or `.onnx` files in git. `release/` is gitignored; the published
  bundle is on Hugging Face.

## What happened, in order

1. **E-029 QAT (2026-11-11, negative).** Custom MPS straight-through quantizer (no PyTorch
   fake-quant on MPS), four 200–500-step probes from E-028. QAT closes the dev INT8-vs-fp32 gap
   (0.21 → 0.05) but the best arm only matches E-028 INT8 on dev and fails the product gates:
   T-600 25.34 / 53.21, rare 10/47, natural 428. The full arm was not funded. Report:
   `runs/E-029/RESULT.md`.
2. **E-030 decoder (2026-11-11).** 12-arm decoder sweep on the unchanged E-028 INT8 graph,
   selected on CV-IT + FLEURS dev. The chosen arm (12/10/0.55/2.7) improves dev 31.14 → 30.95 and
   gives T-600 **24.71 / 54.17**, T-S3 **25.21 / 50.14**, natural **436**, rare **11/47**,
   rare aligned ETA **28.4%**. That passes five of six E-028 gates; the rare gate is two
   occurrences short. Head-fp32 was checked: 24.54 / 54.36, rare 23/81, 20.62 MB, no rare
   recovery. Report: `runs/E-030/RESULT.md`.
3. **0.8.0 release (2026-11-12).** Packaged E-028 + decoder 0.55/2.7 (19,440,761 bytes), re-ran
   the real Chromium/WASM acceptance **4/4** with the new decoder, uploaded the model and Space,
   and verified remote config/model hashes and revisions. Model card:
   `runs/E-028/MODEL_CARD.md`; record: `runs/E-028/release-0.8.0.json`; acceptance:
   `runs/E-028/browser-acceptance/RESULT.md`.
4. **Noise check (2026-11-12).** Re-ran the E-024 pool harness on the E-028 fp32 checkpoint with
   the same 0.55/2.7 decoder as the 0.7.0 control. 0.8.0 is a tie at 10–30 dB (mean WER 27.33 vs
   27.76), tie at 15 dB, and significantly better clean (20.37 vs 23.09, −2.72 [−4.42, −1.02]);
   no significant noise regression. Report: `runs/E-028/noise-0.8.0/RESULT.md`.
5. **Tooling changes:** `runs/E-028/package_release.py` writes 0.55/2.7; `runs/E-022/publish_space.py`
   accepts `--record` (default 0.8.0); `web/acceptance/real-wasm.spec.ts` accepts
   `MOTH_ACCEPTANCE_REPORT_DIR` and records `decode_config`; `runs/E-024/noise_eval.py` accepts
   `--lm-weight`/`--token-bonus`; `tests/test_upload_hf.py` finds the model card matching the
   release version.

## Published numbers (int8, Python ONNX + binary LM)

| Set | 0.7.0 int8 | **0.8.0 int8** |
|---|---:|---:|
| T-600 WER / ETA | 25.14 / 54.65 | **24.71 / 54.17** |
| T-S3 WER / ETA | 27.00 / 48.45 | **25.21 / 50.14** |
| FLEURS-IT | 19.96 (same arm) | **20.25** |
| FLEURS-EN | 49.39 (same arm) | **49.81** |
| Natural terms | 441/817 | 436/817 |
| Rare presence | 8/47 | **11/47** |
| Rare aligned ETA | 22.2% | **28.4%** |
| 16 gold terms | 22/27 | 19/27 |

Rare ETA vs 0.7.0 shipped: **+6.2 [+1.1, +11.8]**; vs 0.7.0 with the same E-030 decoder:
**+7.4 [+1.3, +13.6]**. T-600 vs 0.7.0 (same arm) is a tie: WER −0.23, ETA −0.58. T-S3 is clearly
better. Natural is 4–5 terms lower, still above the 431 gate.

## Where everything lives

- `runs/E-028/MODEL_CARD.md` — published card (HF `README.md`), including the rare-gate exception.
- `runs/E-028/release-0.8.0.json` — release record: hashes, bytes, decode, WER, browser, remote
  revisions, `published: true`.
- `runs/E-028/browser-acceptance/` — `RESULT.md` + 4 per-project JSON (4/4, RTFx, guards).
- `runs/E-028/noise-0.8.0/` — noise report + per-SNR metrics; control in
  `runs/E-028/noise-0.7.0-same-decoder/`.
- `runs/E-030/` — decoder experiment: `PLAN.md`, `RESULT.md`, `compare.json`, `sweep.json`,
  `eval/` (T-600/T-S3/FLEURS hyps/metrics for 0.7.0 and E-028 with the new arm).
- `runs/E-029/` — QAT negative: `PLAN.md`, `RESULT.md`, `probe_qat.py`, probe summaries/logs.
- `runs/E-028/` — fp32 result, quantization sweep, config, training logs.
- `EXPERIMENTS.md` — E-028, E-029, E-030 entries.
- `STATUS.md` — live state + verified table; `docs/11` — decisions D36-D40; `docs/13` — 0.7.0
  audit + 0.8.0 update note; `docs/15` — previous handoff.
- `runs/E-024/space/README.md` — Space card, updated with the 0.8.0 note.

## Reproduction / operations

```bash
# Rebuild and validate the package (from E-028 checkpoint)
uv run python -m stt.export --out runs/E-028/release \
    --checkpoint runs/E-028/checkpoints/ckpt_best.pt \
    --vocab release/vocab.json --head-blocks 2 --version 0.8.0
uv run python runs/E-028/package_release.py        # sets decoder 0.55/2.7, copies LM
uv run --locked --no-dev python -m stt.release release --budget-bytes 20000000

# Browser acceptance (4 projects; keeps 0.7.0 reports separate)
cd web
PYTHON=python3 \
MOTH_RELEASE_DIR="$(cd .. && pwd)/release" \
MOTH_ACCEPTANCE_REPORT_DIR="$(cd .. && pwd)/runs/E-028/browser-acceptance" \
npm run test:acceptance

# Upload model + card, then Space (uses the stored `hf auth login` token)
uv run python runs/E-021/upload_hf.py --repo semantique-team/moth-stt \
    --release release --card runs/E-028/MODEL_CARD.md \
    --record runs/E-028/release-0.8.0.json --push
uv run python runs/E-022/publish_space.py --check
uv run python runs/E-022/publish_space.py --push --repo semantique-team/moth-demo

# Noise check
uv run python runs/E-024/noise_eval.py \
    --checkpoint runs/E-028/checkpoints/ckpt_best.pt --config runs/E-028/config.yaml \
    --out runs/E-028/noise-0.8.0 --clips 100 --skip-clean \
    --lm-weight 0.55 --token-bonus 2.7
```

The local `release/` was staged from `runs/E-028/release/`; the 0.7.0 bundle is backed up at
`~/moth-release-0.7.0-backup` on this machine. A clean clone must download the published bundle
from HF; weights and audio are not in git.

## Known gaps and risks

- **Rare gate exception:** 28.4% vs ≥30%; 23/81 vs 25/81. Documented in the card, record, STATUS
  and D40. Do not call it a gate pass.
- **No independent natural mixed-dev:** T and S3 are reused evidence; S3 is the only held-out
  voice and has been inspected many times. S4 was never recorded.
- **S2 provenance:** 50/50 clips validated and the ZIP hash recorded, but the session log/retakes
  were not returned. The audio ZIP lives only in `~/Downloads` on this machine.
- **Browser acceptance is runtime, not WER:** one 3 s public fixture; no 0.8.0 browser-vs-Python
  WER parity run. The card only claims 4/4 runtime acceptance.
- **Noise check is fp32, same-pool, one speaker:** upper bound, not the E-025 unseen-room curve.
- **Binaries not in git:** release/checkpoints/audio/shards are gitignored; HF is the distribution.
- **Space quality not re-measured:** config and hashes verified, runtime `RUNNING`; no live WER.
- **`docs/12` ledger** covers E-000-E-025 by design; E-028/E-029/E-030 are in `EXPERIMENTS.md`
  and `runs/`, not rewritten into the historical ledger.

## Next steps (in order)

> **Status after this handoff (2026-11-12, later):** the same-size LM route was executed as
> **E-031** and **published as 0.8.1** (rare aligned ETA 46.9%, package 19,706,289 B, browser 4/4,
> noise all ties). The targeted rare addendum was executed as **E-032** and **rejected** (rare up,
> T-S3/FLEURS/natural down). Item 1 below, the independent natural mixed-dev, **remains to do and
> is blocked: no new speaker is available**. See
> [`docs/17`](17-session-handoff-2026-11-12-e031.md).

1. **Independent data (audit #1):** record 2–3 new speakers, natural Italian + English terms,
   20–30 min each; consent, pseudonymous IDs, freeze roles/splits before inspecting errors. Keep
   out of training/tokenizer/LM; plan a separate future holdout. Follow
   `docs/guides/mixed-dev-and-quality-experiments.md`.
2. **Targeted rare addendum (audit #5):** new-speaker rare-term sentences, then a gentle fine-tune
   from E-028 (new experiment, pre-register gates: rare ETA ≥30, T-600/T-S3/natural non-regression,
   package ≤20 MB, browser 4/4).
3. **Optional no-training route:** same-size LM retrain/pruning to favour English terms; lower
   priority, not guaranteed.
4. **Paper/docs:** fold 0.8.0 + the exception + independent data into the paper/ledger; keep
   T/S3 labeled as reused evidence.

## Do not

- Re-run QAT or dynamic-quantization variants: exhausted (E-029, E-028 sweep).
- Re-run contextual bias: rejected (E-027).
- Select checkpoints/decoders/thresholds on T or S3.
- Publish 0.8.1 without independent data and the pre-registered gates.
- Treat browser acceptance as a quality metric.
