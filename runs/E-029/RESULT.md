# E-029 — RESULT: QAT closes the dev quantization gap but not the product gap

> Executed 2026-11-11 from `runs/E-028/checkpoints/ckpt_best.pt`; planned in `PLAN.md`.
> **Negative result: no full arm was funded, no weights have been published.** 0.7.0 remains the
> published model.

## Verdict

The custom straight-through quantizer works on MPS and the 200–500-step probe is mechanically
viable, but **QAT does not produce a publishable INT8 artifact**:

- the best probe closes the dev INT8-vs-fp32 gap from **0.21 to 0.05** points, but the
  quantization-adapted model does not beat the E-028 INT8 baseline on dev
  (**31.13 vs 31.14**);
- on the product readout it **fails the T-600, rare and natural gates**: T-600
  **25.34 / 53.21** (gate ≤25.14 / ≥54.65), rare **10/47** (gate ≥12), rare aligned ETA
  **27.2%** (gate ≥29.6%), natural **428/817** (gate ≥435); T-S3 passes;
- the fp32 control of the same probe is already worse than E-028 fp32
  (**24.97 / 54.07** vs 24.66 / 54.65; natural 434 vs 438), so a longer QAT arm would have to
  reverse the fine-tune drift first.

The pre-registered rule was: if the probe does not improve dev, do not publish. The probe did not
improve dev beyond the E-028 INT8 baseline, so the full arm was **not** funded. **0.8.0 is not
published; keep 0.7.0.**

## Why the probe was necessary

PyTorch fake-quant is not implemented on MPS:

```
NotImplementedError: aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams
```

The probe implements the same arithmetic with plain MPS ops and an identity gradient
(`w + (wq - w).detach()`):

- **weights**: per-output-channel symmetric int8, scale `max|w| / 127`, round/clamp;
- **activations**: per-tensor asymmetric uint8, scale `(max - min) / 255`, zero point
  `-min / scale`, mirroring `DynamicQuantizeLinear`;
- **scope**: 39 `nn.Linear`/`nn.Conv1d` modules (37 Linear + 2 Conv), the same tensors ORT
  quantizes; the 12 attention matmuls stay fp32, exactly like the exported graph;
- **restore**: the original forwards are restored before export, so the ONNX graph remains fp32
  and the INT8 file is produced by the normal release path.

Probe cost: 200–500 steps, no checkpoint schedule, dev only. T was read once, at the end, on the
dev-selected variant.

## Probe arms (dev, selected on CV-IT 600 + FLEURS-IT/EN)

| Arm | Mode | Steps | Encoder | Dev fp32 | Dev INT8 | Step ratio | First loss |
|---|---|---:|---|---:|---:|---:|---:|
| A | weight only | 200 | train | 31.09 | 31.17 | 1.12× | 0.541 |
| B | weight + act | 200 | train | 31.31 | 31.49 | 1.42× | 0.576 |
| C | weight + act | 500 | train | 31.36 | 31.41 | 1.35× | 0.576 |
| D | weight + act | 500 | **frozen** | **31.08** | **31.13** | 1.58× | 0.576 |

Reference: E-028 fp32 **30.93**, E-028 INT8 **31.14**. The fp32 baseline on the same first batch
is 0.537; the activation quantization shock is 0.576 in the first step (+7.3%), not a divergence.
All arms had 0 non-finite steps and nonzero gradients on every trainable quantized module (39/39
unfrozen, 13/13 frozen).

What the arms show:

1. **Weight-only is not enough** (A): no dev recovery, 31.17 vs 31.14.
2. **Activation quantization is the main gap** (B/C): emulating it makes the task harder, and QAT
   closes the gap to its own fp32 (C: 31.41 vs 31.36), but the fine-tune itself drifts fp32 up.
3. **Freezing the encoder is the best trade** (D): the fp32 features stay intact, the INT8 gap is
   0.05, and dev returns to the E-028 INT8 baseline. It still does not improve it.
4. **Step time is acceptable**: 1.12–1.58× the fp32 baseline, below the pre-registered 2× kill
   threshold. The MPS cache must be emptied every 10 steps (as in training), otherwise the step
   time explodes to 40–200 s after ~40 steps.

## Product readout of arm D (the best probe)

The checkpoint is `runs/E-029/probe/frozen_500/ckpt_qat.pt`; the INT8 graph is the normal
`quantize_dynamic(per_channel=True, QInt8)` export of its restored fp32 model. Package
**19,440,760 bytes** (identical to E-028).

| Set | 0.7.0 INT8 | E-028 INT8 | **E-029 QAT INT8** |
|---|---:|---:|---:|
| T-600 WER / ETA | 25.14 / **54.65** | 25.49 / **53.50** | 25.34 / **53.21** |
| T-S3 WER / ETA | 27.00 / 48.45 | 25.89 / 49.01 | **25.55 / 49.30** |
| Rare presence | 8/47 | **12/47** | 10/47 |
| Rare aligned ETA | 22.2% | **29.6%** | 27.2% |
| Natural presence | **441/817** | 428/817 | 428/817 |
| Gold-only | 127/179 | 124/179 | 124/179 |
| 16 gold terms | 22/27 | 19/27 | 19/27 |

The QAT fp32 control (before quantization) is T-600 **24.97 / 54.07**, T-S3 **25.89 / 48.45**,
rare 12/47 (aligned ETA 29.6%), natural 434/817. So even the unquantized QAT model is below the
E-028 fp32 reference (24.66 / 54.65, natural 438); quantization then costs another
0.37 WER / 0.86 ETA on T-600.

## Gates

| Gate | Required | Result | Verdict |
|---|---|---|---|
| T-600 WER | ≤ 25.14 | 25.34 | **FAIL** |
| T-600 ETA | ≥ 54.65 | 53.21 | **FAIL** |
| T-S3 WER / ETA | ≤ 27.00 / ≥ 48.45 | 25.55 / 49.30 | PASS |
| Rare presence | ≥ 12/47 | 10/47 | **FAIL** |
| Rare aligned ETA | ≥ 29.6% | 27.2% | **FAIL** |
| Natural presence | ≥ 435/817 | 428/817 | **FAIL** |
| Package | ≤ 20,000,000 B | 19,440,760 B | PASS |
| Browser acceptance | 4/4 | not run | — (no candidate) |

## Interpretation

The probe answers the question the plan asked, and the answer is negative:

- **The quantizer is not the blocker.** QAT demonstrably removes most of the dev quantization
  error (0.21 → 0.05) with a stable custom STE, 0 non-finite steps and acceptable step time.
- **The product gap is elsewhere.** On T-600 the INT8 gap of the adapted model is still 0.86 ETA
  points, and the adaptation costs rare (12 → 10) and natural (438 → 434) terms relative to
  E-028 fp32. The fine-tune moves the model away from the E-028 optimum faster than it learns
  quantization robustness.
- **Dev remains misaligned with the product.** E-028 fp32 already had worse dev than 0.7.0 but
  better T-600; QAT at 31.13 dev is indistinguishable from E-028 INT8 at 31.14, yet its T-600
  ETA is 0.29 lower than E-028 INT8. Selecting QAT on dev alone cannot resolve this.
- **A longer arm is not justified.** The 500-step probe already degraded its own fp32 control;
  more steps would primarily increase that drift. The pre-registered rule says stop.

The remaining serious routes are the ones already named in `docs/14` and the handoff: a real
natural mixed-dev for product-aligned selection, a **mixed-precision** release (head fp32 plus an
LM trade to stay under 20 MB), or a data/architecture change that reduces activation outliers.
Cheap post-training quantization and QAT are now exhausted.

## Reproduction

```bash
# Probe (dev): the custom STE, 500 steps, frozen encoder — the selected arm
uv run python runs/E-029/probe_qat.py --mode weight_act --steps 500 \
    --baseline-steps 20 --warmup 50 --lr-head 2e-5 --lr-encoder 2e-6 \
    --freeze-encoder --out runs/E-029/probe/frozen_500

# Product readout (T read once, after dev selection)
mkdir -p runs/E-029/probe/frozen_500/release
cp runs/E-029/probe/frozen_500/qat_int8.onnx \
    runs/E-029/probe/frozen_500/release/model.int8.onnx
cp runs/E-028/release/vocab.json runs/E-029/probe/frozen_500/release/vocab.json
uv run python runs/E-022/eval.py --arm frozen \
    --checkpoint runs/E-029/probe/frozen_500/ckpt_qat.pt \
    --int8 --release runs/E-029/probe/frozen_500/release \
    --sets t600,t-s3 --out runs/E-029/probe/frozen_500/eval
uv run python runs/E-015/eval_terms.py \
    --hyp runs/E-029/probe/frozen_500/eval/int8/t600.hyp.jsonl \
    --baseline runs/E-024/eval/gate/int8/t600.hyp.jsonl \
    --out runs/E-029/probe/frozen_500/eval/terms-int8.json
```

The probe directories are gitignored for checkpoints/ONNX (`runs/**/ckpt*.pt`, `*.onnx`); the
JSON logs and this result are committed. The release directory contains only the INT8 graph and
the vocab needed by the evaluator, not a shippable 0.8.0 package.

## Artifacts

- `probe_qat.py` — custom STE, probes A–D, dev export/eval.
- `probe/weight/probe.json` — arm A.
- `probe/weight_act_200/probe.json` — arm B.
- `probe/weight_act_500/probe.json` — arm C.
- `probe/frozen_500/probe.json`, `train.log.jsonl` — arm D (selected).
- `probe/frozen_500/eval/` — T-600/T-S3 hyps, metrics, terms JSON.
