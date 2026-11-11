# E-029 — QAT: recover the INT8 gap without losing the E-028 rare gains

**Status:** executed 2026-11-11. The probe is **negative**: QAT closes the dev quantization gap
but fails the product gates, and the full arm was not funded. See `RESULT.md`.

## Goal

Produce an INT8 release that is **at least as good as 0.7.0 INT8 on T-600/T-S3 and natural
terms**, while keeping the E-028 rare-term gain, inside the **20,000,000-byte** package.

E-028 fp32 passes the gates, but the shipped INT8 artifact gives back the gains:

| Set | 0.7.0 INT8 | E-028 INT8 |
|---|---:|---:|
| T-600 WER / ETA | 25.14 / **54.65** | 25.49 / **53.50** |
| T-S3 WER / ETA | 27.00 / 48.45 | **25.89 / 49.01** |
| Rare presence | 8/47 | **12/47** |
| Natural presence | 441/817 | 428/817 |

Post-training quantization is exhausted (`runs/E-028/QUANTIZATION.md`): default dynamic INT8 is
the best variant inside 20 MB; per-tensor/reduce-range/QUInt8 are worse; head-fp32 is 20.62 MB
for +0.08 dev; static QDQ is broken (91.4% dev WER).

## Constraint discovered

PyTorch's standard fake-quant op is **not implemented on MPS**:

```
NotImplementedError: aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` would run that op on CPU; this is acceptable only for a short
probe, not a full epoch. The default plan is a **custom MPS straight-through quantizer**.

## Proposed method

Start from `runs/E-028/checkpoints/ckpt_best.pt` (E-028 fp32, step 13,283).

1. **Weight-only STE probe.**
   - Wrap every `nn.Linear`/`nn.Conv1d` weight in a forward pre-hook that returns
     `w + (wq - w).detach()`, where
     `wq = clamp(round(w / scale), -127, 127) * scale` and `scale` is per-output-channel
     `max|w| / 127`.
   - Keep activations in fp32 first. Run 100–200 steps from E-028; measure loss, grad norm,
     step time vs fp32. Kill if the model diverges or the step is >2× slower.
   - Export INT8 and evaluate **dev** (CV-IT/FLEURS), not T.
2. **Activation quantization (only if weight-only is promising).**
   - Add a dynamic per-tensor fake quant on activations entering the quantized MatMuls.
   - Keep the same STE; calibrate scales from the current batch or a running observer.
   - This is the part that must mimic ONNX dynamic quantization; do not assume weight-only is
     sufficient.
3. **Fine-tune arm.**
   - Data: same mixture as E-028 (including S1/S2 term x20 and old gold x50).
   - 1 epoch, or a shorter pre-registered schedule if the probe shows convergence;
     head 5e-5 / encoder 5e-6 as the starting point (E-028 is the init, so the update is a
     quantization-adaptation step, not a new task).
   - Dev every 1,000 steps, checkpoints every 10 min, `caffeinate -i`.
   - Strip the STE wrappers before export; the exported graph must be the same architecture.
4. **Artifact gate.**
   - Export INT8 with the E-028 `stt.export`/`package_release.py` path.
   - Browser acceptance 4/4 on the real release.
   - Evaluate T-600/T-S3 and the 47 rare utterances; compare against 0.7.0 INT8 and E-028 INT8.

## Pre-registered gates

A QAT arm is adopted only if **all** hold:

1. INT8 T-600 WER **≤ 25.14** (0.7.0 INT8) and ETA **≥ 54.65** (or a pre-declared tie margin
   of 0.3 with a CI, but not a point regression);
2. INT8 T-S3 WER **≤ 27.00** and ETA **≥ 48.45**;
3. INT8 rare presence **≥ 12/47** and aligned ETA **≥ 29.6%** (the E-028 INT8 level);
4. INT8 natural presence **≥ 435/817** (between E-028 INT8 428 and 0.7.0 441);
5. package **≤ 20,000,000 bytes**; browser acceptance 4/4;
6. dev is reported and not a catastrophic regression; the mixed-dev caveat remains.

If the probe does not improve dev or the full arm misses one gate, **do not publish**: record the
negative result, keep 0.7.0.

## Risks

- Weight-only STE may not model activation quantization; the probe can look good and the exported
  INT8 can still miss.
- MPS support for the custom ops; `round`/`clamp` are available, but check gradients and speed.
- QAT can overfit the quantization noise or damage rare terms; the gates protect the product.
- The true fix may be **mixed precision** (a small fp32/LM trade) rather than QAT; if E-029 fails,
  record that too.

## Commands (probe)

```bash
# 0. Rebuild the fp32 reference if missing
uv run python -m stt.export --out runs/E-029/probe/release \
    --checkpoint runs/E-028/checkpoints/ckpt_best.pt \
    --vocab release/vocab.json --head-blocks 2 --version 0.9.0-dev

# 1. Probe: custom STE, 500 steps, encoder frozen, dev eval (the selected arm)
uv run python runs/E-029/probe_qat.py --mode weight_act --steps 500 \
    --baseline-steps 20 --warmup 50 --lr-head 2e-5 --lr-encoder 2e-6 \
    --freeze-encoder --out runs/E-029/probe/frozen_500

# 2. Full arm only if the probe passes its pre-registered check (it did not)
uv run python -m stt.train --config runs/E-029/config.yaml
```

`probe_qat.py` is implemented; `config.yaml` was deliberately not written because the probe
failed. The result and all four probe arms are in `RESULT.md`.
