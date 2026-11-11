"""E-029 probe: custom straight-through quantization (STE) on MPS.

PyTorch's fake-quant op is not implemented on MPS, so this probe wraps every ``nn.Linear`` and
``nn.Conv1d`` with a forward that fake-quantizes the weight (per-output-channel symmetric int8)
and, optionally, the input activation (per-tensor asymmetric uint8, like ONNX
``DynamicQuantizeLinear``). Gradients use the straight-through estimator, so the wrapped layers
train exactly like the fp32 ones.

The probe answers three questions before a full QAT arm is funded:

1. does the quantized forward train at all on MPS (finite loss, nonzero gradients)?
2. how much does quantization shock the loss at step 0?
3. is the step time acceptable (kill if >2x the fp32 baseline)?

It is deliberately killable: 200 steps, no checkpointing schedule, dev evaluation only. T is not
read. The original forwards are restored before export, so the ONNX graph stays fp32.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import random
import time
import types
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from stt.data.dataset import AugmentConfig, MelDataset, plan_batches
from stt.decode import load_pieces
from stt.eval import score_rows
from stt.export import export_fp32
from stt.lm import BinaryBPELM
from stt.model import MothSTT
from stt.onnx_runner import OnnxMoth
from stt.tokenizer import encode as tokenizer_encode
from stt.tokenizer import load_tokenizer
from stt.train import (
    TrainingObjective,
    _build_optimizer,
    _build_scheduler,
    _encode_targets,
    _prepare_batch,
    load_config,
    load_sources,
)

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "runs/E-028/checkpoints/ckpt_best.pt"
CONFIG = REPO / "runs/E-028/config.yaml"
RELEASE = REPO / "runs/E-028/release"
VOCAB = RELEASE / "vocab.json"
LM = RELEASE / "lm_bpe3.bin"
#: All release files except model.int8.onnx (same constant as runs/E-028/quant_sweep.py).
FIXED_PACKAGE = 7_530_083
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4}
DEV_SETS = (
    ("cv_it_dev", "data/manifests/cv_it_dev.jsonl", "data/shards/cv_it_dev", 600),
    ("fleurs_it_dev", "data/manifests/fleurs_it_dev.jsonl", "data/shards/fleurs_it_dev", 0),
    ("fleurs_en_dev", "data/manifests/fleurs_en_dev.jsonl", "data/shards/fleurs_en_dev", 0),
)
#: E-028 references: fp32 30.93 dev, E-028 INT8 31.14 dev.
FP32_DEV = 0.30929
INT8_DEV = 0.31138


def fake_quant_weight(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric int8 fake quant with an identity gradient."""
    source = weight.float()
    dims = tuple(range(1, source.dim()))
    scale = source.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / 127.0
    quantized = torch.clamp(torch.round(source / scale), -127.0, 127.0) * scale
    return weight + (quantized.to(weight.dtype) - weight).detach()


def fake_quant_activation(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor asymmetric uint8 fake quant, mirroring ``DynamicQuantizeLinear``."""
    source = x.float()
    low = source.amin()
    high = source.amax()
    scale = ((high - low) / 255.0).clamp_min(1e-8)
    zero_point = torch.clamp(torch.round(-low / scale), 0.0, 255.0)
    quantized = torch.clamp(torch.round(source / scale) + zero_point, 0.0, 255.0)
    restored = (quantized - zero_point) * scale
    return x + (restored.to(x.dtype) - x).detach()


def _quantized_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    weight = self._qat_original_weight_quant(self.weight)  # type: ignore[attr-defined]
    if self._qat_mode == "weight_act":  # type: ignore[attr-defined]
        x = fake_quant_activation(x)
    if isinstance(self, nn.Linear):
        return F.linear(x, weight, self.bias)
    return F.conv1d(
        x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
    )


def install_qat(model: MothSTT, mode: str) -> list[nn.Module]:
    """Monkeypatch Linear/Conv forwards; returns the wrapped modules for restoration."""
    wrapped = []
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv1d)):
            continue
        module._qat_original_weight_quant = fake_quant_weight  # type: ignore[attr-defined]
        module._qat_mode = mode  # type: ignore[attr-defined]
        module.forward = types.MethodType(_quantized_forward, module)
        wrapped.append(module)
    return wrapped


def remove_qat(wrapped: list[nn.Module]) -> None:
    for module in wrapped:
        del module.forward
        del module._qat_original_weight_quant
        del module._qat_mode


def batches(dataset: MelDataset, durations: list[float], count: int):
    iterator = plan_batches(
        durations, max_seconds=192, max_clips=32, seed=0, epoch=0
    )
    return itertools.islice(iterator, count), len(iterator)


def train_steps(
    model: MothSTT,
    objective: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    dataset: MelDataset,
    positions_iter,
    tokenize,
    device: str,
    steps: int,
    log_path: Path,
    grad_clip: float = 1.0,
    empty_cache_every: int = 10,
) -> dict:
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    records = []
    skipped = 0
    grad_modules_ok = None
    started = time.perf_counter()
    for index, positions in enumerate(positions_iter):
        if index >= steps:
            break
        step_started = time.perf_counter()
        items = [dataset[position] for position in positions]
        tensor, mel_lengths, batch_rows = _prepare_batch(items, "truncated")
        targets, target_lengths = _encode_targets(batch_rows, tokenize)
        tensor = tensor.to(device)
        mel_lengths = mel_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            loss, main_loss, _ = objective(
                model, tensor, mel_lengths, targets, target_lengths, criterion
            )
        finite = bool(torch.isfinite(loss))
        if finite:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_modules_ok is None:
                grad_modules_ok = sum(
                    1
                    for module in model.modules()
                    if isinstance(module, (nn.Linear, nn.Conv1d))
                    and module.weight.grad is not None
                    and bool(module.weight.grad.abs().sum() > 0)
                )
            grad_norm = float(nn.utils.clip_grad_norm_(parameters, grad_clip))
            optimizer.step()
            scheduler.step()
        else:
            skipped += 1
            grad_norm = float("nan")
        if device.split(":")[0] == "mps":
            if empty_cache_every and (index + 1) % empty_cache_every == 0:
                torch.mps.empty_cache()
            torch.mps.synchronize()
        elapsed = time.perf_counter() - step_started
        record = {
            "step": index + 1,
            "loss": float(loss.detach()),
            "main_loss": float(main_loss.detach()),
            "grad_norm": grad_norm,
            "seconds": elapsed,
            "lr_head": optimizer.param_groups[0]["lr"],
        }
        records.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if (index + 1) % 25 == 0 or index == 0:
            print(
                f"step {index + 1:4d} loss {record['loss']:.4f} "
                f"grad {grad_norm:7.3f} {elapsed:5.2f}s",
                flush=True,
            )
    elapsed_total = time.perf_counter() - started
    losses = [record["loss"] for record in records]
    times = [record["seconds"] for record in records]
    return {
        "steps": len(records),
        "skipped_nonfinite": skipped,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "median_seconds": float(np.median(times)) if times else None,
        "mean_seconds": float(np.mean(times)) if times else None,
        "total_seconds": elapsed_total,
        "grad_modules_ok": grad_modules_ok,
    }


def evaluate_onnx(
    path: Path, dev: dict, pieces, lm
) -> dict:
    spec = importlib.util.spec_from_file_location("e022eval", REPO / "runs/E-022/eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    moth = OnnxMoth(path, VOCAB)
    per_set = {}
    all_rows: list[dict] = []
    all_hyps: dict[str, str] = {}
    for name, (index, rows, root) in dev.items():
        hyps = module.decode_int8(moth, pieces, lm, index, rows, root, PARAMS)
        scored = score_rows(rows, hyps, bootstrap=0)
        per_set[name] = {"n": scored["overall"]["n"], "wer": scored["overall"]["wer"]}
        all_rows.extend(rows)
        all_hyps.update(hyps)
    combined = score_rows(all_rows, all_hyps, bootstrap=0)["overall"]
    return {
        "model_bytes": path.stat().st_size,
        "package_bytes": path.stat().st_size + FIXED_PACKAGE,
        "dev_wer": combined["wer"],
        "per_set": per_set,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("weight", "weight_act"), default="weight")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--baseline-steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--lr-head", type=float, default=5e-5)
    parser.add_argument("--lr-encoder", type=float, default=5e-6)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-eval", action="store_true", help="smoke test only")
    parser.add_argument("--empty-cache-every", type=int, default=10)
    parser.add_argument("--freeze-encoder", action="store_true")
    args = parser.parse_args()

    default_name = f"{args.mode}_{args.steps}" + ("_frozen" if args.freeze_encoder else "")
    out = args.out or (REPO / "runs/E-029/probe" / default_name)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log.jsonl"
    log_path.write_text("", encoding="utf-8")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = load_config(CONFIG)
    config["run"] = f"E-029-probe-{args.mode}"
    config["optimizer"]["lr_head"] = args.lr_head
    config["optimizer"]["lr_encoder"] = args.lr_encoder

    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = MothSTT(vocab_size=1024, head_blocks=2)
    model.load_state_dict(payload["model"], strict=True)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    model.to(args.device).train()

    index, rows, root = load_sources(config["data"]["train"])
    durations = [float(row["duration"]) for row in rows]
    dataset = MelDataset(index, root, rows, augment=AugmentConfig(), seed=seed)
    dataset.epoch = 0

    processor = load_tokenizer(config["tokenizer"])
    tokenize = partial(tokenizer_encode, processor)

    objective = TrainingObjective(model, config).to(args.device)

    # 1. fp32 baseline on the same first batches; it mutates the model, so the state is reloaded.
    optimizer = _build_optimizer(model, config, objective)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    (out / "fp32-baseline.jsonl").write_text("", encoding="utf-8")
    fp32_batches, plan_length = batches(dataset, durations, args.baseline_steps)
    print(f"fp32 baseline: {args.baseline_steps} steps, plan {plan_length} batches", flush=True)
    fp32 = train_steps(
        model,
        objective,
        optimizer,
        _build_scheduler(optimizer, args.warmup, args.baseline_steps, 0.05),
        dataset,
        fp32_batches,
        tokenize,
        args.device,
        args.baseline_steps,
        out / "fp32-baseline.jsonl",
        empty_cache_every=args.empty_cache_every,
    )
    model.load_state_dict(initial_state)

    # 2. QAT arm: reload the E-028 weights, wrap the quantizable modules, repeat the same batches.
    if args.device.split(":")[0] == "mps":
        torch.mps.empty_cache()
    wrapped = install_qat(model, args.mode)
    optimizer = _build_optimizer(model, config, objective)
    scheduler = _build_scheduler(optimizer, args.warmup, args.steps, 0.05)
    qat_batches, _ = batches(dataset, durations, args.steps)
    print(f"QAT {args.mode}: {len(wrapped)} modules, {args.steps} steps", flush=True)
    qat = train_steps(
        model,
        objective,
        optimizer,
        scheduler,
        dataset,
        qat_batches,
        tokenize,
        args.device,
        args.steps,
        log_path,
        empty_cache_every=args.empty_cache_every,
    )

    summary = {
        "mode": args.mode,
        "steps": args.steps,
        "warmup": args.warmup,
        "lr_head": args.lr_head,
        "lr_encoder": args.lr_encoder,
        "wrapped_modules": len(wrapped),
        "freeze_encoder": args.freeze_encoder,
        "fp32_baseline": fp32,
        "qat": qat,
        "references": {"e028_fp32_dev_wer": FP32_DEV, "e028_int8_dev_wer": INT8_DEV},
    }

    # 3. Strip the wrappers, export fp32 and quantize exactly like the release path.
    remove_qat(wrapped)
    model.cpu().eval()
    torch.save({"step": args.steps, "model": model.state_dict()}, out / "ckpt_qat.pt")
    fp32_path = out / "qat_fp32.onnx"
    int8_path = out / "qat_int8.onnx"
    export_fp32(model, fp32_path)
    quantize_dynamic(fp32_path, int8_path, per_channel=True, weight_type=QuantType.QInt8)
    print(f"exported {fp32_path.name} and {int8_path.name}", flush=True)

    if args.skip_eval:
        summary["step_time_ratio"] = (
            qat["median_seconds"] / fp32["median_seconds"] if fp32["median_seconds"] else None
        )
        (out / "probe.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return 0

    pieces = load_pieces(VOCAB)
    lm = BinaryBPELM.load(LM)
    dev = {}
    for name, manifest, shards, limit in DEV_SETS:
        dev_index, dev_rows, dev_root = load_sources(
            [{"manifest": manifest, "shards": shards}]
        )
        if limit:
            dev_rows = dev_rows[:limit]
            keep = {row["id"] for row in dev_rows}
            dev_index = {uid: entry for uid, entry in dev_index.items() if uid in keep}
        dev[name] = (dev_index, dev_rows, dev_root)

    summary["fp32_onnx"] = evaluate_onnx(fp32_path, dev, pieces, lm)
    summary["int8_onnx"] = evaluate_onnx(int8_path, dev, pieces, lm)
    summary["step_time_ratio"] = (
        qat["median_seconds"] / fp32["median_seconds"] if fp32["median_seconds"] else None
    )
    (out / "probe.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
