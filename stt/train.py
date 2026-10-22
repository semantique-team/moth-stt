"""Training loop (design: ``docs/04`` §2–3, §5-H1, §6).

One device-agnostic loop: bf16 autocast on the encoder/head matmuls, ``log_softmax`` and
``CTCLoss(blank=0, zero_infinity=True)`` in fp32, AdamW with linear warm-up → cosine, gradient
clipping at 1.0. Checkpoints (model, optimizer, scheduler, loader position, RNG state) are
written every ``train.checkpoint_minutes`` wall-clock minutes — the last ``train.keep_last`` plus
the best by dev WER — and a resume restarts the same run ID mid-epoch. Guards: a non-finite loss
skips the batch and is counted; more than ``train.max_nonfinite_steps`` consecutive ones abort
the run. A fresh MPS run first checks the same small batch's CTC loss against CPU (gate 1e-3,
fp32) before training starts.

Config: ``runs/<id>/config.yaml``; the JSONL log records git SHA, torch version and device.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from stt.data.build_mel import load_index
from stt.data.dataset import AugmentConfig, MelDataset, collate, plan_batches
from stt.decode import greedy_ids
from stt.eval import score_rows
from stt.manifest import load_manifest
from stt.model import MothSTT
from stt.tokenizer import encode as tokenizer_encode
from stt.tokenizer import load_tokenizer

__all__ = ["TrainState", "iter_batches", "load_config", "load_sources", "main", "train"]

#: CTC loss parity gate between MPS and CPU at the start of a run (docs/04 §6).
PARITY_GATE = 1e-3

#: Clips used by the parity guard.
PARITY_CLIPS = 4

#: 30 s of mel frames (100 frames/s) — the padded-input variant and the model's hard limit.
MAX_MEL_FRAMES = 3000

_MEL_FRAMES_PER_SECOND = 100.0


@dataclass
class TrainState:
    """Global step plus the loader position (epoch and index within that epoch's plan)."""

    step: int = 0
    epoch: int = 0
    batch_cursor: int = 0


def load_config(path: str | Path) -> dict:
    """Load a run config (YAML, ``docs/07`` §3)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_sources(sources: list[dict]) -> tuple[dict[str, dict], list[dict], Path]:
    """Merge ``[{manifest, shards}, ...]`` into one index + row list over a shared shards root.

    Every shard name is rewritten with its source directory as a prefix (``a/shard-000.npy``)
    so the merged index memory-maps correctly from the common root (``data/shards/``).
    """
    if not sources:
        raise ValueError("no data sources configured")
    index: dict[str, dict] = {}
    rows: list[dict] = []
    roots: set[Path] = set()
    for source in sources:
        manifest_path = Path(source["manifest"])
        shards_dir = Path(source["shards"])
        source_rows = load_manifest(manifest_path)
        source_index = load_index(shards_dir / "index.json")
        missing = [row["id"] for row in source_rows if row["id"] not in source_index]
        if missing:
            raise KeyError(
                f"{len(missing)} ids of {manifest_path} missing from {shards_dir}/index.json: "
                f"{missing[:5]}"
            )
        prefix = shards_dir.name
        for uid, entry in source_index.items():
            if uid in index:
                raise ValueError(f"duplicate id across sources: {uid!r}")
            index[uid] = {**entry, "shard": f"{prefix}/{entry['shard']}"}
        rows.extend(source_rows)
        roots.add(shards_dir.parent)
    if len(roots) != 1:
        raise ValueError(f"sources must share one shards root, got {sorted(map(str, roots))}")
    return index, rows, roots.pop()


def iter_batches(
    durations: list[float],
    max_seconds: float,
    max_clips: int,
    seed: int = 0,
    epoch: int = 0,
    batch_cursor: int = 0,
) -> Iterator[tuple[int, int, list[int]]]:
    """Yield ``(epoch, index within that epoch's plan, batch)`` forever, starting at the cursor.

    Restarting with ``(epoch, batch_cursor)`` from a checkpoint reproduces the loader position
    exactly: ``plan_batches`` is deterministic per ``(seed, epoch)``.
    """
    current = epoch
    while True:
        batches = plan_batches(
            durations, max_seconds=max_seconds, max_clips=max_clips, seed=seed, epoch=current
        )
        start = batch_cursor if current == epoch else 0
        for index in range(start, len(batches)):
            yield current, index, batches[index]
        current += 1


def train(
    config: dict,
    device: str | None = None,
    max_steps: int | None = None,
    tokenize: Callable[[str], list[int]] | None = None,
    pieces: Mapping[int, str] | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Run (or resume) training; returns a summary dict.

    ``max_steps`` stops early without changing the schedule (smoke runs, tests); ``tokenize`` and
    ``pieces`` inject the tokenizer (tests) instead of loading ``config["tokenizer"]``.
    """
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    device_kind = device.split(":")[0]

    out_dir = Path(out_dir) if out_dir is not None else Path("runs") / str(config["run"])
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log.jsonl"

    train_cfg = config.get("train", {})
    batch_cfg = config.get("batch", {})
    max_seconds = float(batch_cfg.get("max_seconds", 320.0))
    max_clips = int(batch_cfg.get("max_clips", 64))
    input_mode = str(train_cfg.get("input", "truncated"))
    if input_mode not in ("truncated", "padded30"):
        raise ValueError(f"unknown input mode {input_mode!r}")
    autocast_name = str(train_cfg.get("autocast", "bf16"))
    if autocast_name not in ("bf16", "off"):
        raise ValueError(f"unknown autocast mode {autocast_name!r} (bf16 or off)")
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = max(1, int(train_cfg.get("log_every", 50)))
    max_nonfinite = int(train_cfg.get("max_nonfinite_steps", 20))
    checkpoint_minutes = float(train_cfg.get("checkpoint_minutes", 20))
    keep_last = max(1, int(train_cfg.get("keep_last", 3)))
    # MPS caches freed blocks; with variable-length batches the cache grows to the machine limit
    # and swaps. Releasing it periodically keeps the footprint flat (a small sync, not a slowdown).
    empty_cache_every = int(train_cfg.get("empty_cache_every", 0))
    eval_every = train_cfg.get("eval_every_steps")

    vocab_size = int(config["model"]["vocab_size"])
    index, rows, shards_root = load_sources(config["data"]["train"])
    durations = [float(row["duration"]) for row in rows]

    augment_name = str(train_cfg.get("augment", "full"))
    augment = {"none": None, "light": AugmentConfig.light(), "full": AugmentConfig()}[augment_name]
    dataset = MelDataset(index, shards_root, rows, augment=augment, seed=seed)

    model = MothSTT(vocab_size=vocab_size)
    _apply_model_init(model, config)
    if bool(config["model"].get("freeze_encoder", False)):
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)

    if tokenize is None:
        processor = load_tokenizer(config["tokenizer"])
        if processor.vocab_size() != vocab_size:
            raise ValueError(
                f"tokenizer vocab {processor.vocab_size()} != model vocab_size {vocab_size}"
            )
        tokenize = partial(tokenizer_encode, processor)
        if pieces is None:
            pieces = {
                piece_id + 1: processor.id_to_piece(piece_id)
                for piece_id in range(processor.vocab_size())
            }

    # ``dev`` is the Italian group, ``dev_en`` the optional English one; the selection metric is
    # their macro-average WER so that neither language collapses while the other improves.
    dev_sets: list[tuple[str, MelDataset]] = []
    for name, key in (("it", "dev"), ("en", "dev_en")):
        sources = config["data"].get(key)
        if sources:
            dev_index, dev_rows, dev_root = load_sources(sources)
            dev_sets.append(
                (name, MelDataset(dev_index, dev_root, dev_rows, augment=None, seed=seed))
            )

    steps_per_epoch = len(
        plan_batches(durations, max_seconds=max_seconds, max_clips=max_clips, seed=seed, epoch=0)
    )
    if steps_per_epoch == 0:
        raise ValueError("no training batches — is the manifest empty?")
    epochs = int(train_cfg.get("epochs", 1))
    total_steps = epochs * steps_per_epoch
    step_cap = min(total_steps, int(max_steps)) if max_steps is not None else total_steps

    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(
        optimizer,
        warmup_steps=int(config["schedule"].get("warmup_steps", 1500)),
        total_steps=total_steps,
        min_factor=float(config["schedule"].get("min_factor", 0.05)),
    )

    state = TrainState()
    best_metric: float | None = None
    resumed_from: str | None = None
    last_checkpoint = _latest_checkpoint(checkpoints_dir)
    if last_checkpoint is not None:
        payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        state = TrainState(
            step=int(payload["step"]),
            epoch=int(payload["epoch"]),
            batch_cursor=int(payload["batch_cursor"]),
        )
        best_metric = payload.get("best_metric")
        resumed_from = str(last_checkpoint)

    parity = None
    if resumed_from is None:
        parity = _parity_guard(
            model, dataset, durations, max_seconds, max_clips, seed, tokenize, device_kind
        )
    model.to(device)
    if resumed_from is not None:
        _move_optimizer_state(optimizer, device)

    def save(best: bool = False) -> Path:
        nonlocal last_save
        payload = {
            "step": state.step,
            "epoch": state.epoch,
            "batch_cursor": state.batch_cursor,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": None,  # bf16 autocast needs no GradScaler
            "rng": {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            "config": config,
            "torch": torch.__version__,
            "git_sha": _git_sha(),
            "best_metric": best_metric,
        }
        target = checkpoints_dir / f"ckpt_{state.step:07d}.pt"
        _atomic_torch_save(payload, target)
        if best:
            _atomic_torch_save(payload, checkpoints_dir / "ckpt_best.pt")
        _prune_checkpoints(checkpoints_dir, keep_last)
        last_save = time.monotonic()
        return target

    _append_log(
        log_path,
        {
            "event": "start",
            "run": config.get("run"),
            "device": device,
            "torch": torch.__version__,
            "seed": seed,
            "git_sha": _git_sha(),
            "vocab_size": vocab_size,
            "params": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_params": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "steps_per_epoch": steps_per_epoch,
            "step_cap": step_cap,
            "input": input_mode,
            "autocast": autocast_name,
            "resumed_from": resumed_from,
            "parity": parity,
        },
    )

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    consecutive_nonfinite = 0
    skipped = 0
    window_seconds = 0.0
    window_started = time.monotonic()
    last_save = time.monotonic()
    summary: dict = {
        "run": config.get("run"),
        "device": device,
        "steps": 0,
        "first_loss": None,
        "last_loss": None,
        "dev": None,
        "parity": parity,
        "resumed_from": resumed_from,
    }

    for epoch, batch_index, positions in iter_batches(
        durations,
        max_seconds=max_seconds,
        max_clips=max_clips,
        seed=seed,
        epoch=state.epoch,
        batch_cursor=state.batch_cursor,
    ):
        if state.step >= step_cap or epoch >= epochs:
            break
        state.epoch = epoch
        dataset.epoch = epoch  # augmentation must differ across epochs (docs/10 §4.2.1)
        state.batch_cursor = batch_index + 1

        items = [dataset[position] for position in positions]
        tensor, mel_lengths, batch_rows = _prepare_batch(items, input_mode)
        targets, target_lengths = _encode_targets(batch_rows, tokenize)
        tensor = tensor.to(device)
        mel_lengths = mel_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with torch.autocast(
            device_type=device_kind, dtype=torch.bfloat16, enabled=autocast_name == "bf16"
        ):
            log_probs, out_lengths = model(tensor, mel_lengths)
        loss = criterion(log_probs.transpose(0, 1).contiguous(), targets, out_lengths, target_lengths)

        if not bool(torch.isfinite(loss)):
            consecutive_nonfinite += 1
            skipped += 1
            _append_log(
                log_path,
                {
                    "event": "skip",
                    "step": state.step + 1,
                    "epoch": epoch,
                    "batch": batch_index,
                    "reason": "non_finite",
                    "consecutive_nonfinite": consecutive_nonfinite,
                },
            )
            if consecutive_nonfinite > max_nonfinite:
                raise RuntimeError(
                    f"aborting: {consecutive_nonfinite} consecutive non-finite losses "
                    f"(max {max_nonfinite})"
                )
            continue

        consecutive_nonfinite = 0
        loss_value = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                grad_clip,
            )
        )
        optimizer.step()
        scheduler.step()

        state.step += 1
        if empty_cache_every and device_kind == "mps" and state.step % empty_cache_every == 0:
            torch.mps.empty_cache()
        window_seconds += float(mel_lengths.sum()) / _MEL_FRAMES_PER_SECOND
        if summary["first_loss"] is None:
            summary["first_loss"] = loss_value
        summary["last_loss"] = loss_value

        if state.step % log_every == 0:
            elapsed = time.monotonic() - window_started
            _append_log(
                log_path,
                {
                    "event": "step",
                    "step": state.step,
                    "epoch": epoch,
                    "batch": batch_index,
                    "loss": loss_value,
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_encoder": (
                        optimizer.param_groups[-1]["lr"]
                        if len(optimizer.param_groups) > 1
                        else None
                    ),
                    "grad_norm": grad_norm,
                    "audio_seconds": round(window_seconds, 2),
                    "audio_seconds_per_s": round(window_seconds / elapsed, 3) if elapsed else 0.0,
                    "memory_mb": round(_memory_mb(device_kind), 1),
                    "skipped_nonfinite": skipped,
                },
            )
            window_seconds = 0.0
            window_started = time.monotonic()

        if dev_sets and eval_every and state.step % int(eval_every) == 0:
            metrics = _evaluate_all(
                model, dev_sets, pieces, device, max_seconds, max_clips, input_mode
            )
            _append_log(log_path, {"event": "dev", "step": state.step, **metrics})
            summary["dev"] = metrics
            if best_metric is None or metrics["wer"] < best_metric:
                best_metric = float(metrics["wer"])
                save(best=True)

        if time.monotonic() - last_save >= checkpoint_minutes * 60.0:
            save()

    if dev_sets:
        metrics = _evaluate_all(
            model, dev_sets, pieces, device, max_seconds, max_clips, input_mode, bootstrap=1000
        )
        _append_log(log_path, {"event": "dev", "step": state.step, "final": True, **metrics})
        summary["dev"] = metrics
        improved = best_metric is None or metrics["wer"] < best_metric
        best_metric = float(metrics["wer"]) if improved else best_metric
    else:
        improved = False

    saved = save(best=improved)
    summary |= {
        "steps": state.step,
        "epoch": state.epoch,
        "batch_cursor": state.batch_cursor,
        "skipped_nonfinite": skipped,
        "best_metric": best_metric,
        "checkpoint": str(saved),
        "log": str(log_path),
    }
    _append_log(
        log_path,
        {
            "event": "end",
            "step": state.step,
            "best_metric": best_metric,
            "skipped_nonfinite": skipped,
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train.py", description="Training loop with resume (design: docs/04 §6)."
    )
    parser.add_argument("--config", required=True, help="run config YAML (runs/<id>/config.yaml)")
    parser.add_argument("--device", default=None, choices=["mps", "cuda", "cpu"])
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N steps (smoke)")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)
    summary = train(
        config, device=args.device, max_steps=args.max_steps, out_dir=config_path.parent
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------------------------
# internals


def _build_optimizer(model: MothSTT, config: dict) -> torch.optim.AdamW:
    settings = config["optimizer"]
    groups = [
        {
            "params": [p for p in model.head.parameters() if p.requires_grad],
            "lr": float(settings["lr_head"]),
            "name": "head",
        }
    ]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        groups.append(
            {"params": encoder_params, "lr": float(settings["lr_encoder"]), "name": "encoder"}
        )
    return torch.optim.AdamW(
        groups,
        betas=tuple(settings.get("betas", (0.9, 0.98))),
        eps=float(settings.get("eps", 1e-8)),
        weight_decay=float(settings.get("weight_decay", 0.01)),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_factor: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up → cosine decay to ``min_factor`` of the peak LR (docs/04 §5-H2)."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _prepare_batch(
    items: list[dict], input_mode: str
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    tensor, mel_lengths, rows = collate(items)
    if input_mode == "padded30" and tensor.shape[2] < MAX_MEL_FRAMES:
        tensor = nn.functional.pad(tensor, (0, MAX_MEL_FRAMES - tensor.shape[2]))
    return tensor, mel_lengths, rows


def _encode_targets(
    rows: list[dict], tokenize: Callable[[str], list[int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [tokenize(row["norm"]) for row in rows]
    flat = [identifier for sequence in sequences for identifier in sequence]
    if not flat:
        raise ValueError("empty CTC targets — check the manifest's norm texts")
    if min(flat) <= 0:
        raise ValueError("token ids must be ≥ 1 (0 is the CTC blank)")
    targets = torch.tensor(flat, dtype=torch.long)
    target_lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    return targets, target_lengths


def _ctc_loss(
    model: MothSTT,
    tensor: torch.Tensor,
    mel_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    criterion: nn.CTCLoss,
) -> torch.Tensor:
    log_probs, out_lengths = model(tensor, mel_lengths)
    return criterion(log_probs.transpose(0, 1).contiguous(), targets, out_lengths, target_lengths)


def _parity_guard(
    model: MothSTT,
    dataset: MelDataset,
    durations: list[float],
    max_seconds: float,
    max_clips: int,
    seed: int,
    tokenize: Callable[[str], list[int]],
    device: str,
) -> dict | None:
    """Same small batch, fp32 CTC loss on MPS vs CPU within ``PARITY_GATE`` (docs/04 §6)."""
    if device != "mps":
        return None
    batch = plan_batches(
        durations, max_seconds=max_seconds, max_clips=max_clips, seed=seed, epoch=0
    )[0][:PARITY_CLIPS]
    items = [dataset[position] for position in batch]
    tensor, mel_lengths, rows = _prepare_batch(items, "truncated")
    targets, target_lengths = _encode_targets(rows, tokenize)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    reference = copy.deepcopy(model).to("cpu").eval()
    with torch.no_grad():
        cpu_loss = float(_ctc_loss(reference, tensor, mel_lengths, targets, target_lengths, criterion))
    model.to(device)
    with torch.no_grad():
        mps_loss = float(
            _ctc_loss(
                model,
                tensor.to(device),
                mel_lengths.to(device),
                targets.to(device),
                target_lengths.to(device),
                criterion,
            )
        )
    difference = abs(cpu_loss - mps_loss)
    report = {"cpu_loss": cpu_loss, "mps_loss": mps_loss, "max_abs_diff": difference, "gate": PARITY_GATE}
    if difference > PARITY_GATE:
        raise RuntimeError(f"device parity guard failed: |Δ| = {difference:.2e} > {PARITY_GATE}")
    return report


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    """Move CPU-loaded Adam state to the model device; resume otherwise fails on MPS/CUDA."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _apply_model_init(model: MothSTT, config: dict) -> None:
    """Load the pretrained encoder (``init``) and/or a full Moth checkpoint (``init_checkpoint``).

    H2 starts from the H1 run's encoder **and** head, so the full state dict is loaded; the older
    ``init`` key keeps loading the converted Whisper-tiny encoder alone (H1).
    """
    init_path = config["model"].get("init")
    if init_path:
        init = torch.load(init_path, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(init["state_dict"], strict=True)
    checkpoint = config["model"].get("init_checkpoint")
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])


def _evaluate_all(
    model: MothSTT,
    dev_sets: list[tuple[str, MelDataset]],
    pieces: Mapping[int, str] | None,
    device: str,
    max_seconds: float,
    max_clips: int,
    input_mode: str,
    bootstrap: int = 0,
) -> dict:
    """Score every dev group; ``wer`` is their macro-average (the checkpoint selection metric)."""
    per_set = {
        name: _evaluate(
            model, dataset, pieces, device, max_seconds, max_clips, input_mode, bootstrap
        )
        for name, dataset in dev_sets
    }
    macro = sum(metrics["wer"] for metrics in per_set.values()) / len(per_set)
    result: dict = {
        "wer": macro,
        "n_utterances": sum(metrics["n_utterances"] for metrics in per_set.values()),
        "sets": per_set,
    }
    result["wer_ci"] = next(iter(per_set.values()))["wer_ci"] if len(per_set) == 1 else None
    return result


def _evaluate(
    model: MothSTT,
    dataset: MelDataset,
    pieces: Mapping[int, str] | None,
    device: str,
    max_seconds: float,
    max_clips: int,
    input_mode: str,
    bootstrap: int = 0,
) -> dict:
    if pieces is None:
        raise ValueError("dev evaluation needs decode pieces (a tokenizer vocab)")
    durations = [float(row["duration"]) for row in dataset.rows]
    batches = plan_batches(durations, max_seconds=max_seconds, max_clips=max_clips, seed=0, epoch=0)
    was_training = model.training
    model.eval()
    hyps: dict[str, str] = {}
    with torch.inference_mode():
        for positions in batches:
            items = [dataset[position] for position in positions]
            tensor, mel_lengths, rows = _prepare_batch(items, input_mode)
            log_probs, out_lengths = model(tensor.to(device), mel_lengths.to(device))
            ids = log_probs.argmax(dim=-1)
            for offset, row in enumerate(rows):
                hyps[row["id"]] = greedy_ids(
                    ids[offset, : int(out_lengths[offset])].tolist(), pieces
                )
    if was_training:
        model.train()
    scored = score_rows(dataset.rows, hyps, bootstrap=bootstrap, seed=0)
    return {
        "wer": scored["overall"]["wer"],
        "wer_ci": scored["overall"].get("wer_ci"),
        "n_utterances": scored["n_utterances"],
    }


def _latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    numbered: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("ckpt_*.pt"):
        try:
            numbered.append((int(path.stem.split("_")[-1]), path))
        except ValueError:
            continue  # ckpt_best.pt
    return max(numbered)[1] if numbered else None


def _prune_checkpoints(checkpoints_dir: Path, keep: int) -> None:
    numbered: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("ckpt_*.pt"):
        try:
            numbered.append((int(path.stem.split("_")[-1]), path))
        except ValueError:
            continue  # ckpt_best.pt is never pruned
    numbered.sort()
    for _step, path in numbered[:-keep]:
        path.unlink()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _memory_mb(device_kind: str) -> float:
    if device_kind == "mps" and torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**20
    if device_kind == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
