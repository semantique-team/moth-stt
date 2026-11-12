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
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None

from stt.data.build_mel import load_index
from stt.data.dataset import MAX_MEL_FRAMES, AugmentConfig, MelDataset, collate, plan_batches
from stt.decode import greedy_ids
from stt.eval import score_rows
from stt.manifest import load_manifest, validate_training_rows
from stt.model import EncoderConfig, MothSTT
from stt.tokenizer import encode as tokenizer_encode
from stt.tokenizer import load_tokenizer

__all__ = [
    "TrainState",
    "TrainingObjective",
    "check_training_config",
    "interctc_recipe",
    "iter_batches",
    "load_config",
    "load_sources",
    "main",
    "train",
    "training_recipe_fingerprint",
]

#: CTC loss parity gate between MPS and CPU at the start of a run (docs/04 §6).
PARITY_GATE = 1e-3

#: Clips used by the parity guard.
PARITY_CLIPS = 4

_MEL_FRAMES_PER_SECOND = 100.0


@dataclass
class TrainState:
    """Global step plus the loader position (epoch and index within that epoch's plan)."""

    step: int = 0
    epoch: int = 0
    batch_cursor: int = 0


def interctc_recipe(config: dict) -> dict:
    """Validate the training-only objective, without loading data or starting training."""
    settings = config.get("train", {}).get("interctc", {})
    if not isinstance(settings, dict) or set(settings) - {"weight", "tap"}:
        raise ValueError("train.interctc accepts only weight and tap")
    if isinstance(settings.get("weight"), bool):
        raise ValueError("interctc weight must be a number, not bool")  # noqa: TRY004
    try:
        weight = float(settings.get("weight", 0.0))
    except (TypeError, ValueError) as error:
        raise ValueError("interctc weight must be a finite number") from error
    tap = settings.get("tap", "encoder_output")
    if not math.isfinite(weight) or not 0 <= weight < 1:
        raise ValueError("interctc weight must be finite and satisfy 0 <= weight < 1")
    if tap != "encoder_output":
        raise ValueError("interctc tap must be encoder_output")
    return {"weight": weight, "tap": tap}


class TrainingObjective(nn.Module):
    """Training-owned auxiliary head; never registered on the deployable model."""

    def __init__(self, model: MothSTT, config: dict) -> None:
        super().__init__()
        self.recipe = interctc_recipe(config)
        self.aux_head = (
            nn.Linear(
                model.head.in_features, model.head.out_features,
                device=model.head.weight.device, dtype=model.head.weight.dtype,
            )
            if self.recipe["weight"] else None
        )

    def forward(
        self, model: MothSTT, tensor: torch.Tensor, mel_lengths: torch.Tensor,
        targets: torch.Tensor, target_lengths: torch.Tensor, criterion: nn.CTCLoss,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        encoded = []
        hook = None
        if self.aux_head is not None:
            hook = model.encoder.register_forward_hook(
                lambda _module, _args, output: encoded.append(output[0])
            )
        try:
            log_probs, out_lengths = model(tensor, mel_lengths)
        finally:
            if hook is not None:
                hook.remove()
        aux_probs = None
        if self.aux_head is not None:
            aux_probs = self.aux_head(encoded[0]).float().log_softmax(dim=-1)
        with torch.autocast(device_type=tensor.device.type, enabled=False):
            main = criterion(
                log_probs.float().transpose(0, 1).contiguous(),
                targets, out_lengths, target_lengths,
            )
            aux = None if aux_probs is None else criterion(
                aux_probs.transpose(0, 1).contiguous(), targets, out_lengths, target_lengths,
            )
            weight = self.recipe["weight"]
            loss = main if aux is None else (1.0 - weight) * main + weight * aux
        return loss, main, aux

    def checkpoint_state(self) -> dict:
        return {"recipe": dict(self.recipe), "state_dict": self.state_dict()}

    def restore_checkpoint(self, payload: dict) -> None:
        """Exact resume rejects changed recipes; model-only init remains independent."""
        saved = payload.get("training_objective")
        old_recipe = interctc_recipe(payload.get("config", {}))
        if old_recipe != self.recipe or (saved is not None and saved["recipe"] != self.recipe):
            raise ValueError("interctc recipe changed on resume; use init_checkpoint for a new run")
        if saved is None:
            if self.aux_head is not None:
                raise ValueError("resume checkpoint is missing the auxiliary training objective")
        else:
            self.load_state_dict(saved["state_dict"], strict=True)


def load_config(path: str | Path) -> dict:
    """Load a run config (YAML, ``docs/07`` §3)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_training_config(
    config: dict, *, check_files: bool = True, require_tokenizer: bool = True,
) -> dict:
    """No-training preflight; paths follow training's working-directory semantics."""
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping")
    for key in ("model", "optimizer", "schedule", "data"):
        if not isinstance(config.get(key), dict):
            raise TypeError(f"config.{key} must be a mapping")
    for key in ("train", "batch"):
        if not isinstance(config.get(key, {}), dict):
            raise TypeError(f"config.{key} must be a mapping")
    recipe = interctc_recipe(config)
    settings = config.get("train", {})
    for key, default, choices in (
        ("input", "truncated", ("truncated", "padded30")),
        ("autocast", "bf16", ("bf16", "off")),
        ("augment", "full", ("none", "light", "full")),
    ):
        if settings.get(key, default) not in choices:
            raise ValueError(f"unknown {key}: {settings[key]!r}; expected {choices}")
    for name, value, minimum in (
        ("model.vocab_size", config["model"].get("vocab_size"), 1),
        ("model.head_blocks", config["model"].get("head_blocks", 0), 0),
        ("train.epochs", settings.get("epochs", 1), 1),
        ("batch.max_clips", config.get("batch", {}).get("max_clips", 64), 1),
        ("schedule.warmup_steps", config["schedule"].get("warmup_steps", 1500), 0),
        ("seed", config.get("seed", 0), 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (
        ("optimizer.lr_head", config["optimizer"].get("lr_head")),
        ("batch.max_seconds", config.get("batch", {}).get("max_seconds", 320.0)),
        ("train.grad_clip", settings.get("grad_clip", 1.0)),
        ("optimizer.eps", config["optimizer"].get("eps", 1e-8)),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (
            math.isfinite(value) and value > 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    frozen_encoder = bool(config["model"].get("freeze_encoder", False))
    if not frozen_encoder or "lr_encoder" in config["optimizer"]:
        value = config["optimizer"].get("lr_encoder")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (
            math.isfinite(value) and (value >= 0 if frozen_encoder else value > 0)
        ):
            requirement = "nonnegative" if frozen_encoder else "positive"
            raise ValueError(f"optimizer.lr_encoder must be finite and {requirement}")
    factor = config["schedule"].get("min_factor", 0.05)
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 0 <= factor <= 1:
        raise ValueError("schedule.min_factor must satisfy 0 <= min_factor <= 1")
    betas = config["optimizer"].get("betas", (0.9, 0.98))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2 or any(
        isinstance(beta, bool) or not isinstance(beta, (int, float)) or not 0 <= beta < 1
        for beta in betas
    ):
        raise ValueError("optimizer.betas must contain two numbers in [0, 1)")
    inventory = []
    if require_tokenizer and not config.get("tokenizer"):
        raise ValueError("config.tokenizer is required")
    if config.get("tokenizer"):
        inventory.append(Path(config["tokenizer"]))
    for key in ("init", "init_checkpoint"):
        if config["model"].get(key):
            inventory.append(Path(config["model"][key]))
    if not config["data"].get("train"):
        raise ValueError("data.train must contain at least one source")
    for key in ("train", "dev", "dev_en"):
        sources = config["data"].get(key, [])
        if not isinstance(sources, list):
            raise TypeError(f"data.{key} must be a list of sources")
        for source in sources:
            if not isinstance(source, dict) or not source.get("manifest") or not source.get("shards"):
                raise ValueError(f"data.{key} sources require manifest and shards paths")
            inventory.extend((Path(source["manifest"]), Path(source["shards"]) / "index.json"))
    if check_files:
        missing = [str(path) for path in inventory if not path.is_file()]
        for key in ("train", "dev", "dev_en"):
            for source in config["data"].get(key, []):
                root = Path(source["shards"])
                if (root / "index.json").is_file():
                    for shard in sorted({entry["shard"] for entry in load_index(root / "index.json").values()}):
                        path = root / shard
                        inventory.append(path)
                        if not path.is_file():
                            missing.append(str(path))
        if missing:
            raise ValueError("missing training files (replace placeholders):\n" + "\n".join(missing))
    return {"interctc": recipe, "files": list(dict.fromkeys(map(str, inventory)))}


def training_recipe_fingerprint(config: dict, tokenizer_sha256: str | None = None) -> str:
    """Pin trajectory settings and small input files, not multi-GB mel shard payloads."""
    if tokenizer_sha256 is None and config.get("tokenizer"):
        tokenizer_sha256 = _sha256_file(config["tokenizer"])
    settings = {
        "epochs": 1, "input": "truncated", "autocast": "bf16", "augment": "full",
        "grad_clip": 1.0, "max_nonfinite_steps": 20, "empty_cache_every": 0,
    }
    settings.update(config.get("train", {}))
    for key in ("log_every", "checkpoint_minutes", "keep_last"):
        settings.pop(key, None)
    settings["interctc"] = interctc_recipe(config)
    model = {"head_blocks": 0, "freeze_encoder": False, **config.get("model", {})}
    for key in ("init", "init_checkpoint"):
        model.pop(key, None)  # Initializers are not read when restoring checkpoint weights.
    optimizer = {"betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.01,
                 **config.get("optimizer", {})}
    data = copy.deepcopy(config.get("data", {}))
    for key in ("train", "dev", "dev_en"):
        for source in data.get(key, []):
            source["manifest"] = str(Path(source["manifest"]).resolve())
            source["shards"] = str(Path(source["shards"]).resolve())
            source["manifest_sha256"] = _sha256_file(source["manifest"])
            source["index_sha256"] = _sha256_file(Path(source["shards"]) / "index.json")
    recipe = {
        "version": 1, "augmentation_version": "whole-utterance-tempo-fallback-v1",
        "augmentation_full": asdict(AugmentConfig()),
        "augmentation_light": asdict(AugmentConfig.light()),
        "max_mel_frames": MAX_MEL_FRAMES, "seed": config.get("seed", 0),
        "tokenizer_sha256": tokenizer_sha256, "encoder": asdict(EncoderConfig()),
        "model": model, "train": settings, "optimizer": optimizer,
        "schedule": {"warmup_steps": 1500, "min_factor": 0.05, **config.get("schedule", {})},
        "batch": {"max_seconds": 320.0, "max_clips": 64, **config.get("batch", {})},
        "data": data,
    }
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_resume_recipe(payload: dict, config: dict, tokenizer_sha256: str | None) -> None:
    old_config = payload.get("config")
    if not isinstance(old_config, dict):
        raise ValueError(  # noqa: TRY004
            "resume checkpoint has no recipe; use init_checkpoint for a new run"
        )
    saved_sha = payload.get("tokenizer_sha256")
    if saved_sha is None and old_config.get("tokenizer"):
        try:
            saved_sha = _sha256_file(old_config["tokenizer"])
        except OSError as error:
            raise ValueError("legacy tokenizer file missing; use init_checkpoint for a new run") from error
    if saved_sha != tokenizer_sha256:
        raise ValueError("tokenizer content changed on resume; use init_checkpoint for a new run")
    previous = payload.get("training_recipe_fingerprint")
    if previous is None:
        previous = training_recipe_fingerprint(old_config, saved_sha)
    if previous != training_recipe_fingerprint(config, tokenizer_sha256):
        raise ValueError("training recipe changed on resume; use init_checkpoint for a new run")


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
    check_training_config(config, check_files=False, require_tokenizer=tokenize is None)
    tokenizer_sha256 = _sha256_file(config["tokenizer"]) if config.get("tokenizer") else None
    recipe_fingerprint = training_recipe_fingerprint(config, tokenizer_sha256)
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
    validate_training_rows(rows)
    dev_data = []
    for name, key in (("it", "dev"), ("en", "dev_en")):
        sources = config["data"].get(key)
        if sources:
            dev_index, dev_rows, dev_root = load_sources(sources)
            validate_training_rows(rows, dev_rows)
            dev_data.append((name, dev_index, dev_rows, dev_root))
    durations = [float(row["duration"]) for row in rows]

    augment_name = str(train_cfg.get("augment", "full"))
    augment = {"none": None, "light": AugmentConfig.light(), "full": AugmentConfig()}[augment_name]
    dataset = MelDataset(index, shards_root, rows, augment=augment, seed=seed)

    model = MothSTT(
        vocab_size=vocab_size, head_blocks=int(config["model"].get("head_blocks", 0))
    )
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
    for name, dev_index, dev_rows, dev_root in dev_data:
        dev_sets.append((name, MelDataset(dev_index, dev_root, dev_rows, augment=None, seed=seed)))

    steps_per_epoch = len(
        plan_batches(durations, max_seconds=max_seconds, max_clips=max_clips, seed=seed, epoch=0)
    )
    if steps_per_epoch == 0:
        raise ValueError("no training batches — is the manifest empty?")
    epochs = int(train_cfg.get("epochs", 1))
    total_steps = epochs * steps_per_epoch
    step_cap = min(total_steps, int(max_steps)) if max_steps is not None else total_steps

    objective = TrainingObjective(model, config)
    optimizer = _build_optimizer(model, config, objective)
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
        _validate_resume_recipe(payload, config, tokenizer_sha256)
        objective.restore_checkpoint(payload)
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
    objective.to(device)
    if resumed_from is not None:
        _move_optimizer_state(optimizer, device)

    def save(best: bool = False) -> Path:
        nonlocal last_save
        payload = {
            "step": state.step,
            "epoch": state.epoch,
            "batch_cursor": state.batch_cursor,
            "model": model.state_dict(),
            "training_objective": objective.checkpoint_state(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": None,  # bf16 autocast needs no GradScaler
            "rng": {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            "config": config,
            "tokenizer_sha256": tokenizer_sha256,
            "training_recipe_fingerprint": recipe_fingerprint,
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
            "interctc": objective.recipe,
            "auxiliary_params": sum(p.numel() for p in objective.parameters()),
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
            loss, main_loss, aux_loss = objective(
                model, tensor, mel_lengths, targets, target_lengths, criterion
            )

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
                [p for group in optimizer.param_groups for p in group["params"]],
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
                    "main_loss": float(main_loss.detach()),
                    "aux_loss": None if aux_loss is None else float(aux_loss.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_encoder": (
                        optimizer.param_groups[-1]["lr"]
                        if len(optimizer.param_groups) > 1
                        else None
                    ),
                    "grad_norm": grad_norm,
                    "audio_seconds": round(window_seconds, 2),
                    "audio_seconds_per_s": round(window_seconds / elapsed, 3) if elapsed else 0.0,
                    "memory_mib": _memory_mib(device_kind),
                    "memory_unit": "MiB (2**20 bytes)",
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
    parser.add_argument(
        "--check-config", action="store_true",
        help="validate configuration and file inventory only; never construct or run training",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if args.check_config:
        try:
            config = load_config(config_path)
            report = check_training_config(config)
            report["tokenizer_sha256"] = _sha256_file(config["tokenizer"])
            report["training_recipe_fingerprint"] = training_recipe_fingerprint(
                config, report["tokenizer_sha256"]
            )
        except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as error:
            print(f"config check failed: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"status": "ok", **report}, indent=2))
        return 0
    config = load_config(config_path)
    summary = train(
        config, device=args.device, max_steps=args.max_steps, out_dir=config_path.parent
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------------------------
# internals


def _build_optimizer(
    model: MothSTT, config: dict, objective: TrainingObjective | None = None,
) -> torch.optim.AdamW:
    settings = config["optimizer"]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    head_params += [p for p in model.head_blocks.parameters() if p.requires_grad]
    if objective is not None:
        head_params += [p for p in objective.parameters() if p.requires_grad]
    groups = [
        {
            "params": head_params,
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
    for item in items:
        if item["mel"].shape[1] > MAX_MEL_FRAMES:
            raise ValueError(
                f"{item['row'].get('id')}: input exceeds {MAX_MEL_FRAMES} mel frames; "
                "refusing to crop audio with a complete transcript"
            )
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
        state = payload["model"]
        current = model.state_dict()
        # A tokenizer change (e.g. V=1024 → V=4096) changes only the CTC head shape; the encoder
        # and the transformer blocks are still a valid initialisation. Skip shape-mismatched
        # tensors and let them be initialised fresh, but keep every other mismatch an error.
        skipped = {
            name for name, tensor in state.items()
            if name in current and current[name].shape != tensor.shape
        }
        filtered = {name: tensor for name, tensor in state.items() if name not in skipped}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        allowed_missing = {name for name in missing if name.startswith("head_blocks.")}
        allowed_missing |= skipped
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing {sorted(set(missing) - allowed_missing)}, "
                f"unexpected {sorted(unexpected)}, skipped {sorted(skipped)}"
            )


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


def _memory_mib(device_kind: str) -> float | None:
    """Device allocation or process peak RSS in MiB; unavailable platforms return None."""
    if device_kind == "mps" and torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**20
    if device_kind == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    if resource is None:
        return None
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
