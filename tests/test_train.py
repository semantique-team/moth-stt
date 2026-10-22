"""Tests for stt/train.py — tiny synthetic shards, CPU only (design: `docs/04` §2–6)."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from stt.data.build_mel import build_shards
from stt.data.dataset import MelDataset
from stt.model import MothSTT
from stt.train import (
    _apply_model_init,
    _build_optimizer,
    _move_optimizer_state,
    iter_batches,
    load_config,
    load_sources,
    train,
)


def _sine(seconds: float = 2.0, sample_rate: int = 16000, frequency: float = 220.0) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def _tokenize(text: str) -> list[int]:
    return [1, 2, 3]


def _make_data(
    tmp_path: Path, clips: int = 2, tag: str = "a", shards_dir: Path | None = None
) -> dict:
    base = tmp_path / f"src_{tag}"
    audio_dir = base / "audio"
    audio_dir.mkdir(parents=True)
    rows = []
    for index in range(clips):
        name = f"c{index}.wav"
        sf.write(audio_dir / name, _sine(2.0, frequency=220.0 + 55.0 * index), 16000)
        rows.append(
            {
                "id": f"{tag}-c{index}",
                "audio": name,
                "text": "ciao ciao",
                "norm": "ciao ciao",
                "duration": 2.0,
                "source": "test",
                "split": "train",
                "speaker": "s1",
                "lang": "it",
                "category": None,
                "en_spans": [],
            }
        )
    shards = shards_dir if shards_dir is not None else base / "shards"
    build_shards(rows, audio_dir, shards, workers=1)
    manifest = base / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return {"manifest": manifest, "shards": shards}


def _config(
    sources: list[dict],
    vocab: int = 16,
    epochs: int = 1,
    augment: str = "none",
    **train_overrides,
) -> dict:
    train_section = {
        "epochs": epochs,
        "grad_clip": 1.0,
        "autocast": "off",
        "log_every": 1,
        "checkpoint_minutes": 100000,
        "max_nonfinite_steps": 2,
        "input": "truncated",
        "augment": augment,
    }
    train_section.update(train_overrides)
    return {
        "run": "TEST",
        "seed": 0,
        "model": {"vocab_size": vocab, "freeze_encoder": True},
        "data": {
            "train": [{"manifest": str(s["manifest"]), "shards": str(s["shards"])} for s in sources]
        },
        "batch": {"max_seconds": 2.5, "max_clips": 1},
        "optimizer": {
            "lr_head": 1e-3,
            "lr_encoder": 1e-4,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
        "schedule": {"warmup_steps": 1, "min_factor": 0.05},
        "train": train_section,
    }


def _log_lines(out_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (out_dir / "train.log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _step_lines(out_dir: Path) -> list[dict]:
    return [line for line in _log_lines(out_dir) if line.get("event") == "step"]


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("run: T\nseed: 1\ntrain:\n  epochs: 2\n", encoding="utf-8")
    config = load_config(path)
    assert config["run"] == "T"
    assert config["train"]["epochs"] == 2


def test_iter_batches_resumes_and_advances_epochs() -> None:
    durations = [2.0] * 5
    full = list(
        itertools.islice(
            iter_batches(durations, max_seconds=2.5, max_clips=1, seed=0, epoch=0, batch_cursor=0),
            3,
        )
    )
    resumed = list(
        itertools.islice(
            iter_batches(durations, max_seconds=2.5, max_clips=1, seed=0, epoch=0, batch_cursor=2),
            1,
        )
    )
    assert resumed == full[2:]
    tail = list(
        itertools.islice(
            iter_batches(durations, max_seconds=2.5, max_clips=1, seed=0, epoch=0, batch_cursor=4),
            2,
        )
    )
    assert [epoch for epoch, _index, _batch in tail] == [0, 1]


def test_load_sources_prefixes_shard_paths(tmp_path: Path) -> None:
    root = tmp_path / "shards"
    first = _make_data(tmp_path, clips=2, tag="a", shards_dir=root / "a")
    second = _make_data(tmp_path, clips=2, tag="b", shards_dir=root / "b")
    index, rows, shards_root = load_sources(
        [
            {"manifest": str(first["manifest"]), "shards": str(first["shards"])},
            {"manifest": str(second["manifest"]), "shards": str(second["shards"])},
        ]
    )
    assert shards_root == root
    assert len(rows) == 4 and len(index) == 4
    assert index["a-c0"]["shard"] == "a/shard-000.npy"
    assert index["b-c1"]["shard"] == "b/shard-000.npy"


def test_train_one_step_and_log_schema(tmp_path: Path) -> None:
    data = _make_data(tmp_path)
    out = tmp_path / "run"
    summary = train(_config([data]), device="cpu", max_steps=1, tokenize=_tokenize, out_dir=out)

    assert summary["steps"] == 1
    assert summary["parity"] is None  # nothing to compare on CPU
    lines = _step_lines(out)
    assert len(lines) == 1
    line = lines[0]
    assert line["step"] == 1 and line["epoch"] == 0 and isinstance(line["batch"], int)
    for key in ("loss", "lr", "grad_norm", "audio_seconds", "audio_seconds_per_s", "memory_mb"):
        assert key in line
    assert np.isfinite(line["loss"]) and line["loss"] > 0
    assert np.isfinite(line["grad_norm"]) and line["grad_norm"] > 0  # pre-clip norm
    assert 0.0 < line["lr"] <= 1e-3

    checkpoint = out / "checkpoints" / "ckpt_0000001.pt"
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["step"] == 1 and payload["batch_cursor"] == 1
    assert payload["optimizer"]["state"]  # the optimizer actually stepped
    assert "scheduler" in payload and "rng" in payload
    assert payload["scaler"] is None  # bf16 autocast needs no GradScaler


def test_padded30_input_runs(tmp_path: Path) -> None:
    data = _make_data(tmp_path)
    out = tmp_path / "run"
    summary = train(
        _config([data], input="padded30"),
        device="cpu",
        max_steps=1,
        tokenize=_tokenize,
        out_dir=out,
    )
    assert summary["steps"] == 1
    assert _step_lines(out)[0]["loss"] > 0


def test_resume_reproduces_loader_position(tmp_path: Path) -> None:
    data = _make_data(tmp_path, clips=6)
    config = _config([data], epochs=2)

    straight_out = tmp_path / "straight"
    train(config, device="cpu", max_steps=4, tokenize=_tokenize, out_dir=straight_out)
    straight = _step_lines(straight_out)

    resumed_out = tmp_path / "resumed"
    train(config, device="cpu", max_steps=2, tokenize=_tokenize, out_dir=resumed_out)
    summary = train(config, device="cpu", max_steps=4, tokenize=_tokenize, out_dir=resumed_out)
    resumed = _step_lines(resumed_out)

    assert summary["steps"] == 4
    assert [line["step"] for line in resumed] == [1, 2, 3, 4]
    # the first batch after the resume is the one the straight run processed third
    assert [line["batch"] for line in resumed] == [line["batch"] for line in straight]
    assert [line["loss"] for line in resumed[2:]] == pytest.approx(
        [line["loss"] for line in straight[2:]], rel=1e-5, abs=1e-7
    )


def test_nonfinite_guard_aborts(tmp_path: Path) -> None:
    data = _make_data(tmp_path, clips=4)
    for shard in sorted(data["shards"].glob("shard-*.npy")):
        array = np.load(shard)
        array[:] = np.nan
        np.save(shard, array)

    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="non-finite"):
        train(
            _config([data], max_nonfinite_steps=2),
            device="cpu",
            max_steps=10,
            tokenize=_tokenize,
            out_dir=out,
        )
    skips = [line for line in _log_lines(out) if line.get("event") == "skip"]
    assert len(skips) == 3  # aborts on the third consecutive, max allowed is 2
    assert skips[-1]["consecutive_nonfinite"] == 3


def test_dev_evaluation_line_and_best_checkpoint(tmp_path: Path) -> None:
    data = _make_data(tmp_path, clips=3)
    config = _config([data], eval_every_steps=1)
    config["data"]["dev"] = [
        {"manifest": str(data["manifest"]), "shards": str(data["shards"])}
    ]
    out = tmp_path / "run"
    summary = train(
        config,
        device="cpu",
        max_steps=2,
        tokenize=_tokenize,
        pieces={identifier: f"p{identifier}" for identifier in range(17)},
        out_dir=out,
    )
    dev_lines = [line for line in _log_lines(out) if line.get("event") == "dev"]
    assert dev_lines
    assert np.isfinite(dev_lines[-1]["wer"]) and dev_lines[-1]["wer"] >= 0.0
    assert (out / "checkpoints" / "ckpt_best.pt").exists()
    assert summary["dev"]["wer"] == pytest.approx(dev_lines[-1]["wer"])


def test_dev_en_is_scored_separately_and_selects_on_macro(tmp_path: Path) -> None:
    data = _make_data(tmp_path, clips=3)
    config = _config([data], eval_every_steps=1)
    sources = [{"manifest": str(data["manifest"]), "shards": str(data["shards"])}]
    config["data"]["dev"] = sources
    config["data"]["dev_en"] = sources
    out = tmp_path / "run"
    summary = train(
        config,
        device="cpu",
        max_steps=2,
        tokenize=_tokenize,
        pieces={identifier: f"p{identifier}" for identifier in range(17)},
        out_dir=out,
    )
    dev = summary["dev"]
    assert set(dev["sets"]) == {"it", "en"}
    assert dev["wer"] == pytest.approx(
        (dev["sets"]["it"]["wer"] + dev["sets"]["en"]["wer"]) / 2
    )
    assert dev["wer_ci"] is None


def test_training_updates_the_dataset_epoch(monkeypatch, tmp_path: Path) -> None:
    data = _make_data(tmp_path, clips=3)
    config = _config([data], epochs=2, augment="light")
    seen_epochs: list[int] = []
    original = MelDataset.__getitem__

    def spy(self, position):
        if self.augment_config is not None:
            seen_epochs.append(self.epoch)
        return original(self, position)

    monkeypatch.setattr(MelDataset, "__getitem__", spy)
    train(
        config,
        device="cpu",
        max_steps=4,
        tokenize=_tokenize,
        pieces={identifier: f"p{identifier}" for identifier in range(17)},
        out_dir=tmp_path / "run",
    )
    assert {0, 1} <= set(seen_epochs)


def test_apply_model_init_allows_new_head_blocks(tmp_path: Path) -> None:
    source = MothSTT(vocab_size=16)
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": source.state_dict()}, checkpoint)

    target = MothSTT(vocab_size=16, head_blocks=2)
    _apply_model_init(target, {"model": {"init_checkpoint": str(checkpoint)}})
    assert torch.equal(target.encoder.conv1.weight, source.encoder.conv1.weight)
    assert torch.equal(target.head.weight, source.head.weight)


def test_optimizer_includes_head_blocks() -> None:
    model = MothSTT(vocab_size=16, head_blocks=2)
    optimizer = _build_optimizer(model, {"optimizer": {"lr_head": 1e-3, "lr_encoder": 1e-4}})
    head_group = next(group for group in optimizer.param_groups if group["name"] == "head")
    expected = sum(parameter.numel() for parameter in model.head.parameters())
    expected += sum(parameter.numel() for parameter in model.head_blocks.parameters())
    assert sum(parameter.numel() for parameter in head_group["params"]) == expected


def test_move_optimizer_state_follows_the_model_device() -> None:
    model = MothSTT(vocab_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(parameter),
        }

    _move_optimizer_state(optimizer, "meta")
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device.type == "meta"


def test_apply_model_init_loads_a_full_checkpoint(tmp_path: Path) -> None:
    source = MothSTT(vocab_size=16)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(0.125)
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": source.state_dict()}, checkpoint)

    target = MothSTT(vocab_size=16)
    _apply_model_init(target, {"model": {"init_checkpoint": str(checkpoint)}})
    for expected, actual in zip(source.state_dict().values(), target.state_dict().values()):
        assert torch.equal(expected, actual)
