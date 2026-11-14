"""Tests for stt/train.py — tiny synthetic shards, CPU only (design: `docs/04` §2–6)."""

import hashlib
import importlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from stt.data.build_mel import build_shards
from stt.data.dataset import MelDataset
from stt.model import EncoderConfig, MothSTT
from stt.train import (
    DistillationObjective,
    _apply_model_init,
    _build_optimizer,
    _evaluate,
    _evaluate_all,
    _move_optimizer_state,
    _prepare_batch,
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
    tmp_path: Path, clips: int = 2, tag: str = "a", shards_dir: Path | None = None,
    *, split: str = "train", lang: str = "it",
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
                "text": "ciao ciao" if split == "train" else (
                    "buongiorno mondo" if lang == "it" else "hello world"
                ),
                "norm": "ciao ciao" if split == "train" else (
                    "buongiorno mondo" if lang == "it" else "hello world"
                ),
                "duration": 2.0,
                "source": "test",
                "split": split,
                "role": split,
                "speaker": "s1",
                "lang": lang,
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


@pytest.mark.parametrize("changes", [
    {"split": "test"}, {"split": "dev"}, {"source": "recorded_t"},
    {"source": "recorded_a"}, {"speaker": "S3"},
])
def test_train_rejects_held_out_rows(tmp_path: Path, changes: dict) -> None:
    data = _make_data(tmp_path, clips=1)
    path = data["manifest"]
    row = json.loads(path.read_text(encoding="utf-8")) | changes
    path.write_text(json.dumps(row), encoding="utf-8")
    # Loading remains role-neutral for evaluation callers.
    assert load_sources([data])[1][0] == row
    with pytest.raises(ValueError, match="cannot train"):
        train(_config([data]), device="cpu", max_steps=1, tokenize=_tokenize,
              out_dir=tmp_path / "run")


@pytest.mark.parametrize("key", ["dev", "dev_en"])
@pytest.mark.parametrize("splice", [False, True])
def test_train_rejects_dev_overlap(tmp_path: Path, key: str, splice: bool) -> None:
    data = _make_data(tmp_path, clips=1)
    config = _config([data])
    if splice:
        dev = _make_data(tmp_path, clips=1, tag="dev", split="dev",
                         lang="en" if key == "dev_en" else "it")
        row = json.loads(dev["manifest"].read_text(encoding="utf-8"))
        row.update(source="spliced_it_en", audio="a-c0+en-1", split="dev")
        dev["manifest"].write_text(json.dumps(row), encoding="utf-8")
    else:
        dev = _make_data(tmp_path, clips=1, tag="dev", split="dev",
                         lang="en" if key == "dev_en" else "it")
        row = json.loads(dev["manifest"].read_text(encoding="utf-8"))
        row["source_ids"] = ["a-c0"]
        dev["manifest"].write_text(json.dumps(row), encoding="utf-8")
    config["data"][key] = [dev]
    with pytest.raises(ValueError, match="overlap"):
        train(config, device="cpu", max_steps=1, tokenize=_tokenize, out_dir=tmp_path / "run")


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
    for key in ("loss", "lr", "grad_norm", "audio_seconds", "audio_seconds_per_s", "memory_mib"):
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


@pytest.mark.parametrize("mode", ["skip", "recover", "abort"])
def test_finite_loss_nonfinite_backward_never_steps(monkeypatch, tmp_path, mode):
    module = importlib.import_module("stt.train")
    bad_gradients = [float("nan"), float("inf"), None] * 2 if mode == "recover" else [
        float("nan"), float("inf"), float("nan")
    ]
    data = _make_data(tmp_path, clips=len(bad_gradients))
    model = MothSTT(16, EncoderConfig(width=16, n_layers=1, n_heads=2, ffn_dim=32))
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(module, "MothSTT", lambda **kwargs: model)
    gradients = iter(bad_gradients)

    def corrupt(gradient):
        value = next(gradients)
        return gradient if value is None else torch.full_like(gradient, value)

    model.head.weight.register_hook(corrupt)
    captured = {}
    original_optimizer = module._build_optimizer
    original_scheduler = module._build_scheduler

    def optimizer_spy(*args):
        optimizer = original_optimizer(*args)
        captured["optimizer"] = optimizer

        def check_gradients(optimizer, args, kwargs):
            assert all(p.grad is None or torch.isfinite(p.grad).all()
                       for group in optimizer.param_groups for p in group["params"])

        optimizer.register_step_pre_hook(check_gradients)
        return optimizer

    def scheduler_spy(*args, **kwargs):
        scheduler = original_scheduler(*args, **kwargs)
        captured["scheduler"] = scheduler
        return scheduler

    monkeypatch.setattr(module, "_build_optimizer", optimizer_spy)
    monkeypatch.setattr(module, "_build_scheduler", scheduler_spy)
    out = tmp_path / "run"
    config = _config([data], max_nonfinite_steps=2 if mode == "abort" else 3)
    if mode == "abort":
        with pytest.raises(RuntimeError, match="non-finite losses or gradients"):
            train(config, device="cpu", tokenize=_tokenize, out_dir=out)
    else:
        summary = train(config, device="cpu", tokenize=_tokenize, out_dir=out)
        assert summary["steps"] == (2 if mode == "recover" else 0)
        assert summary["skipped_nonfinite"] == (4 if mode == "recover" else 3)
    skips = [line for line in _log_lines(out) if line["event"] == "skip"]
    assert all(line["stage"] == "gradient" and line["reason"] == "non_finite" for line in skips)
    assert [line["consecutive_nonfinite"] for line in skips] == (
        [1, 2, 1, 2] if mode == "recover" else [1, 2, 3]
    )
    assert captured["scheduler"].last_epoch == (2 if mode == "recover" else 0)
    if mode != "recover":
        assert not captured["optimizer"].state
        assert all(p.grad is None for p in model.parameters())
        for name, value in model.state_dict().items():
            assert torch.equal(value, initial[name])
    else:
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert all(int(state["step"]) == 2 for state in captured["optimizer"].state.values())


@pytest.mark.parametrize("empty", [False, True])
def test_dev_requirements_fail_before_model_construction(monkeypatch, tmp_path, empty):
    module = importlib.import_module("stt.train")
    data = _make_data(tmp_path, clips=1)
    dev = _make_data(tmp_path, clips=1, tag="dev", split="dev")
    if empty:
        dev["manifest"].write_text("", encoding="utf-8")
    config = _config([data])
    config["data"]["dev"] = [dev]
    monkeypatch.setattr(module, "MothSTT", lambda **kwargs: pytest.fail("constructed model"))
    with pytest.raises(ValueError, match="at least one dev row" if empty else "decode pieces"):
        train(config, device="cpu", tokenize=_tokenize, out_dir=tmp_path / "run")


@pytest.mark.parametrize("key", ["dev", "dev_en"])
@pytest.mark.parametrize("changes,match", [
    ({"split": "train"}, "split/role"),
    ({"split": "test"}, "split/role"),
    ({"role": "train"}, "split/role"),
    ({"role": "test"}, "split/role"),
    ({"lang": "mix"}, "wrong-language"),
    ({"norm": "", "text": ""}, "nonempty"),
    ({"text": ""}, "dev speech"),
    ({"duration": 0}, "dev speech"),
    ({"duration": float("nan")}, "dev speech"),
    ({"duration": float("inf")}, "dev speech"),
    ({"source": "recorded_t"}, "historical"),
    ({"source": "recorded_a"}, "historical"),
    ({"source": "recorded_r"}, "historical"),
    ({"speaker": "S3"}, "historical"),
    ({"norm": "ciao ciao", "text": "ciao ciao"}, "overlap"),
])
def test_production_dev_validation_before_model(monkeypatch, tmp_path, key, changes, match):
    module = importlib.import_module("stt.train")
    data = _make_data(tmp_path, clips=1)
    dev = _make_data(tmp_path, clips=1, tag="dev", split="dev",
                     lang="en" if key == "dev_en" else "it")
    row = json.loads(dev["manifest"].read_text(encoding="utf-8")) | changes
    dev["manifest"].write_text(json.dumps(row), encoding="utf-8")
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"synthetic identity")
    config = _config([data]) | {"tokenizer": str(tokenizer)}
    config["data"][key] = [dev]
    monkeypatch.setattr(module, "MothSTT", lambda **kwargs: pytest.fail("constructed model"))
    monkeypatch.setattr(module, "load_tokenizer", lambda *args: pytest.fail("loaded tokenizer"))
    with pytest.raises(ValueError, match=match):
        train(config, device="cpu", out_dir=tmp_path / "run")


@pytest.mark.parametrize("identity", ["id", "source_id", "constituent", "audio"])
def test_dev_languages_must_have_disjoint_identities(monkeypatch, tmp_path, identity):
    module = importlib.import_module("stt.train")
    data = _make_data(tmp_path, clips=1)
    dev = _make_data(tmp_path, clips=1, tag="dev", split="dev")
    dev_en = _make_data(tmp_path, clips=1, tag="dev_en", split="dev", lang="en")
    row = json.loads(dev_en["manifest"].read_text(encoding="utf-8"))
    if identity == "id":
        row["id"] = "dev-c0"
        index_path = dev_en["shards"] / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["dev-c0"] = index.pop("dev_en-c0")
        index_path.write_text(json.dumps(index), encoding="utf-8")
    elif identity == "source_id":
        row["source_ids"] = ["dev-c0"]
    elif identity == "constituent":
        row.update(source="spliced_it_en", audio="dev-c0+another-clip")
    else:
        shared = str(tmp_path / "shared.wav")
        it_row = json.loads(dev["manifest"].read_text(encoding="utf-8")) | {"audio": shared}
        dev["manifest"].write_text(json.dumps(it_row), encoding="utf-8")
        row["audio"] = shared
    dev_en["manifest"].write_text(json.dumps(row), encoding="utf-8")
    config = _config([data])
    config["data"].update(dev=[dev], dev_en=[dev_en])
    monkeypatch.setattr(module, "MothSTT", lambda **kwargs: pytest.fail("constructed model"))
    with pytest.raises(ValueError, match="IT/EN dev identity overlap"):
        train(config, device="cpu", tokenize=_tokenize, pieces={1: "a"},
              out_dir=tmp_path / "run")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_dev_rejects_nonfinite_logits_and_restores_mode(monkeypatch, tmp_path, value):
    data = _make_data(tmp_path, clips=1, tag="dev", split="dev")
    index, rows, root = load_sources([data])
    dataset = MelDataset(index, root, rows, augment=None)
    model = MothSTT(16, EncoderConfig(width=16, n_layers=1, n_heads=2, ffn_dim=32))
    monkeypatch.setattr(model, "forward", lambda *args: (
        torch.full((1, 100, 17), value), torch.tensor([100])
    ))
    with pytest.raises(ValueError, match="non-finite log probabilities"):
        _evaluate(model, dataset, {1: "a"}, "cpu", 2.5, 1, "truncated")
    assert model.training


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_dev_macro_rejects_nonfinite_wer(monkeypatch, value):
    monkeypatch.setattr("stt.train._evaluate", lambda *args: {"wer": value})
    with pytest.raises(ValueError, match="non-finite WER"):
        _evaluate_all(None, [("it", None)], {}, "cpu", 2.5, 1, "truncated")


def test_dev_evaluation_line_and_best_checkpoint(tmp_path: Path) -> None:
    data = _make_data(tmp_path, clips=3)
    dev = _make_data(tmp_path, clips=2, tag="dev", split="dev")
    config = _config([data], eval_every_steps=1)
    config["data"]["dev"] = [
        {"manifest": str(dev["manifest"]), "shards": str(dev["shards"])}
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
    dev = _make_data(tmp_path, clips=2, tag="dev", split="dev")
    dev_en = _make_data(tmp_path, clips=2, tag="dev_en", split="dev", lang="en")
    config = _config([data], eval_every_steps=1)
    sources = [{"manifest": str(dev["manifest"]), "shards": str(dev["shards"])}]
    config["data"]["dev"] = sources
    config["data"]["dev_en"] = [dev_en]
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


def test_apply_model_init_skips_a_different_vocab_head(tmp_path: Path) -> None:
    source = MothSTT(vocab_size=16)
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": source.state_dict()}, checkpoint)

    target = MothSTT(vocab_size=32)
    _apply_model_init(target, {"model": {"init_checkpoint": str(checkpoint)}})
    assert torch.equal(target.encoder.conv1.weight, source.encoder.conv1.weight)
    assert target.head.weight.shape == (33, source.head.weight.shape[1])


@pytest.mark.parametrize("same", [False, True])
def test_init_tokenizer_identity_controls_same_size_head(tmp_path, same):
    source = MothSTT(16, EncoderConfig(width=16, n_layers=1, n_heads=2, ffn_dim=32))
    target = MothSTT(16, source.encoder_config)
    original_head = target.head.weight.clone()
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"same" if same else b"different")
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": source.state_dict(),
                "tokenizer_sha256": hashlib.sha256(b"same").hexdigest()}, checkpoint)
    _apply_model_init(target, {"tokenizer": str(tokenizer),
                              "model": {"init_checkpoint": str(checkpoint)}})
    assert torch.equal(target.encoder.conv1.weight, source.encoder.conv1.weight)
    assert torch.equal(target.head.weight, source.head.weight if same else original_head)


def test_init_rejects_non_head_shape_mismatch(tmp_path):
    source = MothSTT(16, EncoderConfig(width=16, n_layers=1, n_heads=2, ffn_dim=32))
    state = source.state_dict()
    state["encoder.conv1.weight"] = state["encoder.conv1.weight"][:1]
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": state}, checkpoint)
    with pytest.raises(RuntimeError, match="non-head shape mismatch"):
        _apply_model_init(source, {"model": {"init_checkpoint": str(checkpoint)}})


def test_init_vocab_resize_does_not_need_old_tokenizer(tmp_path):
    source = MothSTT(16, EncoderConfig(width=16, n_layers=1, n_heads=2, ffn_dim=32))
    target = MothSTT(32, source.encoder_config)
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": source.state_dict(),
                "config": {"tokenizer": str(tmp_path / "missing.model")}}, checkpoint)
    tokenizer = tmp_path / "new.model"
    tokenizer.write_bytes(b"new")
    _apply_model_init(target, {"tokenizer": str(tokenizer),
                              "model": {"init_checkpoint": str(checkpoint)}})
    assert torch.equal(target.encoder.conv1.weight, source.encoder.conv1.weight)


@pytest.mark.parametrize("identity", ["same", "different", "synthetic"])
def test_distill_tokenizer_compatibility(tmp_path, identity):
    model = MothSTT(16, EncoderConfig(width=16, n_layers=1, n_heads=2, ffn_dim=32))
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"model": model.state_dict(),
                "tokenizer_sha256": hashlib.sha256(b"same").hexdigest()}, checkpoint)
    config = {"train": {"distill": {"weight": 0.5, "checkpoint": str(checkpoint)}}}
    if identity != "synthetic":
        tokenizer = tmp_path / "tokenizer.model"
        tokenizer.write_bytes(identity.encode())
        config["tokenizer"] = str(tokenizer)
    if identity == "different":
        with pytest.raises(ValueError, match="teacher tokenizer differs"):
            DistillationObjective(model, config)
    else:
        objective = DistillationObjective(model, config)
        assert torch.equal(objective.teacher.head.weight, model.head.weight)


@pytest.mark.parametrize("kind", ["init", "kd"])
@pytest.mark.parametrize("old_path", ["missing", "mutated"])
def test_legacy_tokenizer_path_is_unverifiable_not_identity(tmp_path, kind, old_path):
    source = MothSTT(16, EncoderConfig(width=16, n_layers=1, n_heads=2, ffn_dim=32))
    target = MothSTT(16, source.encoder_config)
    historical = tmp_path / "old.model"
    historical.write_bytes(b"original tokenizer")
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": source.state_dict(), "config": {"tokenizer": str(historical)}},
               checkpoint)
    if old_path == "missing":
        historical.unlink()
        current = tmp_path / "current.model"
        current.write_bytes(b"current tokenizer")
    else:
        historical.write_bytes(b"mutated tokenizer")
        current = historical
    config = {"tokenizer": str(current)}
    if kind == "init":
        config["model"] = {"init_checkpoint": str(checkpoint)}
        with pytest.warns(UserWarning, match="historical tokenizer compatibility is unverifiable"):
            _apply_model_init(target, config)
    else:
        config["train"] = {"distill": {"weight": 0.5, "checkpoint": str(checkpoint)}}
        with pytest.warns(UserWarning, match="historical tokenizer compatibility is unverifiable"):
            target = DistillationObjective(target, config).teacher
    assert torch.equal(target.head.weight, source.head.weight)
    assert torch.equal(target.encoder.conv1.weight, source.encoder.conv1.weight)


@pytest.mark.parametrize("kind", ["init", "kd"])
def test_stored_digest_rejects_mutated_same_tokenizer_path(tmp_path, kind):
    source = MothSTT(16, EncoderConfig(width=16, n_layers=1, n_heads=2, ffn_dim=32))
    target = MothSTT(16, source.encoder_config)
    original_head = target.head.weight.clone()
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"original")
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": source.state_dict(), "config": {"tokenizer": str(tokenizer)},
                "tokenizer_sha256": hashlib.sha256(b"original").hexdigest()}, checkpoint)
    tokenizer.write_bytes(b"mutated")
    config = {"tokenizer": str(tokenizer)}
    if kind == "init":
        config["model"] = {"init_checkpoint": str(checkpoint)}
        _apply_model_init(target, config)
        assert torch.equal(target.head.weight, original_head)
    else:
        config["train"] = {"distill": {"weight": 0.5, "checkpoint": str(checkpoint)}}
        with pytest.raises(ValueError, match="teacher tokenizer differs"):
            DistillationObjective(target, config)


@pytest.mark.parametrize("input_mode", ["truncated", "padded30"])
def test_prepare_batch_rejects_a_long_mel(input_mode) -> None:
    items = [{"mel": np.zeros((80, 3248), dtype=np.float32), "row": {"id": "x"}}]
    with pytest.raises(ValueError, match="x: input exceeds 3000"):
        _prepare_batch(items, input_mode)


@pytest.mark.parametrize("input_mode", ["truncated", "padded30"])
@pytest.mark.parametrize("frames", [2999, 3000])
def test_prepare_batch_preserves_complete_audio_and_transcript(input_mode, frames) -> None:
    mel = np.arange(80 * frames, dtype=np.float32).reshape(80, frames)
    row = {"id": "x", "norm": "ciao fino alla fine"}
    tensor, lengths, rows = _prepare_batch([{"mel": mel, "row": row}], input_mode)
    assert tensor.shape[2] == (3000 if input_mode == "padded30" else frames)
    assert lengths.tolist() == [frames]
    np.testing.assert_array_equal(tensor[0, :, :frames].numpy(), mel)
    assert rows[0] is row
