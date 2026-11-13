"""Synthetic correctness only: no training loop or optimizer updates."""

import copy
import importlib
import json
import subprocess
import sys

import pytest
import torch
import yaml
from torch import nn

from stt.model import EncoderConfig, MothSTT
from stt.train import (
    DistillationObjective,
    TrainingObjective,
    _apply_model_init,
    _atomic_torch_save,
    _build_optimizer,
    _build_scheduler,
    _distill_loss,
    _freeze_parameters,
    _move_optimizer_state,
    _sha256_file,
    _validate_resume_recipe,
    check_training_config,
    distill_recipe,
    interctc_recipe,
    training_recipe_fingerprint,
)


def _model():
    return MothSTT(5, EncoderConfig(width=16, n_layers=1, n_heads=2, ffn_dim=32),
                   head_blocks=2)


def _config(weight=0.25, tap="encoder_output"):
    return {"train": {"interctc": {"weight": weight, "tap": tap}},
            "optimizer": {"lr_head": 1e-3, "lr_encoder": 1e-4}}


def _inputs():
    return (torch.randn(2, 80, 20), torch.tensor([20, 15]),
            torch.tensor([1, 2, 3, 2]), torch.tensor([2, 2]),
            nn.CTCLoss(blank=0, zero_infinity=True))


@pytest.mark.parametrize("weight", [
    -0.1, 1.0, 2.0, float("nan"), float("inf"), None, "bad", True, False,
])
def test_invalid_weight_rejected_without_training(weight):
    with pytest.raises(ValueError, match="weight"):
        interctc_recipe(_config(weight))


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.999])
def test_recipe_accepts_valid_weights(weight):
    assert interctc_recipe(_config(weight)) == {"weight": weight, "tap": "encoder_output"}


@pytest.mark.parametrize("settings", [{"tap": "block2"}, {"unknown": True}, None, 0.5])
def test_invalid_interctc_recipe_rejected(settings):
    with pytest.raises(ValueError, match="interctc"):
        interctc_recipe({"train": {"interctc": settings}})


def test_disabled_objective_matches_existing_ctc_and_state():
    model = _model()
    objective = TrainingObjective(model, {})
    assert objective.aux_head is None
    assert not list(objective.parameters())
    args = _inputs()
    log_probs, lengths = model(*args[:2])
    expected = args[-1](log_probs.transpose(0, 1).contiguous(), args[2], lengths, args[3])
    loss, main, aux = objective(model, *args)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    assert loss is main and aux is None


@pytest.mark.parametrize("autocast", [False, True])
@torch.backends.mkldnn.flags(enabled=False)
def test_weighted_fp32_loss_single_encoder_forward_and_gradients(autocast):
    torch.manual_seed(3)
    model = _model()
    objective = TrainingObjective(model, _config())
    captured = []
    hook = model.encoder.register_forward_hook(lambda _m, _a, out: captured.append(out))
    args = _inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        loss, main, aux = objective(model, *args)
    hook.remove()
    assert len(captured) == 1
    assert loss.dtype == main.dtype == aux.dtype == torch.float32
    torch.testing.assert_close(loss, 0.75 * main + 0.25 * aux, rtol=0, atol=0)
    expected_aux = args[-1](
        objective.aux_head(captured[0][0].float()).log_softmax(-1).transpose(0, 1).contiguous(),
        args[2], captured[0][1], args[3],
    )
    if not autocast:
        torch.testing.assert_close(aux, expected_aux, rtol=0, atol=0)
    aux_encoder_grad, = torch.autograd.grad(aux, model.encoder.conv1.weight, retain_graph=True)
    assert torch.isfinite(aux_encoder_grad).all() and aux_encoder_grad.abs().sum() > 0
    assert torch.autograd.grad(aux, model.head_blocks[0].query.weight,
                               retain_graph=True, allow_unused=True) == (None,)
    loss.backward()
    for parameter in (model.encoder.conv1.weight, model.head_blocks[0].query.weight,
                      model.head.weight, objective.aux_head.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert not model.encoder._forward_hooks


def test_hook_removed_when_forward_fails():
    model = _model()
    objective = TrainingObjective(model, _config())
    args = list(_inputs())
    args[0] = torch.randn(2, 79, 20)
    with pytest.raises(ValueError, match="mel bins"):
        objective(model, *args)
    assert not model.encoder._forward_hooks


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_auxiliary_inherits_head_dtype_and_cpu_forward(dtype):
    model = _model().to(dtype=dtype)
    objective = TrainingObjective(model, _config())
    assert objective.aux_head.weight.dtype == dtype
    assert objective.aux_head.weight.device == model.head.weight.device
    args = list(_inputs())
    args[0] = args[0].to(dtype=dtype)
    with torch.no_grad():
        losses = objective(model, *args)
    assert all(loss.dtype == torch.float32 and torch.isfinite(loss) for loss in losses)


def test_auxiliary_created_on_head_device():
    model = _model().to(device="meta", dtype=torch.float64)
    objective = TrainingObjective(model, _config())
    assert objective.aux_head.weight.device.type == "meta"
    assert objective.aux_head.weight.dtype == torch.float64


def test_auxiliary_backward_with_frozen_encoder():
    model = _model()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    objective = TrainingObjective(model, _config())
    loss, _main, _aux = objective(model, *_inputs())
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert objective.aux_head.weight.grad.abs().sum() > 0
    optimizer = _build_optimizer(model, _config(), objective)
    assert [group["name"] for group in optimizer.param_groups] == ["head"]


def test_auxiliary_in_head_group_clipping_and_device():
    model = _model()
    objective = TrainingObjective(model, _config())
    optimizer = _build_optimizer(model, _config(), objective)
    head = next(group for group in optimizer.param_groups if group["name"] == "head")
    aux_ids = {id(p) for p in objective.parameters()}
    assert aux_ids <= {id(p) for p in head["params"]}
    assert head["lr"] == 1e-3
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    for p in params:
        p.grad = torch.ones_like(p)
    nn.utils.clip_grad_norm_(params, 0.1)
    assert all(p.grad.abs().max() < 1 for p in objective.parameters())
    objective.to("meta")
    assert all(p.device.type == "meta" for p in objective.parameters())


def test_checkpoint_roundtrip_keeps_model_clean_and_resume_exact(tmp_path):
    torch.manual_seed(11)
    model = _model()
    config = _config()
    objective = TrainingObjective(model, config)
    optimizer = _build_optimizer(model, config, objective)
    scheduler = _build_scheduler(optimizer, 2, 10, 0.05)
    # Populate synthetic Adam state without taking any optimizer steps.
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer.state[p] = {"step": torch.tensor(2.0),
                                  "exp_avg": torch.full_like(p, 0.125),
                                  "exp_avg_sq": torch.full_like(p, 0.25)}
    payload = {"model": model.state_dict(), "config": config,
               "training_objective": objective.checkpoint_state(),
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "rng": torch.get_rng_state(), "step": 2, "epoch": 1, "batch_cursor": 2}
    expected_random = torch.rand(3)
    path = tmp_path / "checkpoint.pt"
    _atomic_torch_save(payload, path)
    saved = torch.load(path, weights_only=False)
    restored = _model()
    restored.load_state_dict(saved["model"], strict=True)
    assert not any("aux" in key for key in saved["model"])
    restored_objective = TrainingObjective(restored, config)
    restored_objective.restore_checkpoint(saved)
    restored_optimizer = _build_optimizer(restored, config, restored_objective)
    restored_scheduler = _build_scheduler(restored_optimizer, 2, 10, 0.05)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_scheduler.load_state_dict(saved["scheduler"])
    _move_optimizer_state(restored_optimizer, "cpu")
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for p in restored_objective.parameters():
        assert torch.equal(restored_optimizer.state[p]["exp_avg"], torch.full_like(p, 0.125))
    torch.set_rng_state(saved["rng"])
    assert torch.equal(torch.rand(3), expected_random)
    args = _inputs()
    expected = objective(model, *args)
    actual = restored_objective(restored, *args)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert (saved["step"], saved["epoch"], saved["batch_cursor"]) == (2, 1, 2)


@pytest.mark.parametrize("new_weight", [0.0, 0.5])
def test_resume_rejects_recipe_change(new_weight):
    model = _model()
    original = TrainingObjective(model, _config())
    payload = {"config": _config(), "training_objective": original.checkpoint_state()}
    with pytest.raises(ValueError, match="recipe changed"):
        TrainingObjective(model, _config(new_weight)).restore_checkpoint(payload)


def test_legacy_checkpoint_resume_and_new_run_initialization(tmp_path):
    source = _model()
    legacy = {"model": source.state_dict(), "config": {}}
    TrainingObjective(source, {}).restore_checkpoint(legacy)
    with pytest.raises(ValueError, match="recipe changed"):
        TrainingObjective(source, _config()).restore_checkpoint(legacy)
    path = tmp_path / "old.pt"
    _atomic_torch_save(legacy, path)
    target = _model()
    _apply_model_init(target, {"model": {"init_checkpoint": str(path)}})
    objective = TrainingObjective(target, _config())
    assert objective.aux_head is not None
    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value)


def test_resume_rejects_missing_auxiliary_state():
    model = _model()
    objective = TrainingObjective(model, _config())
    with pytest.raises(ValueError, match="missing"):
        objective.restore_checkpoint({"config": _config()})
    payload = {"config": _config(), "training_objective": copy.deepcopy(objective.checkpoint_state())}
    payload["training_objective"]["state_dict"].pop("aux_head.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        objective.restore_checkpoint(payload)


def test_disabled_objective_checkpoint_roundtrip():
    model = _model()
    objective = TrainingObjective(model, {})
    payload = {"config": {}, "training_objective": objective.checkpoint_state()}
    restored = TrainingObjective(model, {})
    restored.restore_checkpoint(payload)
    assert restored.state_dict() == {}


def test_memory_telemetry_units_and_unavailable_resource(monkeypatch):
    module = importlib.import_module("stt.train")
    monkeypatch.setattr(module, "resource", None)
    assert module._memory_mib("cpu") is None

    class Resource:
        RUSAGE_SELF = 0

        @staticmethod
        def getrusage(_who):
            return type("Usage", (), {"ru_maxrss": 2048})()

    monkeypatch.setattr(module, "resource", Resource)
    monkeypatch.setattr(module.sys, "platform", "linux")
    assert module._memory_mib("cpu") == 2.0
    monkeypatch.setattr(module.sys, "platform", "darwin")
    assert module._memory_mib("cpu") == 2048 / 2**20


def test_no_training_import_without_resource_module():
    result = subprocess.run(
        [sys.executable, "-c", ("import sys; sys.modules['resource'] = None; "
         "import stt.train; assert stt.train.resource is None")],
        check=False, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def _preflight_config(tmp_path):
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"synthetic-tokenizer-identity-only")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"x"}\n', encoding="utf-8")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"x": {"shard": "shard.npy"}}), encoding="utf-8")
    (tmp_path / "shard.npy").write_bytes(b"synthetic-inventory-only")
    return {
        **_config(), "run": "NO-TRAIN", "model": {"vocab_size": 5, "head_blocks": 2},
        "tokenizer": str(tokenizer), "schedule": {"warmup_steps": 2},
        "data": {"train": [{"manifest": str(manifest), "shards": str(tmp_path)}]},
    }


def test_check_config_cli_never_constructs_training_objects(tmp_path, monkeypatch, capsys):
    module = importlib.import_module("stt.train")
    config = _preflight_config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        pytest.fail("preflight must not construct or invoke training")

    for name in ("train", "MothSTT", "TrainingObjective", "_build_optimizer", "load_sources",
                 "load_tokenizer"):
        monkeypatch.setattr(module, name, forbidden)
    assert module.main(["--config", str(path), "--check-config"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert str(tmp_path / "shard.npy") in report["files"]
    assert report["tokenizer_sha256"] == _sha256_file(config["tokenizer"])
    assert report["training_recipe_fingerprint"] == training_recipe_fingerprint(config)
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("section,key,value", [
    ("train", "input", "crop"), ("train", "autocast", "fp16"),
    ("train", "augment", "unknown"), ("model", "vocab_size", 0),
    ("model", "head_blocks", -1), ("train", "epochs", True),
    ("optimizer", "lr_head", float("nan")), ("schedule", "warmup_steps", -1),
    ("batch", "max_seconds", 0), ("batch", "max_clips", 0),
    ("schedule", "min_factor", float("nan")), ("optimizer", "betas", [0.9, 1.0]),
    ("optimizer", "eps", 0),
])
def test_preflight_rejects_invalid_key_fields(tmp_path, section, key, value):
    config = _preflight_config(tmp_path)
    config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError, match=key):
        check_training_config(config)


@pytest.mark.parametrize("lr_encoder", [None, 0, 1e-4])
def test_frozen_encoder_preflight_and_head_only_optimizer(tmp_path, lr_encoder):
    config = _preflight_config(tmp_path)
    config["model"]["freeze_encoder"] = True
    if lr_encoder is None:
        config["optimizer"].pop("lr_encoder")
    else:
        config["optimizer"]["lr_encoder"] = lr_encoder
    check_training_config(config)
    model = _model()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = _build_optimizer(model, config)
    assert [group["name"] for group in optimizer.param_groups] == ["head"]
    assert not optimizer.state


def test_trainable_encoder_preflight_and_optimizer(tmp_path):
    config = _preflight_config(tmp_path)
    check_training_config(config)
    optimizer = _build_optimizer(_model(), config)
    assert [group["name"] for group in optimizer.param_groups] == ["head", "encoder"]
    assert optimizer.param_groups[1]["lr"] == config["optimizer"]["lr_encoder"]
    assert not optimizer.state


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("lr_encoder", [-1, float("nan"), float("inf"), True, "bad", None])
def test_preflight_rejects_invalid_encoder_lr(tmp_path, frozen, lr_encoder):
    config = _preflight_config(tmp_path)
    config["model"]["freeze_encoder"] = frozen
    config["optimizer"]["lr_encoder"] = lr_encoder
    with pytest.raises(ValueError, match="optimizer.lr_encoder"):
        check_training_config(config)


@pytest.mark.parametrize("missing", [False, True])
def test_trainable_encoder_requires_positive_lr(tmp_path, missing):
    config = _preflight_config(tmp_path)
    if missing:
        config["optimizer"].pop("lr_encoder")
    else:
        config["optimizer"]["lr_encoder"] = 0
    with pytest.raises(ValueError, match="optimizer.lr_encoder.*positive"):
        check_training_config(config)


def test_preflight_reports_all_missing_placeholders(tmp_path):
    config = _preflight_config(tmp_path)
    config["tokenizer"] = str(tmp_path / "TOKENIZER_PLACEHOLDER")
    config["model"]["init_checkpoint"] = str(tmp_path / "CHECKPOINT_PLACEHOLDER")
    config["data"]["dev"] = [{"manifest": str(tmp_path / "MANIFEST_PLACEHOLDER"),
                              "shards": str(tmp_path / "SHARDS_PLACEHOLDER")}]
    (tmp_path / "shard.npy").unlink()
    with pytest.raises(ValueError, match="missing training files") as caught:
        check_training_config(config)
    for expected in ("TOKENIZER_PLACEHOLDER", "CHECKPOINT_PLACEHOLDER", "MANIFEST_PLACEHOLDER",
                     "SHARDS_PLACEHOLDER", "shard.npy"):
        assert expected in str(caught.value)


def test_check_config_cli_missing_files_returns_failure(tmp_path, capsys):
    module = importlib.import_module("stt.train")
    config = _preflight_config(tmp_path)
    config["tokenizer"] = str(tmp_path / "missing-tokenizer")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert module.main(["--config", str(path), "--check-config"]) == 2
    assert "missing-tokenizer" in capsys.readouterr().err


def test_check_config_cli_missing_yaml_returns_failure(tmp_path, capsys):
    module = importlib.import_module("stt.train")
    assert module.main(["--config", str(tmp_path / "absent.yaml"), "--check-config"]) == 2
    assert "absent.yaml" in capsys.readouterr().err


def test_resume_identity_checkpoint_payload_roundtrip(tmp_path):
    config = _preflight_config(tmp_path)
    sha = _sha256_file(config["tokenizer"])
    payload = {"config": config, "tokenizer_sha256": sha,
               "training_recipe_fingerprint": training_recipe_fingerprint(config, sha)}
    checkpoint = tmp_path / "identity.pt"
    _atomic_torch_save(payload, checkpoint)
    restored = torch.load(checkpoint, weights_only=False)
    _validate_resume_recipe(restored, config, sha)
    assert restored["tokenizer_sha256"] == sha


def test_windows_safe_cli_help_without_resource():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; sys.modules['resource'] = None; "
            "from stt.train import main; main(['--help'])"
        )], check=False, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--check-config" in result.stdout


@pytest.mark.parametrize("section,key,value", [
    ("model", "head_blocks", 3), ("model", "freeze_encoder", True),
    ("train", "augment", "light"), ("train", "epochs", 2),
    ("train", "interctc", {"weight": 0.5}), ("train", "input", "padded30"),
    ("optimizer", "lr_head", 0.002), ("schedule", "warmup_steps", 4),
    ("batch", "max_seconds", 20), ("batch", "max_clips", 2),
])
def test_resume_fingerprint_rejects_changed_trajectory(tmp_path, section, key, value):
    config = _preflight_config(tmp_path)
    sha = _sha256_file(config["tokenizer"])
    payload = {"config": copy.deepcopy(config), "tokenizer_sha256": sha,
               "training_recipe_fingerprint": training_recipe_fingerprint(config, sha)}
    changed = copy.deepcopy(config)
    changed.setdefault(section, {})[key] = value
    with pytest.raises(ValueError, match="recipe changed.*init_checkpoint"):
        _validate_resume_recipe(payload, changed, sha)


@pytest.mark.parametrize("filename", ["manifest.jsonl", "index.json", "tokenizer.model"])
def test_resume_rejects_content_change_at_same_path(tmp_path, filename):
    config = _preflight_config(tmp_path)
    sha = _sha256_file(config["tokenizer"])
    payload = {"config": config, "tokenizer_sha256": sha,
               "training_recipe_fingerprint": training_recipe_fingerprint(config, sha)}
    path = tmp_path / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed on resume"):
        _validate_resume_recipe(payload, config, _sha256_file(config["tokenizer"]))


def test_resume_legacy_declared_tokenizer_identity_and_missing_file(tmp_path):
    config = _preflight_config(tmp_path)
    sha = _sha256_file(config["tokenizer"])
    legacy = {"config": copy.deepcopy(config)}
    _validate_resume_recipe(legacy, config, sha)
    alternate = tmp_path / "other.model"
    alternate.write_bytes(b"different-tokenizer")
    changed = {**config, "tokenizer": str(alternate)}
    with pytest.raises(ValueError, match="tokenizer content changed"):
        _validate_resume_recipe(legacy, changed, _sha256_file(alternate))
    (tmp_path / "tokenizer.model").unlink()
    with pytest.raises(ValueError, match="legacy tokenizer file missing"):
        _validate_resume_recipe(legacy, changed, _sha256_file(alternate))


def test_resume_fingerprint_allows_logging_changes_and_identical_tokenizer_copy(tmp_path):
    config = _preflight_config(tmp_path)
    sha = _sha256_file(config["tokenizer"])
    payload = {"config": config, "tokenizer_sha256": sha,
               "training_recipe_fingerprint": training_recipe_fingerprint(config, sha)}
    copied = tmp_path / "identical.model"
    copied.write_bytes((tmp_path / "tokenizer.model").read_bytes())
    changed = copy.deepcopy(config)
    changed["tokenizer"] = str(copied)
    changed["train"].update(log_every=4, checkpoint_minutes=1, keep_last=1)
    _validate_resume_recipe(payload, changed, _sha256_file(copied))


def test_resume_fingerprint_pins_augmentation_version_and_frame_limit(tmp_path, monkeypatch):
    module = importlib.import_module("stt.train")
    config = _preflight_config(tmp_path)
    original = training_recipe_fingerprint(config)
    monkeypatch.setattr(module, "MAX_MEL_FRAMES", 2999)
    assert training_recipe_fingerprint(config) != original


def _distill_config(path, weight=0.5):
    return {
        "train": {"distill": {"weight": weight, "temperature": 2.0,
                              "checkpoint": str(path)}},
        "optimizer": {"lr_head": 1e-3, "lr_encoder": 1e-4},
    }


def _teacher_checkpoint(tmp_path, model):
    path = tmp_path / "teacher.pt"
    torch.save({"model": copy.deepcopy(model.state_dict())}, path)
    return path


def test_distill_recipe_disabled_and_validation(tmp_path):
    disabled = {"weight": 0.0, "temperature": 1.0, "checkpoint": None,
                "checkpoint_sha256": None}
    assert distill_recipe({}) == disabled
    assert distill_recipe({"train": {"distill": {}}}) == disabled
    assert distill_recipe({"train": {"distill": {"weight": 0.0}}}) == disabled
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    recipe = distill_recipe({"train": {"distill": {
        "weight": 0.5, "temperature": 2.0, "checkpoint": str(teacher)}}})
    assert recipe["weight"] == 0.5 and recipe["temperature"] == 2.0
    assert recipe["checkpoint_sha256"] == _sha256_file(teacher)
    for settings, match in (
        ({"weight": -1, "checkpoint": str(teacher)}, "weight"),
        ({"weight": True, "checkpoint": str(teacher)}, "weight"),
        ({"weight": 0.5, "temperature": 0, "checkpoint": str(teacher)}, "temperature"),
        ({"weight": 0.5}, "checkpoint"),
        ({"weight": 0.5, "checkpoint": str(tmp_path / "missing")}, "missing"),
        ({"weight": 0.5, "checkpoint": str(teacher), "extra": 1}, "distill"),
    ):
        with pytest.raises(ValueError, match=match):
            distill_recipe({"train": {"distill": settings}})


def test_distill_loss_zero_identical_and_masked():
    torch.manual_seed(0)
    log_probs = torch.log_softmax(torch.randn(2, 6, 7), dim=-1)
    lengths = torch.tensor([6, 3])
    assert float(_distill_loss(log_probs, log_probs, lengths, 1.0)) == pytest.approx(
        0.0, abs=1e-7
    )
    teacher = torch.log_softmax(torch.randn(2, 6, 7), dim=-1)
    value = float(_distill_loss(log_probs, teacher, lengths, 1.0))
    assert value > 0
    padded = teacher.clone()
    padded[1, 3:] = torch.log_softmax(torch.randn(3, 7), dim=-1)
    assert float(_distill_loss(log_probs, padded, lengths, 1.0)) == pytest.approx(value)
    assert float(_distill_loss(log_probs, teacher, lengths, 2.0)) > 0


def test_distillation_objective_teacher_is_frozen_and_not_registered(tmp_path):
    torch.manual_seed(0)
    model = _model()
    path = _teacher_checkpoint(tmp_path, model)
    objective = DistillationObjective(model, _distill_config(path))
    assert not list(objective.parameters())
    assert not any(key.startswith("teacher") for key in objective.state_dict())
    assert all(not parameter.requires_grad for parameter in objective.teacher.parameters())
    loss, main, aux = objective(model, *_inputs())
    assert aux is None
    assert objective.last_distill_loss is not None
    assert objective.last_distill_loss == pytest.approx(0.0, abs=1e-6)
    assert float(loss.detach()) == pytest.approx(float(main.detach()), abs=1e-6)
    with torch.no_grad():
        objective.teacher.head.weight.mul_(2.0)
    loss2, main2, _ = objective(model, *_inputs())
    assert objective.last_distill_loss > 0
    assert float(loss2.detach()) > float(main2.detach())
    loss2.backward()
    assert model.head.weight.grad is not None
    assert model.head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in objective.teacher.parameters())


def test_freeze_parameters_modes():
    model = _model()
    report = _freeze_parameters(model, {"model": {}})
    assert report["mode"] == "none"
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())

    model = _model()
    report = _freeze_parameters(model, {"model": {"freeze_encoder": True}})
    assert report["mode"] == "encoder"
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head_blocks.parameters())

    model = MothSTT(5, EncoderConfig(width=16, n_layers=2, n_heads=2, ffn_dim=32),
                    head_blocks=1)
    report = _freeze_parameters(model, {"model": {"trainable_encoder_blocks": 1}})
    assert report["mode"] == "last_1_encoder_blocks"
    assert not model.encoder.blocks[0].query.weight.requires_grad
    assert model.encoder.blocks[1].query.weight.requires_grad
    assert not model.encoder.conv1.weight.requires_grad
    assert not model.encoder.final_ln.weight.requires_grad
    optimizer = _build_optimizer(
        model, {"optimizer": {"lr_head": 1e-3, "lr_encoder": 1e-4}}
    )
    assert [group["name"] for group in optimizer.param_groups] == ["head", "encoder"]
    assert {id(parameter) for parameter in model.encoder.blocks[1].parameters()} == {
        id(parameter) for parameter in optimizer.param_groups[1]["params"]
    }


def test_preflight_partial_encoder_freeze_and_conflict(tmp_path):
    config = _preflight_config(tmp_path)
    config["model"]["trainable_encoder_blocks"] = 2
    check_training_config(config)
    config["optimizer"]["lr_encoder"] = 0
    with pytest.raises(ValueError, match="optimizer.lr_encoder.*positive"):
        check_training_config(config)
    config["optimizer"]["lr_encoder"] = 1e-5
    config["model"]["freeze_encoder"] = True
    with pytest.raises(ValueError, match="conflicts"):
        check_training_config(config)
    for value in (-1, 5, True):
        bad = _preflight_config(tmp_path)
        bad["model"]["trainable_encoder_blocks"] = value
        with pytest.raises(ValueError, match="trainable_encoder_blocks"):
            check_training_config(bad)


def test_resume_fingerprint_pins_distill_recipe_and_teacher(tmp_path):
    config = _preflight_config(tmp_path)
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    config["train"]["distill"] = {
        "weight": 0.5, "temperature": 2.0, "checkpoint": str(teacher)
    }
    sha = _sha256_file(config["tokenizer"])
    payload = {"config": copy.deepcopy(config), "tokenizer_sha256": sha,
               "training_recipe_fingerprint": training_recipe_fingerprint(config, sha)}
    _validate_resume_recipe(payload, config, sha)
    changed = copy.deepcopy(config)
    changed["train"]["distill"]["weight"] = 0.25
    with pytest.raises(ValueError, match="recipe changed"):
        _validate_resume_recipe(payload, changed, sha)
    teacher.write_bytes(b"different")
    with pytest.raises(ValueError, match="recipe changed"):
        _validate_resume_recipe(payload, config, sha)
