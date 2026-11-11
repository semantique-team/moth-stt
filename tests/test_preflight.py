"""Local-only preflight contracts; no torch installation or training required."""

import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stt import preflight


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(preflight.platform, "python_version", lambda: "3.12.test")
    monkeypatch.setattr(preflight.platform, "system", lambda: "TestOS")
    monkeypatch.setattr(preflight.platform, "release", lambda: "test-release")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "test-machine")
    monkeypatch.setattr(preflight.sys, "version_info", (3, 12, 0))

    def version(name):
        if name in preflight.RUNTIME_REQUIRED:
            return "1.test"
        raise preflight.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(preflight.importlib.metadata, "version", version)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: SimpleNamespace(
        total=10000, free=7000,
    ))
    return tmp_path


@pytest.fixture
def recipe(host, monkeypatch):
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda name: "1.test")
    config = {
        "model": {"vocab_size": 10, "init_checkpoint": "weights.pt"},
        "optimizer": {"lr_head": 0.001, "lr_encoder": 0.001},
        "schedule": {}, "tokenizer": "tokenizer.model",
        "data": {"train": [{"manifest": "train.jsonl", "shards": "shards"}]},
        "evaluation": {"decoder": {"mode": "not-valid-for-evaluation"}},
        "private_api_key": "CONFIG_SECRET_VALUE",
    }
    (host / "run.yaml").write_text(json.dumps(config), encoding="utf-8")
    (host / "tokenizer.model").write_bytes(b"tokenizer")
    (host / "weights.pt").write_bytes(b"not a torch checkpoint")
    (host / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (host / "shards").mkdir()
    (host / "shards" / "index.json").write_text(json.dumps({
        "clip1": {"shard": "part.npy"}, "clip2": {"shard": "part.npy"},
    }), encoding="utf-8")
    (host / "shards" / "part.npy").write_bytes(b"not a mel array")
    calls = []

    def check(raw, *, check_files):
        calls.append((raw, check_files))
        assert check_files is False
        return {"files": []}

    def forbidden(*args, **kwargs):
        pytest.fail("training, model construction, optimizer, load or forward invoked")

    monkeypatch.setitem(sys.modules, "yaml", SimpleNamespace(safe_load=json.loads))
    monkeypatch.setitem(sys.modules, "stt.train", SimpleNamespace(
        check_training_config=check, train=forbidden, main=forbidden, MothSTT=forbidden,
    ))
    monkeypatch.setitem(sys.modules, "stt.training_eval", SimpleNamespace(check_config=forbidden))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        load=forbidden, optim=SimpleNamespace(AdamW=forbidden),
    ))
    return config, calls


def test_environment_only_without_torch_or_optional_imports(host, monkeypatch):
    def forbidden(name):
        pytest.fail(f"unexpected optional import: {name}")

    monkeypatch.setattr(preflight.importlib, "import_module", forbidden)
    monkeypatch.setenv("HF_TOKEN", "ENV_SECRET_VALUE")
    report = preflight.build_report()
    assert report["ok"]
    assert report["dependencies"] == {
        "required": list(preflight.RUNTIME_REQUIRED), "missing": [],
        "native_runtime_load": "not_verified",
    }
    assert report["environment"]["python"] == "3.12.test"
    assert report["environment"]["platform"] == {
        "system": "TestOS", "release": "test-release", "machine": "test-machine",
    }
    assert report["environment"]["installed_versions"]["numpy"] == "1.test"
    assert report["environment"]["installed_versions"]["torch"] is None
    assert report["environment"]["accelerators"]["cuda_available"] is None
    assert report["disk"] == [{"existing_parent": {"root": "cwd", "path": "."},
                                "total_bytes": 10000, "free_bytes": 7000}]
    assert "ENV_SECRET_VALUE" not in json.dumps(report)
    assert str(host) not in json.dumps(report)


def test_all_distributions_missing_fails_and_writes_json(host, monkeypatch):
    def missing(name):
        raise preflight.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(preflight.importlib.metadata, "version", missing)
    assert preflight.main(["--out", "empty.json"]) == 2
    report = json.loads((host / "empty.json").read_text(encoding="utf-8"))
    assert not report["ok"]
    assert report["dependencies"]["missing"] == list(preflight.RUNTIME_REQUIRED)
    assert report["checks"]["dependencies"] == "failed"


@pytest.mark.parametrize("name", preflight.RUNTIME_REQUIRED)
def test_each_runtime_dependency_required_for_environment_only(host, monkeypatch, name):
    def version(package):
        if package not in preflight.RUNTIME_REQUIRED or package == name:
            raise preflight.importlib.metadata.PackageNotFoundError(package)
        return "1.test"

    monkeypatch.setattr(preflight.importlib.metadata, "version", version)
    assert preflight.main(["--out", "missing-runtime.json"]) == 2
    report = json.loads((host / "missing-runtime.json").read_text(encoding="utf-8"))
    assert report["dependencies"]["missing"] == [name]


@pytest.mark.parametrize("name", preflight.TRAINING_REQUIRED)
def test_config_requires_explicit_training_distributions(host, recipe, monkeypatch, name):
    def version(package):
        if package == name:
            raise preflight.importlib.metadata.PackageNotFoundError(package)
        return "1.test"

    monkeypatch.setattr(preflight.importlib.metadata, "version", version)
    assert preflight.main(["--config", "run.yaml", "--out", "missing-training.json"]) == 2
    report = json.loads((host / "missing-training.json").read_text(encoding="utf-8"))
    assert report["checks"]["training_config"] == "passed"
    assert report["dependencies"]["missing"] == [name]
    assert "onnx" not in report["dependencies"]["required"]


def test_config_does_not_require_export_distributions(host, recipe, monkeypatch):
    def version(name):
        if name not in (*preflight.RUNTIME_REQUIRED, *preflight.TRAINING_REQUIRED):
            raise preflight.importlib.metadata.PackageNotFoundError(name)
        return "1.test"

    monkeypatch.setattr(preflight.importlib.metadata, "version", version)
    report = preflight.build_report("run.yaml")
    assert report["ok"]
    assert report["dependencies"]["missing"] == []
    assert report["environment"]["installed_versions"]["onnx"] is None


def test_accelerators_informational_and_lazy(host, monkeypatch):
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda name: "test")
    imported = []

    def load(name):
        imported.append(name)
        assert name == "torch"
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )

    monkeypatch.setattr(preflight.importlib, "import_module", load)
    report = preflight.build_report()
    assert imported == ["torch"]
    assert report["ok"]
    assert report["environment"]["accelerators"] == {
        "informational_only": True, "cuda_available": True, "mps_available": False,
    }


def test_training_inventory_no_execution_and_relative_json(host, recipe, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "ENV_SECRET_VALUE")
    assert preflight.main(["--config", "run.yaml", "--out", "new.json"]) == 0
    report = json.loads((host / "new.json").read_text(encoding="utf-8"))
    assert len(recipe[1]) == 1
    assert report["checks"]["training_config"] == "passed"
    assert report["checks"]["evaluation"] == "not_requested"
    files = {item["path"]: item for item in report["files"]}
    assert len(files) == 6
    assert files["tokenizer.model"]["sha256"] == hashlib.sha256(b"tokenizer").hexdigest()
    assert files["weights.pt"]["sha256"] == hashlib.sha256(b"not a torch checkpoint").hexdigest()
    assert files["shards/part.npy"]["sha256"] is None
    assert report["transfer"]["known_required_bytes"] == sum(
        item["bytes"] for item in files.values()
    )
    serialized = json.dumps(report)
    for private in (str(host), "CONFIG_SECRET_VALUE", "ENV_SECRET_VALUE", "private_api_key"):
        assert private not in serialized


def test_missing_paths_reported_even_after_early_validation_error(host, recipe, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError(f"private {host} CONFIG_SECRET_VALUE")

    monkeypatch.setattr(sys.modules["stt.train"], "check_training_config", reject)
    (host / "weights.pt").unlink()
    (host / "shards" / "part.npy").unlink()
    assert preflight.main(["--config", "run.yaml", "--out", "missing.json"]) == 2
    report = json.loads((host / "missing.json").read_text(encoding="utf-8"))
    assert report["checks"]["training_config"] == "failed"
    assert {item["path"] for item in report["files"] if not item["exists"]} == {
        "weights.pt", "shards/part.npy",
    }
    assert report["transfer"]["missing_file_count"] == 2
    assert not report["transfer"]["complete"]
    assert str(host) not in json.dumps(report)
    assert "CONFIG_SECRET_VALUE" not in json.dumps(report)


def test_evaluation_only_explicitly_requested(host, recipe, monkeypatch):
    called = []

    def check(raw):
        called.append(raw)
        return {"files_sha256": {str(host / "private"): "secret"}}

    monkeypatch.setattr(sys.modules["stt.training_eval"], "check_config", check)
    report = preflight.build_report("run.yaml", evaluation=True)
    assert called == [recipe[0]]
    assert report["checks"]["evaluation"] == "passed"
    assert "private" not in json.dumps(report)
    report = preflight.build_report(evaluation=True)
    assert not report["ok"]
    assert report["checks"]["evaluation"] == "failed"


def test_large_checkpoint_opt_in_streaming_no_shard_hash(host, recipe, monkeypatch):
    monkeypatch.setattr(preflight, "SMALL_FILE_BYTES", 1)
    report = preflight.build_report("run.yaml")
    checkpoint = next(item for item in report["files"] if "checkpoint" in item["roles"])
    assert checkpoint["sha256"] is None
    report = preflight.build_report("run.yaml", hash_checkpoints=True)
    checkpoint = next(item for item in report["files"] if "checkpoint" in item["roles"])
    assert checkpoint["sha256"] == hashlib.sha256(b"not a torch checkpoint").hexdigest()
    assert next(item for item in report["files"] if "mel_shard" in item["roles"])["sha256"] is None


@pytest.mark.parametrize("contents", ["null", "{invalid", "[]"])
def test_invalid_config_still_emits_json(host, recipe, contents):
    (host / "run.yaml").write_text(contents, encoding="utf-8")
    assert preflight.main(["--config", "run.yaml", "--out", "invalid.json"]) == 2
    report = json.loads((host / "invalid.json").read_text(encoding="utf-8"))
    assert report["issues"][0]["stage"] == "config"


def test_output_refuses_overwrite_collision_and_missing_parent(host, recipe):
    (host / "exists.json").write_bytes(b"keep")
    assert preflight.main(["--out", "exists.json"]) == 2
    assert (host / "exists.json").read_bytes() == b"keep"
    before = (host / "run.yaml").read_bytes()
    assert preflight.main(["--config", "run.yaml", "--out", "run.yaml"]) == 2
    assert (host / "run.yaml").read_bytes() == before
    assert preflight.main(["--out", "absent/new.json"]) == 2
    assert not (host / "absent").exists()


def test_missing_config_and_nearest_existing_disk_parent(host, recipe, monkeypatch):
    seen = []

    def disk(path):
        assert path.is_dir()
        seen.append(path)
        return SimpleNamespace(total=9000, free=1234)

    monkeypatch.setattr(preflight.shutil, "disk_usage", disk)
    report = preflight.build_report("not/here/run.yaml", out="also/missing/new.json")
    assert not report["ok"]
    assert seen == [host]
    assert report["disk"][0]["free_bytes"] == 1234


def test_missing_training_dependency_is_issue_not_crash(host, recipe, monkeypatch):
    original = preflight.importlib.import_module

    def load(name):
        if name == "stt.train":
            raise ModuleNotFoundError("torch missing")
        return original(name)

    monkeypatch.setattr(preflight.importlib, "import_module", load)
    report = preflight.build_report("run.yaml")
    assert report["checks"]["training_config"] == "failed"
    assert len(report["files"]) == 6


def test_external_paths_opaque_and_config_directory_relative(host, recipe):
    outside = host.parent / (host.name + "-private-config")
    outside.mkdir()
    config = outside / "run.yaml"
    config.write_text(json.dumps({}), encoding="utf-8")
    report = preflight.build_report(config)
    assert any(item["root"] == "config_directory" and item["path"] == "run.yaml"
               for item in report["files"])
    assert str(outside) not in json.dumps(report)
    config.unlink()
    outside.rmdir()


def test_missing_release_inventory_and_validation_errors(host):
    root = host / "release"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "files": {"extra.bin": "sha256:missing"},
        "model": {"file": "model.onnx"}, "tokenizer": {"vocab": "vocab.json"},
        "features": {"mel_filters": "filters.bin"}, "vad": {"file": "vad.onnx"},
        "decode": {"lm": "lm.bin"},
    }), encoding="utf-8")
    report = preflight.build_report(release=root)
    assert report["checks"]["release"] == "failed"
    assert {item["path"] for item in report["files"] if not item["exists"]} == {
        "release/NOTICE", "release/extra.bin", "release/model.onnx", "release/vocab.json",
        "release/filters.bin", "release/vad.onnx", "release/lm.bin",
    }
    assert report["transfer"]["dependency_distribution_bytes"] is None


def test_actual_training_validator_without_importing_training_dependencies(host, recipe, monkeypatch):
    # Execute the existing pure validators, not the torch-dependent module body.
    source = Path(preflight.__file__).with_name("train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"check_training_config", "interctc_recipe"}]
    namespace = {"Path": Path, "math": math}
    exec(compile(tree, "stt/train.py", "exec"), namespace)  # noqa: S102 - trusted repository source
    monkeypatch.setattr(sys.modules["stt.train"], "check_training_config",
                        namespace["check_training_config"])
    assert preflight.build_report("run.yaml")["checks"]["training_config"] == "passed"
    recipe[0]["optimizer"]["lr_head"] = -1
    (host / "run.yaml").write_text(json.dumps(recipe[0]), encoding="utf-8")
    report = preflight.build_report("run.yaml")
    assert report["checks"]["training_config"] == "failed"
    assert len(report["files"]) == 6


@pytest.mark.parametrize("missing_runtime", [None, *preflight.RUNTIME_REQUIRED])
def test_successful_real_release_validation_without_sessions(host, monkeypatch, missing_runtime):
    def version(name):
        if name not in preflight.RUNTIME_REQUIRED or name == missing_runtime:
            raise preflight.importlib.metadata.PackageNotFoundError(name)
        return "1.test"

    monkeypatch.setattr(preflight.importlib.metadata, "version", version)
    root = host / "release"
    root.mkdir()
    assets = {"model.onnx": b"model", "vocab.json": b"{}", "filters.bin": b"filters",
              "vad.onnx": b"vad", "lm.bin": b"MOTHLMB1lm"}
    for name, content in assets.items():
        (root / name).write_bytes(content)
    (root / "NOTICE").write_bytes(b"notice")
    config = {
        "name": "moth-stt", "version": "test", "sample_rate": 16000,
        "features": {"n_fft": 400, "hop": 160, "n_mels": 80, "window": "hann_periodic",
                     "center": True, "pad_mode": "reflect", "drop_last_frame": True,
                     "floor_range": 8.0, "scale": "(x+4)/4", "mel_filters": "filters.bin"},
        "model": {"file": "model.onnx", "input": "mel", "output": "log_probs",
                  "frame_rate_hz": 50, "max_seconds": 30},
        "tokenizer": {"vocab": "vocab.json", "blank_id": 0, "space_marker": "\u2581"},
        "decode": {"default": "beam", "beam": 12, "top_k": 10,
                   "lm_weight": 0.5, "token_bonus": 2.4, "lm": "lm.bin"},
        "vad": {"file": "vad.onnx", "window": 512, "context": 64, "threshold": 0.1},
        "files": {name: "sha256:" + hashlib.sha256(data).hexdigest()
                  for name, data in assets.items()},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    status = preflight.main(["--release", "release", "--out", "release-report.json"])
    report = json.loads((host / "release-report.json").read_text(encoding="utf-8"))
    assert status == (0 if missing_runtime is None else 2)
    assert report["ok"] == (missing_runtime is None)
    assert report["dependencies"]["missing"] == ([] if missing_runtime is None
                                                   else [missing_runtime])
    assert report["checks"]["release"] == "passed"
    assert report["release_budget"]["total_bytes"] == sum(
        path.stat().st_size for path in root.iterdir()
    )
    assert report["release_budget"]["budget_bytes"] == 20_000_000
    assert str(host) not in json.dumps(report)


def test_unsupported_python_fails_without_training(host, monkeypatch):
    monkeypatch.setattr(preflight.sys, "version_info", (3, 11, 0))
    report = preflight.build_report()
    assert not report["ok"]
    assert report["issues"][0]["stage"] == "environment"
