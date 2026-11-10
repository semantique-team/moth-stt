"""Synthetic release checks: no model sessions, training packages or audio required."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from stt.release import main, validate_release


@pytest.fixture
def release(tmp_path):
    assets = {
        "model.int8.onnx": b"synthetic acoustic model",
        "vocab.json": b'{"1": "test"}',
        "mel_filters.bin": b"synthetic filters",
        "silero_vad.onnx": b"synthetic vad",
        "lm_bpe3.bin": b"MOTHLMB1synthetic lm",
    }
    for name, content in assets.items():
        (tmp_path / name).write_bytes(content)
    (tmp_path / "NOTICE").write_text("Synthetic test attribution", encoding="utf-8")
    config = {
        "name": "moth-stt",
        "version": "0.7.0",
        "sample_rate": 16000,
        "features": {
            "n_fft": 400,
            "hop": 160,
            "n_mels": 80,
            "window": "hann_periodic",
            "center": True,
            "pad_mode": "reflect",
            "drop_last_frame": True,
            "floor_range": 8.0,
            "scale": "(x+4)/4",
            "mel_filters": "mel_filters.bin",
        },
        "model": {
            "file": "model.int8.onnx",
            "input": "mel",
            "output": "log_probs",
            "frame_rate_hz": 50,
            "max_seconds": 30,
        },
        "tokenizer": {"vocab": "vocab.json", "blank_id": 0, "space_marker": "\u2581"},
        "decode": {
            "default": "beam",
            "beam": 12,
            "top_k": 10,
            "lm": "lm_bpe3.bin",
            "lm_weight": 0.5,
            "token_bonus": 2.4,
        },
        "vad": {"file": "silero_vad.onnx", "window": 512, "context": 64, "threshold": 0.1},
        "files": {
            name: "sha256:" + hashlib.sha256(data).hexdigest() for name, data in assets.items()
        },
    }
    write_config(tmp_path, config)
    return tmp_path, config


def write_config(root, config):
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_valid_release_counts_all_seven_files(release):
    root, expected = release
    config, report = validate_release(root)
    assert config == expected
    assert report["total_bytes"] == sum(file.stat().st_size for file in root.iterdir())
    assert len(report["files"]) == 7
    assert report["remaining_bytes"] == 20_000_000 - report["total_bytes"]
    assert report["files"]["lm_bpe3.bin"]["hash_verified"]
    assert not report["files"]["config.json"]["hash_verified"]
    assert not report["files"]["NOTICE"]["hash_verified"]


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "NOTICE",
        "model.int8.onnx",
        "vocab.json",
        "mel_filters.bin",
        "silero_vad.onnx",
        "lm_bpe3.bin",
    ],
)
def test_missing_required_file(release, name):
    root, _ = release
    (root / name).unlink()
    with pytest.raises(ValueError, match="missing release file"):
        validate_release(root)


def test_tampering(release):
    root, _ = release
    (root / "model.int8.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_release(root)


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "sample_rate", 48000),
        (None, "sample_rate", True),
        (None, "version", ""),
        (None, "name", "other"),
        (None, "features", []),
        ("features", "center", 1),
        ("features", "n_mels", 81),
        ("model", "max_seconds", 31),
        ("model", "input", "other"),
        ("tokenizer", "blank_id", False),
        ("tokenizer", "space_marker", " "),
        ("decode", "beam", 0),
        ("decode", "beam", 1.5),
        ("decode", "top_k", True),
        ("decode", "lm_weight", -1),
        ("decode", "token_bonus", float("nan")),
        ("decode", "lm_weight", float("inf")),
        ("decode", "default", "unknown"),
        ("decode", "segment", 0),
        ("decode", "segment", 1),
        ("decode", "segment", "false"),
        ("decode", "segment", None),
        ("vad", "window", 513),
        ("vad", "threshold", 1.1),
    ],
)
def test_invalid_semantics(release, section, key, value):
    root, config = release
    (config if section is None else config[section])[key] = value
    write_config(root, config)
    with pytest.raises(ValueError):
        validate_release(root)


@pytest.mark.parametrize("segment", [False, True])
def test_boolean_segment_setting(release, segment):
    root, config = release
    config["decode"]["segment"] = segment
    write_config(root, config)
    assert validate_release(root)[0]["decode"]["segment"] is segment


@pytest.mark.parametrize(
    "name",
    [
        "../outside.bin",
        "/absolute.bin",
        "C:/outside.bin",
        "..\\outside.bin",
        "a/../lm.bin",
        "./lm.bin",
        "a//b.bin",
        "https://example.com/lm.bin",
        "bad\x00.bin",
    ],
)
def test_unsafe_reference(release, name):
    root, config = release
    config["decode"]["lm"] = name
    write_config(root, config)
    with pytest.raises(ValueError, match="unsafe release path"):
        validate_release(root)


def test_unsafe_manifest_entry(release):
    root, config = release
    config["files"]["../outside.bin"] = "sha256:" + "0" * 64
    write_config(root, config)
    with pytest.raises(ValueError, match="unsafe release path"):
        validate_release(root)


def test_symlink_escape(release, tmp_path):
    root, _ = release
    outside = tmp_path.parent / f"{tmp_path.name}-outside.onnx"
    outside.write_bytes(b"outside")
    model = root / "model.int8.onnx"
    model.unlink()
    try:
        model.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks require privileges on Windows")
    with pytest.raises(ValueError, match="escapes root"):
        validate_release(root)


@pytest.mark.parametrize("declared", [None, "0" * 64, "sha256:xyz", 123])
def test_invalid_hash(release, declared):
    root, config = release
    config["files"]["lm_bpe3.bin"] = declared
    write_config(root, config)
    with pytest.raises(ValueError, match="invalid sha256"):
        validate_release(root)


def test_missing_hash(release):
    root, config = release
    del config["files"]["lm_bpe3.bin"]
    write_config(root, config)
    with pytest.raises(ValueError, match="missing hash"):
        validate_release(root)


def test_rejects_self_hash(release):
    root, config = release
    config["files"]["config.json"] = "sha256:" + "0" * 64
    write_config(root, config)
    with pytest.raises(ValueError, match="own hash"):
        validate_release(root)


def test_rejects_shared_asset_path(release):
    root, config = release
    config["vad"]["file"] = config["model"]["file"]
    write_config(root, config)
    with pytest.raises(ValueError, match="distinct"):
        validate_release(root)


def test_rejects_wrong_asset_type(release):
    root, config = release
    config["decode"]["lm"] = "vocab.json"
    write_config(root, config)
    with pytest.raises(ValueError, match="must be a .bin"):
        validate_release(root)


def test_rejects_empty_notice(release):
    root, _ = release
    (root / "NOTICE").write_bytes(b"")
    with pytest.raises(ValueError, match="empty release file"):
        validate_release(root)


@pytest.mark.parametrize("name", ["lm.pkl", "lm.pickle.gz", "lm.PKL.gz"])
def test_no_pickle_reference(release, name):
    root, config = release
    config["decode"]["lm"] = name
    write_config(root, config)
    with pytest.raises(ValueError, match="pickle"):
        validate_release(root)


def test_disguised_pickle(release):
    root, config = release
    content = b"\x80\x04pickle is never loaded"
    (root / "lm_bpe3.bin").write_bytes(content)
    config["files"]["lm_bpe3.bin"] = "sha256:" + hashlib.sha256(content).hexdigest()
    write_config(root, config)
    with pytest.raises(ValueError, match="MOTHLMB1"):
        validate_release(root)


def test_additional_required_files_and_notice_hash(release):
    root, config = release
    (root / "extra.bin").write_bytes(b"extra required data")
    for name in ("extra.bin", "NOTICE"):
        config["files"][name] = "sha256:" + hashlib.sha256((root / name).read_bytes()).hexdigest()
    write_config(root, config)
    _, report = validate_release(root)
    assert len(report["files"]) == 8
    assert report["files"]["NOTICE"]["hash_verified"]
    assert report["total_bytes"] == sum(file.stat().st_size for file in root.iterdir())


def test_budget_boundary_includes_config_and_notice(release):
    root, _ = release
    _, report = validate_release(root)
    total = report["total_bytes"]
    validate_release(root, total)
    with pytest.raises(ValueError, match="exceeds budget"):
        validate_release(root, total - 1)


def test_extended_tier_is_explicit(release):
    root, _ = release
    with (root / "NOTICE").open("ab") as stream:
        stream.truncate(20_000_000)
    with pytest.raises(ValueError, match="exceeds budget"):
        validate_release(root)
    validate_release(root, 25_000_000)
    with (root / "NOTICE").open("ab") as stream:
        stream.truncate(25_000_000)
    with pytest.raises(ValueError, match="exceeds budget"):
        validate_release(root, 25_000_000)


@pytest.mark.parametrize("budget", [0, -1, True, 25_000_001, 20_000_000.0])
def test_invalid_budget(release, budget):
    with pytest.raises(ValueError, match="budget_bytes"):
        validate_release(release[0], budget)


@pytest.mark.parametrize("content", ["not json", "[]", "{}"])
def test_bad_config(release, content):
    root, _ = release
    (root / "config.json").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_release(root)


def test_cli_browser_raw_accounting(release, tmp_path, capsys):
    root, _ = release
    browser = tmp_path / "browser"
    browser.mkdir()
    (browser / "index.html").write_bytes(b"app")
    (browser / "ort-wasm.wasm").write_bytes(b"wasm-runtime")
    assert main([str(root), "--browser-root", str(browser)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"]
    assert report["browser"]["raw_total_bytes"] == 15
    assert report["browser"]["file_count"] == 2
    assert "not HTTP transfer" in report["browser"]["measurement"]


def test_cli_error_is_json(release, capsys):
    root, _ = release
    assert main([str(root), "--budget-bytes", "1"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "exceeds budget" in report["error"]


def test_cli_rejects_missing_browser_root(release, capsys):
    root, _ = release
    assert main([str(root), "--browser-root", str(root / "missing")]) == 1
    assert "not a directory" in json.loads(capsys.readouterr().out)["error"]


def test_module_cli_needs_only_stdlib(release):
    result = subprocess.run(
        [sys.executable, "-S", "-m", "stt.release", str(release[0])],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["ok"]
