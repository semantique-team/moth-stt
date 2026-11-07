"""Tests for the Hugging Face upload guard (M8).

The upload script must refuse to publish anything that is not the verified release: a stale
``release/`` folder, a hand-edited file, or a missing model card. These tests exercise that gate
without touching the network.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "runs" / "E-021" / "upload_hf.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path]:
    """A tiny release/ + record/card pair shaped like the real one."""
    release = root / "release"
    release.mkdir()
    payload = {
        "model.int8.onnx": b"model-bytes",
        "vocab.json": b"{}",
        "mel_filters.bin": b"mf",
        "silero_vad.onnx": b"vad",
        "lm_bpe3.bin": b"lm",
    }
    files = {}
    for name, blob in payload.items():
        (release / name).write_bytes(blob)
        files[name] = {"bytes": len(blob), "sha256": _sha(release / name)}
    for name in ("config.json", "NOTICE"):
        (release / name).write_text("x", encoding="utf-8")
    record = root / "record.json"
    record.write_text(
        json.dumps(
            {
                "release": "test",
                "package_bytes": sum(entry["bytes"] for entry in files.values()),
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    card = root / "CARD.md"
    card.write_text("# card\n", encoding="utf-8")
    return record, card


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _args(root: Path, record: Path, card: Path) -> list[str]:
    return [
        "--repo",
        "someone/moth-test",
        "--release",
        str(root / "release"),
        "--card",
        str(card),
        "--record",
        str(record),
    ]


def test_dry_run_accepts_a_matching_release(tmp_path: Path) -> None:
    record, card = _fixture(tmp_path)
    result = _run(tmp_path, *_args(tmp_path, record, card))
    assert result.returncode == 0, result.stderr
    assert "verified release" in result.stdout
    assert "nothing uploaded" in result.stdout


def test_modified_file_is_refused(tmp_path: Path) -> None:
    record, card = _fixture(tmp_path)
    (tmp_path / "release" / "model.int8.onnx").write_bytes(b"tampered")
    result = _run(tmp_path, *_args(tmp_path, record, card))
    assert result.returncode == 2
    assert "hash mismatch" in result.stderr


def test_missing_file_is_refused(tmp_path: Path) -> None:
    record, card = _fixture(tmp_path)
    (tmp_path / "release" / "vocab.json").unlink()
    result = _run(tmp_path, *_args(tmp_path, record, card))
    assert result.returncode == 2
    assert "missing" in result.stderr


def test_wrong_payload_total_is_refused(tmp_path: Path) -> None:
    record, card = _fixture(tmp_path)
    data = json.loads(record.read_text(encoding="utf-8"))
    data["package_bytes"] += 1
    record.write_text(json.dumps(data), encoding="utf-8")
    result = _run(tmp_path, *_args(tmp_path, record, card))
    assert result.returncode == 2
    assert "payload size" in result.stderr


def test_missing_card_is_refused(tmp_path: Path) -> None:
    record, card = _fixture(tmp_path)
    card.unlink()
    result = _run(tmp_path, *_args(tmp_path, record, card))
    assert result.returncode == 2
    assert "model card" in result.stderr


def test_push_without_token_fails_before_the_network(tmp_path: Path) -> None:
    record, card = _fixture(tmp_path)
    result = _run(tmp_path, *_args(tmp_path, record, card), "--push")
    assert result.returncode == 2
    assert "HF_TOKEN" in result.stderr


@pytest.mark.skipif(not (REPO / "release").is_dir(), reason="release/ not built on this machine")
def test_the_real_release_passes_the_guard() -> None:
    record = REPO / "runs" / "E-021" / "release-0.5.1.json"
    if not record.is_file():
        pytest.skip("no recorded release")
    result = _run(REPO, "--repo", "someone/moth-test")
    assert result.returncode == 0, result.stderr
    assert "verified release 0.5.1" in result.stdout
