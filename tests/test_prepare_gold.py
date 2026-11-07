"""Tests for the E-017 gold oversampling builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "runs" / "E-017" / "prepare_gold.py"


def _write_source(root: Path, name: str, ids: list[str], durations: list[float]) -> None:
    shards = root / "data" / "shards" / name
    shards.mkdir(parents=True)
    (shards / "shard-000.npy").write_bytes(b"stub")
    index = {
        uid: {"shard": "shard-000.npy", "offset": i * 100, "frames": 100, "duration": d}
        for i, (uid, d) in enumerate(zip(ids, durations, strict=True))
    }
    (shards / "index.json").write_text(json.dumps(index), encoding="utf-8")
    manifests = root / "recordings" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    rows = [{"id": uid, "duration": d, "audio": f"{name}/{uid}.wav"} for uid, d in zip(ids, durations, strict=True)]
    (manifests / f"{name}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_two_gold_sets_are_repeated_with_prefixed_shards(tmp_path: Path) -> None:
    _write_source(tmp_path, "R-S1", ["R-S1-0001", "R-S1-0002"], [2.0, 3.0])
    _write_source(tmp_path, "R-S2", ["R-S2-0001"], [4.0])
    result = _run(tmp_path, "--sets", "R-S1,R-S2", "--factor", "3", "--out-dir", "out")
    assert result.returncode == 0, result.stderr

    out = tmp_path / "out"
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert len(index) == 9  # (2 + 1) clips x 3 copies
    # Each set keeps its own shard namespace so equal file names cannot collide.
    assert {entry["shard"] for entry in index.values()} == {"R-S1/shard-000.npy", "R-S2/shard-000.npy"}
    assert (out / "R-S1" / "shard-000.npy").exists()
    assert (out / "R-S2" / "shard-000.npy").exists()

    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 9
    assert {row["id"] for row in rows} == {
        f"{uid}#r{copy:02d}" for uid in ("R-S1-0001", "R-S1-0002", "R-S2-0001") for copy in (1, 2, 3)
    }
    assert sum(float(row["duration"]) for row in rows) == pytest.approx((2 + 3 + 4) * 3)


def test_missing_mel_cache_fails_loudly(tmp_path: Path) -> None:
    _write_source(tmp_path, "R-S1", ["R-S1-0001"], [2.0])
    result = _run(tmp_path, "--sets", "R-S1,R-S2", "--factor", "2", "--out-dir", "out")
    assert result.returncode != 0
    assert "R-S2" in (result.stderr + result.stdout)


def test_index_missing_a_manifest_id_fails_loudly(tmp_path: Path) -> None:
    _write_source(tmp_path, "R-S1", ["R-S1-0001"], [2.0])
    index_path = tmp_path / "data" / "shards" / "R-S1" / "index.json"
    index_path.write_text(json.dumps({}), encoding="utf-8")
    result = _run(tmp_path, "--sets", "R-S1", "--factor", "2", "--out-dir", "out")
    assert result.returncode != 0
    assert "missing from the mel index" in (result.stderr + result.stdout)