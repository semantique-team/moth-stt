"""Tests for the E-019 Whistle LoRA JSONL builder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "runs" / "E-019" / "build_jsonl.py"


def _row(uid: str, speaker: str, source: str = "recorded_r") -> dict:
    return {
        "id": uid,
        "audio": f"{speaker}/{uid}.wav",
        "text": "la mia squadra ha perso il derby",
        "norm": "la mia squadra ha perso il derby",
        "duration": 2.5,
        "source": source,
        "split": "train",
        "speaker": speaker,
        "lang": "mix",
        "category": "single_word",
        "en_spans": [[1, 2]],
    }


def _write_set(root: Path, name: str, rows: list[dict]) -> None:
    manifests = root / "recordings" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{name}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    for row in rows:
        audio = root / "recordings" / "audio" / name / row["audio"]
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"RIFF")


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _two_speakers(root: Path, per_speaker: int = 20) -> None:
    _write_set(root, "R-S1", [_row(f"R-S1-{i:04d}", "S1") for i in range(1, per_speaker + 1)])
    _write_set(root, "R-S2", [_row(f"R-S2-{i:04d}", "S2") for i in range(1, per_speaker + 1)])


def test_split_is_seeded_stratified_and_canonical(tmp_path: Path) -> None:
    _two_speakers(tmp_path)
    result = _run(tmp_path, "--out-dir", "out")
    assert result.returncode == 0, result.stderr

    report = json.loads((tmp_path / "out" / "split.json").read_text(encoding="utf-8"))
    assert report["clips"] == {"train": 36, "val": 2, "test": 2}
    # Both speakers appear in all three splits.
    assert report["per_speaker"]["train"] == {"S1": 18, "S2": 18}
    assert report["per_speaker"]["val"] == {"S1": 1, "S2": 1}
    assert report["per_speaker"]["test"] == {"S1": 1, "S2": 1}

    splits = {name: set(report["split"][name]) for name in ("train", "val", "test")}
    assert not splits["train"] & splits["val"]
    assert not splits["train"] & splits["test"]
    assert not splits["val"] & splits["test"]
    assert sum(len(ids) for ids in splits.values()) == 40

    line = json.loads(
        (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert set(line) == {"audio", "text", "language"}
    assert line["language"] == "it"
    # Paths are relative to the JSONL, so the tree stays portable.
    assert not Path(line["audio"]).is_absolute()
    assert (tmp_path / "out" / line["audio"]).resolve().exists()


def test_rerun_is_bit_identical(tmp_path: Path) -> None:
    _two_speakers(tmp_path)
    assert _run(tmp_path, "--out-dir", "out").returncode == 0
    first = {
        name: (tmp_path / "out" / name).read_bytes()
        for name in ("train.jsonl", "val.jsonl", "test.jsonl")
    }
    assert _run(tmp_path, "--out-dir", "out").returncode == 0
    for name, blob in first.items():
        assert (tmp_path / "out" / name).read_bytes() == blob


def test_language_override_is_written_on_every_line(tmp_path: Path) -> None:
    _two_speakers(tmp_path)
    result = _run(tmp_path, "--out-dir", "out", "--language", "mix")
    assert result.returncode == 0, result.stderr
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        for line in (tmp_path / "out" / name).read_text(encoding="utf-8").splitlines():
            assert json.loads(line)["language"] == "mix"


def test_missing_audio_fails_loudly(tmp_path: Path) -> None:
    _two_speakers(tmp_path)
    (tmp_path / "recordings" / "audio" / "R-S2" / "S2" / "R-S2-0007.wav").unlink()
    result = _run(tmp_path, "--out-dir", "out")
    assert result.returncode != 0
    assert "R-S2-0007" in (result.stderr + result.stdout)


def test_non_gold_source_is_rejected(tmp_path: Path) -> None:
    _write_set(tmp_path, "R-S1", [_row("R-S1-0001", "S1", source="recorded_t")])
    _write_set(tmp_path, "R-S2", [_row("R-S2-0001", "S2")])
    result = _run(tmp_path, "--out-dir", "out")
    assert result.returncode != 0
    assert "not a gold clip" in (result.stderr + result.stdout)
