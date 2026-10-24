import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stt.data.build_noise import SOURCES, build_noise_set
from stt.manifest import load_manifest


def _wav(path: Path, seconds: float, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    signal = 0.05 * rng.standard_normal(int(seconds * 16000)).astype(np.float32)
    sf.write(path, signal, 16000, subtype="PCM_16")
    return path


def _fake_musan(root: Path) -> Path:
    _wav(root / "noise" / "short.wav", 3.0, seed=1)
    _wav(root / "noise" / "long.wav", 12.0, seed=2)
    _wav(root / "music" / "track.wav", 12.0, seed=3)
    _wav(root / "music" / "tiny.wav", 2.0, seed=4)
    return root


def _build(tmp_path: Path, **overrides: object) -> tuple[Path, dict]:
    audio = tmp_path / "noise"
    manifest = tmp_path / "noise.jsonl"
    report = build_noise_set(
        _fake_musan(tmp_path / "musan"),
        audio,
        manifest,
        minutes_per_category=2 / 6,  # 2 clips of 10 s per category
        categories=("silence", "noise", "music"),
        **overrides,
    )
    return manifest, report


def test_builds_balanced_clips_and_valid_manifest(tmp_path: Path) -> None:
    manifest, report = _build(tmp_path)
    rows = load_manifest(manifest)

    assert len(rows) == 6
    assert report["total_clips"] == 6
    assert all(row["duration"] == 10.0 for row in rows)
    assert all((tmp_path / "noise" / row["audio"]).exists() for row in rows)
    assert sum(row["source"] == SOURCES["music"] for row in rows) == 2
    assert sum(row["source"] == SOURCES["silence"] for row in rows) == 2
    assert sum(row["source"] == SOURCES["noise"] for row in rows) == 2
    assert all(row["norm"] == "" and row["lang"] == "it" for row in rows)


def test_silence_clips_are_digital_zero(tmp_path: Path) -> None:
    manifest, report = _build(tmp_path)
    rows = [row for row in load_manifest(manifest) if row["source"] == SOURCES["silence"]]
    for row in rows:
        signal, rate = sf.read(tmp_path / "noise" / row["audio"], dtype="float32")
        assert rate == 16000
        assert signal.shape == (160000,)
        assert not np.any(signal)
    assert report["categories"]["silence"]["non_silent_clips"] == 0


def test_short_sources_are_concatenated_to_the_clip_length(tmp_path: Path) -> None:
    manifest, _ = _build(tmp_path)
    rows = [row for row in load_manifest(manifest) if row["source"] != SOURCES["silence"]]
    for row in rows:
        signal, _rate = sf.read(tmp_path / "noise" / row["audio"], dtype="float32")
        assert len(signal) == 160000
        assert np.any(signal)


def test_same_seed_same_audio(tmp_path: Path) -> None:
    first, _ = _build(tmp_path / "a")
    second, _ = _build(tmp_path / "b")
    first_rows = load_manifest(first)
    load_manifest(second)
    assert first.read_text() == second.read_text()
    for row in first_rows:
        a = (tmp_path / "a" / "noise" / row["audio"]).read_bytes()
        b = (tmp_path / "b" / "noise" / row["audio"]).read_bytes()
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_room_tone_is_cut_and_merged(tmp_path: Path) -> None:
    room = _wav(tmp_path / "room.wav", 4.0, seed=5)
    manifest = tmp_path / "noise.jsonl"
    report = build_noise_set(
        _fake_musan(tmp_path / "musan"),
        tmp_path / "noise",
        manifest,
        room_tone=room,
        minutes_per_category=1 / 6,
        categories=("silence", "room_tone"),
    )
    rows = load_manifest(manifest)
    assert len(rows) == 2
    assert {row["source"] for row in rows} == {SOURCES["silence"], SOURCES["room_tone"]}
    assert report["categories"]["room_tone"]["non_silent_clips"] == 1

    report = build_noise_set(
        _fake_musan(tmp_path / "musan"),
        tmp_path / "noise",
        manifest,
        minutes_per_category=1 / 6,
        categories=("noise",),
    )
    rows = load_manifest(manifest)
    assert len(rows) == 3
    assert {row["source"] for row in rows} == {
        SOURCES["silence"],
        SOURCES["room_tone"],
        SOURCES["noise"],
    }


def test_missing_music_subset_is_an_error(tmp_path: Path) -> None:
    musan = tmp_path / "musan"
    _wav(musan / "noise" / "a.wav", 12.0)

    with pytest.raises(FileNotFoundError, match="music"):
        build_noise_set(
            musan,
            tmp_path / "noise",
            tmp_path / "noise.jsonl",
            minutes_per_category=1 / 6,
            categories=("music",),
        )


def test_report_is_json_serializable(tmp_path: Path) -> None:
    _manifest, report = _build(tmp_path)
    assert json.loads(json.dumps(report))["total_clips"] == 6
