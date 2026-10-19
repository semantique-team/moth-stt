import json
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.data.build_mel import (
    build_shards,
    delete_audio,
    load_audio_16k,
    load_index,
    main,
    read_clip,
    resample_to_target,
    verify_sample,
)
from stt.features import log_mel


def _sine(seconds: float = 2.0, sample_rate: int = 16000, frequency: float = 440.0) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def _manifest_row(uid: str, audio: str) -> dict:
    return {
        "id": uid,
        "audio": audio,
        "text": "ciao come stai",
        "norm": "ciao come stai",
        "duration": 2.0,
        "source": "test",
        "split": "train",
        "speaker": "s1",
        "lang": "it",
        "category": None,
        "en_spans": [],
    }


def test_resample_preserves_length_and_frequency() -> None:
    signal = _sine(1.0, sample_rate=48000, frequency=440.0)
    out = resample_to_target(signal, 48000, 16000)
    assert len(out) == 16000
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = float(np.argmax(spectrum)) * 16000 / len(out)
    assert abs(peak_hz - 440.0) < 2.0


def test_load_audio_16k_stereo_and_resample(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    left = _sine(1.0, sample_rate=44100)
    sf.write(path, np.stack([left, left], axis=1), 44100)
    mono = load_audio_16k(path)
    assert mono.ndim == 1
    assert abs(len(mono) - 16000) <= 1


def test_build_shards_and_read_back_exactly(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for index in range(3):
        name = f"c{index}.wav"
        sf.write(audio_dir / name, _sine(2.0, frequency=220.0 + 55.0 * index), 16000)
        rows.append({"id": f"c{index}", "audio": name, "duration": 2.0})

    out = tmp_path / "shards"
    report = build_shards(rows, audio_dir, out, shard_bytes=64_000, workers=1)
    assert report["clips"] == 3
    assert report["shards"] == 2  # 2 s ≈ 32 KB per clip, 64 KB per shard
    assert report["errors"] == []

    index = load_index(out / "index.json")
    for row in rows:
        got = read_clip(index, out, row["id"])
        expected = log_mel(load_audio_16k(audio_dir / row["audio"])).astype(np.float16)
        assert got.shape == (80, expected.shape[1])
        assert np.array_equal(got, expected.astype(np.float32))
    assert 1.9 < index["c0"]["duration"] < 2.1


def test_build_shards_records_errors_and_continues(tmp_path: Path) -> None:
    rows = [
        {"id": "bad", "audio": "missing.wav", "duration": 1.0},
        {"id": "good", "audio": "good.wav", "duration": 2.0},
    ]
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / "good.wav", _sine(2.0), 16000)
    report = build_shards(rows, audio_dir, tmp_path / "shards", workers=1)
    assert report["clips"] == 1
    assert report["errors"] and report["errors"][0]["id"] == "bad"


def test_verify_sample_detects_corruption(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / "a.wav", _sine(2.0), 16000)
    rows = [{"id": "a", "audio": "a.wav", "duration": 2.0}]
    out = tmp_path / "shards"
    build_shards(rows, audio_dir, out, workers=1)
    index = load_index(out / "index.json")
    rows_by_id = {row["id"]: row for row in rows}

    assert verify_sample(index, out, rows_by_id, audio_dir, n=1) is True

    shard = out / index["a"]["shard"]
    data = np.load(shard)
    data[0] = 0.0
    np.save(shard, data)
    assert verify_sample(index, out, rows_by_id, audio_dir, n=1) is False


def test_delete_audio(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / "a.wav", _sine(1.0), 16000)
    rows = [{"id": "a", "audio": "a.wav"}, {"id": "b", "audio": "missing.wav"}]
    assert delete_audio(rows, audio_dir) == 1
    assert not (audio_dir / "a.wav").exists()


def test_cli_builds_shards_and_index(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / "a.wav", _sine(2.0), 16000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_manifest_row("a", "a.wav")) + "\n", encoding="utf-8")
    out = tmp_path / "shards"
    code = main(
        [
            "--manifest",
            str(manifest),
            "--audio-root",
            str(audio_dir),
            "--out-dir",
            str(out),
            "--workers",
            "1",
        ]
    )
    assert code == 0
    index = load_index(out / "index.json")
    assert "a" in index
    assert (out / "index.json").exists()
