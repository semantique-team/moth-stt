import json
from pathlib import Path

import numpy as np
import pytest
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


def test_build_shards_chunked_consumption(tmp_path: Path) -> None:
    """Batches smaller than the row count must not change the index or the shard layout."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for index in range(3):
        name = f"c{index}.wav"
        sf.write(audio_dir / name, _sine(2.0, frequency=220.0 + 55.0 * index), 16000)
        rows.append({"id": f"c{index}", "audio": name, "duration": 2.0})

    out = tmp_path / "shards"
    report = build_shards(rows, audio_dir, out, shard_bytes=64_000, workers=2, batch_size=1)
    assert report["clips"] == 3
    assert report["shards"] == 2  # 2 s ≈ 32 KB per clip, 64 KB per shard
    assert report["errors"] == []

    index = load_index(out / "index.json")
    for row in rows:
        got = read_clip(index, out, row["id"])
        expected = log_mel(load_audio_16k(audio_dir / row["audio"])).astype(np.float16)
        assert got.shape == (80, expected.shape[1])
        assert np.array_equal(got, expected.astype(np.float32))


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


def _delete_args(tmp_path: Path, rows: list[dict]) -> list[str]:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return [
        "--manifest", str(manifest), "--audio-root", str(tmp_path),
        "--out-dir", str(tmp_path / "shards"), "--workers", "1", "--delete-audio",
    ]


def test_cli_limit_deletes_only_current_indexed_audio(tmp_path: Path) -> None:
    rows = [_manifest_row("a", "a.wav"), _manifest_row("b", "b.wav")]
    for row in rows:
        sf.write(tmp_path / row["audio"], _sine(), 16000)
    args = _delete_args(tmp_path, rows)
    # A previous complete index must not make the limited rebuild delete b.
    build_shards(rows, tmp_path, tmp_path / "shards")
    assert main(args + ["--limit", "1"]) == 0
    assert not (tmp_path / "a.wav").exists()
    assert (tmp_path / "b.wav").exists()


def test_cli_errors_keep_all_audio(tmp_path: Path) -> None:
    sf.write(tmp_path / "a.wav", _sine(), 16000)
    (tmp_path / "bad.wav").write_bytes(b"not audio")
    rows = [_manifest_row("a", "a.wav"), _manifest_row("bad", "bad.wav")]
    assert main(_delete_args(tmp_path, rows)) == 1
    assert (tmp_path / "a.wav").exists()
    assert (tmp_path / "bad.wav").exists()


def test_cli_verifies_every_deletion_candidate(tmp_path: Path, monkeypatch) -> None:
    import stt.data.build_mel as module

    rows = [_manifest_row(str(i), f"{i}.wav") for i in range(5)]
    for row in rows:
        sf.write(tmp_path / row["audio"], _sine(), 16000)
    original = module.build_shards

    def corrupt(*args, **kwargs):
        report = original(*args, **kwargs)
        out = tmp_path / "shards"
        index = load_index(out / "index.json")
        # Seed zero's first three sampled IDs do not include 0.
        shard = out / index["0"]["shard"]
        data = np.load(shard)
        data[index["0"]["offset"] * 80] = 123.0
        np.save(shard, data)
        return report

    monkeypatch.setattr(module, "build_shards", corrupt)
    assert main(_delete_args(tmp_path, rows)) == 1
    assert all((tmp_path / row["audio"]).exists() for row in rows)


def test_delete_audio_prevalidates_all_paths(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    root.mkdir()
    (root / "safe.wav").write_bytes(b"safe")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"keep")
    with pytest.raises(ValueError, match="unsafe"):
        delete_audio([{"audio": "safe.wav"}, {"audio": "../outside.wav"}], root)
    assert (root / "safe.wav").exists() and outside.exists()


def test_verify_sample_rejects_missing_requested_candidate() -> None:
    assert not verify_sample({}, ".", {}, ".", sample_ids=["missing"], n=1)


def test_verify_sample_cannot_silently_skip_requested_audio(tmp_path: Path) -> None:
    sf.write(tmp_path / "a.wav", _sine(), 16000)
    rows = [{"id": "a", "audio": "a.wav"}]
    out = tmp_path / "shards"
    build_shards(rows, tmp_path, out)
    index = load_index(out / "index.json")
    assert not verify_sample(
        index, out, {"a": rows[0]}, tmp_path, sample_ids=["a", "missing"], n=2
    )


def test_verify_sample_requested_membership_is_linear(monkeypatch) -> None:
    import stt.data.build_mel as module

    class CountingId(str):
        comparisons = 0
        __hash__ = str.__hash__

        def __eq__(self, other):
            CountingId.comparisons += 1
            return super().__eq__(other)

    size = 512
    ids = [CountingId(str(i)) for i in range(size)]
    rows = {uid: {"audio": f"{uid}.wav"} for uid in ids}
    index = {uid: {} for uid in ids}
    mel = np.zeros((80, 1), dtype=np.float32)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(module, "load_audio_16k", lambda path: None)
    monkeypatch.setattr(module, "log_mel", lambda wav: mel)
    monkeypatch.setattr(module, "read_clip", lambda *args: mel)
    assert verify_sample(index, ".", rows, ".", sample_ids=ids, n=1)
    assert CountingId.comparisons < size * 4


@pytest.mark.parametrize("rows,limit", [
    ([{"id": "a", "audio": "a.wav"}], -1),
    ([{"id": "a", "audio": "a.wav"}, {"id": "a", "audio": "b.wav"}], None),
])
def test_build_rejects_ambiguous_ids_and_negative_limits(tmp_path: Path, rows, limit) -> None:
    with pytest.raises(ValueError):
        build_shards(rows, tmp_path, tmp_path / "shards", limit=limit)
    assert not (tmp_path / "shards").exists()


def test_limit_zero_never_deletes_audio(tmp_path: Path) -> None:
    sf.write(tmp_path / "a.wav", _sine(), 16000)
    assert main(_delete_args(tmp_path, [_manifest_row("a", "a.wav")]) + ["--limit", "0"]) == 1
    assert (tmp_path / "a.wav").exists()
