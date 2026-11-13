"""Tests for the long-note segmentation helper (no network, no ffmpeg, no real audio)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from recordings.build_manifest import MAX_SECONDS
from recordings.import_manifest import import_manifest
from recordings.segment_notes import main, plan_segments, write_segments

RATE = 16_000


def _tone(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def _span_seconds(samples: np.ndarray, segments: list[tuple[int, int]]) -> float:
    return sum(end - start for start, end in segments) / RATE


def test_splits_three_bursts_on_gaps():
    samples = np.concatenate([_tone(2), _silence(1), _tone(2), _silence(1), _tone(2)])
    segments = plan_segments(samples)
    assert len(segments) == 3
    assert _span_seconds(samples, segments) == pytest.approx(6.0, abs=0.1)


def test_short_gap_stays_inside_one_clip():
    samples = np.concatenate([_tone(1), _silence(0.1), _tone(1)])
    assert len(plan_segments(samples, min_gap_seconds=0.3)) == 1
    assert len(plan_segments(samples, min_gap_seconds=0.05)) == 2


def test_long_speech_is_cut_under_the_import_limit():
    samples = _tone(75)
    segments = plan_segments(samples)
    durations = [(end - start) / RATE for start, end in segments]
    assert len(segments) >= 3
    assert max(durations) <= MAX_SECONDS
    assert min(durations) >= 1.0
    assert sum(durations) == pytest.approx(75.0, abs=0.1)
    assert segments[0][0] == 0
    assert segments[-1][1] == len(samples)


def test_silent_note_is_rejected():
    with pytest.raises(ValueError, match="no speech"):
        plan_segments(_silence(5))


def test_short_burst_below_minimum_is_rejected():
    samples = np.concatenate([_silence(1), _tone(0.3), _silence(1)])
    with pytest.raises(ValueError, match="minimum"):
        plan_segments(samples)


def test_bounds_are_validated():
    with pytest.raises(ValueError, match="need 0 < min < max"):
        plan_segments(_tone(2), min_seconds=2, max_seconds=1)
    with pytest.raises(ValueError, match="need 0 < min < max"):
        plan_segments(_tone(2), max_seconds=MAX_SECONDS + 1)


@pytest.mark.parametrize("gap", [-1, float("nan"), float("inf")])
def test_invalid_gap_is_rejected(gap):
    with pytest.raises(ValueError, match="min_gap_seconds"):
        plan_segments(_tone(2), min_gap_seconds=gap)


@pytest.mark.parametrize("samples", [np.zeros((10, 2)), np.array([float("nan")])])
def test_invalid_samples_are_rejected(samples):
    with pytest.raises(ValueError, match="finite mono"):
        plan_segments(samples)


@pytest.mark.parametrize("segments", [[], [(0, RATE // 2)], [(0, 3 * RATE)],
                                     [(0, RATE), (RATE // 2, 2 * RATE)]])
def test_invalid_segments_leave_no_outputs(tmp_path, segments):
    with pytest.raises(ValueError, match="segments"):
        write_segments(_tone(2), segments, out_dir=tmp_path / "clips", speaker="S1",
                       prefix="s1-note", source_stem="note",
                       transcripts_path=tmp_path / "notes.jsonl")
    assert list(tmp_path.iterdir()) == []


def test_cli_aggregates_notes_and_overwrites_once(tmp_path):
    notes = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path in notes:
        sf.write(path, _tone(2), RATE)
    out = tmp_path / "clips"
    text = tmp_path / "notes.jsonl"
    args = ["--audio", *map(str, notes), "--speaker", "S1", "--out-dir", str(out),
            "--transcripts", str(text)]
    for suffix in ([], ["--overwrite"]):
        assert main(args + suffix) == 0
        rows = [json.loads(line) for line in text.read_text().splitlines()]
        assert [row["id"] for row in rows] == ["s1-note-first-001", "s1-note-second-001"]
        assert {row["audio"] for row in rows} == {path.name for path in out.glob("*.wav")}


@pytest.mark.parametrize("overwrite", [False, True])
def test_cli_stem_collisions_preflight_before_decode(tmp_path, monkeypatch, overwrite):
    monkeypatch.setattr("recordings.segment_notes.load_audio_16k",
                        lambda _: pytest.fail("must reject before decoding"))
    with pytest.raises(SystemExit, match="2"):
        main(["--audio", "A B.wav", "a-b.wav", "--speaker", "S1", "--out-dir",
              str(tmp_path / "clips"), "--transcripts", str(tmp_path / "notes.jsonl"),
              *(["--overwrite"] if overwrite else [])])
    assert list(tmp_path.iterdir()) == []


def test_cli_later_existing_clip_leaves_no_partial_outputs(tmp_path):
    notes = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path in notes:
        sf.write(path, _tone(2), RATE)
    out = tmp_path / "clips"
    out.mkdir()
    existing = out / "s1-note-second-001.wav"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        main(["--audio", *map(str, notes), "--speaker", "S1", "--out-dir", str(out),
              "--transcripts", str(tmp_path / "notes.jsonl")])
    assert list(out.iterdir()) == [existing]
    assert existing.read_bytes() == b"keep"
    assert not (tmp_path / "notes.jsonl").exists()


def test_cli_later_invalid_note_leaves_no_outputs(tmp_path):
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    sf.write(first, _tone(2), RATE)
    sf.write(second, _silence(2), RATE)
    with pytest.raises(ValueError, match="no speech"):
        main(["--audio", str(first), str(second), "--speaker", "S1", "--out-dir",
              str(tmp_path / "clips"), "--transcripts", str(tmp_path / "notes.jsonl")])
    assert not (tmp_path / "clips").exists()
    assert not (tmp_path / "notes.jsonl").exists()


def test_cli_transcript_clip_collision_leaves_no_outputs(tmp_path):
    note = tmp_path / "first.wav"
    sf.write(note, _tone(2), RATE)
    out = tmp_path / "clips"
    with pytest.raises(SystemExit, match="2"):
        main(["--audio", str(note), "--speaker", "S1", "--out-dir", str(out),
              "--transcripts", str(out / "s1-note-first-001.wav")])
    assert not out.exists()


def test_single_note_inspection_failure_leaves_no_files(tmp_path, monkeypatch):
    def fail(_):
        raise ValueError("inspection failed")

    monkeypatch.setattr("recordings.segment_notes.inspect_audio", fail)
    with pytest.raises(ValueError, match="inspection failed"):
        write_segments(_tone(2), [(0, 2 * RATE)], out_dir=tmp_path / "clips", speaker="S1",
                       prefix="s1-note", source_stem="note",
                       transcripts_path=tmp_path / "notes.jsonl")
    assert list((tmp_path / "clips").iterdir()) == []
    assert not (tmp_path / "notes.jsonl").exists()


def test_cli_later_inspection_failure_preserves_existing_outputs(tmp_path, monkeypatch):
    notes = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path in notes:
        sf.write(path, _tone(2), RATE)
    out = tmp_path / "clips"
    out.mkdir()
    existing = out / "s1-note-first-001.wav"
    existing.write_bytes(b"old audio")
    text = tmp_path / "notes.jsonl"
    text.write_text("old transcript")
    from recordings import segment_notes

    inspect = segment_notes.inspect_audio

    def fail_second(path):
        if "second" in path.name:
            raise ValueError("inspection failed")
        return inspect(path)

    monkeypatch.setattr(segment_notes, "inspect_audio", fail_second)
    with pytest.raises(ValueError, match="inspection failed"):
        main(["--audio", *map(str, notes), "--speaker", "S1", "--out-dir", str(out),
              "--transcripts", str(text), "--overwrite"])
    assert list(out.iterdir()) == [existing]
    assert existing.read_bytes() == b"old audio"
    assert text.read_text() == "old transcript"


def test_write_segments_makes_16k_mono_and_a_skeleton(tmp_path):
    samples = np.concatenate([_tone(2), _silence(1), _tone(2)])
    segments = plan_segments(samples)
    out = tmp_path / "clips"
    transcripts = tmp_path / "notes.jsonl"
    summary = write_segments(
        samples, segments, out_dir=out, speaker="S1", prefix="s1-note",
        transcripts_path=transcripts, source_stem="2026-11-12",
    )
    assert summary["clips"] == len(segments) == 2
    assert summary["silent_clips"] == []

    clips = sorted(out.glob("*.wav"))
    assert [path.name for path in clips] == ["s1-note-2026-11-12-001.wav", "s1-note-2026-11-12-002.wav"]
    info = sf.info(clips[0])
    assert (info.samplerate, info.channels, info.subtype) == (RATE, 1, "PCM_16")

    rows = [json.loads(line) for line in transcripts.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["s1-note-2026-11-12-001", "s1-note-2026-11-12-002"]
    assert all(row["text"] == "" and row["lang"] == "mix" and row["speaker"] == "S1" for row in rows)
    assert all(row["audio"].endswith(".wav") and row["en_spans"] == [] for row in rows)

    with pytest.raises(FileExistsError, match="overwrite"):
        write_segments(
            samples, segments, out_dir=out, speaker="S1", prefix="s1-note",
            transcripts_path=transcripts, source_stem="2026-11-12",
        )


def test_filled_skeleton_imports_as_natural_mixed_dev(tmp_path):
    samples = _tone(3)
    segments = plan_segments(samples)
    out = tmp_path / "clips"
    transcripts = tmp_path / "notes.jsonl"
    write_segments(
        samples, segments, out_dir=out, speaker="S2", prefix="s2-note",
        transcripts_path=transcripts, source_stem="session-a",
    )
    rows = [json.loads(line) for line in transcripts.read_text(encoding="utf-8").splitlines()]
    rows[0] |= {"text": "il deploy del meeting", "category": "phrase", "en_spans": [[1, 2], [3, 4]]}
    transcripts.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )

    result = import_manifest(
        transcripts, out, scripts_dir=tmp_path, manifest_dir=tmp_path, exclude=transcripts
    )
    assert result["errors"] == []
    assert len(result["rows"]) == len(rows)
    row = result["rows"][0]
    assert (row["source"], row["split"], row["lang"], row["speaker"]) == (
        "natural_mixed_dev", "dev", "mix", "S2",
    )
    assert row["norm"] == "il deploy del meeting"
    assert 1.0 <= row["duration"] <= MAX_SECONDS
