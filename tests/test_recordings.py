import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recordings.build_manifest import (
    build_manifest,
    build_report,
    cues_for,
    default_audio_dir,
    find_audio,
)
from recordings.generate_scripts import script_path
from recordings.record import append_session_log, next_audio_path, plan_session


def _write_script(
    scripts_dir: Path, kind: str, rows: list[dict], speaker: str | None = None
) -> None:
    path = script_path(scripts_dir, kind, speaker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _cue(uid: str, text: str, category: str, speaker: str | None = None) -> dict:
    row = {
        "id": uid,
        "text": text,
        "category": category,
        "en_spans": [[3, 4]] if category == "single_word" else [],
        "generator": "test",
        "date": "2026-10-19",
        "reviewed": True,
    }
    if speaker is not None:
        row["speaker"] = speaker
    return row


def _wav(path: Path, seconds: float = 2.0, amplitude: float = 0.1, subtype: str = "FLOAT") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 48000
    time = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    data = (amplitude * np.sin(2 * np.pi * 220.0 * time)).astype("float32")
    sf.write(path, data, rate, subtype=subtype)


def test_cues_for_filters_and_orders() -> None:
    rows = [
        _cue("T-0001", "a b c", "single_word", "S2"),
        _cue("T-0002", "a b c", "single_word", "S1"),
        _cue("T-0003", "d e f", "control_it", "S1"),
    ]
    cues = cues_for(rows, "S1")
    assert [row["id"] for row in cues] == ["T-0002", "T-0003"]


def test_default_audio_dirs_separate_t_and_r() -> None:
    assert default_audio_dir("T", "S1") == Path("recordings/audio")
    assert default_audio_dir("A", "S2") == Path("recordings/audio/anchor")
    assert default_audio_dir("R", "S1") == Path("recordings/audio/R-S1")
    assert default_audio_dir("R", "S2") == Path("recordings/audio/R-S2")
    # S4 gets one for E-018; S3 never does (it is the held-out speaker).
    assert default_audio_dir("R", "S4") == Path("recordings/audio/R-S4")
    with pytest.raises(ValueError, match="held out"):
        default_audio_dir("R", "S3")


def test_find_audio_prefers_wav_and_finds_m4a(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    assert find_audio(root, "S1", 1) is None
    (root / "S1").mkdir(parents=True, exist_ok=True)
    (root / "S1" / "S1_0001.m4a").write_bytes(b"x")
    found = find_audio(root, "S1", 1)
    assert found is not None and found.suffix == ".m4a"
    (root / "S1" / "S1_0001.wav").write_bytes(b"x")
    found = find_audio(root, "S1", 1)
    assert found is not None and found.suffix == ".wav"


def test_build_manifest_joins_by_index(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    audio = tmp_path / "audio"
    _write_script(
        scripts,
        "T",
        [
            _cue("T-0001", "ho spostato il meeting a domani", "single_word", "S1"),
            _cue("T-0002", "ciao come stai", "control_it", "S1"),
            _cue("T-0003", "oggi piove forte", "control_it", "S2"),
        ],
    )
    _wav(audio / "S1" / "S1_0001.wav")
    _wav(audio / "S1" / "S1_0002.wav")
    _wav(audio / "S2" / "S2_0001.wav")
    result = build_manifest("T", scripts, audio)
    assert result["errors"] == []
    rows = result["rows"]
    assert [(row["id"], row["speaker"]) for row in rows] == [
        ("T-0001", "S1"),
        ("T-0002", "S1"),
        ("T-0003", "S2"),
    ]
    assert rows[0]["audio"] == "S1/S1_0001.wav"
    assert rows[0]["source"] == "recorded_t"
    assert rows[0]["split"] == "test"
    assert rows[0]["lang"] == "mix"
    assert rows[0]["duration"] == 2.0
    assert result["speakers"]["S1"]["found"] == 2
    assert result["speakers"]["S2"]["found"] == 1


def test_build_manifest_reports_missing_and_extra(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    audio = tmp_path / "audio"
    _write_script(scripts, "T", [_cue("T-0001", "a b c", "control_it", "S1")])
    _wav(audio / "S1" / "S1_0002.wav")  # extra: index 2 without a cue
    result = build_manifest("T", scripts, audio)
    assert any("S1_0001" in error for error in result["errors"])
    assert any("S1_0002" in warning for warning in result["warnings"])
    assert result["rows"] == []


def test_unreviewed_t_rejected_unless_flag(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    audio = tmp_path / "audio"
    row = _cue("T-0001", "a b c", "control_it", "S1")
    row["reviewed"] = False
    _write_script(scripts, "T", [row])
    _wav(audio / "S1" / "S1_0001.wav")
    assert build_manifest("T", scripts, audio)["errors"]
    assert build_manifest("T", scripts, audio, allow_unreviewed=True)["errors"] == []


def test_silent_and_clipping_detected(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    audio = tmp_path / "audio"
    _write_script(
        scripts,
        "T",
        [
            _cue("T-0001", "a b c", "control_it", "S1"),
            _cue("T-0002", "d e f", "control_it", "S1"),
        ],
    )
    _wav(audio / "S1" / "S1_0001.wav", amplitude=0.0)
    _wav(audio / "S1" / "S1_0002.wav", amplitude=1.5, subtype="PCM_16")
    result = build_manifest("T", scripts, audio)
    assert any("is silent" in error for error in result["errors"])
    assert any("clipping" in error for error in result["errors"])


def test_record_builds_r_manifest(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    audio = tmp_path / "audio"
    _write_script(scripts, "R", [_cue("R-S1-0001", "a b c", "control_it")], speaker="S1")
    _wav(audio / "S1" / "S1_0001.wav")
    result = build_manifest("R", scripts, audio, speaker="S1")
    assert result["errors"] == []
    assert result["rows"][0]["source"] == "recorded_r"
    assert result["rows"][0]["split"] == "train"


def test_build_report_renders_counts() -> None:
    report = {
        "speakers": {"S1": {"expected": 3, "found": 2, "total_seconds": 12.5}},
        "errors": [],
        "warnings": [],
    }
    text = build_report(report)
    assert "S1" in text and "2/3" in text and "12.5" in text


def test_plan_session_skips_existing_and_before_start(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "S1" / "S1_0002.wav", seconds=1.0)
    planned = plan_session(audio, "S1", total=4, start=2)
    assert planned == [3, 4]


def test_next_audio_path_zero_pads(tmp_path: Path) -> None:
    assert next_audio_path(tmp_path, "S3", 7).name == "S3_0007.wav"


def test_append_session_log(tmp_path: Path) -> None:
    log = tmp_path / "S1" / "session.log.jsonl"
    append_session_log(log, {"id": "T-0001", "action": "ok"})
    append_session_log(log, {"id": "T-0001", "action": "retake"})
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["action"] == "retake"


def test_anchor_manifest_is_per_speaker(tmp_path: Path) -> None:
    """Kind A: the same 60 anchor sentences recorded by every speaker, in their own folder."""
    scripts = tmp_path / "scripts"
    audio = tmp_path / "anchor"
    _write_script(
        scripts,
        "A",
        [
            _cue("A-0001", "ho spostato il meeting a domani", "single_word"),
            _cue("A-0002", "oggi leggo un libro", "control_it"),
        ],
    )
    _wav(audio / "S1" / "S1_0001.wav")
    _wav(audio / "S1" / "S1_0002.wav")
    result = build_manifest("A", scripts, audio, speaker="S1")
    assert result["errors"] == []
    assert [(row["id"], row["speaker"]) for row in result["rows"]] == [
        ("A-0001", "S1"),
        ("A-0002", "S1"),
    ]
    assert result["rows"][0]["source"] == "recorded_a"
    assert result["rows"][0]["split"] == "test"
    assert result["speakers"]["S1"]["found"] == 2


def test_anchor_requires_a_speaker(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    _write_script(scripts, "A", [_cue("A-0001", "oggi leggo un libro", "control_it")])
    with pytest.raises(ValueError, match="needs --speaker"):
        build_manifest("A", scripts, tmp_path / "anchor")
