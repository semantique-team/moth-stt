from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from recordings.import_manifest import import_manifest, main
from stt.manifest import load_manifest


def _wav(path: Path, *, seconds: float = 1.5, amplitude: float = 0.3, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 48_000
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    mono = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    sf.write(path, data, rate, subtype="PCM_16")
    return path


def _transcripts(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "transcripts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _empty_dirs(tmp_path: Path) -> tuple[Path, Path]:
    scripts, manifests = tmp_path / "scripts", tmp_path / "manifests"
    scripts.mkdir()
    manifests.mkdir()
    return scripts, manifests


def _rows() -> list[dict]:
    return [
        {
            "id": "NM-0001",
            "text": "ho spostato il meeting a domani",
            "speaker": "S9",
            "category": "single_word",
            "en_spans": [[3, 4]],
        },
        {
            "id": "NM-0002",
            "text": "la riunione è finita",
            "speaker": "S9",
            "category": "control_it",
            "en_spans": [],
        },
    ]


def test_imports_valid_collection(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    for row in _rows():
        _wav(audio / f"{row['id']}.wav")
    scripts, manifests = _empty_dirs(tmp_path)

    result = import_manifest(
        _transcripts(tmp_path, _rows()), audio, scripts_dir=scripts, manifest_dir=manifests
    )

    assert result["errors"] == []
    assert result["summary"]["rows"] == 2
    assert result["summary"]["speakers"] == ["S9"]
    first, second = result["rows"]
    assert first["source"] == "natural_mixed_dev"
    assert first["split"] == "dev"
    assert first["lang"] == "mix"
    assert first["norm"] == "ho spostato il meeting a domani"
    assert first["en_spans"] == [[3, 4]]
    assert second["en_spans"] == []


def test_speaker_subdirectory_and_audio_field(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "S9" / "NM-0001.wav")
    _wav(audio / "other" / "take.wav")
    rows = _rows()
    rows[1]["audio"] = "other/take.wav"
    scripts, manifests = _empty_dirs(tmp_path)

    result = import_manifest(
        _transcripts(tmp_path, rows), audio, scripts_dir=scripts, manifest_dir=manifests
    )

    assert result["errors"] == []
    assert [row["audio"] for row in result["rows"]] == ["S9/NM-0001.wav", "other/take.wav"]


def test_refuses_exact_duplicate_within_collection(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "NM-0001.wav")
    _wav(audio / "NM-0002.wav")
    rows = _rows()
    rows[1]["text"] = rows[0]["text"]
    rows[1]["category"] = "single_word"
    rows[1]["en_spans"] = [[3, 4]]
    scripts, manifests = _empty_dirs(tmp_path)

    result = import_manifest(
        _transcripts(tmp_path, rows), audio, scripts_dir=scripts, manifest_dir=manifests
    )

    assert any("duplicate" in error for error in result["errors"])


def test_refuses_bad_spans(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "NM-0001.wav")
    rows = [_rows()[0]]
    rows[0]["en_spans"] = [[9, 10]]
    scripts, manifests = _empty_dirs(tmp_path)

    result = import_manifest(
        _transcripts(tmp_path, rows), audio, scripts_dir=scripts, manifest_dir=manifests
    )

    assert any("en_spans" in error for error in result["errors"])


def test_refuses_silent_and_missing_audio(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "NM-0001.wav", amplitude=0.0)
    scripts, manifests = _empty_dirs(tmp_path)

    result = import_manifest(
        _transcripts(tmp_path, _rows()), audio, scripts_dir=scripts, manifest_dir=manifests
    )

    errors = "\n".join(result["errors"])
    assert "silent audio" in errors
    assert "audio not found" in errors


def test_refuses_stereo(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "NM-0001.wav", channels=2)
    _wav(audio / "NM-0002.wav")
    scripts, manifests = _empty_dirs(tmp_path)

    result = import_manifest(
        _transcripts(tmp_path, _rows()), audio, scripts_dir=scripts, manifest_dir=manifests
    )

    assert any("expected mono" in error for error in result["errors"])


def test_cli_writes_a_loadable_manifest_and_refuses_overwrite(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    for row in _rows():
        _wav(audio / f"{row['id']}.wav")
    scripts, manifests = _empty_dirs(tmp_path)
    out = manifests / "natural_mixed_dev.jsonl"

    assert (
        main(
            [
                "--transcripts", str(_transcripts(tmp_path, _rows())),
                "--audio-dir", str(audio),
                "--out", str(out),
                "--scripts-dir", str(scripts),
                "--manifest-dir", str(manifests),
            ]
        )
        == 0
    )
    loaded = load_manifest(out)
    assert len(loaded) == 2
    assert loaded[0]["source"] == "natural_mixed_dev"

    assert (
        main(
            [
                "--transcripts", str(_transcripts(tmp_path, _rows())),
                "--audio-dir", str(audio),
                "--out", str(out),
                "--scripts-dir", str(scripts),
                "--manifest-dir", str(manifests),
            ]
        )
        == 1
    )
