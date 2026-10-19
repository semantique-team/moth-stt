import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stt.data.sources import (
    SOURCES,
    download,
    plan_records,
    probe_duration,
    read_cv_tsv,
    read_fleurs_tsv,
    read_librispeech_transcripts,
    read_mls_transcripts,
    sample_by_duration,
    sha256_file,
    write_records,
)


def test_registry_has_licenses_and_urls() -> None:
    assert {"cv_it_27", "mls_it", "librispeech", "fleurs_it", "fleurs_en"} <= set(SOURCES)
    for source in SOURCES.values():
        assert source.license
        assert source.splits
    assert SOURCES["cv_it_27"].license == "CC0"
    assert "CC-BY" in SOURCES["mls_it"].license
    assert SOURCES["fleurs_it"].eval_only is True
    assert SOURCES["noise_musan"].license.startswith("verify")


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"moth")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_download_verifies_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"audio bytes")
    dest = tmp_path / "raw" / "copy.bin"
    download(source.as_uri(), dest, sha256=sha256_file(source))
    assert dest.read_bytes() == b"audio bytes"
    sums = (tmp_path / "raw" / "SHA256SUMS").read_text()
    assert sha256_file(source) in sums


def test_download_rejects_bad_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"audio bytes")
    with pytest.raises(ValueError, match="checksum"):
        download(source.as_uri(), tmp_path / "copy.bin", sha256="0" * 64)


def test_download_skips_existing_valid_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"audio bytes")
    dest = tmp_path / "copy.bin"
    digest = sha256_file(source)
    download(source.as_uri(), dest, sha256=digest)
    source.unlink()
    download(source.as_uri(), dest, sha256=digest)  # no network access needed
    assert dest.read_bytes() == b"audio bytes"


CV_TSV = (
    "client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccents\tlocale\tsegment\n"
    "c1\tclip_1.mp3\tCiao come stai\t2\t0\t\t\t\tti\t\n"
    "c2\tclip_2.mp3\t\t1\t0\t\t\t\tti\t\n"
    "c3\tclip_3.mp3\tOggi piove\t1\t0\t\t\t\tti\t\n"
)


def test_read_cv_tsv(tmp_path: Path) -> None:
    path = tmp_path / "train.tsv"
    path.write_text(CV_TSV, encoding="utf-8")
    rows = read_cv_tsv(path, clips_dir="clips")
    assert [(row["id"], row["audio"], row["text"]) for row in rows] == [
        ("cv-c1", "clips/clip_1.mp3", "Ciao come stai"),
        ("cv-c3", "clips/clip_3.mp3", "Oggi piove"),
    ]
    assert rows[0]["speaker"] == "c1"


LS_ROOT = "LibriSpeech/train-clean-100/19/198"


def test_read_librispeech_transcripts(tmp_path: Path) -> None:
    chapter = tmp_path / LS_ROOT
    chapter.mkdir(parents=True)
    (chapter / "19-198.trans.txt").write_text(
        "19-198-0001 HELLO WORLD\n19-198-0002 ANOTHER LINE\n", encoding="utf-8"
    )
    (chapter / "19-198-0001.flac").write_bytes(b"x")
    (chapter / "19-198-0002.flac").write_bytes(b"x")
    rows = read_librispeech_transcripts(tmp_path / "LibriSpeech/train-clean-100")
    assert [(row["id"], row["text"], row["speaker"]) for row in rows] == [
        ("ls-19-198-0001", "HELLO WORLD", "19"),
        ("ls-19-198-0002", "ANOTHER LINE", "19"),
    ]
    assert rows[0]["audio"].endswith("19/198/19-198-0001.flac")


def test_read_mls_transcripts(tmp_path: Path) -> None:
    root = tmp_path / "mls_italian/train"
    root.mkdir(parents=True)
    (root / "transcripts.txt").write_text(
        "1001_1_0001\tciao a tutti\n1001_1_0002\tseconda frase\n", encoding="utf-8"
    )
    audio = root / "audio" / "1001" / "1"
    audio.mkdir(parents=True)
    (audio / "1001_1_0001.flac").write_bytes(b"x")
    (audio / "1001_1_0002.flac").write_bytes(b"x")
    rows = read_mls_transcripts(root)
    assert [(row["id"], row["text"], row["speaker"]) for row in rows] == [
        ("mls-1001_1_0001", "ciao a tutti", "1001"),
        ("mls-1001_1_0002", "seconda frase", "1001"),
    ]
    assert rows[0]["audio"] == "audio/1001/1/1001_1_0001.flac"


def test_read_fleurs_tsv(tmp_path: Path) -> None:
    """Official layout: headerless, 7 columns; ids repeat, filenames are the unique key."""
    root = tmp_path / "it_it"
    (root / "audio" / "test").mkdir(parents=True)
    (root / "test.tsv").write_text(
        "1536\t1.wav\tCiao a tutti\tciao a tutti\ta | b\t16000\tFEMALE\n"
        "1536\t2.wav\t\t\t\t\t\n",
        encoding="utf-8",
    )
    (root / "audio" / "test" / "1.wav").write_bytes(b"x")
    rows = read_fleurs_tsv(root, "test")
    assert [(row["id"], row["audio"], row["text"]) for row in rows] == [
        ("fleurs-1", "audio/test/1.wav", "Ciao a tutti")
    ]


def test_read_fleurs_tsv_with_header_variant(tmp_path: Path) -> None:
    root = tmp_path / "it_it"
    (root / "audio" / "dev").mkdir(parents=True)
    (root / "dev.tsv").write_text(
        "id\tfilename\ttranscription\traw_transcription\n"
        "7\t7.wav\tbuongiorno\tbuongiorno\n",
        encoding="utf-8",
    )
    rows = read_fleurs_tsv(root, "dev")
    assert rows == [
        {"id": "fleurs-7", "audio": "audio/dev/7.wav", "text": "buongiorno", "speaker": None}
    ]


def test_sample_by_duration_is_deterministic_and_respects_target() -> None:
    records = [
        {"id": f"r{index}", "duration": 10.0, "audio": f"{index}.wav", "text": "x"}
        for index in range(10)
    ]
    first = sample_by_duration(records, target_seconds=35.0, seed=7)
    second = sample_by_duration(records, target_seconds=35.0, seed=7)
    assert [row["id"] for row in first["selected"]] == [row["id"] for row in second["selected"]]
    assert len(first["selected"]) == 4
    assert first["selected_seconds"] == 40.0
    assert first["total_seconds"] == 100.0


def test_sample_by_duration_takes_everything_when_short() -> None:
    records = [{"id": "a", "duration": 1.0}, {"id": "b", "duration": 2.0}]
    result = sample_by_duration(records, target_seconds=3600.0, seed=0)
    assert len(result["selected"]) == 2
    assert result["short"] is True


def test_probe_duration(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(24000, dtype="float32"), 16000)
    assert probe_duration(path) == pytest.approx(1.5, abs=1e-3)


def test_plan_records_and_write_manifest(tmp_path: Path) -> None:
    root = tmp_path / "it"
    (root / "clips").mkdir(parents=True)
    (root / "train.tsv").write_text(CV_TSV, encoding="utf-8")
    for name in ("clip_1.mp3", "clip_3.mp3"):
        sf.write(root / "clips" / name, np.zeros(16000, dtype="float32"), 16000)
    rows = plan_records("cv_it_27", root, "train")
    assert len(rows) == 2
    assert rows[0]["lang"] == "it"
    assert rows[0]["split"] == "train"
    assert rows[0]["norm"] == "ciao come stai"
    out = tmp_path / "manifests" / "cv_it_train.jsonl"
    write_records(rows, out)
    loaded = [json.loads(line) for line in out.read_text().splitlines()]
    assert loaded[0]["source"] == "cv_it_27"
    assert loaded[0]["duration"] == 1.0


def test_plan_records_refuses_eval_only_source_for_training(tmp_path: Path) -> None:
    root = tmp_path / "it_it"
    root.mkdir()
    with pytest.raises(ValueError, match="evaluation"):
        plan_records("fleurs_it", root, "train")
