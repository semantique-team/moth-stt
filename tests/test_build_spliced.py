"""Tests for H3b utterance splicing on the mel cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stt.data.build_mel import build_shards, load_index, read_clip
from stt.data.build_spliced import build_spliced, plan_pairs
from stt.manifest import ManifestError
from stt.text import normalize


def _row(identifier: str, text: str, duration: float, lang: str) -> dict:
    return {
        "id": identifier,
        "audio": f"{identifier}.wav",
        "text": text,
        "norm": normalize(text),
        "duration": duration,
        "source": "test",
        "split": "train",
        "lang": lang,
    }


def _wav(path: Path, seconds: float, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(16000 * seconds)) / 16000.0
    signal = 0.3 * np.sin(2 * np.pi * frequency * time).astype(np.float32)
    sf.write(path, signal, 16000)


def _shard_set(root: Path, rows: list[dict]) -> Path:
    out = root / "shards"
    build_shards(rows, root, out)
    return out


def test_plan_pairs_is_deterministic_and_bounded() -> None:
    words = ["uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci"]
    it_rows = [_row(f"it-{i}", f"ciao numero {word}", 3.0, "it") for i, word in enumerate(words)]
    en_rows = [_row(f"en-{i}", f"hello number {word}", 4.0, "en") for i, word in enumerate(words)]

    first = plan_pairs(it_rows, en_rows, hours=0.05, seed=0)
    assert first == plan_pairs(it_rows, en_rows, hours=0.05, seed=0)
    assert first != plan_pairs(it_rows, en_rows, hours=0.05, seed=1)

    assert len({row["id"] for row in first}) == len(first)
    assert sum(row["duration"] for row in first) >= 0.05 * 3600
    for row in first:
        assert row["duration"] <= 27.0
        assert row["lang"] == "mix" and row["category"] == "switch"
        assert row["norm"] == normalize(row["text"])
        (start, end), = row["en_spans"]
        assert row["norm"].split()[start:end] == row["text"].split()[start:end]
        assert start == 3  # "ciao numero N" is three words


def test_plan_pairs_rejects_impossible_bounds() -> None:
    it_rows = [_row("it-0", "a", 20.0, "it")]
    en_rows = [_row("en-0", "b", 20.0, "en")]
    try:
        plan_pairs(it_rows, en_rows, hours=1.0, max_seconds=27.0)
    except ValueError as error:
        assert "duration bounds" in str(error)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("expected ValueError")


def test_build_spliced_concatenates_the_two_cached_mels(tmp_path: Path) -> None:
    it_root, en_root = tmp_path / "it", tmp_path / "en"
    _wav(it_root / "it-0.wav", 1.0, 220.0)
    _wav(en_root / "en-0.wav", 2.0, 440.0)
    it_rows = [_row("it-0", "ciao", 1.0, "it")]
    en_rows = [_row("en-0", "hello", 2.0, "en")]
    it_shards = _shard_set(it_root, it_rows)
    en_shards = _shard_set(en_root, en_rows)

    rows = plan_pairs(it_rows, en_rows, hours=0.0005, seed=0)
    assert len(rows) == 1
    out = tmp_path / "spliced"
    report = build_spliced(
        [(it_shards, load_index(it_shards / "index.json"))],
        [(en_shards, load_index(en_shards / "index.json"))],
        rows,
        out,
    )

    assert report["clips"] == 1 and report["errors"] == []
    index = load_index(out / "index.json")
    entry = index[rows[0]["id"]]
    mel = read_clip(index, out, rows[0]["id"])
    it_mel = read_clip(load_index(it_shards / "index.json"), it_shards, "it-0")
    en_mel = read_clip(load_index(en_shards / "index.json"), en_shards, "en-0")
    assert mel.shape == (80, it_mel.shape[1] + en_mel.shape[1])
    assert entry["frames"] == mel.shape[1]
    assert np.array_equal(mel[:, : it_mel.shape[1]], it_mel)
    assert np.array_equal(mel[:, it_mel.shape[1] :], en_mel)
    assert abs(entry["duration"] - mel.shape[1] / 100.0) < 1e-6
    assert report["hours"] > 0
    assert json.loads((out / "build_report.json").read_text())["clips"] == 1


def test_spliced_manifest_validates(tmp_path: Path) -> None:
    from stt.manifest import load_manifest

    it_rows = [_row("it-0", "ciao", 2.0, "it")]
    en_rows = [_row("en-0", "hello world", 2.0, "en")]
    rows = plan_pairs(it_rows, en_rows, hours=0.001, seed=0)
    path = tmp_path / "spliced.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loaded = load_manifest(path)
    assert loaded[0]["norm"] == "ciao hello world"
    assert loaded[0]["en_spans"] == [[1, 3]]


@pytest.mark.parametrize("side", ["it", "en"])
@pytest.mark.parametrize("changes", [
    {"split": "test"}, {"split": "dev"}, {"source": "recorded_t"},
    {"source": "recorded_a"}, {"speaker": "S3"},
])
def test_plan_pairs_rejects_held_out_sources(side: str, changes: dict) -> None:
    it = _row("it-1", "ciao", 2.0, "it")
    en = _row("en-1", "hello", 2.0, "en")
    (it if side == "it" else en).update(changes)
    with pytest.raises(ManifestError, match="cannot train"):
        plan_pairs([it], [en], hours=0.001)
