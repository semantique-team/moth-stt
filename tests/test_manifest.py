import json
from pathlib import Path

import pytest

from stt.manifest import ManifestError, load_manifest, norm_hash

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "manifest_sample.jsonl"


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "m.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(**overrides: object) -> dict:
    base = {
        "id": "r1",
        "audio": "a.wav",
        "text": "ciao come stai",
        "norm": "ciao come stai",
        "duration": 1.0,
        "source": "test",
        "split": "train",
        "speaker": None,
        "lang": "it",
        "category": None,
        "en_spans": [],
    }
    base.update(overrides)
    return base


def test_loads_the_sample_fixture() -> None:
    rows = load_manifest(SAMPLE)
    assert [row["id"] for row in rows] == [
        "mix-0001",
        "mix-0002",
        "cv-it-0001",
        "ls-en-0001",
        "noise-0001",
    ]


def test_missing_required_key_rejected(tmp_path: Path) -> None:
    row = _row()
    del row["lang"]
    with pytest.raises(ManifestError, match="lang"):
        load_manifest(_write(tmp_path, [row]))


def test_bad_lang_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="lang"):
        load_manifest(_write(tmp_path, [_row(lang="fr")]))


def test_category_on_non_mixed_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="category"):
        load_manifest(_write(tmp_path, [_row(category="single_word")]))


def test_en_spans_on_non_mixed_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="en_spans"):
        load_manifest(_write(tmp_path, [_row(en_spans=[[0, 1]])]))


def test_en_spans_out_of_bounds_rejected(tmp_path: Path) -> None:
    row = _row(lang="mix", category="single_word", en_spans=[[0, 99]])
    with pytest.raises(ManifestError, match="en_spans"):
        load_manifest(_write(tmp_path, [row]))


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(_write(tmp_path, [_row(), _row()]))


def test_norm_must_be_canonical(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="norm"):
        load_manifest(_write(tmp_path, [_row(norm="CIAO, come stai!")]))


def test_negative_duration_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="duration"):
        load_manifest(_write(tmp_path, [_row(duration=-1.0)]))


def test_norm_hash_is_stable_and_normalizes() -> None:
    first = norm_hash("Ho spostato il meeting! a domani")
    second = norm_hash("  ho spostato il meeting  a domani ")
    assert first == second
    assert len(first) == 40
    assert norm_hash("altro testo") != first
