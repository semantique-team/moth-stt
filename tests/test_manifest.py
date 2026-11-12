import json
from pathlib import Path

import pytest

from stt.manifest import ManifestError, load_manifest, norm_hash, validate_training_rows

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


@pytest.mark.parametrize("changes", [
    {"split": "dev"}, {"split": "test"}, {"role": "test"}, {"source": "recorded_t"},
    {"source": "recorded_a"}, {"speaker": "S3"}, {"id": "T-0001"},
    {"id": "A-S1-0001"}, {"id": "R-S3-0001"},
    {"source": "spliced_it_en", "audio": "T-0001+en-1"},
    {"source": "fleurs_it"}, {"source": "fleurs_en"},
])
def test_training_rows_reject_held_out_data(changes: dict) -> None:
    with pytest.raises(ManifestError, match="cannot train"):
        validate_training_rows([_row(**changes)])


@pytest.mark.parametrize("changes", [
    {"id": "dev-1"},
    {"source": "spliced_it_en", "audio": "it-1+dev-1"},
    {"source_ids": ["dev-1"]},
    {"audio": "shared/clip.wav"},
])
def test_training_rows_reject_overlap(changes: dict) -> None:
    with pytest.raises(ManifestError, match="overlap"):
        validate_training_rows([_row(**changes)], [_row(id="dev-1", audio="shared/clip.wav")])


def test_training_rows_allow_gold_and_local_filenames() -> None:
    validate_training_rows(
        [_row(id="R-S1-0001", source="recorded_r", speaker="S1")], [_row(id="dev-1")]
    )


def test_training_rows_allow_repeated_transcripts_with_distinct_identities() -> None:
    validate_training_rows(
        [_row(id="train-1", audio="train/clip.wav")],
        [_row(id="dev-1", audio="dev/clip.wav", split="dev")],
    )


def test_training_rows_detect_dev_splice_constituents() -> None:
    with pytest.raises(ManifestError, match="overlap"):
        validate_training_rows([_row(id="it-1")], [
            _row(id="splice-1", source="spliced_it_en", audio="it-1+en-1")
        ])
