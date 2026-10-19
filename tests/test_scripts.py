import json
from pathlib import Path

import pytest

from recordings.generate_scripts import (
    CATEGORIES,
    QUOTAS_T,
    ScriptError,
    evaluate_stats,
    import_candidates,
    load_script,
    main,
    mark_reviewed,
    review_report,
    stratify_speakers,
    validate_candidate,
)

# A 12-word sentence: changing its first word leaves Jaccard(3-grams) = 9/11 ≈ 0.82 > 0.8.
LONG_A = "oggi parliamo del progetto domani mattina e poi andiamo insieme a prendere un caffe"
LONG_B = "domani parliamo del progetto domani mattina e poi andiamo insieme a prendere un caffe"


def _candidate(text: str, category: str = "phrase", en_spans: list | None = None) -> dict:
    return {
        "text": text,
        "category": category,
        "en_spans": en_spans if en_spans is not None else [],
        "generator": "test",
        "date": "2026-10-19",
    }


def test_accepts_a_valid_candidate() -> None:
    row = validate_candidate(
        _candidate("ho spostato il meeting a domani", "single_word", [[3, 4]])
    )
    assert row["text"] == "ho spostato il meeting a domani"
    assert row["en_spans"] == [[3, 4]]


def test_candidate_must_be_normalized() -> None:
    with pytest.raises(ScriptError, match="normali"):
        validate_candidate(_candidate("Ho spostato il meeting.", "single_word", [[3, 4]]))


def test_bad_category_rejected() -> None:
    with pytest.raises(ScriptError, match="category"):
        validate_candidate(_candidate("il meeting", "meeting", [[1, 2]]))


def test_control_it_rejects_en_spans() -> None:
    with pytest.raises(ScriptError, match="control_it"):
        validate_candidate(_candidate("oggi piove forte", "control_it", [[1, 2]]))


def test_single_word_requires_exactly_one_span_of_one_word() -> None:
    with pytest.raises(ScriptError, match="single_word"):
        validate_candidate(_candidate("il quick update", "single_word", [[1, 3]]))


def test_phrase_requires_at_least_two_english_words() -> None:
    with pytest.raises(ScriptError, match="phrase"):
        validate_candidate(_candidate("il meeting oggi", "phrase", [[1, 2]]))


def test_en_spans_bounds_checked() -> None:
    with pytest.raises(ScriptError, match="en_spans"):
        validate_candidate(_candidate("ho spostato il meeting", "single_word", [[3, 9]]))


def test_exact_duplicate_skipped(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    row = _candidate("ho spostato il meeting a domani", "single_word", [[3, 4]])
    first = import_candidates([row], "T", scripts)
    second = import_candidates([row], "T", scripts)
    assert first["imported"] == 1
    assert second["imported"] == 0
    assert second["duplicates"] == 1
    assert len(load_script(scripts / "T.jsonl")) == 1


def test_near_duplicate_skipped_far_text_kept(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    import_candidates([_candidate(LONG_A, "control_it")], "T", scripts)
    near = import_candidates([_candidate(LONG_B, "control_it")], "T", scripts)
    far = import_candidates(
        [_candidate("oggi leggo un libro sul divano", "control_it")], "T", scripts
    )
    assert near["duplicates"] == 1
    assert far["imported"] == 1


def test_stratified_assignment_balances_speakers() -> None:
    rows = [{"category": "single_word"} for _ in range(150)]
    rows += [{"category": "phrase"} for _ in range(100)]
    rows += [{"category": "acronym"} for _ in range(60)]
    stratify_speakers(rows)
    counts = {speaker: 0 for speaker in ("S1", "S2", "S3")}
    for row in rows:
        counts[row["speaker"]] += 1
    assert sum(counts.values()) == 310
    assert max(counts.values()) - min(counts.values()) <= 1


def test_stats_reports_quota_deltas(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    import_candidates(
        [_candidate("ho spostato il meeting a domani", "single_word", [[3, 4]])], "T", scripts
    )
    import_candidates([_candidate("oggi leggo un libro", "control_it")], "pool", scripts)
    stats = evaluate_stats(scripts)
    assert stats["T"]["total"] == 1
    assert stats["T"]["categories"]["single_word"]["actual"] == 1
    assert stats["T"]["categories"]["single_word"]["quota"] == QUOTAS_T["single_word"]
    assert stats["T"]["categories"]["single_word"]["delta"] == QUOTAS_T["single_word"] - 1
    assert stats["pool"]["count"] == 1
    assert set(stats["T"]["categories"]) == set(CATEGORIES)


def test_review_report_and_mark_reviewed(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    import_candidates(
        [_candidate("ho spostato il meeting a domani", "single_word", [[3, 4]])], "T", scripts
    )
    rows = load_script(scripts / "T.jsonl")
    report = review_report(rows)
    assert report["total"] == 1
    assert report["reviewed"] == 0
    assert len(report["unreviewed"]) == 1
    assert mark_reviewed(rows, "tancredi") == 1
    assert rows[0]["reviewed"] is True
    assert rows[0]["reviewer"] == "tancredi"
    assert mark_reviewed(rows, "tancredi") == 0


def test_import_cli_roundtrip(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(_candidate("ho spostato il meeting a domani", "single_word", [[3, 4]])),
        encoding="utf-8",
    )
    assert main(["import", "--kind", "T", "--in", str(candidates), "--scripts-dir", str(scripts)]) == 0
    rows = load_script(scripts / "T.jsonl")
    assert rows[0]["id"] == "T-0001"
    assert rows[0]["speaker"] in ("S1", "S2", "S3")
    assert rows[0]["reviewed"] is False
    assert main(["stats", "--scripts-dir", str(scripts)]) == 0
