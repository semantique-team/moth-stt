import hashlib
import json
from pathlib import Path

import pytest

from stt.data.freeze_lm_corpus import freeze_lm_corpus, main


def _jsonl(path: Path, *rows: dict) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _freeze(tmp_path: Path, train: Path, exclude: Path, name: str = "corpus") -> dict:
    return freeze_lm_corpus(
        train_manifests=[train],
        train_scripts=[],
        exclusions=[exclude],
        out=tmp_path / f"{name}.jsonl",
        report=tmp_path / f"{name}.json",
    )


def test_deterministic_dedupe_and_hashes(tmp_path: Path) -> None:
    train = _jsonl(
        tmp_path / "train.jsonl",
        {"text": "ignored", "norm": "Ciao, MONDO!", "lang": "it", "split": "train"},
        {"text": "ciao mondo", "lang": "mix", "split": "train"},
        {"text": "2 tickets", "lang": "en", "split": "train"},
        {"text": "!!!", "split": "train"},
    )
    script = _jsonl(tmp_path / "R.jsonl", {"text": "nuovo meeting"})
    exclude = _jsonl(tmp_path / "dev.jsonl", {"text": "test diverso", "split": "dev"})
    kwargs = {"train_manifests": [train], "train_scripts": [script], "exclusions": [exclude]}
    first = freeze_lm_corpus(
        **kwargs,
        out=tmp_path / "a.jsonl",
        report=tmp_path / "a.json",
    )
    second = freeze_lm_corpus(
        **kwargs,
        out=tmp_path / "b.jsonl",
        report=tmp_path / "b.json",
    )
    assert first == second
    content = (tmp_path / "a.jsonl").read_bytes()
    assert content == (tmp_path / "b.jsonl").read_bytes()
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert [json.loads(line) for line in content.splitlines()] == [
        {"text": "ciao mondo", "lang": "it"},
        {"text": "nuovo meeting", "lang": "mix"},
        {"text": "two tickets", "lang": "en"},
    ]
    assert first["corpus_sha256"] == hashlib.sha256(content).hexdigest()
    assert first["corpus_bytes"] == len(content)
    assert first["duplicate_train_rows"] == first["empty_train_rows"] == 1
    assert next(r for r in first["inputs"] if r["role"] == "train_manifest")["sha256"] == (
        hashlib.sha256(train.read_bytes()).hexdigest()
    )
    rows = train.read_text(encoding="utf-8").splitlines()
    train.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
    reordered = freeze_lm_corpus(
        **kwargs,
        out=tmp_path / "c.jsonl",
        report=tmp_path / "c.json",
    )
    assert reordered["corpus_sha256"] == first["corpus_sha256"]
    assert reordered["inputs"] != first["inputs"]


@pytest.mark.parametrize("use_norm", [False, True])
def test_overlap_rejected_before_outputs(tmp_path: Path, use_norm: bool) -> None:
    train = _jsonl(tmp_path / "train.jsonl", {"text": "2 tickets", "lang": "en", "split": "train"})
    row = {"text": "ignored" if use_norm else "TWO-TICKETS!", "lang": "mix"}
    if use_norm:
        row["norm"] = "two tickets"
    exclude = _jsonl(tmp_path / "T.jsonl", row)
    with pytest.raises(ValueError, match="overlaps explicit exclusions"):
        _freeze(tmp_path, train, exclude)
    assert not (tmp_path / "corpus.jsonl").exists()
    assert not (tmp_path / "corpus.json").exists()


def test_lang_hint_and_source_changes(tmp_path: Path) -> None:
    train = _jsonl(tmp_path / "train.jsonl", {"text": "3", "lang_hint": "en", "split": "train"})
    exclude = _jsonl(tmp_path / "dev.jsonl", {"text": "other"})
    first = _freeze(tmp_path, train, exclude, "a")
    assert json.loads((tmp_path / "a.jsonl").read_text())["text"] == "three"
    _jsonl(train, {"text": "4", "lang_hint": "en", "split": "train"})
    changed = _freeze(tmp_path, train, exclude, "b")
    assert first["inputs"] != changed["inputs"]
    assert first["corpus_sha256"] != changed["corpus_sha256"]
    _jsonl(exclude, {"text": "another exclusion"})
    changed_exclusion = _freeze(tmp_path, train, exclude, "c")
    assert changed["inputs"] != changed_exclusion["inputs"]
    assert changed["corpus_sha256"] == changed_exclusion["corpus_sha256"]


@pytest.mark.parametrize(
    "row",
    [
        {"text": "ciao", "split": "test"},
        {"text": "ciao", "split": "dev"},
        {"text": "ciao"},
        {"text": "ciao", "split": "train", "id": "T-0001"},
        {"text": "ciao", "split": "train", "source": "recorded_a"},
    ],
)
def test_manifest_must_be_training(tmp_path: Path, row: dict) -> None:
    train = _jsonl(tmp_path / "input.jsonl", row)
    exclude = _jsonl(tmp_path / "dev.jsonl", {"text": "other"})
    with pytest.raises(ValueError):
        _freeze(tmp_path, train, exclude)


@pytest.mark.parametrize(
    "name,row",
    [
        ("T.jsonl", {"text": "test"}),
        ("renamed.jsonl", {"text": "test", "id": "A-001"}),
        ("renamed.jsonl", {"text": "test", "split": "dev"}),
    ],
)
def test_scripts_cannot_bypass_eval_guard(tmp_path: Path, name: str, row: dict) -> None:
    script = _jsonl(tmp_path / name, row)
    exclude = _jsonl(tmp_path / "exclude.jsonl", {"text": "other"})
    with pytest.raises(ValueError):
        freeze_lm_corpus(
            train_manifests=[],
            train_scripts=[script],
            exclusions=[exclude],
            out=tmp_path / "out.jsonl",
            report=tmp_path / "out.json",
        )


def test_explicit_exclusions_and_cli(tmp_path: Path) -> None:
    train = _jsonl(tmp_path / "R.jsonl", {"text": "training"})
    exclude = _jsonl(tmp_path / "test.jsonl", {"text": "evaluation"})
    out, report = tmp_path / "out.jsonl", tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="explicit exclusion"):
        freeze_lm_corpus(
            train_manifests=[],
            train_scripts=[train],
            exclusions=[],
            out=out,
            report=report,
        )
    args = ["--train-script", str(train), "--out", str(out), "--report", str(report)]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert main(args + ["--exclude", str(exclude)]) == 0
    original = out.read_bytes()
    with pytest.raises(SystemExit):
        main(args + ["--exclude", str(exclude)])
    assert out.read_bytes() == original


@pytest.mark.parametrize(
    "row", [None, {"text": 3}, {"text": "hello", "norm": 3}, {"text": "hello", "lang_hint": "fr"}]
)
def test_invalid_rows_fail_closed(tmp_path: Path, row: dict) -> None:
    if isinstance(row, dict):
        row = {**row, "split": "train"}
    train = _jsonl(tmp_path / "train.jsonl", row)
    exclude = _jsonl(tmp_path / "dev.jsonl", {"text": "other"})
    with pytest.raises(ValueError):
        _freeze(tmp_path, train, exclude)


def test_input_order_and_repeated_paths_do_not_change_freeze(tmp_path: Path) -> None:
    a = _jsonl(tmp_path / "first.jsonl", {"text": "hello", "lang": "mix"})
    b = _jsonl(tmp_path / "second.jsonl", {"text": "HELLO!", "lang": "en"})
    exclude = _jsonl(tmp_path / "exclude.jsonl", {"text": "other"})
    reports = []
    for index, paths in enumerate(([a, b], [b, a, b])):
        reports.append(
            freeze_lm_corpus(
                train_manifests=[],
                train_scripts=paths,
                exclusions=[exclude],
                out=tmp_path / f"out{index}.jsonl",
                report=tmp_path / f"report{index}.json",
            )
        )
    assert reports[0] == reports[1]
    assert reports[0]["duplicate_train_rows"] == 1
    assert json.loads((tmp_path / "out0.jsonl").read_text()) == {"text": "hello", "lang": "en"}


def test_exclusion_lang_hint_and_empty_exclusions(tmp_path: Path) -> None:
    train = _jsonl(tmp_path / "train.jsonl", {"text": "two", "split": "train"})
    exclude = _jsonl(tmp_path / "exclude.jsonl", {"text": "2", "lang_hint": "en"})
    with pytest.raises(ValueError, match="overlaps explicit exclusions"):
        _freeze(tmp_path, train, exclude)
    _jsonl(exclude, {"text": "!!!"})
    with pytest.raises(ValueError, match="exclusions have no nonempty"):
        _freeze(tmp_path, train, exclude)


def test_outputs_cannot_replace_inputs(tmp_path: Path) -> None:
    train = _jsonl(tmp_path / "train.jsonl", {"text": "training", "split": "train"})
    exclude = _jsonl(tmp_path / "exclude.jsonl", {"text": "other"})
    original = train.read_bytes()
    with pytest.raises(ValueError, match="distinct"):
        freeze_lm_corpus(
            train_manifests=[train],
            train_scripts=[],
            exclusions=[exclude],
            out=train,
            report=tmp_path / "receipt.json",
        )
    assert train.read_bytes() == original
