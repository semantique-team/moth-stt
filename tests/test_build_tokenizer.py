import json
import random
from pathlib import Path

import pytest

from stt.data.build_tokenizer import (
    build_corpus,
    main,
    merge_report,
    oversample_to_words,
    read_pool,
    read_terms,
    read_transcripts,
)
from stt.manifest import ManifestError
from stt.tokenizer import load_tokenizer


def _manifest(path: Path, rows: list[tuple[str, float]], source: str, lang: str) -> Path:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"{source}-{index:04d}",
                    "audio": f"{source}/{index:04d}.flac",
                    "text": text,
                    "norm": text,
                    "duration": duration,
                    "source": source,
                    "split": "train",
                    "speaker": "1",
                    "lang": lang,
                }
            )
            for index, (text, duration) in enumerate(rows)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_oversample_is_deterministic_and_hits_the_target() -> None:
    texts = [f"frase numero {index} con qualche parola in piu" for index in range(50)]
    first, added = oversample_to_words(texts, target_words=500, seed=0)
    second, again = oversample_to_words(texts, target_words=500, seed=0)
    assert first == second and added == again
    assert added > 0
    assert sum(len(text.split()) for text in first) >= 500
    # no line is dropped: the original order stays
    assert first[: len(texts)] == texts
    assert oversample_to_words(texts, target_words=10, seed=0)[1] == 0


def test_build_corpus_reaches_the_italian_share() -> None:
    it = ["una frase italiana semplice"] * 10
    en = ["a short english sentence here"] * 10
    pool = ["ho spostato il meeting a domani"]
    lines, stats = build_corpus(it, en, pool, it_share=0.7, seed=0)
    assert abs(stats["it_share_words"] - 0.7) < 0.05
    assert lines[-1] == pool[0]
    assert stats["pool_lines"] == 1
    assert len(lines) == len(it) + stats["it_lines_duplicated"] + len(en) + len(pool)


def test_merge_report_keeps_other_keys(tmp_path: Path) -> None:
    report = tmp_path / "build_report.json"
    report.write_text(json.dumps({"sets": {"x": {"kept": 3}}, "seed": 0}), encoding="utf-8")
    merged = merge_report(report, {"vocab_size": 64})
    assert merged["sets"] == {"x": {"kept": 3}}
    assert merged["seed"] == 0
    assert json.loads(report.read_text(encoding="utf-8"))["tokenizer"] == {"vocab_size": 64}


def test_read_terms_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "terms.txt"
    path.write_text("meeting\n\n# comment\ndeploy\n", encoding="utf-8")
    assert read_terms(path) == ["meeting", "deploy"]


def test_main_builds_a_tokenizer_and_reports(tmp_path: Path) -> None:
    rng = random.Random(0)
    words = ["il", "meeting", "è", "pronto", "con", "deploy", "di", "domani", "kpi", "ticket"]
    it = [(" ".join(rng.choices(words, k=6)), 3.0) for _ in range(40)]
    en = [(" ".join(rng.choices(words, k=6)), 4.0) for _ in range(20)]
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        "\n".join(
            json.dumps({"text": "ho spostato il meeting a domani", "category": "single_word"})
            for _ in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    terms = tmp_path / "terms.txt"
    terms.write_text("meeting\ndeploy\nkpi\n", encoding="utf-8")
    report = tmp_path / "build_report.json"
    report.write_text(json.dumps({"sets": {"mls_it_train": {"kept": 40953}}, "seed": 0}), encoding="utf-8")
    out = tmp_path / "moth_bpe"

    code = main(
        [
            "--it", str(_manifest(tmp_path / "it.jsonl", it, "mls_it", "it")),
            "--en", str(_manifest(tmp_path / "en.jsonl", en, "ls_en", "en")),
            "--pool", str(pool),
            "--terms", str(terms),
            "--out", str(out),
            "--vocab-size", "64",
            "--report", str(report),
        ]
    )

    assert code == 0
    processor = load_tokenizer(out.with_suffix(".model"))
    assert processor.vocab_size() == 64
    vocab = json.loads(out.with_suffix(".vocab.json").read_text(encoding="utf-8"))
    assert sorted(int(key) for key in vocab) == list(range(1, 65))
    section = json.loads(report.read_text(encoding="utf-8"))["tokenizer"]
    assert json.loads(report.read_text(encoding="utf-8"))["sets"] == {"mls_it_train": {"kept": 40953}}
    assert section["vocab_size"] == 64
    assert section["corpus"]["pool_lines"] == 3
    assert section["stats"]["tokens_per_second"] > 0
    assert section["stats"]["terms"] == 3
    assert section["sha256"][str(out.with_suffix(".model"))]


@pytest.mark.parametrize("changes", [
    {"split": "test"}, {"split": "dev"}, {"source": "recorded_t"},
    {"source": "recorded_a"}, {"speaker": "S3"},
])
def test_read_transcripts_rejects_held_out_rows(tmp_path: Path, changes: dict) -> None:
    path = _manifest(tmp_path / "m.jsonl", [("ciao", 1.0)], "mls_it", "it")
    row = json.loads(path.read_text(encoding="utf-8")) | changes
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ManifestError, match="cannot train"):
        read_transcripts(path)


@pytest.mark.parametrize("changes", [
    {"kind": "T"}, {"kind": "A"}, {"kind": "unknown"}, {"id": "T-0001"},
    {"id": "A-0001"}, {"speaker": "S3"}, {"split": "dev"},
    {"source": "recorded_t"},
])
def test_read_pool_rejects_held_out_scripts(tmp_path: Path, changes: dict) -> None:
    path = tmp_path / "renamed.jsonl"
    path.write_text(json.dumps({"text": "ciao"} | changes), encoding="utf-8")
    with pytest.raises(ManifestError, match="training-side"):
        read_pool(path)


@pytest.mark.parametrize("kind", [None, "pool", "R"])
def test_read_pool_normalizes_valid_training_text(tmp_path: Path, kind: str | None) -> None:
    path = tmp_path / "pool.jsonl"
    row = {"text": "HELLO\tworld!"}
    if kind is not None:
        row["kind"] = kind
    path.write_text(json.dumps(row), encoding="utf-8")
    assert read_pool(path) == ["hello world"]


def test_read_pool_rejects_anchor_filename_without_metadata(tmp_path: Path) -> None:
    path = tmp_path / "A.jsonl"
    path.write_text(json.dumps({"text": "ciao"}), encoding="utf-8")
    with pytest.raises(ManifestError, match="T/A"):
        read_pool(path)


def test_existing_pool_script_remains_valid() -> None:
    pool = read_pool(Path(__file__).parents[1] / "recordings/scripts/pool.jsonl")
    assert len(pool) > 3000
