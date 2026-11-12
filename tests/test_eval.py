import json

import pytest

from stt.eval import align, macro_wer_bootstrap, main, op_counts, read_hypotheses, score_rows


def test_align_perfect_match() -> None:
    ops = align(["a", "b", "c"], ["a", "b", "c"])
    assert all(op == "match" for op, _, _ in ops)
    assert op_counts(ops) == (0, 0, 0)


def test_align_substitution() -> None:
    ops = align(["a", "b"], ["a", "x"])
    assert op_counts(ops) == (1, 0, 0)


def test_align_deletion() -> None:
    ops = align(["a", "b"], ["a"])
    assert op_counts(ops) == (0, 1, 0)


def test_align_insertion() -> None:
    ops = align(["a"], ["a", "b"])
    assert op_counts(ops) == (0, 0, 1)


def test_align_tie_prefers_substitution_over_insert_delete() -> None:
    # ref "a b" vs hyp "x": two minimum paths (sub+del or del+sub); no insertions either way.
    ops = align(["a", "b"], ["x"])
    sub, dele, ins = op_counts(ops)
    assert (sub, dele, ins) == (1, 1, 0)


def _rows() -> list[dict]:
    return [
        {
            "id": "m1",
            "text": "ho spostato il meeting a domani",
            "norm": "ho spostato il meeting a domani",
            "category": "single_word",
            "speaker": "S1",
            "lang": "mix",
            "en_spans": [[3, 4]],
        },
        {
            "id": "m2",
            "text": "serve un quick update prima della call",
            "norm": "serve un quick update prima della call",
            "category": "phrase",
            "speaker": "S3",
            "lang": "mix",
            "en_spans": [[2, 4], [6, 7]],
        },
        {
            "id": "m3",
            "text": "abbiamo deployato ieri sera",
            "norm": "abbiamo deployato ieri sera",
            "category": "italianized",
            "speaker": "S1",
            "lang": "mix",
            "en_spans": [[1, 2]],
        },
    ]


def test_score_rows_wer_and_eta() -> None:
    rows = _rows()
    hyps = {
        "m1": "ho spostato il meeting a domani",  # perfect, ETA hit
        "m2": "serve un quick update prima della call",  # perfect, 2 ETA hits
        "m3": "abbiamo deploiato ieri sera",  # 1 sub on the English term -> ETA miss
    }
    result = score_rows(rows, hyps, bootstrap=200, seed=7)
    overall = result["overall"]
    assert overall["ref_words"] == 6 + 7 + 4
    assert overall["sub"] == 1
    assert overall["del"] == 0
    assert overall["ins"] == 0
    assert overall["wer"] == pytest.approx(1 / 17)
    assert overall["eta"] == pytest.approx(4 / 5)
    assert overall["eta_errors"] == [{"id": "m3", "term": "deployato", "got": "deploiato"}]


def test_score_rows_breakdown_and_speaker_groups() -> None:
    result = score_rows(_rows(), {"m1": "x", "m2": "x", "m3": "x"}, bootstrap=0, seed=0)
    breakdown = result["breakdown"]
    assert set(breakdown["category"]) == {"single_word", "phrase", "italianized"}
    assert breakdown["speaker"]["S1"]["n"] == 2
    assert breakdown["speaker_group"]["S1+S2"]["n"] == 2
    assert breakdown["speaker_group"]["S3+S4"]["n"] == 1


def test_score_rows_noise_block() -> None:
    rows = [
        {"id": "n1", "text": "", "norm": "", "category": None, "speaker": None, "lang": "it"},
        {"id": "n2", "text": "", "norm": "", "category": "music", "speaker": None, "lang": "it"},
        {
            "id": "n3",
            "text": "",
            "norm": "",
            "category": None,
            "source": "noise_music",
            "speaker": None,
            "lang": "it",
        },
    ]
    hyps = {"n1": "", "n2": "la la la", "n3": "un due tre"}
    result = score_rows(rows, hyps, bootstrap=0, seed=0)
    noise = result["noise"]
    assert noise["non_music_words"] == 0
    assert noise["non_music_non_empty_clips"] == 0
    assert noise["music_words"] == 6
    assert noise["music_non_empty_clips"] == 2


def test_bootstrap_is_deterministic_and_ties_include_zero() -> None:
    rows = _rows()
    hyps = {"m1": "x y z", "m2": "x", "m3": "q"}
    first = score_rows(rows, hyps, bootstrap=300, seed=42)
    second = score_rows(rows, hyps, bootstrap=300, seed=42)
    assert first["overall"]["wer_ci"] == second["overall"]["wer_ci"]

    perfect = score_rows(_rows(), {r["id"]: r["norm"] for r in _rows()}, bootstrap=300, seed=42)
    lo, hi = perfect["overall"]["wer_ci"]
    assert lo == 0 == hi


def test_cli_writes_metrics_and_vs_block(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in _rows()), encoding="utf-8")

    hyp_a = tmp_path / "a.jsonl"
    hyp_a.write_text(
        "\n".join(
            json.dumps({"id": i, "hyp": h})
            for i, h in {
                "m1": "ho spostato il meeting a domani",
                "m2": "serve un quick update prima della call",
                "m3": "abbiamo deployato ieri sera",
            }.items()
        ),
        encoding="utf-8",
    )
    hyp_b = tmp_path / "b.jsonl"
    hyp_b.write_text(
        "\n".join(
            json.dumps({"id": i, "hyp": h})
            for i, h in {"m1": "x", "m2": "x", "m3": "x"}.items()
        ),
        encoding="utf-8",
    )

    out_a = tmp_path / "a.metrics.json"
    out_b = tmp_path / "b.metrics.json"
    assert main(["--manifest", str(manifest), "--hyp", str(hyp_a), "--out", str(out_a),
                 "--bootstrap", "100", "--seed", "1"]) == 0
    metrics_b = main(["--manifest", str(manifest), "--hyp", str(hyp_b), "--out", str(out_b),
                      "--bootstrap", "100", "--seed", "1", "--vs", str(out_a)])

    assert metrics_b == 0
    data_b = json.loads(out_b.read_text(encoding="utf-8"))
    assert data_b["overall"]["wer"] > 0.5
    assert data_b["vs"]["wer_diff"] > 0
    assert data_b["vs"]["wer_tie"] is False
    data_a = json.loads(out_a.read_text(encoding="utf-8"))
    assert data_a["overall"]["wer"] == 0
    assert data_a["overall"]["eta"] == 1.0
    assert data_a["manifest_sha256"]


def test_duplicate_reference_ids_rejected():
    with pytest.raises(ValueError, match="duplicate reference"):
        score_rows([{"id": "a", "norm": "a"}] * 2, {"a": "a"})


def test_duplicate_hypothesis_ids_rejected(tmp_path):
    path = tmp_path / "hyp.jsonl"
    path.write_text('{"id":"a","hyp":"a"}\n{"id":"a","hyp":"b"}\n')
    with pytest.raises(ValueError, match="duplicate hypothesis"):
        read_hypotheses(path)


def test_invalid_hypothesis_type_rejected(tmp_path):
    path = tmp_path / "hyp.jsonl"
    path.write_text('{"id":"a","hyp":null}\n')
    with pytest.raises(TypeError, match="hyp must be a string"):
        read_hypotheses(path)


@pytest.mark.parametrize("row", [
    {"id": "a", "text": "speech"}, {"id": "a", "text": "speech", "norm": ""},
    {"id": "a", "norm": 1},
    {"id": "a", "text": "speech", "norm": " "},
])
def test_speech_requires_norm(row):
    with pytest.raises(ValueError, match="requires nonempty norm"):
        score_rows([row], {"a": "speech"})


@pytest.mark.parametrize("spans", [
    [[-1, 1]], [[0, 3]], [[1, 1]], [[0, 1.0]], [[True, 2]], [[0]],
    [[0, 2], [1, 2]], [[0, 1], [0, 1]],
    1,
])
def test_invalid_or_overlapping_spans_rejected(spans):
    with pytest.raises(ValueError, match="en_spans"):
        score_rows([{"id": "a", "norm": "a b", "en_spans": spans}], {"a": "a b"})


def test_minimal_historical_references_remain_valid():
    result = score_rows([{"id": "s", "norm": "a"}, {"id": "n"}], {"s": "a", "n": ""})
    assert result["overall"]["wer"] == 0
    assert result["noise"]["clips"] == 1


def test_macro_bootstrap_stratifies_and_pairs_by_language():
    rows = [{"id": "it", "norm": "a", "lang": "it"}] + [
        {"id": f"en{i}", "norm": "b", "lang": "en"} for i in range(9)
    ]
    base = score_rows(rows, {row["id"]: row["norm"] for row in rows}, bootstrap=0)
    new = score_rows(rows, {row["id"]: "x" if row["lang"] == "it" else "b" for row in rows},
                     bootstrap=0)
    assert new["overall"]["wer"] == 0.1
    result = macro_wer_bootstrap(rows, new, base, bootstrap=100)
    assert result["point"] == 0.5
    assert result["ci"] == [0.5, 0.5]
    assert result["strata"] == {"it": 1, "en": 9}
    assert result == macro_wer_bootstrap(rows, new, base, bootstrap=100)
