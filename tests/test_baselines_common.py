import json
import os
import shlex
import sys
from pathlib import Path

import pytest

from baselines.common import (
    dir_size,
    external_transcriber,
    fair_hotwords,
    hotwords_for,
    oracle_hotwords,
    read_terms,
    render_command,
    resolve_audio,
    rtfx,
    run_set,
    run_sets,
    terms_in_rows,
    write_hypotheses,
)


def _row(uid: str, audio: str, duration: float = 2.0, **extra) -> dict:
    row = {
        "id": uid,
        "audio": audio,
        "text": "ciao come stai",
        "norm": "ciao come stai",
        "duration": duration,
        "source": "test",
        "split": "test",
        "speaker": "S1",
        "lang": "it",
        "category": None,
        "en_spans": [],
    }
    row.update(extra)
    return row


def test_run_set_collects_hypotheses_latency_and_audio_seconds() -> None:
    rows = [_row("a", "a.wav", 2.0), _row("b", "b.wav", 3.0)]
    result = run_set(lambda path: f"testo {path.name}", rows, Path("/audio"))
    assert [hyp["id"] for hyp in result["hypotheses"]] == ["a", "b"]
    assert result["hypotheses"][0]["hyp"] == "testo a.wav"
    assert all(hyp["latency_s"] >= 0 for hyp in result["hypotheses"])
    assert result["audio_seconds"] == 5.0
    assert result["wall_seconds"] >= 0
    assert result["errors"] == []


def test_run_set_records_errors_and_continues() -> None:
    def transcribe(path: Path) -> str:
        if path.name == "bad.wav":
            raise RuntimeError("boom")
        return "ok"

    rows = [_row("bad", "bad.wav"), _row("good", "good.wav")]
    result = run_set(transcribe, rows, Path("/audio"))
    assert result["errors"] == [{"id": "bad", "error": "boom"}]
    assert result["hypotheses"][0]["hyp"] == ""
    assert result["hypotheses"][1]["hyp"] == "ok"


def test_run_set_respects_limit() -> None:
    rows = [_row(str(index), f"{index}.wav") for index in range(5)]
    result = run_set(lambda path: "x", rows, Path("/audio"), limit=2)
    assert len(result["hypotheses"]) == 2


def test_write_hypotheses_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "hyp.jsonl"
    write_hypotheses(path, [{"id": "a", "hyp": "ciao", "latency_s": 0.1}])
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row == {"id": "a", "hyp": "ciao", "latency_s": 0.1}


def test_rtfx_and_zero_guard() -> None:
    assert rtfx(10.0, 2.0) == 5.0
    assert rtfx(10.0, 0.0) is None


def test_render_command_substitutes_values() -> None:
    command = render_command(
        "needle transcribe --language {lang} --keywords {keywords} {audio}",
        lang="it",
        keywords_path="/tmp/k.txt",
    )
    quoted = '"/tmp/k.txt"' if os.name == "nt" else "/tmp/k.txt"
    assert command == f"needle transcribe --language it --keywords {quoted} {{audio}}"


@pytest.mark.parametrize("quote", ["", '"', "'"])
def test_render_command_drops_absent_flags(quote: str) -> None:
    command = render_command(
        f"needle transcribe --language {quote}{{lang}}{quote} "
        f"--keywords={quote}{{keywords}}{quote} {{audio}}", lang=""
    )
    assert command == "needle transcribe {audio}"


def test_render_command_keeps_other_placeholders() -> None:
    command = render_command("cli --lang={lang} {audio}", lang="it")
    assert command == "cli --lang=it {audio}"


def test_dir_size_sums_existing_files(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "b.bin").write_bytes(b"y" * 20)
    assert dir_size([tmp_path / "a.bin", tmp_path / "b.bin", tmp_path / "missing.bin"]) == 30


def test_resolve_audio_relative_and_absolute(tmp_path: Path) -> None:
    assert resolve_audio(_row("a", "S1/x.wav"), tmp_path) == tmp_path / "S1/x.wav"
    absolute = tmp_path / "abs.wav"
    assert resolve_audio(_row("a", str(absolute)), Path("/ignored")) == absolute


def _scripts(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    rows = [
        _row("R-S1-0001", "a.wav", en_spans=[[0, 1]], text="meeting", norm="meeting"),
        _row("R-S1-0002", "b.wav", en_spans=[[0, 1]], text="deployato", norm="deployato"),
    ]
    (scripts / "R-S1.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    test_row = _row("T-0001", "c.wav", en_spans=[[0, 1]], text="kpi", norm="kpi")
    (scripts / "T.jsonl").write_text(json.dumps(test_row) + "\n")
    return scripts


def test_fair_hotwords_uses_r_scripts_and_terms_not_test(tmp_path: Path) -> None:
    scripts = _scripts(tmp_path)
    terms = tmp_path / "terms.txt"
    terms.write_text("sprint\n# comment\n\nrouter\n", encoding="utf-8")
    words = fair_hotwords(scripts, terms)
    assert {"meeting", "deployato", "sprint", "router"} <= set(words)
    assert "kpi" not in words
    assert read_terms(terms) == ["sprint", "router"]


def test_terms_in_rows_and_oracle_hotwords(tmp_path: Path) -> None:
    rows = [_row("T-1", "a.wav", en_spans=[[0, 2]], text="quick update", norm="quick update")]
    assert terms_in_rows(rows) == ["quick", "update"]
    fair = ["meeting"]
    oracle = oracle_hotwords(rows, fair)
    assert set(fair) <= set(oracle)
    assert {"quick", "update"} <= set(oracle)
    words, is_oracle = hotwords_for("oracle", rows, tmp_path / "none", tmp_path / "none.txt")
    assert is_oracle is True and "quick" in words
    assert hotwords_for("none", rows, tmp_path / "none", tmp_path / "none.txt") == ([], False)


def _shell_path(path: Path | str) -> str:
    return f'"{path}"' if os.name == "nt" else shlex.quote(str(path))


@pytest.mark.parametrize("placeholder", [" {audio}", ' "{audio}"', " '{audio}'", ""])
@pytest.mark.parametrize("name", ["clip & take's (1).wav", "clip&take.wav"])
def test_external_transcriber_success_and_audio_substitution(
    tmp_path: Path, placeholder: str, name: str
) -> None:
    script = tmp_path / "echo audio.py"
    script.write_text("import sys\nassert len(sys.argv) == 2\nprint(sys.argv[1])\n", encoding="utf-8")
    transcribe = external_transcriber(
        f"{_shell_path(sys.executable)} {_shell_path(script)}{placeholder}"
    )
    audio = tmp_path / name
    assert transcribe(audio) == str(audio)


@pytest.mark.parametrize("quote", ["", '"', "'"])
@pytest.mark.parametrize("name", ["hot words.txt", "hot&words.txt"])
def test_render_command_quotes_keyword_paths_with_spaces(
    tmp_path: Path, quote: str, name: str
) -> None:
    script = tmp_path / "keywords reader.py"
    script.write_text("import sys\nassert len(sys.argv) == 3\nprint(sys.argv[1])\n")
    keywords = tmp_path / name
    command = render_command(
        f"{_shell_path(sys.executable)} {_shell_path(script)} "
        f"{quote}{{keywords}}{quote} {quote}{{audio}}{quote}",
        keywords_path=keywords,
    )
    assert external_transcriber(command)(tmp_path / "audio file.wav") == str(keywords)


@pytest.mark.skipif(os.name != "nt", reason="cmd-specific expansion")
@pytest.mark.parametrize("name", ["%PATH%.wav", "!PATH!.wav", 'bad"name.wav', "bad\nname.wav"])
def test_external_transcriber_rejects_cmd_expansion_in_audio(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        external_transcriber("unused {audio}")(Path(name))


def test_external_transcriber_takes_last_non_empty_line(tmp_path: Path) -> None:
    script = tmp_path / "banner.py"
    script.write_text("print('loading model...')\nprint('')\nprint('ciao come stai')\n")
    transcribe = external_transcriber(f"{_shell_path(sys.executable)} {_shell_path(script)}")
    assert transcribe(Path("x.wav")) == "ciao come stai"


def test_external_transcriber_failure_is_actionable() -> None:
    transcribe = external_transcriber(f'{_shell_path(sys.executable)} -c "import sys; sys.exit(3)"')
    with pytest.raises(RuntimeError, match="exit 3"):
        transcribe(Path("x.wav"))


def test_run_sets_writes_hypotheses_and_meta(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "mixed.jsonl").write_text(
        json.dumps(_row("a", "a.wav")) + "\n" + json.dumps(_row("b", "b.wav")) + "\n"
    )
    out = tmp_path / "run"

    def build_transcriber(hotwords: list[str]):
        assert hotwords == []
        return lambda path: "ciao"

    code = run_sets(
        system="fake",
        variant="v1",
        sets=["mixed"],
        manifests_dir=manifests,
        audio_root=tmp_path / "audio",
        out_dir=out,
        build_transcriber=build_transcriber,
        hotwords_mode="none",
        scripts_dir=tmp_path / "scripts",
        terms_path=tmp_path / "terms.txt",
        meta_extra={"padded_30s": False},
    )
    assert code == 0
    hyp = [json.loads(line) for line in (out / "mixed.jsonl").read_text().splitlines()]
    assert [row["id"] for row in hyp] == ["a", "b"]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["system"] == "fake" and meta["variant"] == "v1"
    assert meta["sets"]["mixed"]["n"] == 2
    assert meta["sets"]["mixed"]["rtfx"] is not None
    assert meta["padded_30s"] is False


def test_run_sets_missing_manifest_is_an_error(tmp_path: Path) -> None:
    code = run_sets(
        system="fake",
        variant="v1",
        sets=["nope"],
        manifests_dir=tmp_path,
        audio_root=tmp_path,
        out_dir=tmp_path / "run",
        build_transcriber=lambda hotwords: (lambda path: ""),
        hotwords_mode="none",
        scripts_dir=tmp_path,
        terms_path=tmp_path / "terms.txt",
        meta_extra={},
    )
    assert code == 1
