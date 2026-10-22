from pathlib import Path

from stt.hotwords import HotwordIndex, load_hotwords


def test_index_matches_terminals_prefixes_and_suffixes() -> None:
    index = HotwordIndex([(1, 2), (3,)])

    assert index.max_len == 2
    assert index.match((1, 2)) == (True, False)
    assert index.match((1,)) == (False, True)
    assert index.match((9, 1)) == (False, True)
    assert index.match((9, 1, 2)) == (True, False)
    assert index.match((3,)) == (True, False)
    assert index.match((9,)) == (False, False)


def test_terminal_can_also_be_a_prefix() -> None:
    index = HotwordIndex([(1,), (1, 2)])

    assert index.match((1,)) == (True, True)
    assert index.match((1, 2)) == (True, False)


def test_load_hotwords_ignores_comments_and_normalizes(tmp_path: Path) -> None:
    path = tmp_path / "terms.txt"
    path.write_text("# comment\n\nMeeting\ncall di\n", encoding="utf-8")
    assert load_hotwords(path) == ["meeting", "call di"]
