import pytest

from stt.text import KEEP_CHARS, normalize

# Fixture table from the design (docs/05-evaluation.md §1) plus edge cases.
FIXTURES: list[tuple[str, str, str]] = [
    # raw, lang_hint, expected
    ("L'APPUNTAMENTO È ALLE 15:30", "it", "l'appuntamento è alle quindici trenta"),
    ("un’idea", "it", "un'idea"),
    ("mail—risposta/slack", "it", "mail risposta slack"),
    ("l' ho", "it", "l'ho"),
    ("città perché ù", "it", "città perché ù"),
    ("über ça ñ", "it", "uber ca n"),
    ("emoji 😀 ok", "it", "emoji ok"),
    ("3,5", "it", "tre virgola cinque"),
    ("3.5", "en", "three point five"),
    ("il KPI del trimestre", "it", "il kpi del trimestre"),
    ("Forty-two is the answer", "en", "forty two is the answer"),
    ("ho spostato il meeting a domani", "it", "ho spostato il meeting a domani"),
    ("Let's deploy, it's a 3.5% change", "en", "let's deploy it's a three point five change"),
    ("“Hira‘” on the mountain", "en", "hira'on the mountain"),
    ("  doppio   spazio \n", "it", "doppio spazio"),
    ("ciao\tcome\nstai\r\noggi", "it", "ciao come stai oggi"),
    ("hello\vworld\fgood\u2028morning", "en", "hello world good morning"),
]


@pytest.mark.parametrize(("raw", "lang", "expected"), FIXTURES)
def test_fixture_table(raw: str, lang: str, expected: str) -> None:
    assert normalize(raw, lang) == expected


@pytest.mark.parametrize(("raw", "lang", "_"), FIXTURES)
def test_idempotent(raw: str, lang: str, _: str) -> None:
    once = normalize(raw, lang)
    assert normalize(once, lang) == once


@pytest.mark.parametrize(("raw", "lang", "_"), FIXTURES)
def test_output_uses_only_the_charset(raw: str, lang: str, _: str) -> None:
    out = normalize(raw, lang)
    assert set(out) <= KEEP_CHARS | {" "}


def test_rejects_unknown_language() -> None:
    with pytest.raises(ValueError):
        normalize("ciao", "fr")
