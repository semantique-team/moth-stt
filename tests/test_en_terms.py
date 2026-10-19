from pathlib import Path

from baselines.common import read_terms
from stt.text import normalize

TERMS_PATH = Path(__file__).parents[1] / "data" / "en_terms.txt"


def test_en_terms_list_exists_and_is_normalized() -> None:
    terms = read_terms(TERMS_PATH)
    assert len(terms) >= 300
    assert all(normalize(term) == term for term in terms)


def test_en_terms_list_has_no_duplicates() -> None:
    lines = [
        line.strip()
        for line in TERMS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) == len(set(lines))
