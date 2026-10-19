"""The one text rule (design: ``docs/05-evaluation.md`` §1).

``normalize()`` is the only place that decides what counts as the same text. It runs on
everything *before comparison*: tokenizer training corpora, training targets, reference
transcripts and every system's hypotheses.

Rules, in order:

1. Unicode NFKC; lowercase.
2. Typographic apostrophes (``’ ‘ ´``) fold to ``'``; whitespace after an apostrophe is removed
   (``l' ho`` → ``l'ho``).
3. Time-like colons between digits become spaces (``15:30`` → ``15 30``) so the number words do
   not merge.
4. Digit runs become words via ``num2words`` in ``lang_hint`` (``15:30`` → ``quindici trenta``).
5. Hyphens, dashes and slashes become spaces so words never merge.
6. Keep ``a-z``, the Italian accented vowels (``à è é ì í ò ó ù ú``) and ``'``; fold other Latin
   diacritics to their ASCII base (``ü``→``u``, ``ç``→``c``, ``ñ``→``n``); drop everything else
   (punctuation, symbols, emoji).
7. Collapse whitespace; strip.

Known limitation, documented in the design: a number in an English context is spelled with the
``lang_hint`` language. Training drops digit transcripts and test scripts spell numbers as words,
so this path mainly exists to compare third-party hypotheses.
"""

from __future__ import annotations

import re
import unicodedata

from num2words import num2words

__all__ = ["KEEP_CHARS", "normalize"]

_LANGS = frozenset({"it", "en"})

#: Every character the normalized text may contain (plus space).
KEEP_CHARS: frozenset[str] = frozenset("abcdefghijklmnopqrstuvwxyzàèéìíòóùú'")

_ASCII_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz")

# Typographic apostrophes that fold to "'".
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u00b4": "'"})

# Hyphens, dashes and slashes become spaces so words never merge.
_SEPARATOR_CHARS = (
    "\u002d"  # -
    "\u2010"  # ‐
    "\u2011"  # ‑
    "\u2012"  # ‒
    "\u2013"  # –
    "\u2014"  # —
    "\u2015"  # ―
    "\u2212"  # −
    "\u2044"  # ⁄
    "\u2215"  # ∕
    "\u002f"  # /
)
_SEPARATORS = re.compile(f"[{re.escape(_SEPARATOR_CHARS)}]")

_TIME_COLON = re.compile(r"(?<=\d):(?=\d)")
_AFTER_APOSTROPHE_SPACE = re.compile(r"'\s+")
_NUMBERS = re.compile(r"\d+(?:[.,]\d+)?")


def normalize(text: str, lang_hint: str = "it") -> str:
    """Normalize ``text`` to the project's canonical form.

    ``lang_hint`` (``"it"`` or ``"en"``) selects the language used to spell out digits.
    """
    if lang_hint not in _LANGS:
        raise ValueError(f"unsupported lang_hint: {lang_hint!r} (expected 'it' or 'en')")

    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(_APOSTROPHES)
    text = _AFTER_APOSTROPHE_SPACE.sub("'", text)
    text = _TIME_COLON.sub(" ", text)
    text = _NUMBERS.sub(lambda match: _number_to_words(match.group(0), lang_hint), text)
    text = _SEPARATORS.sub(" ", text)
    text = "".join(_fold(char) for char in text)
    return " ".join(text.split())


def _number_to_words(raw: str, lang: str) -> str:
    if "." in raw or "," in raw:
        return num2words(float(raw.replace(",", ".")), lang=lang)
    return num2words(int(raw), lang=lang)


def _fold(char: str) -> str:
    """Keep ``char`` if allowed, fold diacritics to ASCII, otherwise drop it."""
    if char == " " or char in KEEP_CHARS:
        return char
    decomposed = unicodedata.normalize("NFKD", char)
    base = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    if base and all(c in _ASCII_LETTERS for c in base):
        return base
    return ""
