"""Decode-time hotword lists for CTC beam search.

Hotwords are stored as CTC token sequences, not strings: the tokenizer decides how a term is
spelled, and matching a suffix of the beam prefix keeps the check cheap. A term can start
anywhere in the utterance; the first piece usually carries the SentencePiece ``▁`` marker, so
matches naturally fall on word boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from stt.text import normalize

__all__ = ["HotwordIndex", "load_hotwords"]


class HotwordIndex:
    """Suffix matcher for hotword token sequences.

    ``terminals`` are complete hotword sequences; ``prefixes`` are their proper prefixes. The
    decoder asks :meth:`match` about the prefix it just extended: a complete match earns the
    full bonus, a partial match can keep a hypothesis alive with a smaller bonus.
    """

    def __init__(self, sequences: Iterable[Sequence[int]]) -> None:
        terminals = {
            tuple(int(token) for token in sequence) for sequence in sequences if len(sequence) > 0
        }
        prefixes: set[tuple[int, ...]] = set()
        for sequence in terminals:
            for size in range(1, len(sequence)):
                prefixes.add(sequence[:size])

        self.terminals = frozenset(terminals)
        self.prefixes = frozenset(prefixes)
        self.max_len = max((len(sequence) for sequence in terminals), default=0)
        self._terminal_last = frozenset(sequence[-1] for sequence in terminals)
        self._prefix_last = frozenset(sequence[-1] for sequence in prefixes)

    def match(self, tokens: Sequence[int]) -> tuple[bool, bool]:
        """Return ``(terminal, prefix)`` for the suffixes of ``tokens``.

        ``terminal`` is true when the last token completes a hotword (possibly a multi-piece
        one); ``prefix`` is true when the suffix is a proper prefix of a longer hotword. Both
        can be true when one hotword is a prefix of another.
        """
        if not tokens:
            return False, False
        last = int(tokens[-1])
        if last not in self._terminal_last and last not in self._prefix_last:
            return False, False

        terminal = False
        prefix = False
        for size in range(min(self.max_len, len(tokens)), 0, -1):
            suffix = tuple(tokens[-size:])
            if suffix in self.terminals:
                terminal = True
            if suffix in self.prefixes:
                prefix = True
            if terminal and prefix:
                break
        return terminal, prefix


def load_hotwords(path: Path | str) -> list[str]:
    """Read one normalized hotword per line; blanks and ``#`` comments are ignored."""
    terms: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        term = normalize(line)
        if term:
            terms.append(term)
    return terms
