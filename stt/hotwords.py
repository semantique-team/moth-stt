"""Decode-time hotword lists for CTC beam search.

Hotwords are stored as CTC token sequences, not strings: the tokenizer decides how a term is
spelled, and matching a suffix of the beam prefix keeps the check cheap. A term can start
anywhere in the utterance; the first piece usually carries the SentencePiece ``▁`` marker, so
matches naturally fall on word boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

from stt.text import normalize

__all__ = ["ContextualBias", "HotwordIndex", "load_hotwords"]


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


class ContextualBias:
    """Experimental bounded single-word bias, separate from historical hotwords.

    Sequences contain CTC IDs (SentencePiece IDs + 1), with an initial word marker.
    ``delta`` is paid only for an emitted label, never a repeated CTC frame. Prefix
    potential is replaced, not accumulated; ``finish`` removes it at EOS and pays
    the completion bonus only for an exact word. No state survives a call.
    """

    def __init__(
        self,
        sequences: Iterable[Sequence[int]],
        pieces: Mapping[int, str] | Sequence[str],
        *,
        bonus: float = 1.0,
        prefix_bonus: float = 0.0,
    ) -> None:
        if not math.isfinite(bonus) or not math.isfinite(prefix_bonus):
            raise ValueError("contextual bonuses must be finite")
        if not 0.0 <= prefix_bonus <= bonus:
            raise ValueError("require 0 <= prefix_bonus <= bonus")
        self.bonus = bonus
        self.prefix_bonus = prefix_bonus
        source = pieces.items() if isinstance(pieces, Mapping) else enumerate(pieces)
        self._pieces = MappingProxyType({i: piece for i, piece in source if i != 0})
        self.boundaries = frozenset(
            identifier
            for identifier in (pieces if isinstance(pieces, Mapping) else range(len(pieces)))
            if pieces[identifier].startswith("\u2581")
        )
        validated = []
        for sequence in sequences:
            ids = tuple(sequence)
            if not ids or any(type(i) is not int or i <= 0 for i in ids):
                raise ValueError("contextual sequences require nonblank integer CTC IDs")
            try:
                spellings = [pieces[i] for i in ids]
            except (KeyError, IndexError) as error:
                raise ValueError("unknown contextual CTC ID") from error
            spelling = "".join(spellings)
            word = spelling[1:]
            if (
                ids[0] not in self.boundaries
                or "\u2581" in word
                or not all(spellings)
                or not word
                or " " in word
                or normalize(word) != word
            ):
                raise ValueError(
                    "contextual terms must be normalized single words with a start marker"
                )
            validated.append(ids)
        index = HotwordIndex(validated)
        self.terminals = index.terminals
        self.prefixes = index.prefixes
        self.max_len = index.max_len

    @property
    def pieces(self) -> Mapping[int, str]:
        """Immutable vocabulary snapshot for runtime compatibility checks (blank omitted)."""
        return self._pieces

    @classmethod
    def from_terms(
        cls,
        terms: Iterable[str],
        pieces: Mapping[int, str] | Sequence[str],
        encode: Callable[[str], Sequence[int]],
        **kwargs: float,
    ) -> ContextualBias:
        """Normalize terms with the canonical rule; caller's encoder returns CTC IDs.

        Use the frozen tokenizer, not a greedy reconstruction from ``vocab.json``.
        An unknown token or a non-roundtripping encoding is rejected.
        """
        sequences = []
        for term in terms:
            word = normalize(term)
            if not word or " " in word:
                raise ValueError("contextual terms must normalize to a single word")
            ids = tuple(encode(word))
            for i in ids:
                if type(i) is not int or i <= 0:
                    raise ValueError("unknown contextual CTC ID")
                known = i in pieces if isinstance(pieces, Mapping) else 0 <= i < len(pieces)
                if not known:
                    raise ValueError("unknown contextual CTC ID")
            if "".join(pieces[i] for i in ids) != "\u2581" + word:
                raise ValueError("contextual encoding must exactly spell the normalized term")
            sequences.append(ids)
        return cls(sequences, pieces, **kwargs)

    def _word(self, tokens: Sequence[int]) -> tuple[int, ...]:
        # A longer current word cannot match; never scan utterance-length history.
        for size in range(1, min(len(tokens), self.max_len) + 1):
            if tokens[-size] in self.boundaries:
                return tuple(tokens[-size:])
        return ()

    def _potential(self, word: tuple[int, ...]) -> float:
        return self.prefix_bonus if word in self.terminals or word in self.prefixes else 0.0

    def delta(self, prefix: Sequence[int], token: int) -> float:
        """Score change for appending one label to an already CTC-collapsed prefix."""
        word = self._word(prefix)
        boundary = token in self.boundaries
        new_word = (token,) if boundary else word + (token,) if word else ()
        completion = self.bonus if boundary and word in self.terminals else 0.0
        return completion + self._potential(new_word) - self._potential(word)

    def finish(self, prefix: Sequence[int]) -> float:
        """EOS delta: confirm an exact final word or refund its pending potential."""
        word = self._word(prefix)
        return (self.bonus if word in self.terminals else 0.0) - self._potential(word)


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
