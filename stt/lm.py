"""Small BPE n-gram language model for shallow fusion at decode time.

The LM is trained on the *training* transcripts only, over the same 1024-piece BPE vocabulary as
the CTC model. A trigram model with stupid backoff is small enough to stay near the edge budget
and fast enough for CPU/browser decoding. ``BOS_ID``/``EOS_ID`` are reserved LM tokens; CTC ids
are shifted by one (``ctc_id - 1``) before scoring.
"""

from __future__ import annotations

import gzip
import math
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

__all__ = ["BOS_ID", "EOS_ID", "LM_VOCAB_SIZE", "BPETrigramLM"]

#: SentencePiece vocabulary is 1024; the two ids after it are LM-only.
BOS_ID = 1024
EOS_ID = 1025
LM_VOCAB_SIZE = 1026


class BPETrigramLM:
    """Trigram LM over BPE ids with stupid backoff.

    ``score(prev2, prev1, token)`` returns a log probability. Counts are pruned at build time
    (rare bigrams/trigrams fall back), while the unigram distribution keeps add-one smoothing.
    """

    def __init__(
        self,
        unigram: Mapping[int, int] | np.ndarray,
        bigram: Mapping[int, Mapping[int, int]],
        trigram: Mapping[tuple[int, int], Mapping[int, int]],
        backoff: float = 0.4,
    ) -> None:
        if backoff <= 0.0:
            raise ValueError("backoff must be positive")

        if isinstance(unigram, np.ndarray):
            counts = np.zeros(LM_VOCAB_SIZE, dtype=np.float64)
            counts[: min(len(unigram), LM_VOCAB_SIZE)] = unigram[:LM_VOCAB_SIZE]
        else:
            counts = np.zeros(LM_VOCAB_SIZE, dtype=np.float64)
            for token, count in unigram.items():
                token = int(token)
                if 0 <= token < LM_VOCAB_SIZE:
                    counts[token] = float(count)
        self._unigram_counts = counts
        self._bigram_counts = {
            int(prev): {int(token): int(count) for token, count in row.items() if count > 0}
            for prev, row in bigram.items()
        }
        self._trigram_counts = {
            (int(prev2), int(prev1)): {
                int(token): int(count) for token, count in row.items() if count > 0
            }
            for (prev2, prev1), row in trigram.items()
        }
        self.backoff = float(backoff)

        total = float(counts.sum())
        self._unigram_log = np.log((counts + 1.0) / (total + LM_VOCAB_SIZE))
        self._unigram_floor = self._unigram_log + 2.0 * math.log(self.backoff)

        self._bigram_log: dict[int, dict[int, float]] = {}
        for prev, row in self._bigram_counts.items():
            denominator = float(sum(row.values()))
            if denominator > 0:
                self._bigram_log[prev] = {
                    token: math.log(self.backoff * count / denominator)
                    for token, count in row.items()
                }

        self._trigram_log: dict[tuple[int, int], dict[int, float]] = {}
        for context, row in self._trigram_counts.items():
            denominator = float(sum(row.values()))
            if denominator > 0:
                self._trigram_log[context] = {
                    token: math.log(count / denominator) for token, count in row.items()
                }

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[Sequence[int]],
        min_bigram_count: int = 2,
        min_trigram_count: int = 2,
        backoff: float = 0.4,
    ) -> BPETrigramLM:
        """Count n-grams over raw token sequences (no BOS/EOS in the input)."""
        unigram: Counter[int] = Counter()
        bigram: dict[int, Counter[int]] = defaultdict(Counter)
        trigram: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)

        for sequence in sequences:
            tokens = [int(token) for token in sequence]
            if any(token < 0 or token >= BOS_ID for token in tokens):
                raise ValueError("sequence tokens must be in [0, 1024)")
            words = [BOS_ID, BOS_ID, *tokens, EOS_ID]
            for position in range(2, len(words)):
                prev2, prev1, token = words[position - 2], words[position - 1], words[position]
                unigram[token] += 1
                bigram[prev1][token] += 1
                trigram[(prev2, prev1)][token] += 1

        pruned_bigrams = {
            prev: {token: count for token, count in row.items() if count >= min_bigram_count}
            for prev, row in bigram.items()
        }
        pruned_trigrams = {
            context: {token: count for token, count in row.items() if count >= min_trigram_count}
            for context, row in trigram.items()
        }
        return cls(unigram, pruned_bigrams, pruned_trigrams, backoff=backoff)

    def score(self, prev2: int, prev1: int, token: int) -> float:
        """Log probability of ``token`` after ``prev2, prev1`` (LM vocabulary ids)."""
        token = int(token)
        if token < 0 or token >= LM_VOCAB_SIZE:
            raise ValueError("token outside LM vocabulary")
        row = self._trigram_log.get((int(prev2), int(prev1)))
        if row is not None:
            value = row.get(token)
            if value is not None:
                return value
        row = self._bigram_log.get(int(prev1))
        if row is not None:
            value = row.get(token)
            if value is not None:
                return value
        return float(self._unigram_floor[token])

    def save(self, path: Path | str) -> None:
        path = Path(path)
        payload = {
            "version": 1,
            "bos": BOS_ID,
            "eos": EOS_ID,
            "backoff": self.backoff,
            "unigram": self._unigram_counts.astype(np.int64),
            "bigram": self._bigram_counts,
            "trigram": self._trigram_counts,
        }
        if path.suffix == ".gz":
            with gzip.open(path, "wb", compresslevel=6) as handle:
                pickle.dump(payload, handle, protocol=4)
        else:
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=4)

    @classmethod
    def load(cls, path: Path | str) -> BPETrigramLM:
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("bos") != BOS_ID or payload.get("eos") != EOS_ID:
            raise ValueError("LM special-token ids do not match this decoder")
        return cls(
            payload["unigram"],
            payload["bigram"],
            payload["trigram"],
            backoff=float(payload.get("backoff", 0.4)),
        )

    def summary(self) -> dict[str, int | float]:
        """Small human-readable description for build logs."""
        return {
            "vocab": LM_VOCAB_SIZE,
            "bigram_contexts": len(self._bigram_log),
            "bigram_entries": sum(len(row) for row in self._bigram_log.values()),
            "trigram_contexts": len(self._trigram_log),
            "trigram_entries": sum(len(row) for row in self._trigram_log.values()),
            "backoff": self.backoff,
        }
