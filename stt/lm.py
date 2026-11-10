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
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

__all__ = ["BOS_ID", "EOS_ID", "LM_VOCAB_SIZE", "BPETrigramLM", "BinaryBPELM"]

#: SentencePiece vocabulary is 1024; the two ids after it are LM-only.
BOS_ID = 1024
EOS_ID = 1025
LM_VOCAB_SIZE = 1026


class BinaryBPELM:
    """Read-only MOTHLMB1 scorer over CSR arrays, without reconstructing counts.

    Construct from bytes or use ``load(path)``. Scores include the writer's int16
    quantization and the browser's float32 unigram fallback rounding.
    """

    def __init__(self, data: bytes) -> None:
        data = bytes(data)
        offset = 0

        def read(dtype: str, count: int) -> np.ndarray:
            nonlocal offset
            size = np.dtype(dtype).itemsize * count
            if size > len(data) - offset:
                raise ValueError("truncated LM buffer")
            array = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
            offset += size
            return array

        if len(data) < 28:
            raise ValueError("truncated LM header")
        if data[:8] != b"MOTHLMB1":
            raise ValueError("bad LM magic")
        version, vocab, backoff, unigram_length, log_scale = struct.unpack_from("<IIfIf", data, 8)
        if version != 1:
            raise ValueError(f"unsupported LM version: {version}")
        if vocab != LM_VOCAB_SIZE or unigram_length != LM_VOCAB_SIZE:
            raise ValueError("LM vocab mismatch")
        if not math.isfinite(backoff) or backoff <= 0:
            raise ValueError("LM backoff must be finite and positive")
        if not math.isfinite(log_scale) or log_scale <= 0:
            raise ValueError("LM log scale must be finite and positive")
        self.backoff = backoff
        self.log_scale = log_scale
        offset = 28
        self.unigram = read("<f4", unigram_length)
        if not np.isfinite(self.unigram).all():
            raise ValueError("LM unigram must be finite")

        def read_rows(contexts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            offsets = read("<u4", contexts + 1)
            if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
                raise ValueError("LM offsets must start at zero and be monotonic")
            entries = int(offsets[-1])
            tokens = read("<u2", entries)
            logs = read("<i2", entries)
            if np.any(tokens >= vocab):
                raise ValueError("LM token outside vocabulary")
            ordered = tokens[1:] > tokens[:-1]
            boundaries = offsets[1:-1]
            boundaries = boundaries[(boundaries > 0) & (boundaries < entries)]
            ordered[boundaries - 1] = True
            if not ordered.all():
                raise ValueError("LM row tokens must be strictly sorted")
            return offsets, tokens, logs

        bigram_contexts = int(read("<u4", 1)[0])
        if bigram_contexts != vocab:
            raise ValueError("LM bigram contexts mismatch")
        self.bigram_offsets, self.bigram_tokens, self.bigram_logq = read_rows(bigram_contexts)
        offset = (offset + 3) // 4 * 4
        trigram_contexts = int(read("<u4", 1)[0])
        if trigram_contexts > vocab * vocab:
            raise ValueError("too many LM trigram contexts")
        self.trigram_keys = read("<u4", trigram_contexts)
        if np.any(self.trigram_keys >= vocab * vocab) or np.any(
            self.trigram_keys[1:] <= self.trigram_keys[:-1]
        ):
            raise ValueError("LM trigram keys must be in range and strictly sorted")
        self.trigram_offsets, self.trigram_tokens, self.trigram_logq = read_rows(trigram_contexts)
        if offset != len(data):
            raise ValueError("trailing LM bytes")
        # JS adds in float64, then stores the fallback in a Float32Array.
        self.unigram_floor = (self.unigram.astype(np.float64) + 2.0 * math.log(backoff)).astype(
            np.float32
        )
        self.unigram_floor.flags.writeable = False

    @classmethod
    def load(cls, path: Path | str) -> BinaryBPELM:
        return cls(Path(path).read_bytes())

    def score(self, prev2: int, prev1: int, token: int) -> float:
        """Log probability using trigram, bigram, then unigram fallback."""
        prev2, prev1, token = int(prev2), int(prev1), int(token)
        if not 0 <= token < LM_VOCAB_SIZE:
            raise ValueError("token outside LM vocabulary")
        token_id = np.uint16(token)
        if 0 <= prev1 < LM_VOCAB_SIZE:
            if 0 <= prev2 < LM_VOCAB_SIZE:
                key = prev2 * LM_VOCAB_SIZE + prev1
                row = int(np.searchsorted(self.trigram_keys, np.uint32(key)))
                if row < len(self.trigram_keys) and self.trigram_keys[row] == key:
                    start, end = self.trigram_offsets[row : row + 2]
                    index = int(start) + int(
                        np.searchsorted(self.trigram_tokens[start:end], token_id)
                    )
                    if index < end and self.trigram_tokens[index] == token:
                        return int(self.trigram_logq[index]) * self.log_scale
            start, end = self.bigram_offsets[prev1 : prev1 + 2]
            index = int(start) + int(np.searchsorted(self.bigram_tokens[start:end], token_id))
            if index < end and self.bigram_tokens[index] == token:
                return int(self.bigram_logq[index]) * self.log_scale
        return float(self.unigram_floor[token])


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

    def save_browser(self, path: Path | str, log_scale: float = 0.01) -> None:
        """Write the compact sparse binary consumed by ``web/lm.js``.

        Layout: magic, header, float32 unigram log-probs, CSR bigrams (context = prev id),
        CSR trigrams (sorted ``prev2 * V + prev1`` keys). Log-probs are int16-quantized;
        absent entries fall back exactly like :meth:`score`.
        """
        if log_scale <= 0.0:
            raise ValueError("log_scale must be positive")

        def quantize(value: float) -> int:
            return max(-32768, min(32767, round(value / log_scale)))

        buffer = bytearray(b"MOTHLMB1")
        buffer += struct.pack(
            "<IIfIf",
            1,
            LM_VOCAB_SIZE,
            self.backoff,
            LM_VOCAB_SIZE,
            log_scale,
        )
        buffer += self._unigram_log.astype("<f4").tobytes()

        bigram_offsets = [0]
        bigram_tokens: list[int] = []
        bigram_logs: list[int] = []
        for prev in range(LM_VOCAB_SIZE):
            row = self._bigram_log.get(prev, {})
            for token in sorted(row):
                bigram_tokens.append(token)
                bigram_logs.append(quantize(row[token]))
            bigram_offsets.append(len(bigram_tokens))
        buffer += struct.pack("<I", LM_VOCAB_SIZE)
        buffer += struct.pack(f"<{len(bigram_offsets)}I", *bigram_offsets)
        if bigram_tokens:
            buffer += struct.pack(f"<{len(bigram_tokens)}H", *bigram_tokens)
            buffer += struct.pack(f"<{len(bigram_logs)}h", *bigram_logs)
        while len(buffer) % 4:
            buffer.append(0)

        contexts = sorted(self._trigram_log)
        keys = [prev2 * LM_VOCAB_SIZE + prev1 for prev2, prev1 in contexts]
        trigram_offsets = [0]
        trigram_tokens: list[int] = []
        trigram_logs: list[int] = []
        for context in contexts:
            row = self._trigram_log[context]
            for token in sorted(row):
                trigram_tokens.append(token)
                trigram_logs.append(quantize(row[token]))
            trigram_offsets.append(len(trigram_tokens))
        buffer += struct.pack("<I", len(contexts))
        if keys:
            buffer += struct.pack(f"<{len(keys)}I", *keys)
        buffer += struct.pack(f"<{len(trigram_offsets)}I", *trigram_offsets)
        if trigram_tokens:
            buffer += struct.pack(f"<{len(trigram_tokens)}H", *trigram_tokens)
            buffer += struct.pack(f"<{len(trigram_logs)}h", *trigram_logs)
        Path(path).write_bytes(buffer)

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
