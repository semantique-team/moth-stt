"""CTC decoders (design: ``docs/02-architecture.md`` §6).

``greedy`` is the cheap default: argmax per frame → collapse consecutive repeats → drop blanks
(``blank = 0``) → join the pieces (``▁`` becomes a space). ``beam_search`` keeps several prefix
hypotheses alive and can add a decode-time hotword bonus; both return the same text interface.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np

from stt.hotwords import HotwordIndex
from stt.lm import BOS_ID, EOS_ID, BPETrigramLM

__all__ = ["BLANK_ID", "beam_search", "greedy", "greedy_ids", "load_pieces"]

BLANK_ID = 0
_SPACE_MARKER = "▁"
_NEG_INF = float("-inf")


def load_pieces(vocab_path: Path | str) -> dict[int, str]:
    """Load ``vocab.json`` (pieces by CTC id, as written by :mod:`stt.tokenizer`)."""
    data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    return {int(identifier): piece for identifier, piece in data.items()}


def _piece_getter(pieces: Mapping[int, str] | Iterable[str]) -> Callable[[int], str]:
    if isinstance(pieces, Mapping):
        return pieces.__getitem__
    values = list(pieces)
    return values.__getitem__


def _ids_to_text(ids: Iterable[int], pieces: Mapping[int, str] | Iterable[str]) -> str:
    """Join a CTC label sequence (repeats are significant) into text."""
    getter = _piece_getter(pieces)
    text = "".join(getter(int(identifier)) for identifier in ids)
    return text.replace(_SPACE_MARKER, " ").strip()


def _logaddexp(a: float, b: float) -> float:
    """Stable ``log(exp(a) + exp(b))`` without NumPy scalar overhead."""
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def greedy_ids(
    frame_ids: Iterable[int], pieces: Mapping[int, str] | Iterable[str]
) -> str:
    """Collapse a per-frame id sequence (already argmaxed) and join the pieces."""
    collapsed: list[int] = []
    previous = -1
    for identifier in frame_ids:
        identifier = int(identifier)
        if identifier != BLANK_ID and identifier != previous:
            collapsed.append(identifier)
        previous = identifier
    return _ids_to_text(collapsed, pieces)


def greedy(log_probs: object, length: int, pieces: Mapping[int, str] | Iterable[str]) -> str:
    """Decode ``log_probs [T, V]`` (torch tensor or ndarray) with the first ``length`` frames."""
    array = log_probs.detach().cpu().numpy() if hasattr(log_probs, "detach") else np.asarray(log_probs)
    return greedy_ids(array[:length].argmax(axis=-1), pieces)


def _hotword_bonus(
    index: HotwordIndex | None,
    prefix: tuple[int, ...],
    terminal_bonus: float,
    prefix_bonus: float,
) -> float:
    if index is None:
        return 0.0
    terminal, is_prefix = index.match(prefix)
    return (terminal_bonus if terminal else 0.0) + (prefix_bonus if is_prefix else 0.0)


def _lm_bonus(
    lm: BPETrigramLM | None,
    prefix: tuple[int, ...],
    token: int,
    weight: float,
) -> float:
    if lm is None or weight == 0.0:
        return 0.0
    prev1 = prefix[-1] - 1 if prefix else BOS_ID
    prev2 = prefix[-2] - 1 if len(prefix) >= 2 else BOS_ID
    return weight * lm.score(prev2, prev1, token - 1)


def beam_search(
    log_probs: object,
    length: int,
    pieces: Mapping[int, str] | Iterable[str],
    beam_width: int = 5,
    top_k: int = 8,
    hotwords: HotwordIndex | None = None,
    hotword_bonus: float = 1.0,
    hotword_prefix_bonus: float = 0.0,
    lm: BPETrigramLM | None = None,
    lm_weight: float = 0.0,
) -> str:
    """CTC prefix beam search over the first ``length`` frames.

    ``top_k`` is the number of non-blank labels expanded per frame; blank is always kept. The
    hotword bonus is added when a prefix completes a term; ``hotword_prefix_bonus`` can keep
    partial terms alive (default 0, since it can accumulate on multi-piece terms).
    """
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    array = log_probs.detach().cpu().numpy() if hasattr(log_probs, "detach") else np.asarray(log_probs)
    if array.ndim != 2:
        raise ValueError("log_probs must have shape [T, V]")
    frames = int(length)
    if frames <= 0:
        return ""
    array = array[:frames]
    vocab = int(array.shape[1])
    beams: dict[tuple[int, ...], list[float]] = {(): [0.0, _NEG_INF]}
    logaddexp = _logaddexp

    for frame in array:
        if top_k >= vocab:
            candidates = list(range(vocab))
        else:
            candidates = np.argpartition(frame, -top_k)[-top_k:].astype(int).tolist()
            if BLANK_ID not in candidates:
                candidates.append(BLANK_ID)
        # Deterministic order: best score first, lowest id on ties.
        candidates.sort(key=lambda token: (-float(frame[token]), token))

        next_beams: dict[tuple[int, ...], list[float]] = {}
        for prefix, (p_blank, p_nonblank) in beams.items():
            total = logaddexp(p_blank, p_nonblank)
            entry = next_beams.get(prefix)
            if entry is None:
                entry = [_NEG_INF, _NEG_INF]
                next_beams[prefix] = entry
            entry[0] = logaddexp(entry[0], total + float(frame[BLANK_ID]))

            last = prefix[-1] if prefix else -1
            for token in candidates:
                if token == BLANK_ID:
                    continue
                logp = float(frame[token])
                if token == last:
                    # Repeating a label without a blank keeps the same prefix.
                    entry[1] = logaddexp(entry[1], p_nonblank + logp)
                    new_prefix = prefix + (token,)
                    bonus = _hotword_bonus(
                        hotwords, new_prefix, hotword_bonus, hotword_prefix_bonus
                    )
                    bonus += _lm_bonus(lm, prefix, token, lm_weight)
                    new_entry = next_beams.get(new_prefix)
                    if new_entry is None:
                        new_entry = [_NEG_INF, _NEG_INF]
                        next_beams[new_prefix] = new_entry
                    new_entry[1] = logaddexp(new_entry[1], p_blank + logp + bonus)
                else:
                    new_prefix = prefix + (token,)
                    bonus = _hotword_bonus(
                        hotwords, new_prefix, hotword_bonus, hotword_prefix_bonus
                    )
                    bonus += _lm_bonus(lm, prefix, token, lm_weight)
                    new_entry = next_beams.get(new_prefix)
                    if new_entry is None:
                        new_entry = [_NEG_INF, _NEG_INF]
                        next_beams[new_prefix] = new_entry
                    new_entry[1] = logaddexp(new_entry[1], total + logp + bonus)

        ranked = sorted(
            next_beams.items(),
            key=lambda item: -logaddexp(item[1][0], item[1][1]),
        )
        beams = dict(ranked[:beam_width])

    if lm is not None and lm_weight != 0.0:

        def final_score(item: tuple[tuple[int, ...], list[float]]) -> float:
            prefix, (p_blank, p_nonblank) = item
            prev1 = prefix[-1] - 1 if prefix else BOS_ID
            prev2 = prefix[-2] - 1 if len(prefix) >= 2 else BOS_ID
            return logaddexp(p_blank, p_nonblank) + lm_weight * lm.score(prev2, prev1, EOS_ID)

        best_prefix = max(beams.items(), key=final_score)[0]
    else:
        best_prefix = max(
            beams.items(), key=lambda item: logaddexp(item[1][0], item[1][1])
        )[0]
    return _ids_to_text(best_prefix, pieces)
