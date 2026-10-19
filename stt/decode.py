"""Greedy CTC decode (design: ``docs/02-architecture.md`` §6).

The default decoder: argmax per frame → collapse consecutive repeats → drop blanks (``blank = 0``)
→ join the pieces (``▁`` becomes a space). A blank between two identical labels keeps them both,
as CTC defines. The hotword beam search (decode-time, optional) is a later, swap-in decoder with
the same interface.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

__all__ = ["BLANK_ID", "greedy", "load_pieces"]

BLANK_ID = 0
_SPACE_MARKER = "▁"


def load_pieces(vocab_path: Path | str) -> dict[int, str]:
    """Load ``vocab.json`` (pieces by CTC id, as written by :mod:`stt.tokenizer`)."""
    data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    return {int(identifier): piece for identifier, piece in data.items()}


def greedy(log_probs: object, length: int, pieces: Mapping[int, str] | Iterable[str]) -> str:
    """Decode ``log_probs [T, V]`` (torch tensor or ndarray) with the first ``length`` frames."""
    array = log_probs.detach().cpu().numpy() if hasattr(log_probs, "detach") else np.asarray(log_probs)
    frame_ids = array[:length].argmax(axis=-1)

    collapsed: list[int] = []
    previous = -1
    for identifier in frame_ids:
        identifier = int(identifier)
        if identifier != BLANK_ID and identifier != previous:
            collapsed.append(identifier)
        previous = identifier

    piece_of = pieces.__getitem__ if isinstance(pieces, Mapping) else lambda index: list(pieces)[index]
    text = "".join(piece_of(identifier) for identifier in collapsed)
    return text.replace(_SPACE_MARKER, " ").strip()
