import json
from pathlib import Path

import numpy as np
import torch

from stt.decode import greedy, load_pieces

PIECES = {1: "▁ho", 2: "▁spostato", 3: "▁il"}


def _log_probs(frames: list[int], vocab: int = 4) -> np.ndarray:
    array = np.full((len(frames), vocab), -10.0, dtype=np.float32)
    for position, identifier in enumerate(frames):
        array[position, identifier] = 0.0
    return array


def test_collapse_repeats_and_drop_blanks() -> None:
    assert greedy(_log_probs([1, 1, 0, 1]), 4, PIECES) == "ho ho"


def test_blank_separates_repeated_labels() -> None:
    assert greedy(_log_probs([1, 0, 1]), 3, PIECES) == "ho ho"
    assert greedy(_log_probs([1, 1, 1]), 3, PIECES) == "ho"


def test_length_limits_the_frames_read() -> None:
    assert greedy(_log_probs([1, 2, 3]), 2, PIECES) == "ho spostato"


def test_all_blank_is_empty() -> None:
    assert greedy(_log_probs([0, 0, 0]), 3, PIECES) == ""


def test_accepts_torch_tensor() -> None:
    tensor = torch.from_numpy(_log_probs([1, 3]))
    assert greedy(tensor, 2, PIECES) == "ho il"


def test_load_pieces_from_vocab_json(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"1": "▁ho", "2": "▁spostato"}), encoding="utf-8")
    pieces = load_pieces(path)
    assert pieces == {1: "▁ho", 2: "▁spostato"}
