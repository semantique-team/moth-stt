import json
from pathlib import Path

import numpy as np
import torch

from stt.decode import beam_search, greedy, greedy_ids, load_pieces
from stt.hotwords import HotwordIndex

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


def test_greedy_ids_matches_greedy() -> None:
    frames = np.array([1, 1, 0, 2, 2, 3, 0])
    assert greedy_ids(frames, PIECES) == greedy(_log_probs(frames.tolist()), len(frames), PIECES)


def test_beam_matches_greedy_on_confident_frames() -> None:
    frames = [1, 1, 0, 2, 2, 3, 0]
    probs = _log_probs(frames)
    assert beam_search(probs, len(frames), PIECES) == greedy(probs, len(frames), PIECES)


def test_beam_preserves_repeated_labels_separated_by_blank() -> None:
    # A label sequence may legitimately contain "ll"; only frame-level greedy collapse would
    # merge the two labels, so the beam must join pieces without collapsing them.
    assert beam_search(_log_probs([1, 0, 1], vocab=2), 3, {1: "l"}, beam_width=3) == "ll"


def test_beam_hotword_bonus_prefers_the_term() -> None:
    probs = np.full((1, 4), -10.0, dtype=np.float32)
    probs[0, 1] = 0.0  # "▁meetin"
    probs[0, 2] = 0.0  # "▁meeting"
    index = HotwordIndex([(2,)])
    pieces = {1: "▁meetin", 2: "▁meeting", 3: "▁x"}
    assert (
        beam_search(probs, 1, pieces, beam_width=2, hotwords=index, hotword_bonus=2.0)
        == "meeting"
    )


def test_beam_accepts_torch_and_length_limit() -> None:
    tensor = torch.from_numpy(_log_probs([1, 2, 3]))
    assert beam_search(tensor, 2, PIECES) == "ho spostato"
    assert beam_search(tensor, 0, PIECES) == ""


def test_load_pieces_from_vocab_json(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"1": "▁ho", "2": "▁spostato"}), encoding="utf-8")
    pieces = load_pieces(path)
    assert pieces == {1: "▁ho", 2: "▁spostato"}
