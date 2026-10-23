import math
from pathlib import Path

import numpy as np
import pytest

from stt.decode import beam_search
from stt.lm import BOS_ID, BPETrigramLM


def test_trigram_prefers_seen_sequences_and_backs_off() -> None:
    lm = BPETrigramLM.from_sequences(
        [[1, 2, 1], [1, 2, 1], [3, 4]],
        min_bigram_count=1,
        min_trigram_count=1,
    )
    seen = lm.score(BOS_ID, BOS_ID, 1)
    unseen = lm.score(BOS_ID, BOS_ID, 9)

    assert math.isfinite(seen) and math.isfinite(unseen)
    assert seen > unseen
    assert lm.score(1, 2, 1) > lm.score(1, 2, 9)
    assert lm.summary()["trigram_entries"] > 0


def test_save_load_roundtrip(tmp_path: Path) -> None:
    lm = BPETrigramLM.from_sequences([[1, 2, 3], [1, 2, 3]], min_bigram_count=1, min_trigram_count=1)
    path = tmp_path / "lm.pkl.gz"
    lm.save(path)
    loaded = BPETrigramLM.load(path)

    for context in [(BOS_ID, BOS_ID), (1, 2)]:
        for token in [1, 2, 3, 9]:
            assert loaded.score(*context, token) == pytest.approx(lm.score(*context, token))


def test_save_browser_writes_the_binary_header(tmp_path: Path) -> None:
    lm = BPETrigramLM.from_sequences([[1, 2], [1, 2]], min_bigram_count=1, min_trigram_count=1)
    path = tmp_path / "lm.bin"
    lm.save_browser(path)
    data = path.read_bytes()

    assert data[:8] == b"MOTHLMB1"
    assert len(data) > 8 + 28 + 1026 * 4


def test_beam_uses_the_lm_to_break_a_tie() -> None:
    probs = np.full((1, 4), -10.0, dtype=np.float32)
    probs[0, 1] = 0.0
    probs[0, 2] = 0.0
    pieces = {1: "▁a", 2: "▁b", 3: "▁c"}
    # CTC id 2 maps to SentencePiece id 1 in the LM.
    lm = BPETrigramLM.from_sequences(
        [[1], [1]], min_bigram_count=1, min_trigram_count=1
    )

    assert beam_search(probs, 1, pieces, beam_width=2, lm=lm, lm_weight=1.0) == "b"
