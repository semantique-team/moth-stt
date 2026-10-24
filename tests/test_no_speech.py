"""Tests for the no-speech guard (E-011 follow-up)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from stt.features import log_mel
from stt.no_speech import SILENCE_MAX, is_no_speech

REPO = Path(__file__).resolve().parents[1]


def _mel_with_peak(peak: float) -> np.ndarray:
    mel = np.full((80, 100), peak - 2.0, dtype=np.float32)
    mel[40, 50] = peak
    return mel


def test_digital_silence_is_guarded() -> None:
    # log_mel of all-zero audio is exactly 1e-10 -> log10 -10 -> (-10 + 4) / 4
    silence = log_mel(np.zeros(16000, dtype=np.float32))
    assert float(silence.max()) == pytest.approx(-1.5, abs=1e-6)
    assert is_no_speech(silence) is True


def test_quiet_speech_is_not_guarded() -> None:
    # The quietest speech clip measured (FLEURS-EN) peaks at +0.10; keep a wider margin.
    assert is_no_speech(_mel_with_peak(0.10)) is False
    assert is_no_speech(_mel_with_peak(SILENCE_MAX + 0.01)) is False


def test_threshold_sits_above_silence_and_below_speech() -> None:
    assert -1.5 < SILENCE_MAX < 0.10


def test_guard_reads_the_loudest_bin_not_the_mean() -> None:
    mel = np.full((80, 100), -1.9, dtype=np.float32)
    mel[40, 50] = -0.5
    assert float(mel.mean()) < SILENCE_MAX
    assert is_no_speech(mel) is False


def test_js_threshold_matches_python() -> None:
    source = (REPO / "web" / "mel.js").read_text(encoding="utf-8")
    match = re.search(r"export const SILENCE_MAX = (-?[0-9.]+);", source)
    assert match, "web/mel.js must export SILENCE_MAX"
    assert float(match.group(1)) == SILENCE_MAX
