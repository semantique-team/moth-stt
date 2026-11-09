"""Tests for the Silero VAD no-speech guard (D25, E-011 addendum)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from stt.data.build_mel import load_audio_16k
from stt.vad import (
    ASSET_PATH,
    ASSET_SHA256,
    CONTEXT,
    GATE_THRESHOLD,
    SAMPLE_RATE,
    STATE_SHAPE,
    WINDOW,
    WINDOW_THRESHOLD,
    _chunks,
    _sha256,
    has_speech,
    segment_bounds,
    speech_fraction,
    speech_probabilities,
)

REPO = Path(__file__).resolve().parents[1]
SPEECH_FIXTURE = REPO / "web" / "tests" / "fixtures" / "speech.wav"
TONE_FIXTURE = REPO / "web" / "tests" / "fixtures" / "clip.wav"


def test_asset_is_the_advertised_file() -> None:
    assert ASSET_PATH.stat().st_size == 2_327_524
    assert _sha256(ASSET_PATH) == ASSET_SHA256


def test_real_speech_is_kept() -> None:
    # 3 s of FLEURS-IT (CC-BY 4.0) chosen for a clean window; see fixtures/README.md.
    audio = load_audio_16k(SPEECH_FIXTURE)
    assert speech_fraction(audio) > 0.5
    assert has_speech(audio) is True


def test_digital_silence_is_gated() -> None:
    assert speech_fraction(np.zeros(SAMPLE_RATE, dtype=np.float32)) == 0.0
    assert has_speech(np.zeros(SAMPLE_RATE * 10, dtype=np.float32)) is False


def test_tone_and_noise_are_gated() -> None:
    # The synthetic 5 s tone of the browser smoke test: no speech-like content at all.
    assert has_speech(load_audio_16k(TONE_FIXTURE)) is False
    noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 10) * 0.1).astype(np.float32)
    assert has_speech(noise) is False


def test_gate_is_strict_above_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    def probabilities(*_args, **_kwargs) -> np.ndarray:
        windows = np.zeros(10, dtype=np.float32)
        windows[0] = 0.9  # fraction = 0.1 is *not* enough
        return windows

    monkeypatch.setattr("stt.vad.speech_probabilities", probabilities)
    assert speech_fraction(np.zeros(1, dtype=np.float32)) == pytest.approx(0.1)
    assert has_speech(np.zeros(1, dtype=np.float32)) is False


def test_gate_accepts_a_clear_majority(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = np.full(10, 0.9, dtype=np.float32)
    windows[:2] = 0.1  # fraction = 0.8
    monkeypatch.setattr("stt.vad.speech_probabilities", lambda *_a, **_k: windows)
    assert has_speech(np.zeros(1, dtype=np.float32)) is True


class _RecordingSession:
    """Minimal stand-in for the ONNX session: records inputs, returns a mid probability."""

    def __init__(self) -> None:
        self.inputs: list[dict[str, np.ndarray]] = []

    def run(self, _outputs, feeds):
        self.inputs.append({key: np.array(value) for key, value in feeds.items()})
        return np.array([[0.75]], dtype=np.float32), feeds["state"]


def test_invocation_contract_512_plus_64(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _RecordingSession()
    monkeypatch.setattr("stt.vad._session", lambda: session)
    audio = np.arange(3 * WINDOW + 7, dtype=np.float32) / 10_000
    probabilities = speech_probabilities(audio)

    assert len(session.inputs) == len(probabilities) == 4  # 3 full windows + 1 padded tail
    for feeds in session.inputs:
        assert feeds["input"].shape == (1, WINDOW + CONTEXT)
        assert feeds["input"].dtype == np.float32
        assert feeds["state"].shape == STATE_SHAPE
        assert feeds["sr"].shape == () and feeds["sr"].dtype == np.int64
        assert int(feeds["sr"]) == SAMPLE_RATE
    # The 64-sample context is the previous window's tail, not zeros.
    expected_tail = audio[:WINDOW][-CONTEXT:]
    assert np.array_equal(session.inputs[1]["input"][0, :CONTEXT], expected_tail)
    # The tail window is zero-padded, never dropped silently.
    tail = session.inputs[3]["input"][0, CONTEXT:]
    assert np.array_equal(tail[:7], audio[-7:])
    assert not tail[7:].any()


def test_chunks_carry_the_previous_tail() -> None:
    audio = np.arange(WINDOW + 3, dtype=np.float32)
    chunks = list(_chunks(audio))
    assert [chunk.shape for chunk in chunks] == [(WINDOW + CONTEXT,), (WINDOW + CONTEXT,)]
    assert np.array_equal(chunks[0][CONTEXT:], audio[:WINDOW])
    assert np.array_equal(chunks[1][:CONTEXT], audio[WINDOW - CONTEXT : WINDOW])
    assert np.array_equal(chunks[1][CONTEXT : CONTEXT + 3], audio[-3:])
    assert not chunks[1][CONTEXT + 3 :].any()  # the short tail is zero-padded


def test_js_rule_matches_python() -> None:
    source = (REPO / "web" / "vad.js").read_text(encoding="utf-8")
    for name, value in (
        ("WINDOW", WINDOW),
        ("CONTEXT", CONTEXT),
        ("WINDOW_THRESHOLD", WINDOW_THRESHOLD),
        ("GATE_THRESHOLD", GATE_THRESHOLD),
    ):
        match = re.search(rf"export const {name} = ([0-9.]+);", source)
        assert match, f"web/vad.js must export {name}"
        assert float(match.group(1)) == value


# --- sentence segmentation (runs/audit-2026-11-09) -------------------------------------------


def test_segment_bounds_returns_one_run_for_a_single_sentence() -> None:
    """A single sentence must come back as one segment, so callers can decode it whole."""
    audio = load_audio_16k(SPEECH_FIXTURE)
    bounds = segment_bounds(audio)
    assert len(bounds) == 1
    start, end = bounds[0]
    assert 0 <= start < end <= audio.size


def test_segment_bounds_splits_at_pauses() -> None:
    """Two sentences separated by silence are two runs, and the margin widens each one."""
    audio = load_audio_16k(SPEECH_FIXTURE)
    quiet = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    joined = np.concatenate([audio, quiet, audio])
    without_margin = segment_bounds(joined, margin_s=0.0)
    with_margin = segment_bounds(joined, margin_s=0.2)
    assert len(without_margin) == 2
    assert len(with_margin) == 2
    # The margin must widen the cuts, never narrow them, and never leave the audio.
    for (s0, e0), (s1, e1) in zip(without_margin, with_margin, strict=True):
        assert s1 <= s0
        assert e1 >= e0
        assert 0 <= s1 and e1 <= joined.size


def test_segment_bounds_margin_is_clamped_at_the_edges() -> None:
    audio = load_audio_16k(SPEECH_FIXTURE)
    bounds = segment_bounds(audio, margin_s=5.0)
    assert len(bounds) == 1
    start, end = bounds[0]
    assert start == 0
    assert end == audio.size


def test_segment_bounds_drops_too_short_runs() -> None:
    """A run shorter than ``min_segment_s`` is dropped rather than decoded on its own."""
    audio = load_audio_16k(SPEECH_FIXTURE)
    assert segment_bounds(audio, min_segment_s=10_000.0) == []


def test_segment_bounds_is_empty_without_speech() -> None:
    assert segment_bounds(np.zeros(int(2.0 * SAMPLE_RATE), dtype=np.float32)) == []
