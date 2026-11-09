"""Silero VAD v5 as the no-speech guard (D25, E-011 addendum).

The EM energy guard in :mod:`stt.no_speech` removes digital silence but cannot separate loud
noise from speech: MUSAN noise still produced 19–35 words on the 240 non-music clips of E-011.
A small pre-trained VAD does separate them, so it replaces the energy guard in the shipping
paths (Python evaluation and ``web/ort.js``) while the energy module stays for reproducing the
E-011/E-016 artifacts with ``--no-speech-guard``.

**Rule** (``docs/plans/2026-10-30-vad-guard.md``): run the VAD on 16 kHz mono PCM, one 32 ms
window at a time; a window is *speech* when ``P(speech) > 0.5``; a clip is decoded when the
**fraction of speech windows** is ``> 0.10``. The fraction is the whole point: a single-window
maximum does not separate (noise transients reach 0.99). Measured separability: speech clips
T/FLEURS are at 0.410/0.420 minimum, MUSAN noise at 0.038 maximum, digital silence at 0.009.

**Invocation trap** (v5 ONNX, cost one afternoon): each call takes **512 new samples plus 64
samples of context** — a 576-sample input — with ``state [2, 1, 128]`` float32 and ``sr`` an
int64 scalar; it returns ``output [1, 1]`` and ``stateN [2, 1, 128]``. Passing 512 samples
*without* the context silently returns ~0.001 even on speech, because the conv stack was trained
on the concatenated window.

**Level note**: our recordings are quiet (~−42 dBFS RMS) but the VAD needs no gain step (max
probability 1.000 on T); do not normalise the input.

The model is Silero VAD v5, MIT licensed, vendored as ``stt/assets/silero_vad.onnx`` (2.33 MB,
sha256 checked on load). The same 576-sample logic and thresholds live in ``web/vad.js``; keep
the two in sync.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "ASSET_PATH",
    "ASSET_SHA256",
    "CONTEXT",
    "GATE_THRESHOLD",
    "MODEL_INPUT",
    "SAMPLE_RATE",
    "SEGMENT_MARGIN_S",
    "SEGMENT_MIN_SEGMENT_S",
    "SEGMENT_MIN_SILENCE_S",
    "STATE_SHAPE",
    "WINDOW",
    "WINDOW_THRESHOLD",
    "has_speech",
    "segment_bounds",
    "speech_fraction",
    "speech_probabilities",
]

SAMPLE_RATE = 16000
#: New samples per VAD call (32 ms at 16 kHz).
WINDOW = 512
#: Samples of the previous window prepended to each call.
CONTEXT = 64
#: ``state`` tensor the v5 graph expects.
STATE_SHAPE = (2, 1, 128)
#: Per-window ``P(speech)`` above which a window counts as speech.
WINDOW_THRESHOLD = 0.5
#: Fraction of speech windows above which the clip is decoded (gate: speech iff fraction > this).
GATE_THRESHOLD = 0.10

#: Silence needed to cut between two sentences when splitting a long recording.
SEGMENT_MIN_SILENCE_S = 0.25
#: Context kept on each side of a cut: cutting flush removes the leading silence the model expects.
SEGMENT_MARGIN_S = 0.20
#: Runs shorter than this are dropped rather than decoded on their own.
SEGMENT_MIN_SEGMENT_S = 0.40

ASSET_PATH = Path(__file__).parent / "assets" / "silero_vad.onnx"
ASSET_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
MODEL_INPUT = "input"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _session():
    """The ONNX session, created on first use (tests that do not need it stay fast)."""
    import onnxruntime as ort

    if not ASSET_PATH.exists():
        raise FileNotFoundError(
            f"{ASSET_PATH} missing: re-download it with\n"
            "  curl -sL -o stt/assets/silero_vad.onnx \\\n"
            "    https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
        )
    actual = _sha256(ASSET_PATH)
    if actual != ASSET_SHA256:
        raise ValueError(f"{ASSET_PATH}: sha256 {actual} != {ASSET_SHA256}")
    return ort.InferenceSession(str(ASSET_PATH), providers=["CPUExecutionProvider"])


def _chunks(audio: np.ndarray) -> Iterator[np.ndarray]:
    """Yield 576-sample chunks: 512 new samples prefixed with the previous window's last 64."""
    samples = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    carry = np.zeros(CONTEXT, dtype=np.float32)
    for start in range(0, samples.size, WINDOW):
        frame = samples[start : start + WINDOW]
        if frame.size < WINDOW:  # zero-pad the tail: a partial window is still a window
            frame = np.pad(frame, (0, WINDOW - frame.size))
        yield np.concatenate([carry, frame])
        carry = frame[-CONTEXT:]


def speech_probabilities(audio: np.ndarray) -> np.ndarray:
    """``P(speech)`` for each 32 ms window of ``audio`` (16 kHz mono float32), in order."""
    session = _session()
    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
    probabilities = []
    for chunk in _chunks(audio):
        output, state = session.run(
            None,
            {MODEL_INPUT: chunk[None, :], "state": state, "sr": sample_rate},
        )
        probabilities.append(float(output[0, 0]))
    return np.asarray(probabilities, dtype=np.float32)


def speech_fraction(audio: np.ndarray) -> float:
    """Fraction of windows with ``P(speech) > 0.5``; 0.0 for empty audio."""
    probabilities = speech_probabilities(audio)
    if probabilities.size == 0:
        return 0.0
    return float((probabilities > WINDOW_THRESHOLD).mean())


def has_speech(audio: np.ndarray, threshold: float = GATE_THRESHOLD) -> bool:
    """True when the clip should be decoded: ``speech_fraction(audio) > threshold``."""
    return speech_fraction(audio) > threshold


def segment_bounds(
    audio: np.ndarray,
    margin_s: float = SEGMENT_MARGIN_S,
    min_silence_s: float = SEGMENT_MIN_SILENCE_S,
    min_segment_s: float = SEGMENT_MIN_SEGMENT_S,
) -> list[tuple[int, int]]:
    """Sample offsets of the speech runs in ``audio``, for splitting a long recording.

    Why this exists: the model is trained on sentence-sized gold clips (median 2.7 s) and reads a
    recording of several sentences as one unpunctuated stream. Measured on T-S3, five sentences
    concatenated into one 12.5 s clip score **36.6% WER** against **26.6%** when decoded
    separately; cutting at the pauses recovers **80%** of that gap
    (``runs/audit-2026-11-09/``).

    ``margin_s`` keeps audio on each side of a cut: cutting flush against the first voiced window
    removes the leading silence the model expects and costs +1.6 to +2.9 WER on single clips, so
    callers must not segment at all when only one run is found — see ``unwrap_or_whole``.

    Returns ``[]`` when there is no speech. ``mirror`` of ``web/vad.js::segments``; keep in sync.
    """
    probabilities = speech_probabilities(audio)
    if probabilities.size == 0:
        return []
    voiced = probabilities > WINDOW_THRESHOLD

    runs: list[tuple[int, int]] = []
    index = 0
    total = int(voiced.size)
    while index < total:
        if not voiced[index]:
            index += 1
            continue
        start = index
        while index < total and voiced[index]:
            index += 1
        runs.append((start, index))
    if not runs:
        return []

    seconds_per_window = WINDOW / SAMPLE_RATE
    merged = [runs[0]]
    for start, end in runs[1:]:
        if (start - merged[-1][1]) * seconds_per_window < min_silence_s:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    margin = int(margin_s * SAMPLE_RATE)
    bounds: list[tuple[int, int]] = []
    for start, end in merged:
        if (end - start) * seconds_per_window < min_segment_s:
            continue
        bounds.append(
            (max(0, start * WINDOW - margin), min(int(audio.size), end * WINDOW + margin))
        )
    return bounds
