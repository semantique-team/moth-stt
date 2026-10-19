from pathlib import Path

import numpy as np
import pytest

from stt.features import extract_mel_filters, load_mel_filters, log_mel

SAMPLE_RATE = 16000


def _clip(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    signal = 0.1 * np.sin(2 * np.pi * 220.0 * time) + 0.02 * rng.standard_normal(len(time))
    return signal.astype(np.float32)


def test_shape_dtype_and_finite() -> None:
    mel = log_mel(_clip(2.0))
    assert mel.shape == (80, 2 * SAMPLE_RATE // 160)
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all()


def test_empty_or_too_short_rejected() -> None:
    with pytest.raises(ValueError):
        log_mel(np.zeros(100, dtype=np.float32))


def test_parity_with_whisper() -> None:
    whisper = pytest.importorskip("whisper")
    torch = pytest.importorskip("torch")
    clip = _clip(5.0)
    ours = log_mel(clip)
    reference = whisper.log_mel_spectrogram(torch.from_numpy(clip)).numpy()
    assert ours.shape == reference.shape
    assert float(np.max(np.abs(ours - reference))) < 1e-4


def test_padded_vs_unpadded_clamp_equivalence() -> None:
    """Whisper pads to 30 s before the STFT; a per-clip clamp on the unpadded clip is equivalent."""
    whisper = pytest.importorskip("whisper")
    torch = pytest.importorskip("torch")
    clip = _clip(3.0, seed=1)
    ours = log_mel(clip)
    reference = whisper.log_mel_spectrogram(torch.from_numpy(clip)).numpy()
    # Only the last frames (window crossing the clip end) can differ from the reflect pad.
    overlap = ours.shape[1] - 3
    assert float(np.max(np.abs(ours[:, :overlap] - reference[:, :overlap]))) < 1e-4


def test_extract_and_load_mel_filters(tmp_path: Path) -> None:
    whisper = pytest.importorskip("whisper")
    npz = Path(whisper.__file__).parent / "assets" / "mel_filters.npz"
    dest = tmp_path / "mel_filters.bin"
    extract_mel_filters(npz, dest)
    filters = load_mel_filters(dest)
    reference = np.load(npz)["mel_80"].astype(np.float32)
    assert filters.shape == (80, 201)
    assert np.array_equal(filters, reference)


def test_shipped_filters_exist_and_match_whisper() -> None:
    whisper = pytest.importorskip("whisper")
    npz = Path(whisper.__file__).parent / "assets" / "mel_filters.npz"
    reference = np.load(npz)["mel_80"].astype(np.float32)
    assert np.array_equal(load_mel_filters(), reference)
