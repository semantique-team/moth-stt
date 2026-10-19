"""Reference log-mel front-end (design: ``docs/02-architecture.md`` §2).

Must match what the pretrained Whisper encoder saw, "bit for bit in definition": 16 kHz mono
float input, STFT with ``n_fft=400`` / ``hop=160`` (periodic Hann, ``center=True`` with 200-sample
reflect padding, **last STFT frame dropped** → ``frames = N // 160``), power spectrum, Whisper's
80×201 Slaney filterbank, ``log10(max(mel, 1e-10))``, per-clip ``max(x, x.max() − 8)``, then
``(x + 4) / 4``.

The filterbank ships as ``stt/assets/mel_filters.bin`` (80×201 float32, ≈ 64 KB) so runtime never
re-derives it; ``extract_mel_filters`` regenerates it from the ``openai-whisper`` package (MIT).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["ASSETS_DIR", "extract_mel_filters", "load_mel_filters", "log_mel"]

#: Whisper front-end constants (``docs/02`` §2).
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
FLOOR_RANGE = 8.0

ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_FILTERS = ASSETS_DIR / "mel_filters.bin"

_MIN_SAMPLES = N_FFT + HOP_LENGTH  # shorter clips are filtered out upstream anyway


def extract_mel_filters(whisper_npz: Path | str, dest: Path | str = DEFAULT_FILTERS) -> Path:
    """Write the 80-bin filterbank from ``openai-whisper``'s ``mel_filters.npz`` as a flat bin."""
    data = np.load(Path(whisper_npz))
    filters = np.asarray(data["mel_80"], dtype=np.float32)
    if filters.shape != (N_MELS, N_FFT // 2 + 1):
        raise ValueError(f"unexpected filterbank shape {filters.shape}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    filters.tofile(dest)
    return dest


def load_mel_filters(path: Path | str = DEFAULT_FILTERS) -> np.ndarray:
    """Load the shipped 80×201 filterbank."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — regenerate it from openai-whisper: "
            "python -c \"from stt.features import extract_mel_filters; "
            "import whisper, pathlib; "
            "extract_mel_filters(pathlib.Path(whisper.__file__).parent/'assets'/'mel_filters.npz')\""
        )
    filters = np.fromfile(path, dtype=np.float32)
    return filters.reshape(N_MELS, N_FFT // 2 + 1)


def _periodic_hann() -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)


def log_mel(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Log-mel spectrogram ``[80, N // 160]`` for mono float audio in [-1, 1]."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sample_rate}")
    audio = np.asarray(wav, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError("expected mono audio (1-D array)")
    if len(audio) < _MIN_SAMPLES:
        raise ValueError(f"clip too short: {len(audio)} samples < {_MIN_SAMPLES}")

    pad = N_FFT // 2
    padded = np.pad(audio, (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    frames = frames[: len(audio) // HOP_LENGTH]  # center=True then drop the last STFT frame
    spectrum = np.fft.rfft(frames * _periodic_hann(), n=N_FFT, axis=1)
    power = np.abs(spectrum) ** 2  # [T, 201]

    filters = load_mel_filters()
    mel = filters @ power.T  # [80, T]
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - FLOOR_RANGE)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)
