"""E-025 — record a room the model has never heard, and check it is usable.

The hold-out control for E-024 needs noise that is not in the 217-clip pool. This records one
new room (built-in microphone, 16 kHz mono after resampling), checks it for speech and transients,
and writes a small JSON report so the measurement can say what the noise actually was.

    uv run python runs/E-025/record_room.py --seconds 120
    uv run python runs/E-025/record_room.py --from-wav data/raw/noise/room_holdout/room_2026-11-09.wav

Audio stays under ``data/`` (gitignored); only the report is committed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.data.build_mel import TARGET_RATE, load_audio_16k
from stt.vad import GATE_THRESHOLD, speech_probabilities

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data/raw/noise/room_holdout"
REPORT = REPO / "runs/E-025/room_report.json"

FRAME = 1024
HOP = 512


def _dbfs(signal: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(signal, dtype=np.float64))) + 1e-12))


def _spectrum_stats(audio: np.ndarray) -> dict[str, float]:
    """Mean spectral centroid, flatness and low-band share over Hann-windowed frames."""
    if audio.size < FRAME:
        audio = np.pad(audio, (0, FRAME - audio.size))
    window = np.hanning(FRAME).astype(np.float32)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / TARGET_RATE)
    centroids, flatness, low_share = [], [], []
    for start in range(0, audio.size - FRAME + 1, HOP):
        magnitude = np.abs(np.fft.rfft(audio[start : start + FRAME] * window)) + 1e-12
        power = magnitude**2
        centroids.append(float((freqs * power).sum() / power.sum()))
        flatness.append(float(np.exp(np.log(magnitude).mean()) / magnitude.mean()))
        low_share.append(float(power[freqs < 200.0].sum() / power.sum()))
    return {
        "centroid_hz": round(float(np.mean(centroids)), 1),
        "flatness": round(float(np.mean(flatness)), 4),
        "low_band_share_lt_200hz": round(float(np.mean(low_share)), 4),
    }


def _stationarity(audio: np.ndarray, block_s: float = 10.0) -> dict[str, float]:
    """How much level and timbre drift across the recording, in 10 s blocks."""
    block = int(block_s * TARGET_RATE)
    levels = [_dbfs(audio[start : start + block]) for start in range(0, audio.size, block)]
    levels = [value for value in levels if np.isfinite(value)]
    centroids = []
    for start in range(0, audio.size - FRAME + 1, block):
        chunk = audio[start : start + block]
        if chunk.size >= FRAME:
            centroids.append(_spectrum_stats(chunk)["centroid_hz"])
    return {
        "block_seconds": block_s,
        "level_db_std": round(float(np.std(levels)), 2),
        "level_db_range": round(float(max(levels) - min(levels)), 2),
        "centroid_hz_std": round(float(np.std(centroids)), 1),
    }


def _transients(audio: np.ndarray, over_db: float = 15.0) -> dict[str, float]:
    """Frames whose level jumps far above the median: typing, doors, a passing voice."""
    frame = int(0.05 * TARGET_RATE)
    levels = np.array(
        [_dbfs(audio[start : start + frame]) for start in range(0, audio.size - frame + 1, frame)]
    )
    median = float(np.median(levels))
    loud = levels > median + over_db
    return {
        "frame_seconds": 0.05,
        "median_db": round(median, 1),
        "over_db": over_db,
        "frames": int(loud.sum()),
        "total_frames": int(loud.size),
        "seconds_over": round(float(loud.sum()) * 0.05, 2),
    }


def record(path: Path, seconds: int, device: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "avfoundation", "-i", device,
        "-t", str(seconds), "-ar", "48000", "-ac", "1", "-c:a", "pcm_s24le",
        "-y", str(path),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(prog="record_room.py")
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--device", default=":0")
    parser.add_argument("--from-wav", type=Path, default=None, help="analyse an existing recording")
    parser.add_argument("--trim-start", type=float, default=0.0, help="seconds to cut from the start")
    parser.add_argument("--trim-end", type=float, default=None, help="seconds to keep (default: end)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    raw = args.from_wav or args.out_dir / "room_2026-11-09.wav"
    if args.from_wav is None:
        print(f"recording {args.seconds} s from {args.device} -> {raw}")
        record(raw, args.seconds, args.device)

    audio = load_audio_16k(raw)
    source_duration = audio.size / TARGET_RATE
    end = args.trim_end if args.trim_end is not None else source_duration
    start_sample = int(max(0.0, args.trim_start) * TARGET_RATE)
    end_sample = int(min(source_duration, end) * TARGET_RATE)
    trimmed = end_sample - start_sample != audio.size
    if trimmed:
        audio = audio[start_sample:end_sample]
    probabilities = speech_probabilities(audio)
    fraction = float((probabilities > 0.5).mean()) if probabilities.size else 0.0
    report = {
        "raw_wav": str(raw.relative_to(REPO)) if raw.is_relative_to(REPO) else str(raw),
        "sample_rate": TARGET_RATE,
        "source_duration_s": round(source_duration, 2),
        "trim": [round(args.trim_start, 2), round(end, 2)] if trimmed else None,
        "duration_s": round(audio.size / TARGET_RATE, 2),
        "rms_dbfs": round(_dbfs(audio), 1),
        "peak_dbfs": round(float(20.0 * np.log10(np.max(np.abs(audio)) + 1e-12)), 1),
        "vad": {
            "speech_fraction": round(fraction, 4),
            "max_probability": round(float(probabilities.max()), 4) if probabilities.size else 0.0,
            "gate_threshold": GATE_THRESHOLD,
            "has_speech": bool(fraction > GATE_THRESHOLD),
            "windows": int(probabilities.size),
        },
        "spectrum": _spectrum_stats(audio),
        "stationarity": _stationarity(audio),
        "transients": _transients(audio),
    }
    # A 16 kHz copy is what the mixer loads; keep it next to the 48 kHz original.
    suffix = "trimmed_16k" if trimmed else "16k"
    target = args.out_dir / f"{raw.stem}_{suffix}.wav"
    sf.write(target, audio, TARGET_RATE, subtype="PCM_16")
    report["mix_wav"] = str(target.relative_to(REPO)) if target.is_relative_to(REPO) else str(target)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    if report["vad"]["has_speech"]:
        print("WARNING: the recording contains speech; do not use it as room tone")
    if report["transients"]["seconds_over"] > 2.0:
        print("WARNING: more than 2 s of transients; the room was not quiet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
