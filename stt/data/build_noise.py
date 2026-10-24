"""Build the silence/noise/music set (design: ``docs/03-data.md`` §9).

The set is deliberately boring audio: 10 s clips of digital silence, own room tone, and
permissively licensed non-music noise and music (MUSAN, CC BY 4.0). A speech-to-text model must
emit **nothing** on the non-music half; music is reported separately and never gated.

Example:

```bash
uv run python -m stt.data.build_noise --musan data/raw/musan --room-tone room_tone.wav \
    --out-audio data/raw/noise --out-manifest data/manifests/noise.jsonl
```

Rows carry ``text``/``norm`` empty; ``stt.eval`` scores them in the noise block. Music rows are
tagged with ``source = "noise_music"`` (the schema reserves ``category`` for mixed rows).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.data.build_mel import TARGET_RATE, load_audio_16k

__all__ = ["SOURCES", "build_noise_set", "main"]

#: Manifest ``source`` per category. ``noise_music`` is what ``stt.eval`` counts as music.
SOURCES = {
    "silence": "noise_silence",
    "room_tone": "noise_room_tone",
    "noise": "noise_musan",
    "music": "noise_music",
}

SUBDIRS = {"noise": "noise", "music": "music"}


def _clips(minutes: float, clip_seconds: float) -> int:
    return max(1, round(minutes * 60.0 / clip_seconds))


def _rms_dbfs(signal: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)) + 1e-12))
    return 20.0 * np.log10(rms)


def _apply_level(signal: np.ndarray, target_dbfs: float | None) -> np.ndarray:
    """Optionally bring a clip to a fixed RMS level (``None`` keeps the natural level)."""
    if target_dbfs is None:
        return signal
    current = _rms_dbfs(signal)
    if current <= -120.0:
        return signal
    scaled = signal * (10.0 ** ((target_dbfs - current) / 20.0))
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def _window_from_files(
    paths: list[Path], need: int, rng: random.Random, target_dbfs: float | None
) -> np.ndarray:
    """A window built by concatenating consecutive files when a single one is too short."""
    if not paths:
        return np.zeros(need, dtype=np.float32)
    start = rng.randrange(len(paths))
    ordered = paths[start:] + paths[:start]
    chunks: list[np.ndarray] = []
    total = 0
    for path in ordered:
        signal = load_audio_16k(path)
        if len(signal):
            chunks.append(signal)
            total += len(signal)
        if total >= need:
            break
    if not chunks:
        return np.zeros(need, dtype=np.float32)
    joined = np.concatenate(chunks)
    if len(joined) > need:
        offset = rng.randrange(0, len(joined) - need + 1)
        joined = joined[offset : offset + need]
    else:
        joined = np.tile(joined, int(np.ceil(need / len(joined))))[:need]
    return _apply_level(joined.astype(np.float32), target_dbfs)


def _source_files(root: Path, subdir: str) -> list[Path]:
    base = root / subdir
    return sorted(path for path in base.rglob("*.wav") if path.is_file())


def _write_clip(path: Path, signal: np.ndarray, sample_rate: int = TARGET_RATE) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sample_rate, subtype="PCM_16")
    return len(signal) / sample_rate


def _row(uid: str, audio: str, duration: float, source: str) -> dict:
    return {
        "id": uid,
        "audio": audio,
        "text": "",
        "norm": "",
        "duration": round(duration, 3),
        "source": source,
        "split": "test",
        "speaker": None,
        "lang": "it",
        "category": None,
        "en_spans": [],
    }


def build_noise_set(
    musan_dir: Path | str,
    out_audio: Path | str,
    out_manifest: Path | str,
    *,
    room_tone: Path | str | None = None,
    minutes_per_category: float = 20.0,
    clip_seconds: float = 10.0,
    seed: int = 0,
    target_rms_dbfs: float | None = None,
    categories: tuple[str, ...] = ("silence", "room_tone", "noise", "music"),
    merge_existing: bool = True,
) -> dict:
    """Build the set and write the manifest; returns a report dict.

    ``target_rms_dbfs`` is ``None`` by default: MUSAN keeps its natural level (the set tests the
    runtime as a user would hit it). Pass e.g. ``-30`` to make every clip equally loud.
    """
    musan_dir = Path(musan_dir)
    out_audio = Path(out_audio)
    out_manifest = Path(out_manifest)
    need = round(clip_seconds * TARGET_RATE)
    n_clips = _clips(minutes_per_category, clip_seconds)
    rng = random.Random(seed)

    unknown = set(categories) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown categories: {sorted(unknown)}")
    if "room_tone" in categories and room_tone is None:
        categories = tuple(category for category in categories if category != "room_tone")
        print("warning: no --room-tone given; building without room tone")

    files = {category: _source_files(musan_dir, subdir) for category, subdir in SUBDIRS.items()}
    for category, paths in files.items():
        if category in categories and not paths:
            raise FileNotFoundError(f"no WAV files under {musan_dir / SUBDIRS[category]}")
    room_paths: list[Path] = []
    if "room_tone" in categories and room_tone is not None:
        room_input = Path(room_tone)
        room_paths = sorted(room_input.rglob("*.wav")) if room_input.is_dir() else [room_input]
        if not room_paths:
            raise FileNotFoundError(f"no room-tone WAV files under {room_input}")

    rows: list[dict] = []
    report: dict = {
        "seed": seed,
        "clip_seconds": clip_seconds,
        "minutes_per_category": minutes_per_category,
        "target_rms_dbfs": target_rms_dbfs,
        "musan_dir": str(musan_dir),
        "categories": {},
    }

    for category in ("silence", "room_tone", "noise", "music"):
        if category not in categories:
            continue
        levels: list[float] = []
        valid = 0
        for index in range(1, n_clips + 1):
            uid = f"{SOURCES[category]}-{index:04d}"
            relative = f"{SOURCES[category]}/{uid}.wav"
            if category == "silence":
                signal = np.zeros(need, dtype=np.float32)
            elif category == "room_tone":
                signal = _window_from_files(room_paths, need, rng, target_rms_dbfs)
            else:
                signal = _window_from_files(files[category], need, rng, target_rms_dbfs)
            duration = _write_clip(out_audio / relative, signal)
            if np.any(signal):
                valid += 1
                levels.append(_rms_dbfs(signal))
            rows.append(_row(uid, relative, duration, SOURCES[category]))
        report["categories"][category] = {
            "clips": n_clips,
            "non_silent_clips": valid,
            "seconds": round(n_clips * clip_seconds, 1),
            "rms_dbfs_min": round(min(levels), 1) if levels else None,
            "rms_dbfs_median": round(float(np.median(levels)), 1) if levels else None,
            "rms_dbfs_max": round(max(levels), 1) if levels else None,
            "source_files": len(files.get(category, room_paths)) if category != "silence" else 0,
        }

    if merge_existing and out_manifest.exists():
        rebuilt = {SOURCES[category] for category in categories}
        existing = [
            json.loads(line)
            for line in out_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = [row for row in existing if row.get("source") not in rebuilt] + rows
        order = {source: rank for rank, source in enumerate(SOURCES.values())}
        rows.sort(key=lambda row: (order.get(row.get("source"), 99), row["id"]))

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    total_seconds = sum(float(row["duration"]) for row in rows)
    report["total_clips"] = len(rows)
    report["total_seconds"] = round(total_seconds, 1)
    report["total_minutes"] = round(total_seconds / 60.0, 1)
    report["manifest"] = str(out_manifest)
    report["audio_root"] = str(out_audio)
    parts = ", ".join(f"{name} {data['clips']}" for name, data in report["categories"].items())
    print(
        f"noise set: {len(rows)} clips, {total_seconds / 60.0:.1f} min ({parts})"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_noise.py", description="Build the noise set")
    parser.add_argument("--musan", default="data/raw/musan", help="MUSAN root (noise/ and music/)")
    parser.add_argument("--room-tone", default=None, help="room-tone WAV file or directory")
    parser.add_argument("--out-audio", default="data/raw/noise")
    parser.add_argument("--out-manifest", default="data/manifests/noise.jsonl")
    parser.add_argument("--minutes-per-category", type=float, default=20.0)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-rms-dbfs",
        type=float,
        default=None,
        help="optional fixed clip level; default keeps the natural level",
    )
    parser.add_argument(
        "--categories",
        default="silence,room_tone,noise,music",
        help="comma list of categories to (re)build; existing others are kept",
    )
    parser.add_argument("--report", default="data/manifests/noise_report.json")
    args = parser.parse_args(argv)

    report = build_noise_set(
        args.musan,
        args.out_audio,
        args.out_manifest,
        room_tone=args.room_tone,
        minutes_per_category=args.minutes_per_category,
        clip_seconds=args.clip_seconds,
        seed=args.seed,
        target_rms_dbfs=args.target_rms_dbfs,
        categories=tuple(name.strip() for name in args.categories.split(",") if name.strip()),
    )
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
