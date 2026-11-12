"""Split long voice notes into importable 1-30 s clips for natural mixed dev.

``recordings/import_manifest.py`` accepts one row per mono file of ``MIN_SECONDS``-``MAX_SECONDS``
(1-30 s) with a human transcript, ``category`` and ``en_spans``. A 5-10 minute note does not fit
that shape, so it has to be cut first. This helper cuts at low-energy gaps, writes 16 kHz mono
PCM16 wavs and a transcript skeleton to fill by hand. It never guesses text, category or spans,
never writes a manifest and never reuses T/A/R scripts.

Pipeline for one person (see ``recordings/README.md``):

1. record 5-10 minutes of spontaneous speech with any tool;
2. convert to wav if needed (``afconvert`` / ``ffmpeg``);
3. ``segment_notes.py`` -> wav clips + ``notes.jsonl`` skeleton;
4. fill ``text``, ``category`` and ``en_spans``;
5. ``import_manifest.py --transcripts notes.jsonl --audio-dir <clips> --out <manifest>``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from recordings.build_manifest import MAX_SECONDS, MIN_SECONDS, inspect_audio
from stt.data.build_mel import load_audio_16k

__all__ = ["main", "plan_segments", "write_segments"]

RATE = 16_000
FRAME_SECONDS = 0.02
DEFAULT_SILENCE_DBFS = -35.0
DEFAULT_MIN_GAP = 0.30
SEARCH_SECONDS = 1.0
_ID_SAFE = re.compile(r"[^A-Za-z0-9]+")


def _frame_db(samples: np.ndarray, frame: int) -> np.ndarray:
    count = len(samples) // frame
    if count == 0:
        raise ValueError(f"audio shorter than one {FRAME_SECONDS:g} s frame")
    blocks = samples[: count * frame].reshape(count, frame).astype(np.float64)
    rms = np.sqrt((blocks**2).mean(axis=1))
    return 20.0 * np.log10(rms + 1e-12)


def _speech_runs(samples: np.ndarray, frame: int, silence_dbfs: float) -> list[tuple[int, int]]:
    speech = _frame_db(samples, frame) >= silence_dbfs
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(speech):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start * frame, index * frame))
            start = None
    if start is not None:
        runs.append((start * frame, len(speech) * frame))
    return runs


def _merge_runs(runs: list[tuple[int, int]], min_gap: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _best_cut(samples: np.ndarray, start: int, end: int, frame: int, low: int, high: int) -> int:
    """Quietest frame boundary in the ``[low, high)`` offset window, else the window end."""
    low = max(start + low, start)
    high = min(start + high, end)
    if high - low < frame:
        return max(low, min(high, (start + end) // 2))
    first = (low - start) // frame
    last = max(first + 1, (high - start) // frame)
    window = samples[start:end]
    db = _frame_db(window, frame)[first:last]
    if db.size == 0:
        return (start + end) // 2
    return start + (first + int(np.argmin(db))) * frame


def plan_segments(
    samples: np.ndarray,
    *,
    rate: int = RATE,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    silence_dbfs: float = DEFAULT_SILENCE_DBFS,
    min_gap_seconds: float = DEFAULT_MIN_GAP,
) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` sample spans of 1-30 s covering the speech in ``samples``.

    Gaps shorter than ``min_gap_seconds`` stay inside a clip; runs longer than ``max_seconds``
    are cut at their quietest frame near the limit. Spans shorter than ``min_seconds`` are
    dropped, so a note that is mostly silence raises :class:`ValueError`.
    """
    if not 0 < min_seconds < max_seconds <= MAX_SECONDS:
        raise ValueError(f"need 0 < min < max <= {MAX_SECONDS:g} s, got {min_seconds}/{max_seconds}")
    if not np.isfinite(silence_dbfs):
        raise ValueError("silence_dbfs must be finite")
    frame = max(1, round(rate * FRAME_SECONDS))
    min_samples = round(min_seconds * rate)
    max_samples = round(max_seconds * rate)

    runs = _merge_runs(
        _speech_runs(samples, frame, silence_dbfs), round(min_gap_seconds * rate)
    )
    if not runs:
        raise ValueError(f"no speech above {silence_dbfs:g} dBFS; check the note and the threshold")

    pieces: list[tuple[int, int]] = []
    for start, end in runs:
        while end - start > max_samples:
            if end - start < 2 * min_samples:
                # No room for two legal clips: split in the middle.
                cut = start + (end - start) // 2
            else:
                # Leave at least ``min_samples`` behind, and prefer the quietest frame in the
                # second before the limit so clips stay long without a hard cut.
                target = min(max_samples, end - start - min_samples)
                cut = _best_cut(
                    samples, start, end, frame,
                    max(min_samples, target - round(SEARCH_SECONDS * rate)), target,
                )
            pieces.append((start, cut))
            start = cut
        pieces.append((start, end))

    segments = [(start, end) for start, end in pieces if end - start >= min_samples]
    if not segments:
        raise ValueError(f"no segment reaches the {min_seconds:g} s minimum")
    return segments


def _safe_stem(path: Path) -> str:
    stem = _ID_SAFE.sub("-", path.stem).strip("-").lower()
    return stem or "note"


def write_segments(
    samples: np.ndarray,
    segments: list[tuple[int, int]],
    *,
    out_dir: Path | str,
    speaker: str,
    prefix: str,
    transcripts_path: Path | str,
    source_stem: str,
    rate: int = RATE,
    overwrite: bool = False,
) -> dict:
    """Write the clips plus a transcript skeleton. Refuses to overwrite without ``overwrite``."""
    out_dir = Path(out_dir)
    transcripts_path = Path(transcripts_path)
    paths = []
    rows = []
    for index, (start, end) in enumerate(segments, start=1):
        uid = f"{prefix}-{source_stem}-{index:03d}"
        path = out_dir / f"{uid}.wav"
        paths.append(path)
        rows.append(
            {
                "id": uid,
                "audio": path.name,
                "speaker": speaker,
                "lang": "mix",
                "category": None,
                "text": "",
                "en_spans": [],
            }
        )
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{existing[0]} exists; use --overwrite to re-cut these clips")
    if transcripts_path.exists() and not overwrite:
        raise FileExistsError(f"{transcripts_path} exists; use --overwrite to re-cut this skeleton")

    out_dir.mkdir(parents=True, exist_ok=True)
    measured = []
    for (start, end), path in zip(segments, paths, strict=True):
        sf.write(path, samples[start:end], rate, subtype="PCM_16")
        info = inspect_audio(path)
        measured.append(
            {
                "id": path.stem,
                "seconds": round(float(info["duration"]), 3),
                "clipping_ratio": round(float(info["clipping_ratio"]), 6),
                "silent": bool(info["silent"]),
            }
        )
    transcripts_path.parent.mkdir(parents=True, exist_ok=True)
    transcripts_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return {
        "clips": len(rows),
        "seconds": round(sum(item["seconds"] for item in measured), 1),
        "silent_clips": [item["id"] for item in measured if item["silent"]],
        "clipping_clips": [item["id"] for item in measured if item["clipping_ratio"] > 0.01],
        "out_dir": str(out_dir),
        "transcripts": str(transcripts_path),
        "duration_bounds": [round(measured[0]["seconds"], 3), round(measured[-1]["seconds"], 3)]
        if measured
        else [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="segment_notes.py", description=__doc__)
    parser.add_argument("--audio", required=True, nargs="+", type=Path, help="note file(s)")
    parser.add_argument("--speaker", required=True, help="pseudonymous speaker id, e.g. S2")
    parser.add_argument("--out-dir", required=True, type=Path, help="destination for the wav clips")
    parser.add_argument("--transcripts", required=True, type=Path, help="JSONL skeleton to fill")
    parser.add_argument("--prefix", default=None, help="id prefix (default: <speaker>-note)")
    parser.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--silence-dbfs", type=float, default=DEFAULT_SILENCE_DBFS)
    parser.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP)
    parser.add_argument("--dry-run", action="store_true", help="only report the cut plan")
    parser.add_argument("--overwrite", action="store_true", help="replace existing clips/skeleton")
    args = parser.parse_args(argv)

    speaker = args.speaker.strip()
    if not speaker or not re.fullmatch(r"[A-Za-z0-9_-]+", speaker):
        parser.error("--speaker must be alphanumeric/underscore/dash")
    prefix = args.prefix or f"{speaker.lower()}-note"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        parser.error("--prefix must be alphanumeric/underscore/dash/dot")

    plans = []
    for path in args.audio:
        if not path.is_file():
            parser.error(f"audio not found: {path}")
        try:
            samples = load_audio_16k(path)
        except RuntimeError as error:
            parser.error(
                f"cannot decode {path}: {error}. Convert first, e.g. "
                f"`afconvert -f WAVE -d LEI16@16000 -c 1 {path} out.wav`"
            )
        plans.append((path, samples, plan_segments(
            samples,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            silence_dbfs=args.silence_dbfs,
            min_gap_seconds=args.min_gap,
        )))

    for path, samples, segments in plans:
        durations = [round((end - start) / RATE, 2) for start, end in segments]
        spread = f"{min(durations):.2f}-{max(durations):.2f} s" if durations else "n/a"
        print(
            f"{path}: {len(samples) / RATE:7.1f} s -> {len(segments)} clip(s) {spread}",
            file=sys.stderr,
        )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "clips": sum(len(s) for _, _, s in plans)}, indent=2))
        return 0

    summaries = [
        write_segments(
            samples,
            segments,
            out_dir=args.out_dir,
            speaker=speaker,
            prefix=prefix,
            transcripts_path=args.transcripts,
            source_stem=_safe_stem(path),
            overwrite=args.overwrite,
        )
        for path, samples, segments in plans
    ]
    print(json.dumps(summaries, indent=2))
    print(
        "next: fill text/category/en_spans in "
        f"{args.transcripts}, then\n"
        f"  uv run recordings/import_manifest.py --transcripts {args.transcripts} "
        f"--audio-dir {args.out_dir} --out <manifest.jsonl>",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
