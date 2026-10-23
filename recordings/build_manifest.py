"""Build monolingual recordings: validate the audio and join it with the scripts (design: ``docs/03-data.md`` §7).

The manifest row produced here is the manifest fragment (``docs/03-data.md`` §4) for the
recorded sets: ``recordings/manifests/T.jsonl``, ``recordings/manifests/R-S1.jsonl``,
``recordings/manifests/R-S2.jsonl``.

Files and sentences are joined **by position**: for speaker ``S``, the i-th cue (1-based) of
the script is ``S_<i:04d>.wav`` (or ``.m4a``/``.flac``). The cue order is the script file order,
filtered by speaker for T. Every file is measured (duration, peak, clipping) and checked
against the data rules: 1–30 s, not silent, clipping ≤ 1%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from recordings.generate_scripts import load_script, script_path

__all__ = [
    "AUDIO_EXTENSIONS",
    "build_manifest",
    "build_report",
    "cues_for",
    "default_audio_dir",
    "find_audio",
    "inspect_audio",
    "main",
]

AUDIO_EXTENSIONS = (".wav", ".m4a", ".flac")

MIN_SECONDS = 1.0
MAX_SECONDS = 30.0
SILENCE_PEAK = 1e-3
CLIPPING_RATIO = 0.01


def default_audio_dir(kind: str, speaker: str | None = None) -> Path:
    """Where a recording kind lives: T at the legacy root, A and R in separate trees.

    T and R are different scripts for the same speaker; sharing ``recordings/audio/S1`` would
    make ``<speaker>_<index>.wav`` ambiguous (and made a resume skip/overwrite the wrong cues).
    """
    if kind == "A":
        return Path("recordings/audio/anchor")
    if kind == "R":
        if speaker not in ("S1", "S2"):
            raise ValueError("kind R needs speaker S1 or S2")
        return Path("recordings/audio") / f"R-{speaker}"
    return Path("recordings/audio")


def cues_for(rows: list[dict], speaker: str | None) -> list[dict]:
    """The ordered cues for one speaker: T is filtered by ``speaker``, R is already per speaker."""
    if speaker is None:
        return rows
    return [row for row in rows if row.get("speaker") == speaker]


def find_audio(audio_dir: Path | str, speaker: str, index: int) -> Path | None:
    """Find ``<speaker>_<index:04d>`` with one of the allowed extensions, in preference order."""
    directory = Path(audio_dir) / speaker
    for extension in AUDIO_EXTENSIONS:
        candidate = directory / f"{speaker}_{index:04d}{extension}"
        if candidate.exists():
            return candidate
    return None


def inspect_audio(path: Path) -> dict:
    """Measure a file: ``{"duration", "peak", "clipping_ratio", "silent"}``."""
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    peak = float(np.abs(data).max()) if data.size else 0.0
    clipping = float((np.abs(data) >= 0.999).mean()) if data.size else 0.0
    return {
        "duration": data.shape[0] / rate,
        "peak": peak,
        "clipping_ratio": clipping,
        "silent": peak < SILENCE_PEAK,
    }


def build_manifest(
    kind: str,
    scripts_dir: Path | str,
    audio_dir: Path | str,
    speaker: str | None = None,
    allow_unreviewed: bool = False,
) -> dict:
    """Validate the audio for one recording set and return ``{"rows", "errors", "warnings", "speakers"}``."""
    if kind not in ("T", "R", "A"):
        raise ValueError("kind must be 'T', 'R' or 'A' (the pool has no audio)")
    if kind == "A" and speaker is None:
        raise ValueError("kind 'A' needs --speaker (each speaker records the same anchor sentences)")
    scripts_dir, audio_dir = Path(scripts_dir), Path(audio_dir)
    script_rows = load_script(script_path(scripts_dir, kind, speaker))
    if not script_rows:
        raise ValueError(f"no cues found in {script_path(scripts_dir, kind, speaker)}")

    speakers = sorted({str(row.get("speaker", speaker)) for row in script_rows})
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    per_speaker: dict[str, dict] = {}

    for current in speakers:
        cues = script_rows if kind in ("R", "A") else cues_for(script_rows, current)
        found = 0
        seconds = 0.0
        for index, cue in enumerate(cues, start=1):
            if kind in ("T", "A") and not cue.get("reviewed") and not allow_unreviewed:
                errors.append(f"{current}: cue {cue['id']} is not reviewed (see the review step)")
                continue
            path = find_audio(audio_dir, current, index)
            if path is None:
                errors.append(f"{current}: missing audio {current}_{index:04d}.wav")
                continue
            try:
                measured = inspect_audio(path)
            except Exception as exc:  # noqa: BLE001 - the error text is the report
                errors.append(f"{current}: cannot read {path.name} ({exc})")
                continue
            if measured["silent"]:
                errors.append(f"{current}: {path.name} is silent")
                continue
            if measured["clipping_ratio"] > CLIPPING_RATIO:
                errors.append(
                    f"{current}: {path.name} clipping on {measured['clipping_ratio']:.1%} of samples"
                )
                continue
            if not MIN_SECONDS <= measured["duration"] <= MAX_SECONDS:
                errors.append(
                    f"{current}: {path.name} duration {measured['duration']:.2f}s outside "
                    f"{MIN_SECONDS}-{MAX_SECONDS}s"
                )
                continue
            found += 1
            seconds += measured["duration"]
            rows.append(
                {
                    "id": cue["id"],
                    "audio": f"{current}/{path.name}",
                    "text": cue["text"],
                    "norm": cue["text"],
                    "duration": round(measured["duration"], 3),
                    "source": {"T": "recorded_t", "A": "recorded_a", "R": "recorded_r"}[kind],
                    "split": "train" if kind == "R" else "test",
                    "speaker": current,
                    "lang": "mix",
                    "category": cue["category"],
                    "en_spans": cue["en_spans"],
                }
            )
        per_speaker[current] = {"expected": len(cues), "found": found, "total_seconds": round(seconds, 1)}
        for extra in sorted((audio_dir / current).glob("*")):
            if extra.suffix.lower() in AUDIO_EXTENSIONS:
                stem = extra.stem
                try:
                    index = int(stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    warnings.append(f"{current}: file {extra.name} does not match the naming scheme")
                    continue
                if index > len(cues):
                    warnings.append(f"{current}: extra file {extra.name} (no cue {index})")

    return {"rows": rows, "errors": errors, "warnings": warnings, "speakers": per_speaker}


def build_report(result: dict) -> str:
    lines = []
    for speaker, section in result["speakers"].items():
        lines.append(
            f"{speaker}: {section['found']}/{section['expected']} clips, "
            f"{section['total_seconds']:.1f}s total"
        )
    lines.append(f"errors: {len(result['errors'])}, warnings: {len(result['warnings'])}")
    lines.extend(f"  error: {error}" for error in result["errors"][:10])
    lines.extend(f"  warning: {warning}" for warning in result["warnings"][:10])
    return "\n".join(lines)


def write_cues(rows: list[dict], speaker: str, path: Path) -> int:
    """Write the printable cue sheet ``<NNNN> <id> <text>`` for one speaker."""
    cues = cues_for(rows, speaker) if any("speaker" in row for row in rows) else rows
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{index:04d}\t{cue['id']}\t{cue['text']}" for index, cue in enumerate(cues, start=1))
        + "\n",
        encoding="utf-8",
    )
    return len(cues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_manifest.py",
        description="Validate recorded audio against the scripts and write a manifest fragment.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="validate audio and write the manifest")
    build.add_argument("--kind", required=True, choices=["T", "R", "A"])
    build.add_argument("--speaker", default=None, help="S1-S4 for kind A, S1/S2 for kind R")
    build.add_argument("--scripts-dir", default="recordings/scripts")
    build.add_argument(
        "--audio-dir",
        default=None,
        help="audio root (default recordings/audio, or recordings/audio/anchor for kind A)",
    )
    build.add_argument("--out", default=None, help="output manifest (default recordings/manifests/<kind>.jsonl)")
    build.add_argument("--allow-unreviewed", action="store_true")

    cue = sub.add_parser("cue", help="write the printable cue sheet for one speaker")
    cue.add_argument("--kind", required=True, choices=["T", "R", "A"])
    cue.add_argument("--speaker", required=True)
    cue.add_argument("--scripts-dir", default="recordings/scripts")
    cue.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.command == "cue":
        rows = load_script(script_path(Path(args.scripts_dir), args.kind, args.speaker))
        count = write_cues(rows, args.speaker, Path(args.out))
        if count == 0:
            print("no cues found for this speaker")
            return 1
        print(f"wrote {count} cues to {args.out}")
        return 0

    if args.kind == "R" and not args.speaker:
        parser.error("--speaker is required for kind R")
    audio_dir = args.audio_dir or default_audio_dir(args.kind, args.speaker)
    result = build_manifest(
        args.kind,
        args.scripts_dir,
        audio_dir,
        speaker=args.speaker,
        allow_unreviewed=args.allow_unreviewed,
    )
    print(build_report(result))
    out = Path(args.out) if args.out else Path("recordings/manifests") / (
        f"{args.kind}.jsonl" if args.kind == "T" else f"{args.kind}-{args.speaker}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result["rows"]) + ("\n" if result["rows"] else ""),
        encoding="utf-8",
    )
    print(f"wrote {len(result['rows'])} rows to {out}")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
