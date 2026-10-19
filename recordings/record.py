"""Guided recorder: one sentence per file, no fiddling with a DAW (design: ``docs/03-data.md`` §7).

For every cue it shows the sentence to read, records with Enter (through ``ffmpeg`` /
AVFoundation on the default microphone), and files it as ``<speaker>_<NNNN>.wav`` under
``recordings/audio/<speaker>/``. Keys: Enter = record/next, ``r`` = redo the same sentence,
``p`` = play back, ``s`` = skip, ``q`` = quit. Retakes and skips are appended to
``session.log.jsonl`` next to the audio.

Prerequisites: ``brew install ffmpeg``; macOS asks for microphone permission the first time (if
denied: System Settings → Privacy & Security → Microphone → enable for the terminal app).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from recordings.build_manifest import find_audio, inspect_audio
from recordings.generate_scripts import load_script, script_path

__all__ = ["append_session_log", "main", "next_audio_path", "plan_session"]

MIN_TAKE_SECONDS = 0.5


def next_audio_path(audio_dir: Path | str, speaker: str, index: int) -> Path:
    """Path of the next take: ``<audio_dir>/<speaker>/<speaker>_<index:04d>.wav``."""
    return Path(audio_dir) / speaker / f"{speaker}_{index:04d}.wav"


def plan_session(
    audio_dir: Path | str,
    speaker: str,
    total: int,
    start: int = 1,
    overwrite: bool = False,
) -> list[int]:
    """Cue indices (1-based) still to record, skipping what already exists unless ``overwrite``."""
    planned: list[int] = []
    for index in range(start, total + 1):
        if not overwrite and find_audio(audio_dir, speaker, index) is not None:
            continue
        planned.append(index)
    return planned


def append_session_log(log_path: Path | str, entry: dict) -> None:
    """Append one JSON line to the session log created by the recorder."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **entry}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _record_once(path: Path, device: str, max_seconds: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "avfoundation",
        "-i",
        device,
        "-ac",
        "1",
        "-ar",
        "48000",
        "-t",
        str(max_seconds),
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    input("    ● registrando… premi Enter per fermare ")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    return path.exists()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="record.py",
        description="Guided recording session: one sentence per file.",
    )
    parser.add_argument("--speaker", required=True, choices=["S1", "S2", "S3", "S4"])
    parser.add_argument("--kind", required=True, choices=["T", "R", "A"])
    parser.add_argument("--scripts-dir", default="recordings/scripts")
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="audio root (default recordings/audio, or recordings/audio/anchor for kind A)",
    )
    parser.add_argument("--start", type=int, default=1, help="first cue index (1-based)")
    parser.add_argument("--overwrite", action="store_true", help="re-record existing files")
    parser.add_argument("--dry-run", action="store_true", help="only list the cues to record")
    parser.add_argument("--max-seconds", type=int, default=40, help="safety cap per take")
    parser.add_argument("--device", default=":0", help="AVFoundation input (':0' = default mic)")
    args = parser.parse_args(argv)

    audio_dir = args.audio_dir or (
        "recordings/audio/anchor" if args.kind == "A" else "recordings/audio"
    )
    script_speaker = args.speaker if args.kind == "R" else None
    rows = load_script(script_path(Path(args.scripts_dir), args.kind, script_speaker))
    cues = [
        row
        for row in rows
        if args.kind in ("R", "A") or row.get("speaker") == args.speaker
    ]
    if not cues:
        print(f"no cues for {args.speaker} in the {args.kind} script — import the sentences first")
        return 1

    planned = plan_session(audio_dir, args.speaker, len(cues), args.start, args.overwrite)
    if args.dry_run:
        print(f"{args.speaker}: {len(cues)} cues, {len(planned)} to record")
        for index in planned:
            cue = cues[index - 1]
            print(f"  {index:04d}\t{cue['id']}\t{cue['text']}")
        return 0
    if not planned:
        print("nothing to record — all files exist (use --overwrite to redo)")
        return 0
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found — install it first: brew install ffmpeg")
        return 1

    log_path = Path(audio_dir) / args.speaker / "session.log.jsonl"
    recorded = retakes = skips = 0
    print(f"{args.speaker} — {args.kind}: {len(planned)} sentences to record. Ctrl-C to stop early.\n")
    try:
        for index in planned:
            cue = cues[index - 1]
            print(f"[{index}/{len(cues)}] {cue['id']}  ({cue['category']})")
            print(f'    "{cue["text"]}"')
            key = input("    [Enter] registra, [s] salta, [q] esci > ").strip().lower()
            if key == "q":
                break
            if key == "s":
                skips += 1
                append_session_log(log_path, {"id": cue["id"], "action": "skip"})
                continue
            while True:
                path = next_audio_path(audio_dir, args.speaker, index)
                if not _record_once(path, args.device, args.max_seconds):
                    print("    niente file salvato — riprova")
                    continue
                take = inspect_audio(path)["duration"]
                if take < MIN_TAKE_SECONDS:
                    print(f"    registrazione troppo corta ({take:.1f}s) — riprova")
                    continue
                print(f"    salvata {path.name} ({take:.1f}s)")
                choice = input("    [Enter] avanti, [p] ascolta, [r] rifai > ").strip().lower()
                if "p" in choice:
                    subprocess.run(["afplay", str(path)], check=False)
                    choice = input("    [Enter] avanti, [r] rifai > ").strip().lower()
                if "r" in choice:
                    retakes += 1
                    append_session_log(
                        log_path, {"id": cue["id"], "action": "retake", "duration": round(take, 2)}
                    )
                    continue
                break
            else:
                continue
            recorded += 1
            append_session_log(
                log_path, {"id": cue["id"], "action": "ok", "duration": round(take, 2)}
            )
    except KeyboardInterrupt:
        print("\ninterrotto")

    print(f"\nsessione: {recorded} registrate, {retakes} rifatte, {skips} saltate")
    print(f"log: {log_path}")
    print("quando hai finito: uv run recordings/build_manifest.py build --kind " + args.kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
