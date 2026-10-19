"""Shared plumbing for the baseline runners (design: ``docs/05-evaluation.md`` §7).

A baseline is any program that turns audio into text. Each runner wraps one system, chooses a
variant (e.g. Whisper tiny with the language forced to Italian), and writes hypothesis files that
``stt/eval.py`` scores. This module owns the parts that must be identical across systems:

- the ``transcribe(wav) -> str`` contract (``external_transcriber`` runs a CLI and takes the last
  non-empty stdout line, so Whisper/Whistle/Parakeet wrappers stay tiny);
- hypothesis writing (``{"id", "hyp", "latency_s"}``, one JSON object per line);
- timing (RTFx = audio seconds / wall seconds) and the system's size on disk;
- **hotword fairness** (``docs/05`` §7): the fair list comes from the R scripts and the curated
  English-terms list only; anything that includes the test set's terms is an *oracle* and is
  labelled as such.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stt.manifest import load_manifest

__all__ = [
    "Transcriber",
    "dir_size",
    "external_transcriber",
    "fair_hotwords",
    "hotwords_for",
    "oracle_hotwords",
    "read_terms",
    "render_command",
    "resolve_audio",
    "rtfx",
    "run_set",
    "run_sets",
    "terms_in_rows",
    "write_hypotheses",
]

#: The one contract every baseline implements: audio file in, text out.
Transcriber = Callable[[Path], str]

#: ``--flag {lang}`` / ``--flag={keywords}`` pairs, removed when the value is absent.
_FLAG_PLACEHOLDER = re.compile(r"\s-{1,2}[\w-]+(?:=|\s+)\{(lang|keywords)\}")


def resolve_audio(row: dict, audio_root: Path | str) -> Path:
    """Audio path for a manifest row: absolute paths win, relative ones are under ``audio_root``."""
    path = Path(row["audio"])
    return path if path.is_absolute() else Path(audio_root) / path


def render_command(
    command: str, *, lang: str = "", keywords_path: Path | str | None = None
) -> str:
    """Fill a CLI template for the Whistle/Parakeet runners.

    ``{lang}`` and ``{keywords}`` are substituted; when a value is absent (auto language, no
    hotwords) the whole ``--flag {placeholder}`` / ``--flag={placeholder}`` pair is removed, so
    the audio path can never be swallowed as a flag value.
    """
    def drop(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "lang" and not lang:
            return ""
        if name == "keywords" and keywords_path is None:
            return ""
        return match.group(0)

    command = _FLAG_PLACEHOLDER.sub(drop, command)
    command = command.replace("{lang}", lang)
    command = command.replace("{keywords}", "" if keywords_path is None else str(keywords_path))
    return command.strip()


def external_transcriber(
    command: str, timeout_s: float = 600.0
) -> Transcriber:
    """Wrap a CLI as a transcriber: ``{audio}`` is substituted, the last stdout line is the text."""
    if "{audio}" not in command:
        command += " {audio}"

    def transcribe(path: Path) -> str:
        full = command.replace("{audio}", str(path))
        proc = subprocess.run(
            full,
            shell=True,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip().splitlines()
            detail = stderr[-1] if stderr else "no stderr"
            raise RuntimeError(f"command failed with exit {proc.returncode}: {detail}")
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        return lines[-1] if lines else ""

    return transcribe


def write_hypotheses(path: Path | str, hypotheses: list[dict]) -> None:
    """Write the hypothesis file consumed by ``stt/eval.py``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in hypotheses) + ("\n" if hypotheses else ""),
        encoding="utf-8",
    )


def rtfx(audio_seconds: float, wall_seconds: float) -> float | None:
    """Real-time factor: audio seconds processed per wall-clock second."""
    if wall_seconds <= 0:
        return None
    return audio_seconds / wall_seconds


def dir_size(paths: list[Path]) -> int:
    """Total bytes of the given files (missing files count as zero)."""
    return sum(path.stat().st_size for path in paths if path.exists())


def run_set(
    transcribe: Transcriber,
    rows: list[dict],
    audio_root: Path | str,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Transcribe one manifest and collect hypotheses, timings and failures.

    A failing clip yields an empty hypothesis and an entry in ``errors``; the run continues.
    """
    selected = rows[:limit] if limit else rows
    hypotheses: list[dict] = []
    errors: list[dict] = []
    audio_seconds = 0.0
    wall_start = time.perf_counter()

    for index, row in enumerate(selected, start=1):
        path = resolve_audio(row, audio_root)
        audio_seconds += float(row.get("duration") or 0.0)
        started = time.perf_counter()
        try:
            hyp = transcribe(path)
        except Exception as exc:  # noqa: BLE001 - recorded and reported per clip
            errors.append({"id": row["id"], "error": str(exc)})
            hyp = ""
        latency = time.perf_counter() - started
        hypotheses.append({"id": row["id"], "hyp": hyp, "latency_s": round(latency, 4)})
        if progress is not None:
            progress(index, len(selected))

    return {
        "hypotheses": hypotheses,
        "errors": errors,
        "audio_seconds": audio_seconds,
        "wall_seconds": time.perf_counter() - wall_start,
    }


def read_terms(path: Path | str) -> list[str]:
    """Read a curated term list (one per line, ``#`` comments and blanks ignored, deduped)."""
    path = Path(path)
    if not path.exists():
        return []
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#") and term not in terms:
            terms.append(term)
    return terms


def terms_in_rows(rows: list[dict]) -> list[str]:
    """English-origin reference words of mixed rows, from ``en_spans`` (deduped, ordered)."""
    terms: list[str] = []
    for row in rows:
        words = (row.get("norm") or row.get("text") or "").split()
        for start, end in row.get("en_spans") or []:
            for word in words[start:end]:
                if word not in terms:
                    terms.append(word)
    return terms


def fair_hotwords(scripts_dir: Path | str, terms_path: Path | str) -> list[str]:
    """The fair hotword list: R-script English terms + the curated list. Never test terms."""
    scripts_dir = Path(scripts_dir)
    terms: list[str] = []
    for path in sorted(scripts_dir.glob("R-*.jsonl")):
        for row in _read_jsonl(path):
            for term in terms_in_rows([row]):
                if term not in terms:
                    terms.append(term)
    for term in read_terms(terms_path):
        if term not in terms:
            terms.append(term)
    return terms


def oracle_hotwords(rows: list[dict], fair: list[str]) -> list[str]:
    """The oracle list: fair terms plus the test set's own English terms (labelled in reports)."""
    terms = list(fair)
    for term in terms_in_rows(rows):
        if term not in terms:
            terms.append(term)
    return terms


def hotwords_for(
    mode: str,
    rows: list[dict],
    scripts_dir: Path | str,
    terms_path: Path | str,
) -> tuple[list[str], bool]:
    """Resolve ``--hotwords {none,fair,oracle}`` into ``(words, is_oracle)``."""
    if mode == "none":
        return [], False
    fair = fair_hotwords(scripts_dir, terms_path)
    if mode == "fair":
        return fair, False
    if mode == "oracle":
        return oracle_hotwords(rows, fair), True
    raise ValueError(f"unknown hotwords mode {mode!r}")


def run_sets(
    system: str,
    variant: str,
    sets: list[str],
    manifests_dir: Path | str,
    audio_root: Path | str,
    out_dir: Path | str,
    build_transcriber: Callable[[list[str]], Transcriber],
    hotwords_mode: str,
    scripts_dir: Path | str,
    terms_path: Path | str,
    meta_extra: dict[str, Any] | None = None,
    limit: int | None = None,
    size_files: list[Path] | None = None,
) -> int:
    """Run one variant over several sets and write ``<out_dir>/<set>.jsonl`` + ``meta.json``.

    ``build_transcriber`` receives the resolved hotword list (empty unless requested) and returns
    the transcriber. Returns 0 on success, 1 if any set failed.
    """
    manifests_dir = Path(manifests_dir)
    out_dir = Path(out_dir)
    meta: dict[str, Any] = {
        "system": system,
        "variant": variant,
        "hotwords_mode": hotwords_mode,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sets": {},
        **(meta_extra or {}),
    }
    code = 0
    for set_name in sets:
        manifest_path = manifests_dir / f"{set_name}.jsonl"
        if not manifest_path.exists():
            print(f"error: manifest not found: {manifest_path}")
            code = 1
            continue
        rows = load_manifest(manifest_path)
        hotwords, is_oracle = hotwords_for(hotwords_mode, rows, scripts_dir, terms_path)
        if is_oracle:
            print("warning: oracle hotwords include the test set's English terms")
        transcribe = build_transcriber(hotwords)
        result = run_set(transcribe, rows, audio_root, limit=limit)
        write_hypotheses(out_dir / f"{set_name}.jsonl", result["hypotheses"])
        meta["sets"][set_name] = {
            "manifest": str(manifest_path),
            "n": len(result["hypotheses"]),
            "errors": result["errors"],
            "audio_seconds": round(result["audio_seconds"], 2),
            "wall_seconds": round(result["wall_seconds"], 3),
            "rtfx": rtfx(result["audio_seconds"], result["wall_seconds"]),
            "hotwords": len(hotwords),
            "oracle_hotwords": is_oracle,
        }
        print(
            f"{set_name}: {len(result['hypotheses'])} clips, "
            f"{result['audio_seconds']:.0f}s audio, "
            f"RTFx {meta['sets'][set_name]['rtfx']:.1f}, "
            f"{len(result['errors'])} errors"
            if meta["sets"][set_name]["rtfx"]
            else f"{set_name}: {len(result['hypotheses'])} clips, {len(result['errors'])} errors"
        )
        if result["errors"]:
            code = 1
    if size_files:
        meta["size_bytes"] = dir_size(size_files)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}")
    return code


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
