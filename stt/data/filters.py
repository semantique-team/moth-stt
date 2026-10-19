"""Filters and build report (design: ``docs/03-data.md`` §3).

Every rule is a named reason; a dropped utterance is counted by reason and the build report is
committed next to the manifests. Order of checks: empty → charset → digits → duration →
speaking rate → CTC feasibility → sentence overlap with dev/test → audio health.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.manifest import norm_hash
from stt.text import KEEP_CHARS, normalize

__all__ = [
    "FilterConfig",
    "apply_filters",
    "build_report",
    "charset_ok",
    "ctc_feasible",
    "filter_row",
    "speaking_rate_ok",
]

_SILENCE_PEAK = 1e-3
_CLIPPING_RATIO = 0.01


@dataclass(frozen=True)
class FilterConfig:
    """Thresholds from ``docs/03-data.md`` §3; the speaking-rate range is set from the histogram."""

    min_seconds: float = 1.0
    max_seconds: float = 30.0
    warn_above_seconds: float = 20.0
    min_chars_per_second: float = 4.0
    max_chars_per_second: float = 30.0
    vocab_size: int = 1024
    check_audio: bool = True


def _char_is_textual(character: str) -> bool:
    """True for characters a transcript may legitimately contain (or that normalize folds)."""
    if character in KEEP_CHARS or character.isspace() or character.isdigit() or character.isascii():
        return True
    if unicodedata.category(character).startswith("P"):  # punctuation: dropped, never a problem
        return True
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", character) if not unicodedata.combining(c)
    ).lower()
    return bool(folded) and all(c in "abcdefghijklmnopqrstuvwxyz" for c in folded)


def charset_ok(text: str) -> bool:
    """True when the transcript is Latin-script text (emoji, CJK, Cyrillic → False).

    Punctuation and diacritics are fine (``normalize`` handles them); the check exists to drop
    utterances whose content would be silently lost by normalization.
    """
    normalized = normalize(text)
    if not normalized:
        return False
    return all(_char_is_textual(character) for character in text)


def speaking_rate_ok(text: str, duration: float, config: FilterConfig) -> bool:
    if duration <= 0:
        return False
    rate = len(text) / duration
    return config.min_chars_per_second <= rate <= config.max_chars_per_second


def ctc_feasible(text: str, duration: float, vocab_size: int = 1024) -> bool:
    """A rough CTC length check: encoder frames (50/s) must exceed the label length.

    With V pieces the true label length is unknown before the tokenizer exists; the worst case
    is one token per character, which is what this conservative check uses.
    """
    del vocab_size  # kept in the signature: the tokenizer-aware version will use it
    frames = int(duration * 50)
    return len(text) <= frames


def _audio_reason(path: Path) -> str | None:
    try:
        data, _ = sf.read(path, dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 - any decode failure is "unreadable"
        return "unreadable"
    if data.size == 0:
        return "unreadable"
    peak = float(np.abs(data).max())
    if peak < _SILENCE_PEAK:
        return "silent"
    if float((np.abs(data) >= 0.999).mean()) > _CLIPPING_RATIO:
        return "clipping"
    return None


def filter_row(
    row: dict,
    config: FilterConfig | None = None,
    eval_hashes: set[str] | None = None,
    audio_path: Path | str | None = None,
) -> str | None:
    """Return the drop reason for one row, or ``None`` when it is kept."""
    config = config or FilterConfig()
    text = row.get("text") or ""
    normalized = normalize(text, row.get("lang") if row.get("lang") in ("it", "en") else "it")

    if not normalized:
        return "empty"
    if not charset_ok(text):
        return "charset"
    if any(character.isdigit() for character in text):
        return "digits"
    duration = float(row.get("duration") or 0.0)
    if not config.min_seconds <= duration <= config.max_seconds:
        return "duration"
    if not speaking_rate_ok(normalized, duration, config):
        return "speaking_rate"
    if not ctc_feasible(normalized, duration, config.vocab_size):
        return "ctc_length"
    if eval_hashes and norm_hash(text) in eval_hashes:
        return "overlap"
    if config.check_audio and audio_path is not None:
        reason = _audio_reason(Path(audio_path))
        if reason is not None:
            return reason
    return None


def apply_filters(
    rows: list[dict],
    config: FilterConfig | None = None,
    eval_hashes: set[str] | None = None,
    audio_root: Path | str | None = None,
) -> tuple[list[dict], dict]:
    """Keep the valid rows and count every drop by reason."""
    config = config or FilterConfig()
    kept: list[dict] = []
    dropped: dict[str, int] = {}
    seconds = 0.0
    for row in rows:
        audio_path = None
        if audio_root is not None and row.get("audio"):
            audio_path = Path(audio_root) / row["audio"]
        reason = filter_row(row, config, eval_hashes, audio_path)
        if reason is None:
            kept.append(row)
            seconds += float(row.get("duration") or 0.0)
        else:
            dropped[reason] = dropped.get(reason, 0) + 1
    report = {
        "total": len(rows),
        "kept": len(kept),
        "dropped": dict(sorted(dropped.items())),
        "seconds": seconds,
    }
    return kept, report


def build_report(name: str, report: dict, kept: list[dict]) -> str:
    """Human-readable build report (also the basis of the ``EXPERIMENTS.md`` entry)."""
    hours = report["seconds"] / 3600.0
    speakers = len({row.get("speaker") for row in kept if row.get("speaker")})
    lines = [
        (
            f"{name}: kept {report['kept']}/{report['total']} utterances, "
            f"{hours:.2f} h, {speakers} speakers"
        ),
    ]
    if report["dropped"]:
        reasons = ", ".join(f"{reason}: {count}" for reason, count in report["dropped"].items())
        lines.append(f"  dropped: {reasons}")
    else:
        lines.append("  dropped: none")
    return "\n".join(lines)
