"""Manifest schema: one JSONL format for every split (design: ``docs/03-data.md`` §4).

A manifest is the only input to training and evaluation. One row:

```json
{"id": "cvit27-000123", "audio": "path/or/shard-ref", "text": "raw transcript",
 "norm": "normalized transcript", "duration": 4.37, "source": "cv_it_27", "split": "train",
 "speaker": "abc123", "lang": "it", "category": null, "en_spans": []}
```

``lang`` is ``it``, ``en`` or ``mix``. ``category`` and ``en_spans`` are filled only for the
recorded mixed sets; ``en_spans`` holds ``[start_word, end_word)`` index pairs into
``norm.split()``. ``norm`` must already be in canonical form, i.e. ``normalize(norm) == norm``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stt.text import normalize

__all__ = ["ManifestError", "load_manifest", "norm_hash"]

#: Keys every row must carry (``audio``/``speaker``/``category``/``en_spans`` are optional).
REQUIRED_KEYS = ("id", "text", "norm", "duration", "source", "split", "lang")

_LANGS = ("it", "en", "mix")
_STRING_KEYS = ("text", "norm", "source", "split")
_OPTIONAL_STRING_KEYS = ("audio", "speaker", "category")


class ManifestError(ValueError):
    """Raised when a manifest file does not match the schema."""


def norm_hash(text: str, lang_hint: str = "it") -> str:
    """sha1 of the normalized text — the sentence-dedupe key (``docs/03-data.md`` §3)."""
    return hashlib.sha1(normalize(text, lang_hint).encode("utf-8")).hexdigest()


def load_manifest(path: str | Path, max_errors: int = 10) -> list[dict]:
    """Load and validate a manifest JSONL file.

    Collects problems line by line and raises :class:`ManifestError` listing the first
    ``max_errors`` of them; returns the validated rows in file order.
    """
    errors: list[str] = []
    rows: list[dict] = []
    seen: set[str] = set()

    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
        else:
            if not isinstance(row, dict):
                errors.append(f"line {line_no}: not a JSON object")
            else:
                row_errors = _validate_row(row, line_no)
                errors.extend(row_errors)
                if not row_errors:
                    uid = row["id"]
                    if uid in seen:
                        errors.append(f"line {line_no}: duplicate id {uid!r}")
                    else:
                        seen.add(uid)
                        rows.append(row)
        if len(errors) >= max_errors:
            break

    if errors:
        raise ManifestError("invalid manifest:\n" + "\n".join(f"- {error}" for error in errors))
    return rows


def _validate_row(row: dict, line_no: int) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in row:
            errors.append(f"line {line_no}: missing required key {key!r}")
    if errors:
        return errors

    uid = row["id"]
    if not isinstance(uid, str) or not uid:
        errors.append(f"line {line_no}: 'id' must be a non-empty string")
    if row["lang"] not in _LANGS:
        errors.append(f"line {line_no}: 'lang' must be one of {_LANGS}, got {row['lang']!r}")
    for key in _STRING_KEYS:
        if not isinstance(row[key], str):
            errors.append(f"line {line_no}: {key!r} must be a string")
    duration = row["duration"]
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
        errors.append(f"line {line_no}: 'duration' must be a non-negative number")
    for key in _OPTIONAL_STRING_KEYS:
        value = row.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(f"line {line_no}: {key!r} must be a string or null")

    if isinstance(row["norm"], str) and normalize(row["norm"]) != row["norm"]:
        errors.append(f"line {line_no}: 'norm' is not in canonical form (normalize(norm) != norm)")

    en_spans = row.get("en_spans")
    if en_spans is not None:
        n_words = len(row["norm"].split()) if isinstance(row["norm"], str) else 0
        if not isinstance(en_spans, list):
            errors.append(f"line {line_no}: 'en_spans' must be a list of [start, end) pairs")
        else:
            for span in en_spans:
                is_pair = (
                    isinstance(span, list)
                    and len(span) == 2
                    and all(isinstance(x, int) and not isinstance(x, bool) for x in span)
                    and 0 <= span[0] <= span[1] <= n_words
                )
                if not is_pair:
                    errors.append(
                        f"line {line_no}: bad en_spans entry {span!r} (bounds 0..{n_words})"
                    )
                    break

    if row["lang"] != "mix":
        if row.get("category"):
            errors.append(f"line {line_no}: 'category' is only valid on mixed rows (lang='mix')")
        if row.get("en_spans"):
            errors.append(f"line {line_no}: 'en_spans' is only valid on mixed rows (lang='mix')")
    return errors
