"""Import, validate, dedupe and stratify the mixed IT+EN sentence scripts (design: ``docs/03-data.md`` §6).

The sentence *texts* are produced externally — by any LLM or by hand — following the format and
prompt in ``recordings/README.md``. This tool is the gatekeeper: it checks every candidate against
the schema and the category rules, removes exact and near duplicates across the whole scripts
directory, assigns speaker stratification for the test script T, and reports quotas and review
status.

Candidate format (one JSON object per line):

```json
{"text": "ho spostato il meeting a domani", "category": "single_word",
 "en_spans": [[3, 4]], "generator": "claude-…", "date": "2026-10-19"}
```

``text`` must already be in canonical form (``normalize(text) == text``); ``en_spans`` are
``[start_word, end_word)`` index pairs into ``text.split()``. The stored scripts are JSONL files
under the scripts directory (``T.jsonl``, ``R-S1.jsonl``, ``R-S2.jsonl``, ``pool.jsonl``) whose
rows are exactly the manifest fragment the recordings step needs.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from stt.manifest import norm_hash
from stt.text import normalize

__all__ = [
    "CATEGORIES",
    "NEAR_DUP_JACCARD",
    "POOL_RANGE",
    "QUOTAS_T",
    "R_RANGE",
    "T_SPEAKERS",
    "ScriptError",
    "evaluate_stats",
    "import_candidates",
    "load_script",
    "main",
    "mark_reviewed",
    "review_report",
    "stratify_speakers",
    "validate_candidate",
]

#: Every category from ``docs/03-data.md`` §6, in canonical order.
CATEGORIES = (
    "single_word",
    "phrase",
    "brand",
    "acronym",
    "italianized",
    "switch",
    "control_it",
    "control_en",
)

#: Fixed quotas for the test script T (sum = 800, four speakers × 200).
QUOTAS_T = {
    "single_word": 200,
    "phrase": 133,
    "brand": 133,
    "acronym": 80,
    "italianized": 107,
    "switch": 80,
    "control_it": 40,
    "control_en": 27,
}

#: Accepted size ranges for the other scripts.
R_RANGE = (600, 1000)
POOL_RANGE = (3000, 5000)

T_SPEAKERS = ("S1", "S2", "S3", "S4")

#: Word-3-gram Jaccard above which two sentences are considered near-duplicates.
NEAR_DUP_JACCARD = 0.8

_SCRIPT_FILES = {"T": "T.jsonl", "R": "R-{speaker}.jsonl", "A": "A.jsonl", "pool": "pool.jsonl"}

#: Categories that must contain at least one English word, with the minimum span size.
_EN_CATEGORY_MIN_WORDS = {"phrase": 2, "switch": 2, "brand": 1, "acronym": 1, "italianized": 1}


class ScriptError(ValueError):
    """Raised when a candidate sentence does not match the script schema."""


def script_path(scripts_dir: Path | str, kind: str, speaker: str | None = None) -> Path:
    """Path of one script file inside ``scripts_dir`` (``kind`` is ``T``, ``R`` or ``pool``)."""
    if kind not in _SCRIPT_FILES:
        raise ScriptError(f"unknown kind {kind!r} (expected one of {sorted(_SCRIPT_FILES)})")
    if kind == "R":
        if speaker not in ("S1", "S2"):
            raise ScriptError("kind 'R' needs --speaker S1 or S2 (S3/S4 are held out from training)")
        return Path(scripts_dir) / _SCRIPT_FILES["R"].format(speaker=speaker)
    return Path(scripts_dir) / _SCRIPT_FILES[kind]


def validate_candidate(candidate: dict) -> dict:
    """Validate one candidate row and return the normalized record.

    Category rules: ``control_it`` carries no English words; ``single_word`` exactly one
    one-word span; ``phrase`` and ``switch`` at least two English words; ``brand``,
    ``acronym`` and ``italianized`` at least one; ``control_en`` is unrestricted.
    """
    if not isinstance(candidate, dict):
        raise ScriptError("candidate is not a JSON object")
    text = candidate.get("text")
    if not isinstance(text, str) or not text:
        raise ScriptError("'text' must be a non-empty string")
    category = candidate.get("category")
    if category not in CATEGORIES:
        raise ScriptError(f"'category' must be one of {CATEGORIES}, got {category!r}")
    if normalize(text) != text:
        raise ScriptError(
            f"text is not normalized: {normalize(text)!r} != {text!r} "
            "(generate already-normalized lowercase text, no punctuation)"
        )

    words = text.split()
    spans = candidate.get("en_spans") or []
    if not isinstance(spans, list):
        raise ScriptError("'en_spans' must be a list of [start, end) pairs")
    en_words = 0
    for span in spans:
        is_pair = (
            isinstance(span, list)
            and len(span) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in span)
            and 0 <= span[0] <= span[1] <= len(words)
        )
        if not is_pair:
            raise ScriptError(f"bad en_spans entry {span!r} (bounds 0..{len(words)})")
        en_words += span[1] - span[0]

    if category == "control_it" and en_words:
        raise ScriptError("control_it must not carry en_spans")
    if category == "single_word" and (len(spans) != 1 or en_words != 1):
        raise ScriptError("single_word must carry exactly one en_spans pair of exactly one word")
    minimum = _EN_CATEGORY_MIN_WORDS.get(category)
    if minimum is not None and en_words < minimum:
        raise ScriptError(f"{category} needs at least {minimum} English word(s) in en_spans")

    return {
        "text": text,
        "category": category,
        "en_spans": [list(span) for span in spans],
        "generator": str(candidate.get("generator", "unknown")),
        "date": str(candidate.get("date", date.today().isoformat())),
    }


def _trigrams(text: str) -> frozenset[tuple[str, str, str]]:
    words = text.split()
    return frozenset(zip(words, words[1:], words[2:]))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class _DedupeIndex:
    """Exact (norm-hash) and near-duplicate (3-gram Jaccard) index over stored scripts."""

    def __init__(self) -> None:
        self._hashes: set[str] = set()
        self._trigrams: list[frozenset] = []

    def add(self, text: str) -> None:
        self._hashes.add(norm_hash(text))
        self._trigrams.append(_trigrams(text))

    def is_duplicate(self, text: str) -> bool:
        if norm_hash(text) in self._hashes:
            return True
        candidate = _trigrams(text)
        return any(_jaccard(candidate, other) > NEAR_DUP_JACCARD for other in self._trigrams)


def load_script(path: Path) -> list[dict]:
    """Load a script JSONL file (empty list when the file does not exist yet)."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save_script(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


def stratify_speakers(rows: list[dict], speakers: tuple[str, ...] = T_SPEAKERS) -> None:
    """Balance T rows across speakers, **keeping the assignments already stored**.

    Existing rows must keep their speaker: the recorded files are mapped to cues by position
    (``S1_0001.wav`` is the first S1 row), so re-stratifying them would break every take.
    Unassigned rows (sentences added later) go to the least-loaded speaker, in script order.
    """
    loads = {speaker: 0 for speaker in speakers}
    for row in rows:
        if row.get("speaker") in loads:
            loads[row["speaker"]] += 1
    for category in CATEGORIES:
        for row in rows:
            if row.get("category") != category or row.get("speaker") in loads:
                continue
            speaker = min(speakers, key=lambda s: (loads[s], speakers.index(s)))
            row["speaker"] = speaker
            loads[speaker] += 1


def import_candidates(
    candidates: list[dict],
    kind: str,
    scripts_dir: Path | str,
    speaker: str | None = None,
) -> dict:
    """Validate and append candidates to one script, deduping against every script in the dir.

    Returns ``{"imported", "duplicates", "invalid": [(text, reason)], "ids": [...]}``. Invalid
    candidates are reported but never stored; duplicates are counted and skipped.
    """
    target = script_path(scripts_dir, kind, speaker)
    scripts_dir = Path(scripts_dir)

    index = _DedupeIndex()
    for existing_path in sorted(scripts_dir.glob("*.jsonl")):
        for row in load_script(existing_path):
            index.add(row["text"])

    rows = load_script(target)
    next_number = len(rows) + 1
    imported, duplicates, invalid, ids = 0, 0, [], []
    for candidate in candidates:
        try:
            row = validate_candidate(candidate)
        except ScriptError as exc:
            invalid.append({"text": str(candidate.get("text", "")), "reason": str(exc)})
            continue
        if index.is_duplicate(row["text"]):
            duplicates += 1
            continue
        row["id"] = f"{kind}-{speaker}-{next_number:04d}" if kind == "R" else f"{kind}-{next_number:04d}"
        row["reviewed"] = False
        next_number += 1
        index.add(row["text"])
        rows.append(row)
        ids.append(row["id"])
        imported += 1

    if kind == "T":
        stratify_speakers(rows)
    _save_script(target, rows)
    return {"imported": imported, "duplicates": duplicates, "invalid": invalid, "ids": ids}


def evaluate_stats(scripts_dir: Path | str) -> dict:
    """Counts and quota deltas for every script in the directory."""
    scripts_dir = Path(scripts_dir)
    stats: dict = {}
    t_rows = load_script(script_path(scripts_dir, "T"))
    if t_rows:
        per_category = {
            category: sum(1 for row in t_rows if row["category"] == category)
            for category in CATEGORIES
        }
        stats["T"] = {
            "total": len(t_rows),
            "per_speaker": {
                speaker: sum(1 for row in t_rows if row["speaker"] == speaker)
                for speaker in T_SPEAKERS
            },
            "categories": {
                category: {
                    "quota": QUOTAS_T[category],
                    "actual": per_category[category],
                    "delta": QUOTAS_T[category] - per_category[category],
                }
                for category in CATEGORIES
            },
            "reviewed": sum(1 for row in t_rows if row["reviewed"]),
        }
    for speaker in ("S1", "S2"):
        rows = load_script(script_path(scripts_dir, "R", speaker))
        if rows:
            stats[f"R-{speaker}"] = {
                "count": len(rows),
                "range": list(R_RANGE),
                "reviewed": sum(1 for row in rows if row["reviewed"]),
            }
    pool = load_script(script_path(scripts_dir, "pool"))
    if pool:
        stats["pool"] = {
            "count": len(pool),
            "range": list(POOL_RANGE),
            "reviewed": sum(1 for row in pool if row["reviewed"]),
        }
    return stats


def review_report(rows: list[dict]) -> dict:
    """Review status of one script: totals and the list of unreviewed rows."""
    unreviewed = [
        {"id": row["id"], "text": row["text"]} for row in rows if not row.get("reviewed")
    ]
    return {"total": len(rows), "reviewed": len(rows) - len(unreviewed), "unreviewed": unreviewed}


def mark_reviewed(rows: list[dict], reviewer: str, ids: list[str] | None = None) -> int:
    """Mark rows as reviewed (all when ``ids`` is None); returns how many were marked."""
    wanted = set(ids) if ids is not None else None
    marked = 0
    for row in rows:
        if wanted is not None and row["id"] not in wanted:
            continue
        if not row.get("reviewed"):
            row["reviewed"] = True
            row["reviewer"] = reviewer
            row["review_date"] = date.today().isoformat()
            marked += 1
    return marked


def _print_stats(stats: dict) -> None:
    for name, section in stats.items():
        if name == "T":
            print(f"T: {section['total']}/{sum(QUOTAS_T.values())} sentences, reviewed {section['reviewed']}")
            for speaker, count in section["per_speaker"].items():
                print(f"  {speaker}: {count} (target 200)")
            for category, entry in section["categories"].items():
                flag = "" if entry["delta"] == 0 else f"  (missing {entry['delta']})"
                print(f"  {category}: {entry['actual']}/{entry['quota']}{flag}")
        else:
            low, high = section["range"]
            print(f"{name}: {section['count']} (range {low}-{high}), reviewed {section['reviewed']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_scripts.py",
        description="Import, dedupe and report the mixed IT+EN sentence scripts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    import_parser = sub.add_parser("import", help="import candidate sentences")
    import_parser.add_argument("--kind", required=True, choices=["T", "R", "pool"])
    import_parser.add_argument("--speaker", default=None, help="S1 or S2 (required for kind R)")
    import_parser.add_argument("--in", dest="input", required=True, help="candidates JSONL")
    import_parser.add_argument("--scripts-dir", default="recordings/scripts")

    stats_parser = sub.add_parser("stats", help="counts and quota deltas")
    stats_parser.add_argument("--scripts-dir", default="recordings/scripts")

    review_parser = sub.add_parser("review", help="list unreviewed sentences")
    review_parser.add_argument("--kind", required=True, choices=["T", "R", "pool"])
    review_parser.add_argument("--speaker", default=None, help="S1 or S2 (required for kind R)")
    review_parser.add_argument("--scripts-dir", default="recordings/scripts")
    review_parser.add_argument("--limit", type=int, default=20)

    mark_parser = sub.add_parser("mark-reviewed", help="mark sentences as reviewed")
    mark_parser.add_argument("--kind", required=True, choices=["T", "R", "pool"])
    mark_parser.add_argument("--speaker", default=None, help="S1 or S2 (required for kind R)")
    mark_parser.add_argument("--scripts-dir", default="recordings/scripts")
    mark_parser.add_argument("--ids", nargs="*", default=None, help="ids to mark (default: all)")
    mark_parser.add_argument("--reviewer", required=True)

    args = parser.parse_args(argv)
    scripts_dir = Path(args.scripts_dir)

    if args.command == "review":
        rows = load_script(script_path(scripts_dir, args.kind, args.speaker))
        report = review_report(rows)
        print(f"{args.kind}: {report['reviewed']}/{report['total']} reviewed")
        for row in report["unreviewed"][: args.limit]:
            print(f"  {row['id']}: {row['text']}")
        return 0

    if args.command == "mark-reviewed":
        path = script_path(scripts_dir, args.kind, args.speaker)
        rows = load_script(path)
        marked = mark_reviewed(rows, args.reviewer, args.ids)
        _save_script(path, rows)
        print(f"marked {marked} sentence(s) as reviewed by {args.reviewer}")
        return 0

    if args.command == "import":
        candidates = load_script(Path(args.input))
        result = import_candidates(candidates, args.kind, scripts_dir, speaker=args.speaker)
        print(
            f"imported {result['imported']}, duplicates {result['duplicates']}, "
            f"invalid {len(result['invalid'])}"
        )
        for bad in result["invalid"]:
            print(f"  invalid: {bad['reason']}: {bad['text'][:60]!r}")
        _print_stats(evaluate_stats(scripts_dir))
        return 1 if result["invalid"] else 0

    _print_stats(evaluate_stats(scripts_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
