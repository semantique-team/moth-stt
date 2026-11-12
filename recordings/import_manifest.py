"""Import a natural mixed IT/EN dev collection into a validated manifest.

The guided recorder and `build_manifest.py` support T/A/R only, and the mixed-dev protocol
(`docs/guides/mixed-dev-and-quality-experiments.md`) explicitly forbids relabelling a natural
collection as R or T. This tool is the separate path: it joins an audio directory with
human transcripts, enforces the manifest schema and the recording rules, audits exact and
near-duplicate text against the stored scripts, and writes
``source="natural_mixed_dev"`` / ``split="dev"`` rows.

Transcript JSONL rows:

    {"id": "NM-0001", "audio": "S5/nm-0001.wav", "text": "ho spostato il meeting",
     "speaker": "S5", "category": "single_word", "en_spans": [[3, 4]]}

``audio`` is optional when the file is ``<audio-dir>/<id>.wav`` (or .m4a/.flac) or
``<audio-dir>/<speaker>/<id>.<ext>``. ``lang`` is always ``mix`` for this protocol. Exact
normalized duplicates are refused; near duplicates are reported for human review.

    uv run python recordings/import_manifest.py \
        --transcripts data/natural_mixed_dev/transcripts.jsonl \
        --audio-dir data/natural_mixed_dev/audio \
        --out data/manifests/natural_mixed_dev.jsonl

No audio leaves the machine; nothing is created unless every row validates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from recordings.build_manifest import (
    AUDIO_EXTENSIONS,
    CLIPPING_RATIO,
    MAX_SECONDS,
    MIN_SECONDS,
    inspect_audio,
)
from recordings.generate_scripts import (
    NEAR_DUP_JACCARD,
    _trigrams,
    load_script,
    validate_candidate,
)
from stt.manifest import norm_hash
from stt.text import normalize

__all__ = ["import_manifest", "main"]

SAMPLE_RATES = range(16_000, 96_001)


def _find_audio(root: Path, row: dict) -> Path | None:
    declared = row.get("audio")
    if declared:
        path = root / declared
        return path if path.is_file() else None
    candidates = [root]
    if row.get("speaker"):
        candidates.insert(0, root / str(row["speaker"]))
    for directory in candidates:
        for extension in AUDIO_EXTENSIONS:
            path = directory / f"{row['id']}{extension}"
            if path.is_file():
                return path
    return None


def _check_spans(row: dict, norm: str, errors: list[str]) -> list[list[int]]:
    spans = row.get("en_spans", [])
    words = norm.split()
    if not isinstance(spans, list):
        errors.append("'en_spans' must be a list")
        return []
    cleaned: list[list[int]] = []
    previous_end = 0
    for span in spans:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in span)
        ):
            errors.append(f"bad en_spans entry {span!r}")
            continue
        start, end = span
        if not (0 <= start < end <= len(words)):
            errors.append(f"en_spans {span!r} outside 1..{len(words)} words")
        elif start < previous_end:
            errors.append(f"en_spans {span!r} overlaps or is out of order")
        else:
            cleaned.append([start, end])
            previous_end = end
    try:
        validate_candidate({"text": norm, "category": row.get("category"), "en_spans": cleaned})
    except ValueError as exc:
        errors.append(str(exc))
    return cleaned


def import_manifest(
    transcripts: Path | str,
    audio_dir: Path | str,
    *,
    source: str = "natural_mixed_dev",
    split: str = "dev",
    scripts_dir: Path | str = "recordings/scripts",
    manifest_dir: Path | str = "recordings/manifests",
    exclude: Path | str | None = None,
) -> dict:
    """Validate a transcript JSONL and return ``{"rows", "errors", "warnings", "summary"}``."""
    transcripts, audio_dir = Path(transcripts), Path(audio_dir)
    rows_in = [
        json.loads(line)
        for line in transcripts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    warnings: list[str] = []

    known_hashes: dict[str, str] = {}
    trigrams: list[frozenset] = []
    for path in [
        *(Path(scripts_dir).glob("*.jsonl")),
        *(Path(manifest_dir).glob("*.jsonl")),
    ]:
        if exclude is not None and path.resolve() == Path(exclude).resolve():
            continue
        for stored in load_script(path):
            text = stored.get("norm") or stored.get("text")
            if isinstance(text, str) and text:
                known_hashes[norm_hash(text)] = f"{path.name}:{stored.get('id')}"
                trigrams.append(_trigrams(normalize(text)))

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for number, row in enumerate(rows_in, start=1):
        where = f"row {number}"
        if not isinstance(row, dict):
            errors.append(f"{where}: not a JSON object")
            continue
        uid = row.get("id")
        if not isinstance(uid, str) or not uid:
            errors.append(f"{where}: missing non-empty 'id'")
            continue
        where = f"row {number} ({uid})"
        if uid in seen_ids:
            errors.append(f"{where}: duplicate id")
            continue
        seen_ids.add(uid)

        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{where}: missing non-empty 'text'")
            continue
        norm = normalize(text)
        if not norm:
            errors.append(f"{where}: text normalizes to empty")
            continue

        speaker = row.get("speaker")
        if not isinstance(speaker, str) or not speaker:
            errors.append(f"{where}: missing non-empty pseudonymous 'speaker'")
        expected_lang = "mix"
        if row.get("lang", expected_lang) != expected_lang:
            errors.append(f"{where}: lang must be 'mix' for the mixed protocol")
        row_errors: list[str] = []
        spans = _check_spans(row, norm, row_errors)
        errors.extend(f"{where}: {error}" for error in row_errors)

        audio = _find_audio(audio_dir, row)
        if audio is None:
            errors.append(f"{where}: audio not found")
            duration = None
        else:
            info = sf.info(audio)
            if info.channels != 1:
                errors.append(f"{where}: expected mono, found {info.channels} channels")
            if info.samplerate not in SAMPLE_RATES:
                errors.append(f"{where}: unsupported sample rate {info.samplerate}")
            measured = inspect_audio(audio)
            duration = round(float(measured["duration"]), 3)
            if not MIN_SECONDS <= duration <= MAX_SECONDS:
                errors.append(f"{where}: duration {duration:.2f}s outside {MIN_SECONDS}-{MAX_SECONDS}s")
            if measured["silent"]:
                errors.append(f"{where}: silent audio")
            if measured["clipping_ratio"] > CLIPPING_RATIO:
                errors.append(
                    f"{where}: clipping {measured['clipping_ratio']:.1%} > {CLIPPING_RATIO:.0%}"
                )

        digest = norm_hash(norm)
        if digest in known_hashes:
            errors.append(f"{where}: exact normalized duplicate of {known_hashes[digest]}")
        else:
            candidate = _trigrams(norm)
            near = [
                other
                for other in trigrams
                if other and candidate and len(candidate & other) / len(candidate | other)
                > NEAR_DUP_JACCARD
            ]
            if near:
                warnings.append(f"{where}: {len(near)} near-duplicate sentence(s) (Jaccard > 0.8)")
            known_hashes[digest] = uid
            trigrams.append(candidate)

        if duration is None:
            continue
        relative = audio.relative_to(audio_dir) if audio.is_relative_to(audio_dir) else audio
        rows.append(
            {
                "id": uid,
                "audio": relative.as_posix(),
                "text": text,
                "norm": norm,
                "duration": duration,
                "source": source,
                "split": split,
                "speaker": speaker,
                "lang": expected_lang,
                "category": row.get("category"),
                "en_spans": spans,
            }
        )

    summary = {
        "rows": len(rows),
        "seconds": round(sum(row["duration"] for row in rows), 1),
        "speakers": sorted({row["speaker"] for row in rows}),
        "categories": {
            category: sum(1 for row in rows if row.get("category") == category)
            for category in sorted({row.get("category") for row in rows})
        },
        "errors": len(errors),
        "warnings": len(warnings),
    }
    return {"rows": rows, "errors": errors, "warnings": warnings, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="import_manifest.py", description=__doc__)
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", default="natural_mixed_dev")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--scripts-dir", default="recordings/scripts")
    parser.add_argument("--manifest-dir", default="recordings/manifests")
    parser.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = parser.parse_args(argv)

    result = import_manifest(
        args.transcripts,
        args.audio_dir,
        source=args.source,
        split=args.split,
        scripts_dir=args.scripts_dir,
        manifest_dir=args.manifest_dir,
        exclude=args.out,
    )
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"error: {error}")
        print(f"refused: {len(result['errors'])} error(s); no manifest written")
        return 1
    if args.out.exists() and not args.force:
        print(f"refused: {args.out} exists (use --force to overwrite)")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result["rows"]),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
