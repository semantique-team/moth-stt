"""Freeze explicitly selected training text, guarded by explicit evaluation exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from stt import text as text_module
from stt.manifest import ManifestError, norm_hash
from stt.text import normalize


def freeze_lm_corpus(
    *,
    train_manifests: list[Path],
    train_scripts: list[Path],
    exclusions: list[Path],
    out: Path,
    report: Path,
) -> dict:
    """Write new corpus/receipt files; reject overlap rather than silently filtering it.

    Manifests use ``norm`` when nonempty, otherwise ``text``; scripts use ``text``.
    Exclusions accept either format. No directory discovery or mutable repo defaults.
    """
    groups = {
        "exclusion": sorted({Path(p).resolve() for p in exclusions}),
        "train_manifest": sorted({Path(p).resolve() for p in train_manifests}),
        "train_script": sorted({Path(p).resolve() for p in train_scripts}),
    }
    if not groups["exclusion"]:
        raise ValueError("explicit exclusion manifests/scripts are required")
    if not groups["train_manifest"] and not groups["train_script"]:
        raise ValueError("at least one explicit training manifest/script is required")
    out, report = Path(out).resolve(), Path(report).resolve()
    inputs = {path for paths in groups.values() for path in paths}
    if out == report or out in inputs or report in inputs:
        raise ValueError("corpus and report must be distinct from each other and all inputs")
    for path in (out, report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        if not path.parent.is_dir():
            raise ValueError(f"output parent does not exist: {path.parent}")

    excluded: set[str] = set()
    sentences: dict[str, tuple[str, str]] = {}
    receipts = []
    train_rows = empty_rows = duplicates = 0
    for kind, paths in groups.items():
        for path in paths:
            training = kind != "exclusion"
            if training and path.stem.upper() in ("T", "A"):
                raise ValueError(f"evaluation script cannot be training input: {path}")
            # Hash the same byte snapshot that is parsed, not a second read of mutable inputs.
            content = path.read_bytes()
            count = 0
            for line_no, line in enumerate(content.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                location = f"{path}:{line_no}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ManifestError(f"{location}: expected JSON object")
                if training:
                    split = row.get("split")
                    if (kind == "train_manifest" or "split" in row) and split != "train":
                        raise ValueError(f"{location}: training input requires split='train'")
                    if row.get("source") in ("recorded_t", "recorded_a") or str(
                        row.get("id", "")
                    ).startswith(("T-", "A-")):
                        raise ValueError(f"{location}: evaluation row cannot be training input")
                lang = row.get("lang", "mix")
                if lang not in ("it", "en", "mix"):
                    raise ValueError(f"{location}: lang must be it, en or mix")
                hint = row.get("lang_hint", "en" if lang == "en" else "it")
                raw = row.get("text")
                if kind != "train_script":
                    norm = row.get("norm")
                    if norm is not None and not isinstance(norm, str):
                        raise ManifestError(f"{location}: norm must be a string")
                    raw = norm or raw
                if not isinstance(raw, str):
                    raise ManifestError(f"{location}: text must be a string")
                text = normalize(raw, hint)
                count += 1
                if training:
                    train_rows += 1
                if not text:
                    empty_rows += int(training)
                    continue
                key = norm_hash(text, hint)
                if not training:
                    excluded.add(key)
                elif key in excluded:
                    raise ValueError(f"{location}: normalized text overlaps explicit exclusions")
                else:
                    candidate = (text, lang)
                    if key in sentences:
                        duplicates += 1
                        # Language labels do not duplicate a sentence; ties are order-independent.
                        candidate = min(sentences[key], candidate)
                    sentences[key] = candidate
            receipts.append(
                {
                    "role": kind,
                    "path": str(path),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "rows": count,
                }
            )

    if not excluded:
        raise ValueError("exclusions have no nonempty normalized text")
    if not sentences:
        raise ValueError("training corpus has no nonempty normalized text")
    corpus = "".join(
        json.dumps({"text": text, "lang": lang}, ensure_ascii=False, separators=(",", ":")) + "\n"
        for text, lang in sorted(sentences.values())
    ).encode("utf-8")
    receipt = {
        "format_version": 1,
        "inputs": receipts,
        "normalizer": "stt.text.normalize",
        "normalizer_sha256": hashlib.sha256(Path(text_module.__file__).read_bytes()).hexdigest(),
        "num2words_version": version("num2words"),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "corpus_bytes": len(corpus),
        "sentences": len(sentences),
        "train_rows": train_rows,
        "empty_train_rows": empty_rows,
        "duplicate_train_rows": duplicates,
        "excluded_sentences": len(excluded),
    }
    with out.open("xb") as handle:
        handle.write(corpus)
    with report.open("xb") as handle:
        handle.write((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", action="append", type=Path, default=[])
    parser.add_argument("--train-script", action="append", type=Path, default=[])
    parser.add_argument("--exclude", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        freeze_lm_corpus(
            train_manifests=args.train_manifest,
            train_scripts=args.train_script,
            exclusions=args.exclude,
            out=args.out,
            report=args.report,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
