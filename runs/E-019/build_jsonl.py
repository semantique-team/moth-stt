"""Build the Whistle fine-tuning JSONL for E-019 from the two gold manifests (D28).

One line per clip, in the format ``needle whistle finetune`` reads::

    {"audio": "../../recordings/audio/R-S1/S1/S1_0001.wav", "text": "la mia ...", "language": "it"}

Rules (plan: ``docs/plans/2026-11-07-whistle-lora-probe.md``, step 2):

- Sources are **only** the gold sets R-S1 and R-S2. T and A are never read here: they are the
  evaluation sets and reading them for training, validation, token mining or hyper-parameters
  would be leakage.
- The text is the manifest's ``norm`` field — already canonical per ``stt.text.normalize`` — so
  every system in the repo is scored on the same text rule.
- The split is 90/5/5, seeded, stratified by speaker, so both speakers appear in train,
  validation and test. S3 has no gold set, so it is structurally absent: it stays the honest
  held-out speaker.
- Audio paths are written relative to the JSONL's own directory, which is how
  ``needle.model.whistle_finetune.read_clips`` resolves them (``dirname(jsonl) + path``). That
  keeps the files portable inside the repo instead of pinning absolute paths.

Usage::

    uv run python runs/E-019/build_jsonl.py
    uv run python runs/E-019/build_jsonl.py --language mix --out-dir runs/E-019/data-mix
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from stt.manifest import load_manifest

MANIFESTS = Path("recordings/manifests")
AUDIO_ROOT = Path("recordings/audio")

#: The only gold sets that exist. Both speakers are training speakers in E-017/E-019.
GOLD_SETS = ("R-S1", "R-S2")


def audio_path(row: dict, set_dir: Path, audio_root: Path) -> Path:
    """Resolve a manifest ``audio`` entry.

    Kind R keeps each recording session in its own tree (``recordings/audio/R-S1/S1/...``) while
    T shares one root (``recordings/audio/S3/...``), so the per-set directory is tried first.
    """
    for root in (set_dir, audio_root):
        candidate = root / row["audio"]
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(root / row["audio"]) for root in (set_dir, audio_root))
    raise SystemExit(f"{row['id']!r}: audio not found (tried {tried})")


def read_gold(manifests: Path, audio_root: Path, names: tuple[str, ...]) -> list[dict]:
    """Every clip of the gold sets, with ``_audio_path`` (absolute) checked on disk."""
    rows: list[dict] = []
    for name in names:
        path = manifests / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing: the gold manifests are the only allowed source")
        set_rows = load_manifest(path)  # schema, canonical ``norm``, duplicate ids
        if not set_rows:
            raise SystemExit(f"{path} is empty")
        set_dir = audio_root / name
        for row in set_rows:
            if row["source"] != "recorded_r":
                raise SystemExit(f"{path}: {row['id']!r} is {row['source']!r}, not a gold clip")
            if row["speaker"] not in row["id"]:
                raise SystemExit(f"{path}: {row['id']!r} is not tagged with its speaker")
            if not row.get("audio"):
                raise SystemExit(f"{path}: {row['id']!r} has no audio path")
            rows.append({**row, "_audio_path": audio_path(row, set_dir, audio_root)})
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids across the gold sets")
    return rows


def split_counts(n: int, fraction: float) -> tuple[int, int]:
    """``(n_val, n_test)`` for one speaker: at least one clip each, truncating, train gets the rest."""
    holdout = max(1, int(n * fraction))
    if 2 * holdout >= n:
        raise SystemExit(f"not enough clips to hold out {2 * fraction:.0%} of {n}")
    return holdout, holdout


def make_split(rows: list[dict], seed: int, fraction: float = 0.05) -> dict[str, list[dict]]:
    """Seeded, speaker-stratified 90/5/5 split.

    Each speaker is shuffled independently with a stream seeded by ``(seed, speaker)``, so adding
    a third gold speaker later does not reshuffle the existing two.
    """
    by_speaker: dict[str, list[dict]] = {}
    for row in rows:
        by_speaker.setdefault(row["speaker"], []).append(row)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for speaker in sorted(by_speaker):
        clips = sorted(by_speaker[speaker], key=lambda row: row["id"])
        random.Random(f"{seed}:{speaker}").shuffle(clips)
        n_val, n_test = split_counts(len(clips), fraction)
        splits["test"] += clips[:n_test]
        splits["val"] += clips[n_test : n_test + n_val]
        splits["train"] += clips[n_test + n_val :]
    for split in splits.values():
        split.sort(key=lambda row: row["id"])
    return splits


def write_jsonl(path: Path, rows: list[dict], language: str) -> None:
    """Write one line per clip; the audio path is relative to ``path.parent``."""
    root = path.parent.resolve()
    lines = []
    for row in rows:
        lines.append(
            json.dumps(
                {
                    "audio": os.path.relpath(row["_audio_path"], root),
                    "text": row["norm"],
                    "language": language,
                },
                ensure_ascii=False,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """sha256 of a file, for the reproducibility record."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifests-dir", default=str(MANIFESTS))
    parser.add_argument("--audio-root", default=str(AUDIO_ROOT))
    parser.add_argument("--sets", default=",".join(GOLD_SETS), help="comma list of gold set names")
    parser.add_argument("--out-dir", default="runs/E-019/data")
    parser.add_argument(
        "--language", default="it", help="language code on every line (L1/L2: it, L3: mix)"
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = tuple(name.strip() for name in args.sets.split(",") if name.strip())
    rows = read_gold(Path(args.manifests_dir), Path(args.audio_root), names)
    splits = make_split(rows, args.seed, args.val_fraction)

    out_dir = Path(args.out_dir)
    for name in ("train", "val", "test"):
        write_jsonl(out_dir / f"{name}.jsonl", splits[name], args.language)

    counts = {name: len(split) for name, split in splits.items()}
    per_speaker = {
        name: dict(sorted(Counter(row["speaker"] for row in split).items()))
        for name, split in splits.items()
    }
    seconds = {
        name: round(sum(float(row["duration"]) for row in split), 2)
        for name, split in splits.items()
    }
    categories = dict(sorted(Counter(row["category"] for row in splits["train"]).items()))
    hashes = {name: sha256_file(out_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    report = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "language": args.language,
        "sets": list(names),
        "sha256": hashes,
        "clips": counts,
        "per_speaker": per_speaker,
        "seconds": seconds,
        "train_categories": categories,
        "split": {name: [row["id"] for row in split] for name, split in splits.items()},
    }
    (out_dir / "split.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    total = sum(counts.values())
    print(f"{total} clips from {', '.join(names)}")
    for name in ("train", "val", "test"):
        share = counts[name] / total
        detail = ", ".join(f"{speaker} {n}" for speaker, n in per_speaker[name].items())
        print(
            f"  {name:<5} {counts[name]:>4} clips ({share:>5.1%})  {seconds[name] / 60:>5.1f} min  {detail}"
        )
    print(f"  train categories: {categories}")
    print(f"  language code on every line: {args.language!r}")
    print(f"wrote {out_dir}/{{train,val,test}}.jsonl and split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
