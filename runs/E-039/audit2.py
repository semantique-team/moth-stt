"""E-039 second pass — raw corpus audio hashes and cross-corpus near-duplicate text.

Read-only. Extends the first pass, which deliberately excluded these two checks (see ``PLAN-2.md``).

    uv run python runs/E-039/audit2.py --pass audio
    uv run python runs/E-039/audit2.py --pass text
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from stt.data.sources import sha256_file
from stt.text import normalize

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")
NEAR_THRESHOLD = 0.8
PERMUTATIONS = 64
BANDS = 16
BUCKET_CAP = 400

TRAIN_SOURCES = {
    "cv_it_train": "data/manifests/cv_it_train.jsonl",
    "mls_it_train": "data/manifests/mls_it_train.jsonl",
    "ls_en_train": "data/manifests/ls_en_train.jsonl",
    "spliced_it_en": "data/manifests/spliced_it_en.jsonl",
}
OWN_SOURCES = {
    "rec/T": "recordings/manifests/T.jsonl",
    "rec/A-S1": "recordings/manifests/A-S1.jsonl",
    "rec/R-S1": "recordings/manifests/R-S1.jsonl",
    "rec/R-S2": "recordings/manifests/R-S2.jsonl",
    "script/T": "recordings/scripts/T.jsonl",
    "script/R-S1": "recordings/scripts/R-S1.jsonl",
    "script/R-S2": "recordings/scripts/R-S2.jsonl",
    "script/pool": "recordings/scripts/pool.jsonl",
}
EVAL_SOURCES = {
    "cv_it_dev": "data/manifests/cv_it_dev.jsonl",
    "cv_it_test": "data/manifests/cv_it_test.jsonl",
    "mls_it_dev": "data/manifests/mls_it_dev.jsonl",
    "mls_it_test": "data/manifests/mls_it_test.jsonl",
    "ls_en_dev": "data/manifests/ls_en_dev.jsonl",
    "fleurs_it_dev": "data/manifests/fleurs_it_dev.jsonl",
    "fleurs_it_test": "data/manifests/fleurs_it_test.jsonl",
    "fleurs_en_dev": "data/manifests/fleurs_en_dev.jsonl",
    "fleurs_en_test": "data/manifests/fleurs_en_test.jsonl",
}


def _hash_file(path: str) -> tuple[str, str, int]:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return path, digest.hexdigest(), os.path.getsize(path)


def _source_label(relative: Path) -> str:
    parts = relative.parts
    return parts[0] if len(parts) < 2 else f"{parts[0]}/{parts[1]}"


def audio_pass() -> dict:
    """Hash only supported loose files currently present under data/raw, not decoded audio."""
    root = REPO / "data/raw"
    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        results = list(pool.map(_hash_file, [str(path) for path in files], chunksize=256))
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    hashes: dict[str, list[str]] = defaultdict(list)
    for path, digest, size in results:
        relative = Path(path).relative_to(root)
        label = _source_label(relative)
        by_source[label]["files"] += 1
        by_source[label]["bytes"] += size
        hashes[digest].append(f"{label}:{relative.as_posix()}")
    repeated = {digest: paths for digest, paths in hashes.items() if len(paths) > 1}
    cross = {
        digest: paths for digest, paths in repeated.items()
        if len({path.split(":", 1)[0] for path in paths}) > 1
    }
    return {
        "files": len(results),
        "bytes": sum(size for _, _, size in results),
        "sources": {label: dict(counts) for label, counts in sorted(by_source.items())},
        "repeated_hashes": len(repeated),
        "cross_source_collisions": cross,
        "within_source_repeats": {
            digest: paths for digest, paths in repeated.items() if digest not in cross
        },
        "note": "byte hashes of supported loose files currently present under data/raw only; "
                "labels are the first two path components, not manifest train/dev/test splits. "
                "Excludes archives, missing/deleted files, recordings/audio, mel shards and "
                "decoded-signal equivalence; not an exhaustive training/evaluation leakage check",
    }


def _shingles(text: str) -> list[int]:
    words = text.split()
    if len(words) < 3:
        return [zlib.crc32(text.encode("utf-8"))]
    return [zlib.crc32(" ".join(words[i:i + 3]).encode("utf-8")) for i in range(len(words) - 2)]


def _signature(text: str, permutations: np.ndarray, prime: int) -> np.ndarray:
    hashes = np.array(_shingles(text), dtype=np.uint64)
    if hashes.size == 0:
        return np.zeros(permutations.shape[1], dtype=np.uint64)
    # Independent affine permutations; 32-bit inputs/coefficients fit exactly in uint64.
    multipliers, offsets = permutations
    values = (hashes[:, None] * multipliers[None, :] + offsets[None, :]) % prime
    return values.min(axis=0)


def text_pass() -> dict:
    documents: list[dict] = []
    for source, path in {**TRAIN_SOURCES, **EVAL_SOURCES, **OWN_SOURCES}.items():
        for line in (REPO / path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            norm = row.get("norm") or normalize(row.get("text") or "")
            if norm:
                documents.append({"source": source, "id": str(row["id"]), "norm": norm})

    prime = (1 << 32) - 5
    rng = np.random.default_rng(0)
    permutations = np.stack([
        rng.integers(1, prime, size=PERMUTATIONS, dtype=np.uint64),
        rng.integers(0, prime, size=PERMUTATIONS, dtype=np.uint64),
    ])
    signatures = np.stack([_signature(doc["norm"], permutations, prime) for doc in documents])

    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index in range(len(documents)):
        for band in range(BANDS):
            start = band * (PERMUTATIONS // BANDS)
            buckets[(band, *signatures[index, start:start + PERMUTATIONS // BANDS].tolist())].append(index)

    skipped = 0
    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) > BUCKET_CAP:
            skipped += 1
            continue
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                a, b = members[left], members[right]
                if documents[a]["source"] == documents[b]["source"]:
                    continue
                pair = (min(a, b), max(a, b))
                candidates.add(pair)

    pairs: dict[str, dict] = defaultdict(lambda: {"count": 0, "examples": []})
    checked = 0
    for a, b in sorted(candidates):
        left, right = documents[a], documents[b]
        checked += 1
        grams_a = set(_shingles(left["norm"]))
        grams_b = set(_shingles(right["norm"]))
        score = len(grams_a & grams_b) / len(grams_a | grams_b)
        if score <= NEAR_THRESHOLD:
            continue
        key = "|".join(sorted((left["source"], right["source"])))
        entry = pairs[key]
        entry["count"] += 1
        if len(entry["examples"]) < 10:
            entry["examples"].append(
                {"a": f"{left['source']}:{left['id']}", "b": f"{right['source']}:{right['id']}",
                 "jaccard": round(score, 4), "text_a": left["norm"], "text_b": right["norm"]}
            )
    return {
        "documents": len(documents),
        "candidate_pairs": len(candidates),
        "verified_pairs": checked,
        "threshold": NEAR_THRESHOLD,
        "oversized_buckets_skipped": skipped,
        "by_pair": dict(sorted(pairs.items(), key=lambda item: -item[1]["count"])),
        "total_pairs": sum(entry["count"] for entry in pairs.values()),
        "note": "approximate cross-source candidate search, not exhaustive; oversized buckets "
                "are skipped and LSH can miss pairs. Jaccard uses CRC32 word-trigram hashes "
                "(whole-text hash for fewer than three words), not collision-free strings. "
                "Text overlap alone does not establish audio leakage or speaker disjointness",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass", dest="which", choices=["audio", "text"], required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = {"pass": args.which, "assets": {
        "audit2_py_sha256": sha256_file(Path(__file__)),
    }}
    if args.which == "audio":
        result["audio"] = audio_pass()
    else:
        result["text"] = text_pass()
    target = Path(args.out) if args.out else OUT / f"audit2-{args.which}.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.which == "audio":
        audio = result["audio"]
        print(f"audio: {audio['files']} files, {audio['bytes'] / 1e9:.1f} GB, "
              f"{audio['repeated_hashes']} repeated hashes, "
              f"{len(audio['cross_source_collisions'])} cross-source collisions")
    else:
        text = result["text"]
        print(f"text: {text['documents']} docs, {text['candidate_pairs']} candidates, "
              f"{text['total_pairs']} near-duplicate pairs > {text['threshold']}, "
              f"skipped buckets {text['oversized_buckets_skipped']}")
        for key, entry in text["by_pair"].items():
            print(f"  {key:40s} {entry['count']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
