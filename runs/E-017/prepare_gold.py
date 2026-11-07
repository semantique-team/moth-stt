"""Build the E-017 gold oversampling: repeat the gold speakers' clips N times, no audio duplication.

The gold sets are small (R-S1 29.6 min, R-S2 20.6 min); to give them a meaningful share of the
batches the same cached mels are repeated under suffixed ids (``R-S1-0001#r07``). Every source's
shard files are copied **once** into the output directory and the index entries keep pointing at
the same offsets, so disk cost is one copy of the mel cache, not N.

Generalises ``runs/E-016/prepare_gold_oversample.py`` (one hardcoded source) to any number of
gold sets, which E-017 needs for S1 + S2.

Usage:
    python runs/E-017/prepare_gold.py --sets R-S1,R-S2 --factor 50
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MANIFEST_DIR = Path("recordings/manifests")
SHARDS_DIR = Path("data/shards")


def load_source(name: str) -> tuple[dict[str, dict], list[dict], list[str]]:
    """``(index, rows, shard files)`` of one gold set, validating that the two agree."""
    index_path = SHARDS_DIR / name / "index.json"
    manifest_path = MANIFEST_DIR / f"{name}.jsonl"
    if not index_path.exists():
        raise SystemExit(f"{index_path} missing: build the mel cache for {name} first")
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing: build the manifest for {name} first")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    missing = [row["id"] for row in rows if row["id"] not in index]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} ids missing from the mel index, e.g. {missing[:3]}")
    return index, rows, sorted({entry["shard"] for entry in index.values()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", default="R-S1,R-S2", help="comma list of gold set names")
    parser.add_argument("--factor", type=int, default=50)
    parser.add_argument("--out-dir", default="data/shards/gold-x50")
    args = parser.parse_args()

    names = [name.strip() for name in args.sets.split(",") if name.strip()]
    if not names:
        parser.error("--sets is empty")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repeated: dict[str, dict] = {}
    manifest: list[dict] = []
    copied: list[str] = []
    seconds = 0.0
    per_set: dict[str, int] = {}
    for name in names:
        index, rows, shards = load_source(name)
        by_id = {row["id"]: row for row in rows}
        for shard in shards:
            # Prefix with the set name: several gold sets can reuse the same shard file names.
            target_name = f"{name}/{shard}"
            target = out_dir / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(SHARDS_DIR / name / shard, target)
            copied.append(target_name)
        for copy in range(1, args.factor + 1):
            for uid, entry in index.items():
                new_id = f"{uid}#r{copy:02d}"
                if new_id in repeated:
                    raise SystemExit(f"duplicate id across gold sets: {new_id!r}")
                repeated[new_id] = {**entry, "shard": f"{name}/{entry['shard']}"}
                manifest.append({**by_id[uid], "id": new_id})
        per_set[name] = len(rows)
        seconds += sum(float(row["duration"]) for row in rows) * args.factor

    (out_dir / "index.json").write_text(json.dumps(repeated) + "\n", encoding="utf-8")
    (out_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
    )
    detail = ", ".join(f"{name} {count}" for name, count in per_set.items())
    print(
        f"{len(manifest)} rows ({args.factor}x of {sum(per_set.values())}: {detail}), "
        f"{seconds / 3600:.2f} h of gold, {len(copied)} shard file(s) copied to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())