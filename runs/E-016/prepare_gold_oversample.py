"""Build the E-016 gold oversampling: repeat the R-S1 clips N times without duplicating audio.

The gold set is 29 minutes; to give it a meaningful share of the batches the same cached mels are
repeated under suffixed ids (``R-S1-0001#r07``). The shard file is copied once — the index entries
point at the same offsets, so disk cost is one copy of the mel cache, not N.

Usage: python runs/E-016/prepare_gold_oversample.py --factor 50
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SOURCE_SHARDS = Path("data/shards/R-S1")
SOURCE_MANIFEST = Path("recordings/manifests/R-S1.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=50)
    parser.add_argument("--out-dir", default="data/shards/R-S1-x50")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads((SOURCE_SHARDS / "index.json").read_text())
    rows = [json.loads(line) for line in SOURCE_MANIFEST.read_text().splitlines() if line.strip()]

    shards = sorted({entry["shard"] for entry in index.values()})
    for name in shards:
        target = out_dir / name
        if not target.exists():
            shutil.copy2(SOURCE_SHARDS / name, target)

    repeated: dict[str, dict] = {}
    manifest: list[dict] = []
    for copy in range(1, args.factor + 1):
        for uid, entry in index.items():
            new_id = f"{uid}#r{copy:02d}"
            repeated[new_id] = {**entry}
            row = next(row for row in rows if row["id"] == uid)
            manifest.append({**row, "id": new_id})
    (out_dir / "index.json").write_text(json.dumps(repeated) + "\n", encoding="utf-8")
    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest) + "\n", encoding="utf-8"
    )
    hours = sum(float(row["duration"]) for row in manifest) / 3600
    print(
        f"{len(manifest)} rows ({args.factor}x of {len(rows)}), {hours:.2f} h of gold, "
        f"{len(shards)} shard file(s) copied to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
