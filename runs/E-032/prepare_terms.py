"""E-032 — isolate the 50 new S1 term rows and print the shard/x20 commands.

The rows are `R-S1-0692`–`R-S1-0741`, appended to `R-S1.jsonl` after human review. This script
validates that the audio exists and writes the subset manifest; mel and x20 sources are built with
the project tools:

    uv run python runs/E-032/prepare_terms.py
    uv run python -m stt.data.build_mel --manifest recordings/manifests/R-S1-terms3.jsonl \
        --audio-root recordings/audio/R-S1 --out-dir data/shards/R-S1-terms3
    uv run python runs/E-017/prepare_gold.py --sets R-S1-terms3 --factor 20 \
        --out-dir data/shards/gold-terms3-x20
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "recordings/manifests/R-S1.jsonl"
TARGET = REPO / "recordings/manifests/R-S1-terms3.jsonl"
AUDIO_ROOT = REPO / "recordings/audio/R-S1"
START = 692


def main() -> int:
    rows = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terms = [row for row in rows if int(row["id"].rsplit("-", 1)[1]) >= START]
    if len(terms) != 50:
        raise SystemExit(f"expected 50 rows from {START}, found {len(terms)}")
    missing = [row["id"] for row in terms if not (AUDIO_ROOT / row["audio"]).is_file()]
    if missing:
        raise SystemExit(f"audio missing: {missing[:3]} ({len(missing)} total)")
    seconds = sum(float(row["duration"]) for row in terms)
    TARGET.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in terms), encoding="utf-8"
    )
    print(
        f"wrote {TARGET.relative_to(REPO)}: {len(terms)} clips, {seconds / 60:.1f} min, "
        f"x20 = {seconds * 20 / 3600:.2f} h"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
