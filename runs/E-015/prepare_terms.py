"""E-015 — isolate the 50 new S1 term sentences and build their x20 gold source.

The old `gold-x50` source is a snapshot of 641 R-S1 + 622 R-S2 clips; the new term sentences must
not be folded into it at x50 (that would repeat 50 templates fifty times). They become a separate
source at x20, added to E-024's mixture.

    uv run python runs/E-015/prepare_terms.py
    uv run python -m stt.data.build_mel --manifest recordings/manifests/R-S1-terms.jsonl \
        --audio-root recordings/audio/R-S1 --out-dir data/shards/R-S1-terms
    uv run python runs/E-017/prepare_gold.py --sets R-S1-terms --factor 20 \
        --out-dir data/shards/gold-terms-x20
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "recordings/manifests/R-S1.jsonl"
TARGET = REPO / "recordings/manifests/R-S1-terms.jsonl"
AUDIO_ROOT = REPO / "recordings/audio/R-S1"


def main() -> int:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    terms = [row for row in rows if int(row["id"].rsplit("-", 1)[1]) >= 642]
    if len(terms) != 50:
        raise SystemExit(f"expected 50 term sentences, found {len(terms)}")
    missing = [row["id"] for row in terms if not (AUDIO_ROOT / row["audio"]).is_file()]
    if missing:
        raise SystemExit(f"audio missing: {missing[:3]}")
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
