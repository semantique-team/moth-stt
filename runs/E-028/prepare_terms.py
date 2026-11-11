"""Isolate the term-addendum rows of R-S1 or R-S2 and validate their audio.

E-028 (the second E-015 arm) adds the 50 S2 term sentences next to the already-built 50 S1
sentences. This script only selects and validates the manifest rows; mel and x20 sources are built
with the project tools.

    uv run python runs/E-028/prepare_terms.py --set R-S2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO / "recordings/manifests"
AUDIO_DIR = {
    "R-S1": REPO / "recordings/audio/R-S1",
    "R-S2": REPO / "recordings/audio/R-S2",
}
START = {"R-S1": 642, "R-S2": 624}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=tuple(START), default="R-S2")
    args = parser.parse_args()

    source = MANIFEST_DIR / f"{args.set}.jsonl"
    target = MANIFEST_DIR / f"{args.set}-terms.jsonl"
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terms = [
        row for row in rows if int(row["id"].rsplit("-", 1)[1]) >= START[args.set]
    ]
    if len(terms) != 50:
        raise SystemExit(f"expected 50 term rows in {source}, found {len(terms)}")
    missing = [
        row["id"]
        for row in terms
        if not (AUDIO_DIR[args.set] / row["audio"]).is_file()
    ]
    if missing:
        raise SystemExit(f"audio missing for {missing[:5]} ({len(missing)} total)")
    seconds = sum(float(row["duration"]) for row in terms)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in terms),
        encoding="utf-8",
    )
    print(
        f"wrote {target.relative_to(REPO)}: {len(terms)} clips, {seconds / 60:.1f} min, "
        f"x20 = {seconds * 20 / 3600:.2f} h"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
