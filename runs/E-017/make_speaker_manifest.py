"""Derive a single-speaker view of T, for cross-speaker reporting (E-017).

The full T manifest is the evaluation set; E-017 needs the same rows sliced by speaker so that
baselines and paired differences can be scored on one held-out speaker. The slice keeps every
field unchanged, so hypotheses scored on it are the full-run hypotheses.

    uv run python runs/E-017/make_speaker_manifest.py --speaker S3 \
        --out runs/E-017/manifests/T-S3.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="recordings/manifests/T.jsonl")
    parser.add_argument("--speaker", required=True, choices=["S1", "S2", "S3", "S4"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("speaker") == args.speaker]
    if not selected:
        raise SystemExit(f"no rows for {args.speaker} in {args.manifest}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    seconds = sum(float(row["duration"]) for row in selected)
    print(f"{args.speaker}: {len(selected)} clips, {seconds:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
