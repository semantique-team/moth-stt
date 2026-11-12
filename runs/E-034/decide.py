"""E-034 — apply the pre-registered advance rule to the dev sweep and the builds.

Reads `build.json` (variant sizes) and `dev/sweep.json` (dev macro WER, paired CIs) and prints one
row per arm with the package arithmetic and the verdict. It implements PLAN.md literally:

* advance requires a lower macro point, a paired 95% CI upper bound < 0, and a binary small
  enough to keep the package at or below 20,000,000 bytes;
* an over-budget arm is a diagnostic, even when its interval is below zero.

    uv run python runs/E-034/decide.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
PACKAGE_LIMIT = 20_000_000
NON_LM_BYTES = 19_440_761 - 5_116_464  # 0.8.0 package minus its LM; the rest is shared
CONTROL = "A3_shipped"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=OUT / "dev/dev-only/sweep.json")
    parser.add_argument("--build", type=Path, default=OUT / "build.json")
    args = parser.parse_args(argv)

    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))["arms"]
    if CONTROL not in sweep:
        raise SystemExit(f"{CONTROL} is missing from the sweep; it is the control arm")
    builds = json.loads(args.build.read_text(encoding="utf-8"))["variants"]
    shipped = json.loads(
        (OUT.parent / "E-031/build.json").read_text(encoding="utf-8")
    )["arms"]["frozen_1_2_terms"]

    rows = []
    for name, arm in sweep.items():
        binary = shipped["binary_bytes"] if name == CONTROL else builds[name]["binary_bytes"]
        package = NON_LM_BYTES + binary
        fits = package <= PACKAGE_LIMIT
        if name == CONTROL:
            verdict = "control"
        else:
            lower = arm["macro"] < sweep[CONTROL]["macro"]
            upper = arm["macro_vs_control"]["ci"][1]
            advances = lower and upper < 0 and fits
            if advances:
                verdict = "advance"
            elif not lower or upper >= 0:
                verdict = "no dev gain"
            else:
                verdict = "wins dev, over budget: follow-up only"
            if lower and upper < 0 and not fits:
                verdict = "wins dev, over budget: follow-up only"
        rows.append(
            {
                "arm": name,
                "macro_wer": arm["macro"],
                "macro_delta_pp": (arm["macro"] - sweep[CONTROL]["macro"]) * 100,
                "ci_upper_pp": (arm.get("macro_vs_control", {}).get("ci") or [None, None])[1],
                "it": arm["it"],
                "en": arm["en"],
                "binary_bytes": binary,
                "package_bytes": package,
                "fits_budget": fits,
                "verdict": verdict,
            }
        )

    print(f"{'arm':16s} {'macro':>8s} {'Δ pp':>8s} {'ci upper':>9s} {'binary':>11s} "
          f"{'package':>11s} {'fits':>5s}  verdict")
    for row in rows:
        delta = row["macro_delta_pp"]
        upper = row["ci_upper_pp"]
        ci = "" if upper is None else f"{upper * 100:+.3f}"
        print(
            f"{row['arm']:16s} {row['macro_wer']:8.4f} {delta:+8.4f} {ci:>9s} "
            f"{row['binary_bytes']:11,d} {row['package_bytes']:11,d} "
            f"{row['fits_budget']!s:>5s}  {row['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
