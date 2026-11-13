"""E-035 — apply the pre-registered advance rule to the budgeted arm.

Reads `build.json` (binary size and package arithmetic) and the dev sweep (macro WER, paired CIs)
and applies `PLAN.md` literally: a lower macro point, a paired 95% CI upper bound below zero, a
binary within 5,675,703 B and a package within 20,000,000 B.

    uv run python runs/E-035/decide.py --sweep dev/budgeted-1/sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
PACKAGE_LIMIT = 20_000_000
CONTROL = "A3_shipped"
ARM = "B1_budgeted"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=OUT / "dev/budgeted-1/sweep.json")
    parser.add_argument("--build", type=Path, default=OUT / "build.json")
    args = parser.parse_args(argv)

    sweep_doc = json.loads(args.sweep.read_text(encoding="utf-8"))
    sweep = sweep_doc["arms"]
    for name in (CONTROL, ARM):
        if name not in sweep:
            raise SystemExit(f"{name} is missing from the sweep")
    build = json.loads(args.build.read_text(encoding="utf-8"))
    package = build["arms"][ARM]["package_bytes"]
    binary = build["arms"][ARM]["binary_bytes"]

    lower = sweep[ARM]["macro"] < sweep[CONTROL]["macro"]
    upper = sweep[ARM]["macro_vs_control"]["ci"][1]
    fits = binary <= build["binary_limit"] and package <= PACKAGE_LIMIT
    advances = bool(lower and upper < 0 and fits)
    decision = {
        "rule": "macro lower AND paired CI upper < 0 AND binary <= 5,675,703 B "
                "AND package <= 20,000,000 B",
        "control": {
            "arm": CONTROL,
            "macro": sweep[CONTROL]["macro"],
            "it": sweep[CONTROL]["it"],
            "en": sweep[CONTROL]["en"],
        },
        "arm": {
            "arm": ARM,
            "macro": sweep[ARM]["macro"],
            "it": sweep[ARM]["it"],
            "en": sweep[ARM]["en"],
            "macro_delta_pp": (sweep[ARM]["macro"] - sweep[CONTROL]["macro"]) * 100,
            "macro_ci_pp": [value * 100 for value in sweep[ARM]["macro_vs_control"]["ci"]],
            "pooled_delta_pp": sweep[ARM]["vs_control"]["wer_diff"] * 100,
            "pooled_ci_pp": [value * 100 for value in sweep[ARM]["vs_control"]["wer_diff_ci"]],
            "binary_bytes": binary,
            "package_bytes": package,
        },
        "lower_macro": bool(lower),
        "paired_ci_upper_below_zero": bool(upper < 0),
        "fits_budget": bool(fits),
        "control_check": sweep_doc.get("control_check"),
        "advances": advances,
        "decision": (
            "advance to the product readout (separate, unauthorized here)"
            if advances
            else "no advance: keep 0.8.1 and record the headroom question as answered"
        ),
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{ARM}: macro {decision['arm']['macro'] * 100:.4f} vs "
        f"{decision['control']['macro'] * 100:.4f} "
        f"({decision['arm']['macro_delta_pp']:+.4f} pp, "
        f"CI [{decision['arm']['macro_ci_pp'][0]:+.4f}, "
        f"{decision['arm']['macro_ci_pp'][1]:+.4f}])"
    )
    print(f"binary {binary:,} B, package {package:,} B, fits: {fits}")
    print(decision["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
