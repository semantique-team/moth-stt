"""E-022 — paired differences, arm vs the shipped decode, on every set.

Reads the metrics produced by ``eval.py`` for each arm and writes ``diffs/``. The comparison is
paired by utterance through ``stt.eval._compare``, seed 0, 1,000 resamples — the project's rule.
"""

from __future__ import annotations

import json
from pathlib import Path

from stt.eval import _compare

OUT = Path(__file__).resolve().parent
SETS = ["t600", "t-s3", "t-s1s2", "fleurs_it_test", "fleurs_en_test", "cv_it_dev"]


def load(arm: str, name: str) -> dict | None:
    path = OUT / "eval" / arm / f"{name}.metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    rows = []
    for arm in ("frozen", "cheap"):
        for name in SETS:
            ours = load(arm, name)
            base = load("shipped", name)
            if ours is None or base is None:
                continue
            diff = _compare(ours, base, bootstrap=1000, seed=0)
            record = {
                "arm": arm,
                "set": name,
                "shipped_wer": base["overall"]["wer"],
                "arm_wer": ours["overall"]["wer"],
                "wer_diff": diff["wer_diff"],
                "wer_ci": diff["wer_diff_ci"],
                "wer_tie": diff["wer_tie"],
                "shipped_eta": base["overall"]["eta"],
                "arm_eta": ours["overall"]["eta"],
                "eta_diff": diff["eta_diff"],
                "eta_ci": diff["eta_diff_ci"],
                "eta_tie": diff["eta_tie"],
            }
            rows.append(record)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "set": name,
                        "wer": round(record["arm_wer"], 4),
                        "delta": round(record["wer_diff"], 4),
                        "ci": [round(x, 4) for x in record["wer_ci"]],
                        "tie": record["wer_tie"],
                        "eta_delta": None
                        if record["eta_diff"] is None
                        else round(record["eta_diff"], 4),
                        "eta_tie": record["eta_tie"],
                    }
                ),
                flush=True,
            )

    target = OUT / "diffs"
    target.mkdir(parents=True, exist_ok=True)
    (target / "vs-shipped.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())