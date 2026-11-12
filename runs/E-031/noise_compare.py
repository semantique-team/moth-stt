"""E-031 — paired noise comparison: E-028 checkpoint with the shipped LM vs the E-031 LM.

Both runs use the same fp32 checkpoint, decoder (0.55/2.7), 100 S3 clips, noise pool and seed; only
the LM changes. This isolates the LM's effect on the E-024 noise curve.

    uv run python runs/E-031/noise_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

from stt.eval import _compare

REPO = Path(__file__).resolve().parents[2]
NEW = REPO / "runs/E-031/noise-e031-lm"
CONTROL = REPO / "runs/E-028/noise-0.8.0"
NAMES = {"clean": "clean", "30.0": "30", "20.0": "20", "15.0": "15", "10.0": "10",
         "5.0": "5"}


def main() -> int:
    result = {"new": str(NEW.relative_to(REPO)), "control": str(CONTROL.relative_to(REPO)),
              "conditions": {}}
    for label, name in NAMES.items():
        new = json.loads((NEW / f"{name}.metrics.json").read_text(encoding="utf-8"))
        control = json.loads((CONTROL / f"{name}.metrics.json").read_text(encoding="utf-8"))
        diff = _compare(new, control, bootstrap=1000, seed=0)
        result["conditions"][label] = {
            "new_wer": new["overall"]["wer"],
            "control_wer": control["overall"]["wer"],
            "new_eta": new["overall"]["eta"],
            "control_eta": control["overall"]["eta"],
            "wer_diff": diff["wer_diff"],
            "wer_ci": diff["wer_diff_ci"],
            "wer_tie": diff["wer_tie"],
            "eta_diff": diff["eta_diff"],
            "eta_ci": diff["eta_diff_ci"],
            "eta_tie": diff["eta_tie"],
        }
        print(
            f"{label:6s} wer {new['overall']['wer']*100:6.2f} vs "
            f"{control['overall']['wer']*100:6.2f}  "
            f"Δ {diff['wer_diff']*100:+.2f} "
            f"[{diff['wer_diff_ci'][0]*100:+.2f}, {diff['wer_diff_ci'][1]*100:+.2f}] "
            f"{'tie' if diff['wer_tie'] else 'SIGNIFICANT'}",
            flush=True,
        )
    noise = [result["conditions"][key]["new_wer"] for key in ("5.0", "10.0", "15.0", "20.0", "30.0")]
    control = [result["conditions"][key]["control_wer"] for key in ("5.0", "10.0", "15.0", "20.0", "30.0")]
    result["mean_5_30"] = {"new": sum(noise) / len(noise), "control": sum(control) / len(control)}
    (NEW / "compare.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"mean 5-30 dB: {result['mean_5_30']['new']*100:.2f} vs "
          f"{result['mean_5_30']['control']*100:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
