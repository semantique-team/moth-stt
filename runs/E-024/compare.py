"""E-024 — paired comparison of the two noise curves.

Reads the per-condition metrics written by ``noise_eval.py`` for both arms and reports, for each
SNR: the absolute WER of both, the paired difference and its CI. The decision rule in ``PLAN.md``
is expressed on the mean over 10-30 dB, so that number is computed explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

from stt.eval import _compare

OUT = Path(__file__).resolve().parent
CONDITIONS = [("clean", "clean"), ("30", "30 dB"), ("20", "20 dB"), ("15", "15 dB"), ("10", "10 dB"), ("5", "5 dB")]
CLEAN_SETS = ["t600", "fleurs_it_test", "cv_it_test"]


def load_metrics(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> int:
    base_dir = OUT / "eval" / "base"
    noisy_dir = OUT / "eval" / "noisy"
    if not (base_dir / "result.json").is_file() or not (noisy_dir / "result.json").is_file():
        raise SystemExit("run noise_eval.py for both arms first (runs/E-024/eval.sh)")

    rows = []
    print(f"{'condition':10s} {'base':>7s} {'noisy':>7s} {'delta':>8s}  CI")
    for tag, label in CONDITIONS:
        base = load_metrics(base_dir / f"{tag}.metrics.json")
        noisy = load_metrics(noisy_dir / f"{tag}.metrics.json")
        if base is None or noisy is None:
            continue
        diff = _compare(noisy, base, bootstrap=1000, seed=0)
        record = {
            "condition": label,
            "base_wer": base["overall"]["wer"],
            "noisy_wer": noisy["overall"]["wer"],
            "base_eta": base["overall"]["eta"],
            "noisy_eta": noisy["overall"]["eta"],
            "wer_diff": diff["wer_diff"],
            "wer_ci": diff["wer_diff_ci"],
            "wer_tie": diff["wer_tie"],
        }
        rows.append(record)
        print(
            f"{label:10s} {record['base_wer'] * 100:6.2f} {record['noisy_wer'] * 100:7.2f} "
            f"{record['wer_diff'] * 100:+8.2f}  "
            f"[{diff['wer_diff_ci'][0] * 100:+.2f}, {diff['wer_diff_ci'][1] * 100:+.2f}]"
        )

    noise_rows = [r for r in rows if r["condition"] != "clean"]
    mean_gain = sum(r["wer_diff"] for r in noise_rows) / len(noise_rows)
    print(f"\nmean delta over the noise conditions: {mean_gain * 100:+.2f} points")

    clean_gate = {}
    for name in CLEAN_SETS:
        base = load_metrics(base_dir / f"clean-{name}.metrics.json")
        noisy = load_metrics(noisy_dir / f"clean-{name}.metrics.json")
        if base is None or noisy is None:
            continue
        diff = _compare(noisy, base, bootstrap=1000, seed=0)
        clean_gate[name] = {
            "base_wer": base["overall"]["wer"],
            "noisy_wer": noisy["overall"]["wer"],
            "wer_diff": diff["wer_diff"],
            "wer_ci": diff["wer_diff_ci"],
            "wer_tie": diff["wer_tie"],
        }
        print(
            f"clean {name:16s} {base['overall']['wer'] * 100:6.2f} -> "
            f"{noisy['overall']['wer'] * 100:6.2f}  {diff['wer_diff'] * 100:+6.2f}  "
            f"[{diff['wer_diff_ci'][0] * 100:+.2f}, {diff['wer_diff_ci'][1] * 100:+.2f}]"
        )

    # The decision rule, evaluated mechanically so the verdict is not a judgement call.
    clean_ok = all(v["wer_diff"] <= 0.01 for k, v in clean_gate.items() if k in ("t600",))
    verdict = "KILL"
    if mean_gain <= -0.04 and clean_ok:
        verdict = "ADOPT"
    elif mean_gain <= -0.04:
        verdict = "PARTIAL — retry with 25% of CV noised"
    elif mean_gain <= -0.03:
        verdict = "WEAK — record, do not adopt"
    print(f"\nplan rule: mean gain <= -4.0 points and clean T-600 within +1.0 -> {verdict}")

    summary = {
        "conditions": rows,
        "mean_noise_delta": mean_gain,
        "clean_gate": clean_gate,
        "verdict": verdict,
    }
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())