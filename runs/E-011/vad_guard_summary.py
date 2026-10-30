"""E-011 addendum — speech-fraction distribution of the shipped VAD guard (D25).

Runs the exact code path of ``stt.vad`` (not the ``silero-vad`` wrapper used for the feasibility
study) on every clip of the guard-relevant sets and writes quantiles + gated counts to
``runs/E-011/vad_guard_summary.json``.

    uv run python runs/E-011/vad_guard_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stt.data.build_mel import load_audio_16k
from stt.vad import GATE_THRESHOLD, WINDOW_THRESHOLD, speech_fraction

SETS = {
    "recorded_t": ("recordings/manifests/T.jsonl", "recordings/audio"),
    "fleurs_it_test": ("data/manifests/fleurs_it_test.jsonl", "data/raw"),
    "noise": ("data/manifests/noise.jsonl", "data/raw/noise"),
}
OUT = Path("runs/E-011/vad_guard_summary.json")


def rows_of(manifest: str) -> list[dict]:
    with Path(manifest).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fractions(rows: list[dict], root: str) -> dict[str, float]:
    return {row["id"]: speech_fraction(load_audio_16k(Path(root) / row["audio"])) for row in rows}


def summary(values: dict[str, float]) -> dict:
    array = np.array(list(values.values()), dtype=np.float64)
    gated = array <= GATE_THRESHOLD
    return {
        "clips": int(array.size),
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "gated": int(gated.sum()),
        "gated_pct": float(100.0 * gated.mean()),
    }


def main() -> int:
    report: dict = {
        "thresholds": {"window": WINDOW_THRESHOLD, "gate": GATE_THRESHOLD},
        "sets": {},
    }
    for name, (manifest, root) in SETS.items():
        rows = rows_of(manifest)
        values = fractions(rows, root)
        report["sets"][name] = summary(values)
        if name == "noise":
            for source in ("noise_silence", "noise_musan", "noise_music"):
                subset = {row["id"]: values[row["id"]] for row in rows if row["source"] == source}
                if subset:
                    report["sets"][source] = summary(subset)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
