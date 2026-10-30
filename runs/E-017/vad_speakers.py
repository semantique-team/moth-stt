"""Per-speaker Silero VAD fractions on T, for the cross-speaker check (E-017).

Reports the speech-window fraction of every T clip grouped by speaker, so the guard's margin can
be read per speaker: S1 is in the training gold, S2/S3 are unseen.

    uv run python runs/E-017/vad_speakers.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stt.data.build_mel import load_audio_16k
from stt.vad import GATE_THRESHOLD, speech_fraction

MANIFEST = Path("recordings/manifests/T.jsonl")
AUDIO_ROOT = Path("recordings/audio")
OUT = Path("runs/E-017/vad_speakers.json")


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "clips": int(array.size),
        "min": round(float(array.min()), 4),
        "p10": round(float(np.percentile(array, 10)), 4),
        "median": round(float(np.median(array)), 4),
        "max": round(float(array.max()), 4),
        "gated": int((array <= GATE_THRESHOLD).sum()),
    }


def main() -> int:
    per_speaker: dict[str, list[float]] = defaultdict(list)
    weakest: dict[str, tuple[float, str]] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("speaker") is None:
            continue
        fraction = speech_fraction(load_audio_16k(AUDIO_ROOT / row["audio"]))
        speaker = row["speaker"]
        per_speaker[speaker].append(fraction)
        if speaker not in weakest or fraction < weakest[speaker][0]:
            weakest[speaker] = (fraction, row["id"])

    report = {
        "gate": GATE_THRESHOLD,
        "speakers": {
            speaker: {
                **summary(values),
                "weakest_id": weakest[speaker][1],
                "weakest_fraction": round(weakest[speaker][0], 4),
            }
            for speaker, values in sorted(per_speaker.items())
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
