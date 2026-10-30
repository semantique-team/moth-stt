"""E-011 feasibility: can a small pre-trained VAD separate our speech from the noise set?

Run with the silero-vad package available (it is not a project dependency):

    uv run --with silero-vad python runs/E-011/vad_feasibility.py

Silero VAD v5 (MIT, 2.33 MB) is stateful: each call takes 512 new samples **plus 64 samples of
context** from previous calls (the wrapper handles it) and returns P(speech) for a 32 ms window.
The gate rule under test is the **fraction of windows above 0.5**: it separates cleanly, while a
single-window maximum does not (transients in noise reach 0.99).

Measured 2026-10-30 (macOS M5, onnxruntime CPU): threshold 0.10 keeps 100% of the speech clips and
gates 100% of the 120 MUSAN noise clips and 100% of the silence clips (music, reported not gated,
is 86% below 0.02). See ``docs/plans/2026-10-30-vad-guard.md`` for the full table.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from stt.data.build_mel import load_audio_16k

WINDOW = 512
THRESHOLD = 0.5


def speech_fraction(model, audio: np.ndarray) -> float:
    import torch

    model.reset_states()
    probs = []
    for start in range(0, len(audio) - WINDOW + 1, WINDOW):
        chunk = torch.from_numpy(audio[start : start + WINDOW][None, :].astype(np.float32))
        with torch.no_grad():
            probs.append(float(model(chunk, 16000).item()))
    return float((np.array(probs) > THRESHOLD).mean()) if probs else 0.0


SETS = {
    "T S1+S2": "recordings/audio/S*/*.wav",
    "FLEURS-IT": "data/raw/fleurs/it_it/audio/test/*.wav",
    "noise MUSAN": "data/raw/noise/noise_musan/*.wav",
    "music MUSAN": "data/raw/noise/noise_music/*.wav",
    "silence": "data/raw/noise/noise_silence/*.wav",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default=",".join(SETS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gate", type=float, default=0.10, help="fraction below which no speech")
    args = parser.parse_args()

    from silero_vad import load_silero_vad

    model = load_silero_vad(onnx=True)
    print(f"{'set':<14} {'n':>5} {'min':>6} {'p10':>6} {'med':>6} {'p90':>6} {'max':>6} {'gated':>7}")
    for name in args.sets.split(","):
        files = sorted(glob.glob(SETS[name]))
        if args.limit:
            files = files[: args.limit]
        if not files:
            print(f"{name}: no files")
            continue
        fr = np.array([speech_fraction(model, load_audio_16k(f)) for f in files])
        print(
            f"{name:<14} {len(files):>5} {fr.min():>6.3f} {np.percentile(fr, 10):>6.3f} "
            f"{np.median(fr):>6.3f} {np.percentile(fr, 90):>6.3f} {fr.max():>6.3f} "
            f"{100 * (fr < args.gate).mean():>6.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
