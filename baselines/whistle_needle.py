"""Cactus Whistle baseline through ``cactus-needle`` (design: ``docs/05-evaluation.md`` §7).

Unlike ``whistle_runner.py`` this keeps one model in the process: the needle engine is loaded once
and every clip reuses it, which matters on sets with thousands of utterances.

Install (separate venv, never a project dependency)::

    uv venv .venv-baselines --python 3.12
    uv pip install --python .venv-baselines/bin/python "cactus-needle" soxr

Example::

    PYTHONPATH=. .venv-baselines/bin/python baselines/whistle_needle.py \
        --variant it --hotwords none --sets fleurs_it_test,fleurs_en_test \
        --audio-root data/raw --out baselines/runs/whistle-tiny-it
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import soundfile as sf
import soxr

from baselines.common import run_sets

__all__ = ["main"]

#: Whistle supports these languages; ``auto`` lets the engine detect (``None``).
LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whistle_needle.py", description="Whistle baseline")
    parser.add_argument("--variant", default="it", choices=["auto", "it"])
    parser.add_argument("--hotwords", default="none", choices=["none", "fair", "oracle"])
    parser.add_argument("--sets", required=True, help="comma list of manifest names")
    parser.add_argument("--manifests-dir", default="data/manifests")
    parser.add_argument("--audio-root", default="data/raw")
    parser.add_argument("--scripts-dir", default="recordings/scripts")
    parser.add_argument("--terms", default="data/en_terms.txt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    import needle  # heavy import: only when the runner actually runs

    language = None if args.variant == "auto" else args.variant
    if language is not None and language not in LANGUAGES:
        parser.error(f"Whistle does not support {language!r}")

    def load_audio(path) -> np.ndarray:
        """Mono float32 at 16 kHz; needle reads WAV only, CV is mp3."""
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        if sample_rate != 16000:
            mono = soxr.resample(mono, sample_rate, 16000).astype(np.float32)
        return np.ascontiguousarray(mono)

    def build_transcriber(hotwords: list[str]):
        def transcribe(path) -> str:
            result = needle.transcribe(
                load_audio(path), language=language, keywords=hotwords or None
            )
            return str(result.get("text", "")).strip()

        return transcribe

    return run_sets(
        system="whistle",
        variant=f"{args.variant}-{args.hotwords}",
        sets=[name for name in args.sets.split(",") if name],
        manifests_dir=args.manifests_dir,
        audio_root=args.audio_root,
        out_dir=args.out,
        build_transcriber=build_transcriber,
        hotwords_mode=args.hotwords,
        scripts_dir=args.scripts_dir,
        terms_path=args.terms,
        limit=args.limit,
        meta_extra={"engine": "cactus-needle", "language": args.variant},
    )


if __name__ == "__main__":
    raise SystemExit(main())
