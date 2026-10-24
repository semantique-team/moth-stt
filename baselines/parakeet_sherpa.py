"""NVIDIA Parakeet TDT 0.6B v3 baseline through ``sherpa-onnx`` (design: ``docs/05`` §7).

The offline accuracy ceiling: ≈ 670 MB int8, 25 European languages, no shipping constraint. One
recognizer is created per process and reused for every clip.

Model: https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
(converted from ``nvidia/parakeet-tdt-0.6b-v3``, CC-BY-4.0).

Example::

    PYTHONPATH=. .venv-baselines/bin/python baselines/parakeet_sherpa.py \
        --sets fleurs_it_test --audio-root data/raw --out baselines/runs/parakeet-v3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from baselines.common import run_sets

__all__ = ["main"]

DEFAULT_MODEL_DIR = "models/parakeet-tdt-0.6b-v3-int8"
MODEL_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parakeet_sherpa.py", description="Parakeet baseline")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--sets", required=True, help="comma list of manifest names")
    parser.add_argument("--manifests-dir", default="data/manifests")
    parser.add_argument("--audio-root", default="data/raw")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args(argv)

    import sherpa_onnx  # heavy import: only when the runner actually runs

    model_dir = Path(args.model_dir)
    missing = [name for name in MODEL_FILES if not (model_dir / name).exists()]
    if missing:
        parser.error(f"missing model files under {model_dir}: {', '.join(missing)}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=args.num_threads,
        model_type="nemo_transducer",
        decoding_method="greedy_search",
    )

    def transcribe(path) -> str:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, mono)
        recognizer.decode_stream(stream)
        return str(stream.result.text).strip()

    return run_sets(
        system="parakeet-tdt-0.6b-v3",
        variant="multilingual",
        sets=[name for name in args.sets.split(",") if name],
        manifests_dir=args.manifests_dir,
        audio_root=args.audio_root,
        out_dir=args.out,
        build_transcriber=lambda _hotwords: transcribe,
        hotwords_mode="none",
        scripts_dir="recordings/scripts",
        terms_path="data/en_terms.txt",
        limit=args.limit,
        size_files=[model_dir / name for name in MODEL_FILES],
        meta_extra={"engine": "sherpa-onnx", "num_threads": args.num_threads},
    )


if __name__ == "__main__":
    raise SystemExit(main())
