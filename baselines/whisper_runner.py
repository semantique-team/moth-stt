"""Whisper tiny/base baseline (design: ``docs/05-evaluation.md`` §7).

Runs ``openai-whisper`` locally (CPU/MPS), with the language auto-detected or forced to Italian,
greedy or beam-5 decoding. Whisper always processes 30 s padded audio — recorded in ``meta.json``
as ``padded_30s: true``. No hotword list: Whisper has no biasing interface.

Example:

```bash
uv run baselines/whisper_runner.py --model tiny --variant it --decode beam5 \
    --sets fleurs_it_test,fleurs_en_test --out baselines/runs/whisper-tiny-it-beam5
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.common import run_sets

__all__ = ["main", "whisper_beam", "whisper_language"]

WHISPER_MODELS = ("tiny", "base")


def whisper_language(variant: str) -> str | None:
    """``auto`` → let Whisper detect; ``it`` → force Italian."""
    return None if variant == "auto" else "it"


def whisper_beam(decode: str) -> int:
    """``greedy`` → beam 1; ``beam5`` → beam 5 (Whisper's default)."""
    return 1 if decode == "greedy" else 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whisper_runner.py", description="Whisper baseline")
    parser.add_argument("--model", required=True, choices=WHISPER_MODELS)
    parser.add_argument("--variant", default="it", choices=["auto", "it"])
    parser.add_argument("--decode", default="beam5", choices=["greedy", "beam5"])
    parser.add_argument("--sets", required=True, help="comma list of manifest names")
    parser.add_argument("--manifests-dir", default="data/manifests")
    parser.add_argument("--audio-root", default="data/audio")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    import whisper  # heavy import: only when the runner actually runs

    model = whisper.load_model(args.model)
    language = whisper_language(args.variant)
    beam_size = whisper_beam(args.decode)

    def transcribe(path: Path) -> str:
        result = model.transcribe(
            str(path), language=language, fp16=False, beam_size=beam_size, verbose=None
        )
        return str(result.get("text", "")).strip()

    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    return run_sets(
        system=f"whisper-{args.model}",
        variant=f"{args.variant}-{args.decode}",
        sets=[name.strip() for name in args.sets.split(",") if name.strip()],
        manifests_dir=args.manifests_dir,
        audio_root=args.audio_root,
        out_dir=args.out,
        build_transcriber=lambda hotwords: transcribe,
        hotwords_mode="none",
        scripts_dir="recordings/scripts",
        terms_path="data/en_terms.txt",
        limit=args.limit,
        meta_extra={
            "language": args.variant,
            "decode": args.decode,
            "beam_size": beam_size,
            "padded_30s": True,
            "params_bytes": params_bytes,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
