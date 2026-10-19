"""NVIDIA Parakeet TDT 0.6B v3 baseline — the offline accuracy ceiling (design: ``docs/05`` §7).

Parakeet is ≈ 640 MB int8 and cannot ship in a browser; it is here to measure what a strong model
does on our sets under our ``normalize()``, with and without hotwords. Like the Whistle runner it
takes a command template (``{audio}``, ``{lang}``, ``{keywords}``) so it adapts to sherpa-onnx /
NeMo CLI differences.

Example (adapt to your sherpa-onnx install):

```bash
uv run baselines/parakeet_runner.py --hotwords fair \
    --command "sherpa-onnx-offline-transducer --tokens=tokens.txt --encoder=encoder.onnx \
--decoder=decoder.onnx --joiner=joiner.onnx --hotwords-file={keywords} {audio}" \
    --sets fleurs_it_test --out baselines/runs/parakeet-v3-fair
```
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from baselines.common import external_transcriber, render_command, run_sets

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parakeet_runner.py", description="Parakeet TDT v3 baseline")
    parser.add_argument("--command", required=True, help="CLI template with {audio}, {lang}, {keywords}")
    parser.add_argument("--hotwords", default="none", choices=["none", "fair", "oracle"])
    parser.add_argument("--sets", required=True, help="comma list of manifest names")
    parser.add_argument("--manifests-dir", default="data/manifests")
    parser.add_argument("--audio-root", default="data/audio")
    parser.add_argument("--scripts-dir", default="recordings/scripts")
    parser.add_argument("--terms", default="data/en_terms.txt")
    parser.add_argument("--size-file", action="append", default=[], help="model file(s) for size_bytes")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    keyword_dir = Path(tempfile.mkdtemp(prefix="parakeet-keywords-"))

    def build_transcriber(hotwords: list[str]):
        if hotwords and "{keywords}" not in args.command:
            print("warning: the command template has no {keywords}; hotwords are ignored")
        keyword_path = keyword_dir / "keywords.txt"
        keyword_path.write_text("\n".join(hotwords) + ("\n" if hotwords else ""), encoding="utf-8")
        command = render_command(args.command, lang="", keywords_path=keyword_path if hotwords else None)
        return external_transcriber(command)

    return run_sets(
        system="parakeet-tdt-0.6b-v3",
        variant=f"{args.hotwords}",
        sets=[name.strip() for name in args.sets.split(",") if name.strip()],
        manifests_dir=args.manifests_dir,
        audio_root=args.audio_root,
        out_dir=args.out,
        build_transcriber=build_transcriber,
        hotwords_mode=args.hotwords,
        scripts_dir=args.scripts_dir,
        terms_path=args.terms,
        limit=args.limit,
        size_files=[Path(path) for path in args.size_file],
        meta_extra={"command": args.command, "padded_30s": None},
    )


if __name__ == "__main__":
    raise SystemExit(main())
