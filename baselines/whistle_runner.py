"""Cactus Whistle baseline (design: ``docs/05-evaluation.md`` §7).

Whistle is the strongest small model for the brief (16.9 MB, Apache-2.0, IT+EN, runs in a browser
tab) and the main baseline. Its CLI evolves, so this runner takes a **command template** instead of
hard-coding flags:

- ``{audio}`` is appended if missing (substituted per clip);
- ``{lang}`` is the forced language (``it``) or empty for auto detection;
- ``{keywords}`` is a file with one hotword per line (written by the runner when hotwords are on).

Example (adapt the flags to your ``needle`` version):

```bash
uv run baselines/whistle_runner.py --variant it --hotwords fair \
    --command "needle transcribe --language {lang} --keywords {keywords} {audio}" \
    --sets fleurs_it_test --out baselines/runs/whistle-it-fair
```
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from baselines.common import external_transcriber, render_command, run_sets

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whistle_runner.py", description="Cactus Whistle baseline")
    parser.add_argument("--command", required=True, help="CLI template with {audio}, {lang}, {keywords}")
    parser.add_argument("--variant", default="it", choices=["auto", "it"])
    parser.add_argument("--hotwords", default="none", choices=["none", "fair", "oracle"])
    parser.add_argument("--sets", required=True, help="comma list of manifest names")
    parser.add_argument("--manifests-dir", default="data/manifests")
    parser.add_argument("--audio-root", default="data/audio")
    parser.add_argument("--scripts-dir", default="recordings/scripts")
    parser.add_argument("--terms", default="data/en_terms.txt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    language = "" if args.variant == "auto" else args.variant
    keyword_dir = Path(tempfile.mkdtemp(prefix="whistle-keywords-"))

    def build_transcriber(hotwords: list[str]):
        if hotwords and "{keywords}" not in args.command:
            print("warning: the command template has no {keywords}; hotwords are ignored")
        keyword_path = keyword_dir / "keywords.txt"
        keyword_path.write_text("\n".join(hotwords) + ("\n" if hotwords else ""), encoding="utf-8")
        command = render_command(
            args.command, lang=language, keywords_path=keyword_path if hotwords else None
        )
        return external_transcriber(command)

    return run_sets(
        system="whistle",
        variant=f"{args.variant}-{args.hotwords}",
        sets=[name.strip() for name in args.sets.split(",") if name.strip()],
        manifests_dir=args.manifests_dir,
        audio_root=args.audio_root,
        out_dir=args.out,
        build_transcriber=build_transcriber,
        hotwords_mode=args.hotwords,
        scripts_dir=args.scripts_dir,
        terms_path=args.terms,
        limit=args.limit,
        meta_extra={"language": args.variant, "command": args.command, "padded_30s": None},
    )


if __name__ == "__main__":
    raise SystemExit(main())
