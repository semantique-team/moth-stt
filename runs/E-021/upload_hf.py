"""Publish the Moth release to a Hugging Face model repository (M8).

Uploads the *verified* `release/` folder exactly as `stt/export.py` built it, plus the model card
and the NOTICE. Refuses to run when the artifacts do not match the recorded hashes, so a stale or
hand-edited `release/` can never be published.

Dry run by default: it prints the plan and the file hashes without touching the network.

    # 1. check what would be uploaded
    uv run python runs/E-021/upload_hf.py --repo semantique-team/moth-stt

    # 2. publish (needs a token with write access)
    HF_TOKEN=hf_... uv run python runs/E-021/upload_hf.py --repo semantique-team/moth-stt --push

Token: create a *write* token at https://huggingface.co/settings/tokens. Use `--private` for a
staged repo; switch it public later from the model page. The token is read from the environment
only — never from a file in the repository (AGENTS.md: no secrets in the repo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RELEASE = Path("release")
#: The card moves with the release it documents (0.5.1 in runs/E-021/, 0.6.0 in runs/E-022/).
CARD = Path("runs/E-022/MODEL_CARD.md")
RECORD = Path("runs/E-022/release-0.6.0.json")

#: Images the card references, uploaded next to it. The card cannot switch image by colour scheme,
#: so it uses the white tile, which is legible on both light and dark themes.
CARD_ASSETS = ("logo-tile.png",)
CARD_ASSETS_DIR = Path("docs/assets")

#: Files the runtime downloads. `config.json` and `NOTICE` travel with them.
PAYLOAD = ("model.int8.onnx", "vocab.json", "mel_filters.bin", "silero_vad.onnx", "lm_bpe3.bin")
ALSO = ("config.json", "NOTICE")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(
    release: Path = RELEASE, card: Path = CARD, record_path: Path = RECORD
) -> tuple[dict, list[str]]:
    """Verify a release folder against the recorded hashes; return the record and any problems."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for name, entry in record["files"].items():
        path = release / name
        if not path.is_file():
            problems.append(f"missing: {path}")
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            problems.append(f"hash mismatch: {name}: {actual[:12]}… != {entry['sha256'][:12]}…")
    for name in ALSO:
        if not (release / name).is_file():
            problems.append(f"missing: {release / name}")
    if not card.is_file():
        problems.append(f"missing model card: {card}")
    else:
        # A card that documents a different release is worse than no card: it publishes true
        # numbers for the wrong artifact. Required, because 0.6.0 changed the system without
        # changing the weights (E-022).
        card_text = card.read_text(encoding="utf-8")
        heading = next(
            (line for line in card_text.splitlines() if line.startswith("# ")),
            "",
        )
        if record["release"] not in heading:
            problems.append(
                f"model card is for a different release: its title is {heading.strip()!r}, "
                f"expected it to name {record['release']}"
            )
        # Only check an asset the card actually references, so a caller with its own card is not
        # forced to ship our images.
        for name in CARD_ASSETS:
            if name in card_text and not (CARD_ASSETS_DIR / name).is_file():
                problems.append(f"card references {name}, missing from {CARD_ASSETS_DIR}")
    total = sum((release / name).stat().st_size for name in PAYLOAD if (release / name).is_file())
    declared = record["package_bytes"]
    if total != declared and not problems:
        problems.append(
            f"payload size {total} != recorded {declared} "
            "(package_bytes covers the five runtime files, not config.json/NOTICE)"
        )
    return record, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="target repo, e.g. semantique-team/moth-stt")
    parser.add_argument("--push", action="store_true", help="actually upload (default: dry run)")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--release", default=str(RELEASE))
    parser.add_argument("--card", default=str(CARD))
    parser.add_argument("--record", default=str(RECORD))
    args = parser.parse_args(argv)

    release, card = Path(args.release), Path(args.card)
    record, problems = check(release, card, Path(args.record))
    if problems:
        print("refusing to publish — the release folder is not the verified one:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("\nrebuild it first:", file=sys.stderr)
        print(
            "  uv run python stt/export.py --out release --checkpoint "
            "runs/E-017/checkpoints/ckpt_best.pt --vocab release/vocab.json --head-blocks 2 "
            f"--version {record['release']}",
            file=sys.stderr,
        )
        return 2

    total = sum((release / name).stat().st_size for name in (*PAYLOAD, *ALSO))
    print(f"verified release {record['release']} — {total} bytes ({total / 1e6:.2f} MB)")
    for name in (*PAYLOAD, *ALSO):
        path = release / name
        print(f"  {name:<20} {path.stat().st_size:>10}  sha256 {sha256(path)[:16]}…")
    print(f"  README.md (from {card})")
    for name in CARD_ASSETS:
        if (CARD_ASSETS_DIR / name).is_file():
            print(f"  {name} (from {CARD_ASSETS_DIR})")
    print(
        f"\ntarget: https://huggingface.co/{args.repo}  ({'private' if args.private else 'public'})"
    )

    if not args.push:
        print("\ndry run: nothing uploaded. Re-run with --push and HF_TOKEN set.")
        return 0

    # Three ways in, in order: the environment, then the token `hf auth login` stored (which keeps
    # the secret out of shell history and out of any chat), then refuse.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        from huggingface_hub import get_token

        token = get_token()
    if not token:
        print(
            "error: --push needs HF_TOKEN in the environment, or a token from `uv run hf auth login`",
            file=sys.stderr,
        )
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"model card: Moth {record['release']}",
    )
    for name in CARD_ASSETS:
        source = CARD_ASSETS_DIR / name
        if not source.is_file():
            continue
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"card: {name}",
        )
        print(f"  uploaded {name}", flush=True)
    for name in (*PAYLOAD, *ALSO):
        api.upload_file(
            path_or_fileobj=str(release / name),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"release {record['release']}: {name}",
        )
        print(f"  uploaded {name}", flush=True)
    print(f"\ndone: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
