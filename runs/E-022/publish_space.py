"""Publish the static browser demo as a Hugging Face Space.

Assembles a staging directory from files that are already in the repository (the demo page, the JS
runtime, the verified `release/`, and the three ONNX Runtime files the WASM backend actually needs)
and uploads it. Nothing is duplicated in the repo: only the page and the Space card are versioned,
the 34 MB of binaries are copied from `release/` and `node_modules/` at build time.

    uv run python runs/E-022/publish_space.py --check          # stage + verify hashes, no network
    uv run python runs/E-022/publish_space.py --serve          # also serve it locally for testing
    HF_TOKEN=hf_... uv run python runs/E-022/publish_space.py --push --repo semantique-team/moth-demo

Why a **static** Space and not Gradio: the point of this model is that it runs in the user's
browser and their audio never leaves it. A Gradio Space would run the model on Hugging Face's CPU
and quietly tell the opposite story. Serving without COOP/COEP means no SharedArrayBuffer, so the
runtime falls back to 1 thread — measured at 20x real time on 20 s clips, which is plenty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPACE_SRC = REPO / "runs/E-022/space"
RELEASE = REPO / "release"
RECORD = REPO / "runs/E-022/release-0.6.0.json"

#: The demo page, its runtime, and the ORT build the WASM backend loads. `ort.wasm.min.mjs`
#: dynamically imports `ort-wasm-simd-threaded.mjs`, which loads the `.wasm` — all three must sit
#: in the same directory, which is why the Space keeps the `node_modules` layout.
JS_FILES = (
    "ort.js",
    "decode.js",
    "lm.js",
    "mel.js",
    "mic.js",
    "mic-worklet.js",
    "vad.js",
)
ORT_DIR = "node_modules/onnxruntime-web/dist"
ORT_FILES = (
    "ort.wasm.min.mjs",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
)
RELEASE_FILES = (
    "model.int8.onnx",
    "vocab.json",
    "mel_filters.bin",
    "silero_vad.onnx",
    "lm_bpe3.bin",
    "config.json",
    "NOTICE",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(destination: Path) -> dict:
    """Build the Space tree; returns a report of what went in and whether it verifies."""
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "release").mkdir(parents=True)
    (destination / ORT_DIR).mkdir(parents=True)

    problems: list[str] = []
    for name in ("index.html", "README.md"):
        shutil.copy2(SPACE_SRC / name, destination / name)
    for name in JS_FILES:
        source = REPO / "web" / name
        if not source.is_file():
            problems.append(f"missing {source}")
            continue
        shutil.copy2(source, destination / name)
    for name in ORT_FILES:
        source = REPO / "web" / ORT_DIR / name
        if not source.is_file():
            problems.append(f"missing {source}")
            continue
        shutil.copy2(source, destination / ORT_DIR / name)

    # The demo loads ./release/ and verifies these sha256 values in the browser before running.
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    for name in RELEASE_FILES:
        source = RELEASE / name
        if not source.is_file():
            problems.append(f"missing {source}")
            continue
        shutil.copy2(source, destination / "release" / name)
        expected = record["files"].get(name, {}).get("sha256")
        if expected and sha256(source) != expected:
            problems.append(f"{name}: does not match the recorded release")
    for name in record["files"]:
        if name not in RELEASE_FILES:
            problems.append(f"record has {name}, staging does not")

    total = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    return {"files": sum(1 for p in destination.rglob("*") if p.is_file()),
            "bytes": total, "problems": problems, "version": record["release"]}


def main() -> int:
    parser = argparse.ArgumentParser(prog="publish_space.py")
    parser.add_argument("--repo", default="semantique-team/moth-demo")
    parser.add_argument("--push", action="store_true", help="upload (default: stage and check only)")
    parser.add_argument("--check", action="store_true", help="stage and verify, no network")
    parser.add_argument("--serve", action="store_true", help="serve the staged site locally")
    parser.add_argument("--dest", default=None, help="staging directory (default: a temp dir)")
    parser.add_argument("--private", action="store_true", help="create the Space private")
    args = parser.parse_args()

    destination = Path(args.dest) if args.dest else Path(tempfile.mkdtemp(prefix="moth-space-"))
    report = stage(destination)
    print(json.dumps(report, indent=1))
    print(f"staged in {destination}")
    if report["problems"]:
        for problem in report["problems"]:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    if args.serve:
        print(f"\nserving {destination} on http://127.0.0.1:8767 (Ctrl-C to stop)")
        return subprocess.call(
            [sys.executable, "-m", "http.server", "8767", "--bind", "127.0.0.1"],
            cwd=destination,
        )

    if not args.push:
        print("\nnothing uploaded (staged only). Use --push and HF_TOKEN to publish.")
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
    api.create_repo(args.repo, repo_type="space", space_sdk="static",
                    private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(destination),
        repo_id=args.repo,
        repo_type="space",
        commit_message=f"Moth {report['version']} — static browser demo",
    )
    print(f"\ndone: https://huggingface.co/spaces/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())