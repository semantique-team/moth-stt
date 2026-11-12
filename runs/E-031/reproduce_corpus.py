"""Reproduce E-031's corpus from pinned git snapshots and the original receipt."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from audit import INPUT_REVISION, REPO, digest, git_bytes, require_new

RECEIPT_PATH = "runs/E-031/frozen/lm-corpus.receipt.json"


def receipt_inputs(items, root=None):
    result = {}
    for item in items:
        relative = (
            Path(item["path"]).resolve().relative_to(root.resolve()).as_posix() if root is not None
            else item["path"].split("/moth-stt/", 1)[1]
        )
        key = (relative, item["role"])
        if key in result:
            raise ValueError(f"duplicate receipt input: {key}")
        result[key] = (item["sha256"], item["rows"])
    return result


def reproduce(out):
    receipt = json.loads(git_bytes(RECEIPT_PATH))
    historical_receipt = REPO / RECEIPT_PATH
    if json.loads(historical_receipt.read_bytes()) != receipt:
        raise ValueError("historical receipt differs from pinned E-031 receipt")
    if out.resolve() == historical_receipt.resolve():
        raise ValueError("output must not replace the historical receipt")
    if out.exists() and digest(out.read_bytes()) != receipt["corpus_sha256"]:
        raise ValueError("existing corpus differs from original receipt; choose a new --out")
    if version("num2words") != receipt["num2words_version"]:
        raise ValueError(f"reproduction requires num2words=={receipt['num2words_version']}")
    with tempfile.TemporaryDirectory(prefix="e031-") as directory:
        root = Path(directory)
        groups = {"exclusion": [], "train_manifest": [], "train_script": []}
        for item in receipt["inputs"]:
            relative = item["path"].split("/moth-stt/", 1)[1]
            content = git_bytes(relative, INPUT_REVISION)
            if digest(content) != item["sha256"]:
                raise ValueError(f"{relative}: pinned input differs from original receipt")
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            groups[item["role"]].append(path)
        corpus = root / "corpus.jsonl"
        for name in ("stt/__init__.py", "stt/text.py", "stt/manifest.py", "stt/data/__init__.py",
                     "stt/data/freeze_lm_corpus.py"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(git_bytes(name))
        if digest((root / "stt/text.py").read_bytes()) != receipt["normalizer_sha256"]:
            raise ValueError("pinned normalizer differs from original receipt")
        command = [sys.executable, "-m", "stt.data.freeze_lm_corpus"]
        flags = {"train_manifest": "--train-manifest", "train_script": "--train-script",
                 "exclusion": "--exclude"}
        for role, paths in groups.items():
            for path in paths:
                command.extend([flags[role], str(path)])
        command.extend(["--out", str(corpus), "--report", str(root / "receipt.json")])
        subprocess.run(command, cwd=root, env={**os.environ, "PYTHONPATH": str(root)}, check=True)
        generated = json.loads((root / "receipt.json").read_bytes())
        for key, expected in receipt.items():
            if key == "inputs":
                if receipt_inputs(generated[key], root) != receipt_inputs(expected):
                    raise ValueError("reproduced input receipt mismatch")
            elif generated[key] != expected:
                raise ValueError(f"reproduced {key} differs from original receipt")
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            require_new(out)
            with out.open("xb") as handle:
                handle.write(corpus.read_bytes())
    return {"corpus_sha256": receipt["corpus_sha256"], "input_revision": INPUT_REVISION,
            "out": str(out), "historical_receipt": "preserved"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs/E-031/frozen/lm-corpus.jsonl")
    args = parser.parse_args()
    print(json.dumps(reproduce(args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
