"""Portable, local-only readiness report. Never loads weights or runs a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

SMALL_FILE_BYTES = 16 * 1024 * 1024
RUNTIME_REQUIRED = ("numpy", "onnxruntime", "soundfile", "num2words")
TRAINING_REQUIRED = ("torch", "sentencepiece", "pyyaml")
PACKAGES = (
    "moth-stt", "numpy", "onnxruntime", "soundfile", "num2words", "torch",
    "sentencepiece", "pyyaml", "onnx", "transformers",
)


def environment() -> dict:
    """Allowlisted metadata only; no hostname, environment variables or package locations."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    accelerator = {"informational_only": True, "cuda_available": None, "mps_available": None}
    if versions["torch"] is not None:
        try:
            torch = importlib.import_module("torch")
            accelerator.update(
                cuda_available=bool(torch.cuda.is_available()),
                mps_available=bool(torch.backends.mps.is_available()),
            )
        except Exception as error:  # noqa: BLE001 - optional native probe must not block reporting
            accelerator["probe_error"] = type(error).__name__
    return {
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 12),
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "installed_versions": versions,
        "accelerators": accelerator,
    }


def build_report(
    config: str | Path | None = None,
    release: str | Path | None = None,
    *,
    evaluation: bool = False,
    hash_checkpoints: bool = False,
    out: str | Path | None = None,
) -> dict:
    """Return JSON-ready evidence, including failures. Config paths retain cwd semantics.

    Large checkpoints are streamed only on opt-in; mel shards are never hashed here.
    Strict evaluation explicitly delegates to its existing (potentially expensive) checker.
    """
    cwd = Path.cwd().resolve()
    config_path = Path(config).resolve() if config is not None else None
    roots = [("cwd", cwd)]
    if config_path is not None and not config_path.parent.is_relative_to(cwd):
        roots.append(("config_directory", config_path.parent))
    external = {}

    def location(path):
        path = Path(path).resolve()
        for label, root in roots:
            if path.is_relative_to(root):
                return {"root": label, "path": path.relative_to(root).as_posix()}
        # Do not expose private ancestor names, filenames or cross-drive absolute paths.
        return {"root": "external", "path": external.setdefault(path, str(len(external) + 1))}

    report = {
        "schema_version": 1, "ok": True, "source_root": ".",
        "path_roots": [label for label, _ in roots],
        "environment": environment(), "checks": {}, "files": [], "issues": [],
        "limitations": [
            "No training, model/optimizer construction, checkpoint deserialization or forward.",
            "Hashes establish byte identity, not consent, authenticity or independent evidence.",
            "No automatic transfer, archive, upload or recursive data scan.",
            "Trusted configs/indices may reference external files; path reporting is not a sandbox.",
        ],
    }

    def issue(stage, message, error=None):
        item = {"stage": stage, "message": message}
        if error is not None:
            # Validator exceptions can contain transcripts, secrets and absolute paths.
            item["error_type"] = type(error).__name__
        report["issues"].append(item)
        report["ok"] = False

    if not report["environment"]["python_supported"]:
        issue("environment", "Python 3.12 or newer is required.")

    required = list(RUNTIME_REQUIRED)
    if config_path is not None:
        required.extend(TRAINING_REQUIRED)
    missing = [name for name in required
               if report["environment"]["installed_versions"][name] is None]
    report["dependencies"] = {
        "required": required,
        "missing": missing,
        "native_runtime_load": "not_verified",
    }
    report["checks"]["dependencies"] = "failed" if missing else "passed"
    for name in missing:
        issue("dependencies", "Required installed distribution missing: " + name)

    inventory = {}

    def add(path, role):
        path = Path(path).resolve()
        inventory.setdefault(path, set()).add(role)
        return path

    raw = None
    if config_path is not None:
        add(config_path, "config")
        try:
            yaml = importlib.import_module("yaml")
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise TypeError("config must be a mapping")
        except Exception as error:  # noqa: BLE001 - optional YAML loader errors become issues
            issue("config", "Cannot read YAML mapping; check file and training dependencies.", error)
        if isinstance(raw, dict):
            try:
                checker = importlib.import_module("stt.train").check_training_config
                checker(raw, check_files=False)
                report["checks"]["training_config"] = "passed"
            except Exception as error:  # noqa: BLE001 - imported validator failures become issues
                report["checks"]["training_config"] = "failed"
                issue("training_config", "Training config validation unavailable or rejected; "
                      "review recipe fields with check_training_config(check_files=False).", error)
            # Inventory independently: the validator can reject the recipe before listing paths.
            try:
                for key in ("tokenizer", "vocab"):
                    if raw.get(key):
                        add(raw[key], key)
                for key in ("init", "init_checkpoint"):
                    if raw.get("model", {}).get(key):
                        add(raw["model"][key], "checkpoint")
                for split in ("train", "dev", "dev_en", "dev_mixed"):
                    for source in raw.get("data", {}).get(split, []):
                        if source.get("manifest"):
                            add(source["manifest"], "manifest")
                        if not source.get("shards"):
                            continue
                        index = add(Path(source["shards"]) / "index.json", "shard_index")
                        if not index.is_file():
                            continue
                        try:
                            entries = json.loads(index.read_text(encoding="utf-8"))
                            for entry in entries.values():
                                add(index.parent / entry["shard"], "mel_shard")
                        except (OSError, UnicodeError, ValueError, TypeError, KeyError,
                                AttributeError) as error:
                            issue("inventory", "Cannot read a declared shard index.", error)
                decoder = raw.get("evaluation", {}).get("decoder", {})
                if decoder.get("lm_path"):
                    add(decoder["lm_path"], "evaluation_lm")
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                issue("inventory", "Invalid config file inventory structure.", error)
    if evaluation:
        try:
            if not isinstance(raw, dict):
                raise TypeError("evaluation requires config")
            importlib.import_module("stt.training_eval").check_config(raw)
            report["checks"]["evaluation"] = "passed"
        except Exception as error:  # noqa: BLE001 - imported validator failures become issues
            report["checks"]["evaluation"] = "failed"
            issue("evaluation", "Strict evaluation check unavailable or rejected; "
                  "review splits, tokenizer, LM and evaluation assets.", error)
    else:
        report["checks"]["evaluation"] = "not_requested"

    if release is not None:
        root = Path(release).resolve()
        add(root / "config.json", "release_asset")
        add(root / "NOTICE", "release_asset")
        try:
            packaged = json.loads((root / "config.json").read_text(encoding="utf-8"))
            names = set(packaged.get("files", {}))
            for section, key in (("model", "file"), ("tokenizer", "vocab"),
                                 ("features", "mel_filters"), ("vad", "file"), ("decode", "lm")):
                names.add(packaged[section][key])
            for name in sorted(names):
                target = (root / name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("release path escapes root")
                add(target, "release_asset")
        except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as error:
            issue("release_inventory", "Cannot inventory release config and referenced assets.", error)
        try:
            from stt.release import validate_release

            _, validated = validate_release(root)
            report["checks"]["release"] = "passed"
            report["release_budget"] = {
                key: validated[key] for key in ("total_bytes", "budget_bytes", "remaining_bytes")
            }
        except Exception as error:  # noqa: BLE001 - imported validator failures become issues
            report["checks"]["release"] = "failed"
            issue("release", "Release validation unavailable or rejected (20 MB asset tier).", error)

    disk_parents = set()
    for path, roles in inventory.items():
        item = {**location(path), "roles": sorted(roles), "exists": path.is_file(),
                "bytes": None, "sha256": None}
        try:
            if not item["exists"]:
                issue("files", "Required file missing: " + item["root"] + ":" + item["path"])
            else:
                item["bytes"] = path.stat().st_size
                if "mel_shard" not in roles and (
                    item["bytes"] <= SMALL_FILE_BYTES
                    or (hash_checkpoints and "checkpoint" in roles)
                ):
                    with path.open("rb") as stream:
                        item["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                else:
                    item["hash_skipped"] = "mel_shard_or_large_file"
        except OSError as error:
            issue("files", "Cannot stat/hash a required file.", error)
        report["files"].append(item)
        disk_parents.add(path.parent)
    disk_parents.add(Path(out).resolve().parent if out is not None else cwd)
    report["disk"] = []
    for parent in sorted(disk_parents):
        while not parent.is_dir() and parent != parent.parent:
            parent = parent.parent
        if any(location(parent) == entry["existing_parent"] for entry in report["disk"]):
            continue
        try:
            usage = shutil.disk_usage(parent)
            report["disk"].append({"existing_parent": location(parent),
                                   "total_bytes": usage.total, "free_bytes": usage.free})
        except OSError as error:
            issue("disk", "Cannot inspect free space at an existing parent.", error)
    files = report["files"]
    report["transfer"] = {
        "known_required_bytes": sum(item["bytes"] or 0 for item in files),
        "missing_file_count": sum(not item["exists"] for item in files),
        "complete": not report["issues"],
        "measurement": "Unique declared files, raw uncompressed bytes; not a transfer operation.",
        "additional_disk_needed_bytes": None,
        "dependency_distribution_bytes": None,
        "note": "Missing sizes, raw audio, dependency wheels/caches, optimizer outputs and peak "
                "workspace are unknown. Release asset budget excludes Python/ORT/WASM distribution.",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New JSON file; parent must exist")
    parser.add_argument("--evaluation", action="store_true", help="Strict check, reads/hashes dev data")
    parser.add_argument("--hash-checkpoints", action="store_true", help="Stream large checkpoint hashes")
    args = parser.parse_args(argv)
    if args.out.exists() or not args.out.parent.is_dir():
        print("Output must be a new file in an existing directory.", file=sys.stderr)
        return 2
    report = build_report(args.config, args.release, evaluation=args.evaluation,
                          hash_checkpoints=args.hash_checkpoints, out=args.out)
    try:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except OSError:
        print("Cannot create new JSON output; no existing file was overwritten.", file=sys.stderr)
        return 2
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
