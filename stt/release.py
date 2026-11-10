"""Validate a packaged release without importing inference or training dependencies.

``validate_release`` returns ``(config, report)`` and raises ``ValueError`` for an
invalid release (including missing/unreadable files). MB budgets use decimal bytes.
Hashes prove consistency with config, not publisher authenticity. Config and NOTICE
hashes are reported even when no external trusted manifest is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath


def validate_release(path: Path | str, budget_bytes: int = 20_000_000) -> tuple[dict, dict]:
    """Check config semantics, safe paths, required assets, hashes and actual size.

    Every entry in ``config.files`` is required and counted, in addition to the five
    referenced runtime assets, config and NOTICE. An explicit budget may select the
    extended tier, but must never exceed 25,000,000 bytes. Binary format parsing is
    left to the runtime readers; the LM must at least carry the MOTHLMB1 signature.
    """
    if type(budget_bytes) is not int or not 0 < budget_bytes <= 25_000_000:
        raise ValueError("budget_bytes must be a positive integer <= 25000000")
    root = Path(path).resolve()

    def safe_file(name: str) -> Path:
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or ":" in name
            or any(ord(char) < 32 for char in name)
            or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
        ):
            raise ValueError(f"unsafe release path: {name!r}")
        if any(suffix.lower() in (".pkl", ".pickle") for suffix in Path(name).suffixes):
            raise ValueError(f"pickle is not a release asset: {name}")
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"release path escapes root: {name}")
        if not target.is_file():
            raise ValueError(f"missing release file: {name}")
        return target

    def object_field(parent: dict, key: str) -> dict:
        value = parent.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be an object")  # noqa: TRY004 - invalid JSON data
        return value

    def number(parent: dict, key: str, *, integer: bool = False) -> float:
        value = parent.get(key)
        if type(value) not in ((int,) if integer else (int, float)) or (
            type(value) is float and not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite {'integer' if integer else 'number'}")
        return value

    try:
        config = json.loads(safe_file("config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config must be an object")  # noqa: TRY004 - invalid JSON data
        if config.get("name") != "moth-stt":
            raise ValueError("name must be moth-stt")
        if not isinstance(config.get("version"), str) or not config["version"].strip():
            raise ValueError("version must be a nonempty string")
        if number(config, "sample_rate", integer=True) != 16000:
            raise ValueError("sample_rate must be 16000")

        features = object_field(config, "features")
        model = object_field(config, "model")
        tokenizer = object_field(config, "tokenizer")
        decode = object_field(config, "decode")
        vad = object_field(config, "vad")
        expected = (
            (
                features,
                {
                    "n_fft": 400,
                    "hop": 160,
                    "n_mels": 80,
                    "window": "hann_periodic",
                    "center": True,
                    "pad_mode": "reflect",
                    "drop_last_frame": True,
                    "floor_range": 8.0,
                    "scale": "(x+4)/4",
                },
            ),
            (
                model,
                {"input": "mel", "output": "log_probs", "frame_rate_hz": 50, "max_seconds": 30},
            ),
            (tokenizer, {"blank_id": 0, "space_marker": "\u2581"}),
            (vad, {"window": 512, "context": 64}),
        )
        for section, fields in expected:
            for key, value in fields.items():
                actual = section.get(key)
                if actual != value or isinstance(actual, bool) != isinstance(value, bool):
                    raise ValueError(f"unsupported {key}: {actual!r}; expected {value!r}")
        if decode.get("default") not in ("beam", "greedy"):
            raise ValueError("decode.default must be beam or greedy")
        if "segment" in decode and type(decode["segment"]) is not bool:
            raise ValueError("decode.segment must be a boolean")
        for key in ("beam", "top_k"):
            if number(decode, key, integer=True) <= 0:
                raise ValueError(f"{key} must be positive")
        if number(decode, "lm_weight") < 0:
            raise ValueError("lm_weight must be nonnegative")
        number(decode, "token_bonus")
        if not 0 <= number(vad, "threshold") <= 1:
            raise ValueError("vad.threshold must be between 0 and 1")

        assets = [
            (model.get("file"), ".onnx"),
            (tokenizer.get("vocab"), ".json"),
            (features.get("mel_filters"), ".bin"),
            (vad.get("file"), ".onnx"),
            (decode.get("lm"), ".bin"),
        ]
        for name, suffix in assets:
            target = safe_file(name)
            if target.suffix != suffix:
                raise ValueError(f"{name} must be a {suffix} asset")
        names = [name for name, _ in assets]
        if len(set(names + ["config.json", "NOTICE"])) != 7:
            raise ValueError("the five runtime assets, config and NOTICE must be distinct")
        hashes = object_field(config, "files")
        for name in names:
            if name not in hashes:
                raise ValueError(f"missing hash for {name}")
        if "config.json" in hashes:
            raise ValueError("config.json cannot contain its own hash")

        files = {}
        for name in sorted(set(names + ["config.json", "NOTICE", *hashes])):
            target = safe_file(name)
            size = target.stat().st_size
            if size == 0:
                raise ValueError(f"empty release file: {name}")
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            declared = hashes.get(name)
            if name in hashes:
                if not isinstance(declared, str) or not re.fullmatch(
                    r"sha256:[0-9a-f]{64}", declared
                ):
                    raise ValueError(f"invalid sha256 declaration for {name}")
                if declared != f"sha256:{digest}":
                    raise ValueError(f"sha256 mismatch for {name}")
            files[name] = {"bytes": size, "sha256": digest, "hash_verified": name in hashes}
        with safe_file(decode["lm"]).open("rb") as stream:
            if stream.read(8) != b"MOTHLMB1":
                raise ValueError("LM must use the MOTHLMB1 binary format, not pickle")
        total = sum(entry["bytes"] for entry in files.values())
        if total > budget_bytes:
            raise ValueError(f"release size {total} bytes exceeds budget {budget_bytes} bytes")
        return config, {
            "release": str(root),
            "files": files,
            "total_bytes": total,
            "budget_bytes": budget_bytes,
            "remaining_bytes": budget_bytes - total,
        }
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read release: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="directory containing config.json")
    parser.add_argument(
        "--budget-bytes",
        type=int,
        default=20_000_000,
        help="asset budget in bytes; explicit extended tier up to 25000000",
    )
    parser.add_argument(
        "--browser-root",
        type=Path,
        help="separately count ALL raw files, including vendored ORT/WASM",
    )
    args = parser.parse_args(argv)
    try:
        config, report = validate_release(args.path, args.budget_bytes)
        report["version"] = config["version"]
        report["ok"] = True
        if args.browser_root is not None:
            root = args.browser_root.resolve()
            if not root.is_dir():
                raise ValueError(f"browser root is not a directory: {root}")
            files = [entry for entry in root.rglob("*") if entry.is_file()]
            if any(not entry.resolve().is_relative_to(root) for entry in files):
                raise ValueError("browser root contains a file symlink escaping its root")
            report["browser"] = {
                "root": str(root),
                "raw_total_bytes": sum(entry.stat().st_size for entry in files),
                "file_count": len(files),
                "measurement": "raw on-disk files, not HTTP transfer size",
            }
    except (ValueError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
