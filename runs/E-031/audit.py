"""Pinned E-031 evidence and fail-closed artifact checks (no network).

Requires full Git history (CI checkout fetch-depth: 0); never fetches missing revisions.
"""

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E031_REVISION = "8c2f449a89ed0c6024b3f0c8b73b51856d951d06"
RELEASE_REVISION = "9bc2379227ff8470079138c140514d80b11a556a"
INPUT_REVISION = "bd13c19ad45cef63c4588b73e5c8e001d0b4b2da"


def git_bytes(path, revision=E031_REVISION):
    completed = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{revision}:{path}"],
        check=False, capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"cannot read pinned Git evidence {revision}:{path}; full Git history is required "
            "(CI actions/checkout fetch-depth: 0). Restore history locally; no automatic fetch. "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout


def digest(content):
    return hashlib.sha256(content).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def release_record(version):
    path = f"runs/{'E-028' if version == '0.8.0' else 'E-031'}/release-{version}.json"
    return json.loads(git_bytes(path, RELEASE_REVISION))


def verify_file(path, record, name):
    expected = record["files"][name]
    if not path.is_file() or digest(path.read_bytes()) != expected["sha256"]:
        raise ValueError(f"{path}: expected verified {record['release']} {name} "
                         f"sha256 {expected['sha256']}")


def verify_source(source):
    record = release_record("0.8.0")
    for name in ("model.int8.onnx", "vocab.json", "mel_filters.bin", "silero_vad.onnx", "NOTICE"):
        verify_file(source / name, record, name)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config["decode"] != record["decode"] or config["vad"] != record["vad"]:
        raise ValueError("source decoder/VAD differs from pinned 0.8.0 record")
    return record


def read_pinned_hypotheses(path):
    result = {}
    for line in git_bytes(path).decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = str(row["id"])
        if uid in result:
            raise ValueError(f"duplicate pinned hypothesis id {uid}")
        result[uid] = row["hyp"]
    return result


def require_new(path):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}; use a new --name/output path")


def verified_parity(parity, assets, params, code, audio_sha256):
    return (
        parity.get("assets") == assets and parity.get("params") == params
        and parity.get("code") == code and parity.get("audio_sha256") == audio_sha256
        and parity.get("exact_match") is True
        and isinstance(parity.get("python_text"), str)
        and parity.get("python_text") == parity.get("js_text")
        and parity.get("scope") == "same-logits decoder only"
        and isinstance(parity.get("frames"), int) and parity["frames"] > 0
        and isinstance(parity.get("logits_sha256"), str) and len(parity["logits_sha256"]) == 64
    )
