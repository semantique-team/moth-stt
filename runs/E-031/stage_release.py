"""Stage the 0.8.1 release candidate: 0.8.0 assets + the E-031 A3 LM.

Copies the published 0.8.0 bundle into `runs/E-031/release/`, replaces `lm_bpe3.bin` with the
adopted E-031 binary, sets the version and the LM hash, and runs the release validator against the
20,000,000-byte budget. It does **not** touch the published `release/` directory and does not
upload anything.

    uv run python runs/E-031/stage_release.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from audit import release_record, verify_file, verify_source

from stt.data.sources import sha256_file
from stt.release import validate_release

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "release"
TARGET = REPO / "runs/E-031/release"
LM = REPO / "runs/E-031/build/frozen_1_2_terms.bin"
VERSION = "0.8.1"


def main() -> int:
    if not LM.is_file():
        raise SystemExit(f"missing E-031 LM: {LM}")
    verify_source(SOURCE)
    verify_file(LM, release_record("0.8.1"), "lm_bpe3.bin")
    TARGET.mkdir(parents=True, exist_ok=True)
    for item in SOURCE.iterdir():
        if item.is_file():
            shutil.copyfile(item, TARGET / item.name)
    shutil.copyfile(LM, TARGET / "lm_bpe3.bin")

    config_path = TARGET / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["version"] = VERSION
    config["files"]["lm_bpe3.bin"] = f"sha256:{sha256_file(TARGET / 'lm_bpe3.bin')}"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")

    _, report = validate_release(TARGET, budget_bytes=20_000_000)
    total = sum(path.stat().st_size for path in TARGET.iterdir() if path.is_file())
    summary = {
        "release": VERSION,
        "path": str(TARGET.relative_to(REPO)),
        "total_bytes": total,
        "model_sha256": sha256_file(TARGET / "model.int8.onnx"),
        "lm_sha256": sha256_file(TARGET / "lm_bpe3.bin"),
        "lm_bytes": (TARGET / "lm_bpe3.bin").stat().st_size,
        "report": report,
    }
    print(json.dumps(summary, indent=2))
    if total > 20_000_000:
        raise SystemExit(f"package {total} exceeds 20,000,000 bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
