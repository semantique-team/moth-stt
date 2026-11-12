"""Add the unchanged 0.7.0 LM to the E-028 export and set the published decode parameters."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from stt.data.sources import sha256_file
from stt.release import validate_release

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "runs/E-028/release"
LM = REPO / "release/lm_bpe3.bin"


def main() -> int:
    if not LM.is_file():
        raise SystemExit(f"missing unchanged LM: {LM}")
    shutil.copyfile(LM, RELEASE / "lm_bpe3.bin")
    config_path = RELEASE / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["decode"] = {
        "default": "beam",
        "beam": 12,
        "top_k": 10,
        "lm": "lm_bpe3.bin",
        "lm_weight": 0.55,
        "token_bonus": 2.7,
    }
    config["files"]["lm_bpe3.bin"] = f"sha256:{sha256_file(RELEASE / 'lm_bpe3.bin')}"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _, report = validate_release(RELEASE, budget_bytes=20_000_000)
    total = sum(path.stat().st_size for path in RELEASE.iterdir() if path.is_file())
    print(json.dumps({"release": report["release"], "total_bytes": total, "report": report}, indent=2))
    if total > 20_000_000:
        raise SystemExit(f"package {total} exceeds 20,000,000 bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
