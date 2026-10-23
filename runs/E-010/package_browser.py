"""Package the E-008 LM and decoder settings into the browser release.

Run after ``stt/export.py`` and after the E-008 LM exists. The release stays gitignored; the
binary LM and the patched config are covered by ``runs/E-010/``.
"""

from __future__ import annotations

import json
from pathlib import Path

from stt.data.sources import sha256_file
from stt.lm import BPETrigramLM

RELEASE = Path("release")
LM_PICKLE = Path("runs/E-008/lm_bpe3.pkl.gz")
LM_BINARY = Path("release/lm_bpe3.bin")


def main() -> int:
    config_path = RELEASE / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not (RELEASE / "model.int8.onnx").exists():
        raise SystemExit("release/model.int8.onnx missing: run stt/export.py first")

    lm = BPETrigramLM.load(LM_PICKLE)
    lm.save_browser(LM_BINARY)
    config["decode"] = {
        "default": "beam",
        "beam": 5,
        "top_k": 8,
        "lm": LM_BINARY.name,
        "lm_weight": 0.2,
        "hotword_bonus": 2.0,
    }
    config["files"][LM_BINARY.name] = f"sha256:{sha256_file(LM_BINARY)}"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{LM_BINARY}: {LM_BINARY.stat().st_size / 1e6:.2f} MB")
    print(f"decode: {json.dumps(config['decode'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
