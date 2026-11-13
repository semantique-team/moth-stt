"""E-037 — export the fp32 graph of the E-028 checkpoint for the paired readout.

The shipped INT8 graph is not re-quantized: this script only produces the fp32 reference from the
same weights and verifies it against PyTorch with the standard export gate.

    uv run python runs/E-037/export_fp32.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from stt.data.sources import sha256_file
from stt.export import export_fp32, verify_fp32
from stt.model import MothSTT

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CHECKPOINT = REPO / "runs/E-028/checkpoints/ckpt_best.pt"
INT8 = REPO / "runs/E-028/release/model.int8.onnx"
FP32 = OUT / "model.fp32.onnx"


def main() -> int:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = MothSTT(vocab_size=1024, head_blocks=2)
    model.load_state_dict(payload["model"])
    export_fp32(model, FP32)
    verification = verify_fp32(model, FP32, seed=0)
    report = {
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_step": int(payload["step"]),
        "fp32": {"path": str(FP32.relative_to(REPO)),
                 "bytes": FP32.stat().st_size,
                 "sha256": sha256_file(FP32)},
        "int8": {"path": str(INT8.relative_to(REPO)),
                 "bytes": INT8.stat().st_size,
                 "sha256": sha256_file(INT8)},
        "verification": verification,
    }
    (OUT / "fp32.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
