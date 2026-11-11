"""Static QDQ quantization of the E-028 graph, calibrated on CV-IT dev only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from stt.decode import load_pieces
from stt.eval import score_rows
from stt.lm import BinaryBPELM
from stt.onnx_runner import OnnxMoth
from stt.train import load_sources

REPO = Path(__file__).resolve().parents[2]
FP32 = Path("/tmp/moth-e028-fp32.onnx")
OUT = REPO / "runs/E-028/quant"
RELEASE = REPO / "runs/E-028/release"
VOCAB = RELEASE / "vocab.json"
LM = RELEASE / "lm_bpe3.bin"
FIXED_PACKAGE = 7_530_083
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4}
DEV_SETS = (
    ("cv_it_dev", "data/manifests/cv_it_dev.jsonl", "data/shards/cv_it_dev", 600),
    ("fleurs_it_dev", "data/manifests/fleurs_it_dev.jsonl", "data/shards/fleurs_it_dev", 0),
    ("fleurs_en_dev", "data/manifests/fleurs_en_dev.jsonl", "data/shards/fleurs_en_dev", 0),
)


class Reader(CalibrationDataReader):
    def __init__(self, samples: list[np.ndarray]) -> None:
        self._samples = iter(samples)

    def get_next(self):
        try:
            return {"mel": next(self._samples)}
        except StopIteration:
            return None


def _decode():
    spec = importlib.util.spec_from_file_location("e022eval", REPO / "runs/E-022/eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_int8


def _calibration() -> list[np.ndarray]:
    index, rows, root = load_sources(
        [{"manifest": "data/manifests/cv_it_dev.jsonl", "shards": "data/shards/cv_it_dev"}]
    )
    rows = rows[:50]
    shards: dict[str, np.ndarray] = {}
    samples = []
    for row in rows:
        entry = index[row["id"]]
        shard = Path(entry["shard"]).name
        array = shards.get(shard)
        if array is None:
            array = np.load(root / entry["shard"], mmap_mode="r")
            shards[shard] = array
        start = entry["offset"] * 80
        mel = np.asarray(
            array[start : start + entry["frames"] * 80], dtype=np.float32
        ).reshape(entry["frames"], 80).T
        samples.append(mel[None])
    return samples


def main() -> int:
    target = OUT / "static_qdq.onnx"
    if not target.is_file():
        quantize_static(
            FP32,
            target,
            Reader(_calibration()),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
    pieces = load_pieces(VOCAB)
    lm = BinaryBPELM.load(LM)
    decode_int8 = _decode()
    dev = {}
    for name, manifest, shards, limit in DEV_SETS:
        index, rows, root = load_sources([{"manifest": manifest, "shards": shards}])
        if limit:
            rows = rows[:limit]
            keep = {row["id"] for row in rows}
            index = {uid: entry for uid, entry in index.items() if uid in keep}
        dev[name] = (index, rows, root)

    moth = OnnxMoth(target, VOCAB)
    per_set = {}
    all_rows, all_hyps = [], {}
    for name, (index, rows, root) in dev.items():
        hyps = decode_int8(moth, pieces, lm, index, rows, root, PARAMS)
        scored = score_rows(rows, hyps, bootstrap=0)
        per_set[name] = {"n": scored["overall"]["n"], "wer": scored["overall"]["wer"]}
        all_rows.extend(rows)
        all_hyps.update(hyps)
    combined = score_rows(all_rows, all_hyps, bootstrap=0)["overall"]
    result = {
        "model_bytes": target.stat().st_size,
        "package_bytes": target.stat().st_size + FIXED_PACKAGE,
        "dev_wer": combined["wer"],
        "per_set": per_set,
    }
    (OUT / "static.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
