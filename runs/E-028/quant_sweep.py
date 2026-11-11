"""Quantization sweep for the E-028 checkpoint, selected on dev only.

The fp32 graph is the reference; variants differ in quantization settings. The head-fp32
variant is included to test whether the output layer is the sensitive part, even though it
is over the 20 MB package budget. T is not read here: a variant is selected on dev WER, then
the chosen one is read out on T separately.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from stt.decode import load_pieces
from stt.eval import score_rows
from stt.export import export_fp32
from stt.lm import BinaryBPELM
from stt.model import MothSTT
from stt.onnx_runner import OnnxMoth
from stt.train import load_sources

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "runs/E-028/checkpoints/ckpt_best.pt"
FP32 = Path("/tmp/moth-e028-fp32.onnx")
OUT = REPO / "runs/E-028/quant"
RELEASE = REPO / "runs/E-028/release"
VOCAB = RELEASE / "vocab.json"
LM = RELEASE / "lm_bpe3.bin"
FIXED_PACKAGE = 7_530_083  # all release files except model.int8.onnx
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4}
DEV_SETS = (
    ("cv_it_dev", "data/manifests/cv_it_dev.jsonl", "data/shards/cv_it_dev", 600),
    ("fleurs_it_dev", "data/manifests/fleurs_it_dev.jsonl", "data/shards/fleurs_it_dev", 0),
    ("fleurs_en_dev", "data/manifests/fleurs_en_dev.jsonl", "data/shards/fleurs_en_dev", 0),
)


def _decode():
    spec = importlib.util.spec_from_file_location("e022eval", REPO / "runs/E-022/eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_int8


def _fp32_export() -> Path:
    if FP32.is_file():
        return FP32
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = MothSTT(vocab_size=1024, head_blocks=2)
    model.load_state_dict(payload["model"], strict=True)
    export_fp32(model, FP32)
    return FP32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    fp32 = _fp32_export()
    variants = {
        "fp32_reference": None,
        "int8_default": {"per_channel": True, "reduce_range": False},
        "int8_per_tensor": {"per_channel": False, "reduce_range": False},
        "int8_reduce_range": {"per_channel": True, "reduce_range": True},
        "int8_quint8": {
            "per_channel": True,
            "reduce_range": False,
            "weight_type": QuantType.QUInt8,
        },
        "int8_head_fp32": {
            "per_channel": True,
            "reduce_range": False,
            "nodes_to_exclude": ["/head/MatMul"],
        },
    }
    for name, settings in variants.items():
        target = out / f"{name}.onnx"
        if settings is None:
            target = fp32
        elif not target.is_file():
            quantize_dynamic(
                fp32,
                target,
                per_channel=settings["per_channel"],
                reduce_range=settings["reduce_range"],
                weight_type=settings.get("weight_type", QuantType.QInt8),
                nodes_to_exclude=settings.get("nodes_to_exclude"),
            )
        variants[name] = target

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

    results = {}
    for name, path in variants.items():
        moth = OnnxMoth(path, VOCAB)
        per_set = {}
        all_rows = []
        all_hyps = {}
        for set_name, (index, rows, root) in dev.items():
            hyps = decode_int8(moth, pieces, lm, index, rows, root, PARAMS)
            scored = score_rows(rows, hyps, bootstrap=0)
            per_set[set_name] = {
                "n": scored["overall"]["n"],
                "wer": scored["overall"]["wer"],
            }
            all_rows.extend(rows)
            all_hyps.update(hyps)
        combined = score_rows(all_rows, all_hyps, bootstrap=0)["overall"]
        results[name] = {
            "model_bytes": path.stat().st_size,
            "package_bytes": path.stat().st_size + FIXED_PACKAGE,
            "per_set": per_set,
            "dev_wer": combined["wer"],
        }
        print(
            f"{name:18s} dev WER {combined['wer']*100:6.3f} "
            f"model {path.stat().st_size/1e6:5.2f} MB package {results[name]['package_bytes']/1e6:5.2f} MB "
            + " ".join(f"{k} {v['wer']*100:.2f}" for k, v in per_set.items()),
            flush=True,
        )

    ordered = sorted(results, key=lambda name: (results[name]["dev_wer"], name))
    best = ordered[0]
    in_budget = [
        name
        for name in ordered
        if name != "fp32_reference" and results[name]["package_bytes"] <= 20_000_000
    ]
    chosen_in_budget = in_budget[0]
    (out / "sweep.json").write_text(
        json.dumps(
            {
                "best_dev_overall": best,
                "chosen_in_budget": chosen_in_budget,
                "variants": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"best dev overall: {best}; chosen in budget: {chosen_in_budget}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
