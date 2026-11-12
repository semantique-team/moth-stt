"""E-030 product readout of the dev-selected decoder arm.

The dev sweep selected beam 12 / top-k 10 / LM 0.55 / bonus 2.7 on CV-IT + FLEURS dev. This
script decodes T-600, T-S3 and FLEURS test once for E-028 INT8 and, for a fair comparison, for 0.7.0 INT8
with the same arm. T is not used for selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from stt.decode import load_pieces
from stt.eval import score_rows
from stt.lm import BinaryBPELM
from stt.onnx_runner import OnnxMoth

REPO = Path(__file__).resolve().parents[2]
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}


def _decode():
    spec = importlib.util.spec_from_file_location("e022eval", REPO / "runs/E-022/eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_int8, module.sets, module.write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs/E-030/eval")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    decode_int8, sets, write_jsonl = _decode()
    available = sets()
    lm = BinaryBPELM.load(REPO / "release/lm_bpe3.bin")
    models = {
        "0.7.0-int8": REPO / "release",
        "e028-int8": REPO / "runs/E-028/release",
    }
    summary = {}
    for model_name, model_dir in models.items():
        moth = OnnxMoth(model_dir / "model.int8.onnx", model_dir / "vocab.json")
        model_pieces = load_pieces(model_dir / "vocab.json")
        for set_name in ("t600", "t-s3", "fleurs_it_test", "fleurs_en_test"):
            index, rows, root = available[set_name]
            hyps = decode_int8(moth, model_pieces, lm, index, rows, root, PARAMS)
            out_dir = args.out / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(out_dir / f"{set_name}.hyp.jsonl", hyps)
            scored = score_rows(rows, hyps, bootstrap=1000, seed=0)
            (out_dir / f"{set_name}.metrics.json").write_text(
                json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            overall = scored["overall"]
            summary[f"{model_name}/{set_name}"] = {
                "n": overall["n"],
                "wer": overall["wer"],
                "eta": overall["eta"],
                "del": overall["del"] / overall["ref_words"],
                "ins": overall["ins"] / overall["ref_words"],
                "sub": overall["sub"] / overall["ref_words"],
            }
            eta_text = "  n/a" if overall["eta"] is None else f"{overall['eta']*100:6.2f}"
            print(
                f"{model_name:11s} {set_name:15s} wer {overall['wer']*100:6.2f} "
                f"eta {eta_text} del {summary[f'{model_name}/{set_name}']['del']*100:5.2f} "
                f"ins {summary[f'{model_name}/{set_name}']['ins']*100:5.2f}",
                flush=True,
            )
    (args.out / "summary.json").write_text(
        json.dumps({"params": PARAMS, "results": summary}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
