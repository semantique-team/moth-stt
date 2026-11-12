"""E-030 decoder sweep on the E-028 INT8 artifact, selected on dev only.

The E-028 INT8 model deletes more words than 0.7.0 INT8 (8.49% vs 7.88% on T-600) and inserts
fewer (1.03% vs 1.29%). The shipped decode arm (beam 12 / top-k 10 / LM 0.5 / bonus 2.4) was
tuned for 0.7.0, so this sweep tests whether a slightly different operating point recovers the
missing ETA/WER without touching the weights. T is not read here: the selected arm is evaluated
once afterwards.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

from stt.decode import load_pieces
from stt.eval import score_rows
from stt.lm import BinaryBPELM
from stt.onnx_runner import OnnxMoth
from stt.train import load_sources

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "runs/E-028/release"
VOCAB = RELEASE / "vocab.json"
LM = RELEASE / "lm_bpe3.bin"
OUT = REPO / "runs/E-030"

#: (beam, top_k, lm_weight, token_bonus); frozen is the shipped arm.
ARMS = {
    "frozen": (12, 10, 0.50, 2.4),
    "bonus_2.0": (12, 10, 0.50, 2.0),
    "bonus_2.7": (12, 10, 0.50, 2.7),
    "bonus_3.0": (12, 10, 0.50, 3.0),
    "bonus_3.3": (12, 10, 0.50, 3.3),
    "lm_0.45_b2.7": (12, 10, 0.45, 2.7),
    "lm_0.55_b2.7": (12, 10, 0.55, 2.7),
    "lm_0.45_b3.0": (12, 10, 0.45, 3.0),
    "lm_0.55_b3.0": (12, 10, 0.55, 3.0),
    "beam16_b2.7": (16, 10, 0.50, 2.7),
    "topk12_b2.7": (12, 12, 0.50, 2.7),
    "cheap_b2.7": (8, 8, 0.50, 2.7),
}

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    dev = {}
    for name, manifest, shards, limit in DEV_SETS:
        index, rows, root = load_sources([{"manifest": manifest, "shards": shards}])
        if limit:
            rows = rows[:limit]
            keep = {row["id"] for row in rows}
            index = {uid: entry for uid, entry in index.items() if uid in keep}
        dev[name] = (index, rows, root)
        print(f"{name}: {len(rows)} rows", flush=True)

    pieces = load_pieces(VOCAB)
    lm = BinaryBPELM.load(LM)
    decode_int8 = _decode()
    moth = OnnxMoth(RELEASE / "model.int8.onnx", VOCAB)

    results = {}
    for name, (beam, top_k, lm_weight, token_bonus) in ARMS.items():
        params = {
            "beam": beam,
            "top_k": top_k,
            "lm_weight": lm_weight,
            "token_bonus": token_bonus,
        }
        started = time.perf_counter()
        per_set = {}
        all_rows: list[dict] = []
        all_hyps: dict[str, str] = {}
        for set_name, (index, rows, root) in dev.items():
            hyps = decode_int8(moth, pieces, lm, index, rows, root, params)
            scored = score_rows(rows, hyps, bootstrap=0)["overall"]
            per_set[set_name] = {
                "n": scored["n"],
                "wer": scored["wer"],
                "eta": scored["eta"],
                "del": scored["del"] / scored["ref_words"],
                "ins": scored["ins"] / scored["ref_words"],
                "sub": scored["sub"] / scored["ref_words"],
            }
            all_rows.extend(rows)
            all_hyps.update(hyps)
        combined = score_rows(all_rows, all_hyps, bootstrap=0)["overall"]
        results[name] = {
            "params": params,
            "dev_wer": combined["wer"],
            "dev_eta": combined["eta"],
            "dev_del": combined["del"] / combined["ref_words"],
            "dev_ins": combined["ins"] / combined["ref_words"],
            "dev_sub": combined["sub"] / combined["ref_words"],
            "per_set": per_set,
            "seconds": time.perf_counter() - started,
        }
        eta_text = "  n/a" if combined["eta"] is None else f"{combined['eta']*100:6.2f}"
        print(
            f"{name:15s} dev {combined['wer']*100:6.3f} eta {eta_text} "
            f"del {results[name]['dev_del']*100:5.2f} ins {results[name]['dev_ins']*100:5.2f} "
            f"{results[name]['seconds']:5.1f}s",
            flush=True,
        )

    ordered = sorted(results, key=lambda name: (results[name]["dev_wer"], name))
    (out / "sweep.json").write_text(
        json.dumps(
            {
                "model": "runs/E-028/release/model.int8.onnx",
                "selection": "dev WER: cv_it_dev 600 + fleurs_it_dev + fleurs_en_dev",
                "ordered": ordered,
                "arms": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"best dev: {ordered[0]} ({results[ordered[0]]['dev_wer']*100:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
