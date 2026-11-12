"""E-031 — the single gate readout for the advancing LM arm.

Selection happened on full dev (`dev_sweep.py`). This script decodes the shipped int8 0.8.0
graph with the candidate **browser binary** and compares it, once, with the published 0.8.0
numbers in `runs/E-030/eval/e028-int8/`. It also scores the 47 rare rows (aligned ETA) and the
natural/gold term groups through `runs/E-015/eval_terms.py`.

    uv run python runs/E-031/gate.py --lm runs/E-031/build/frozen_1_2.bin --name frozen_1_2
    uv run python runs/E-031/gate.py --lm ... --name ... --noise
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from stt.decode import load_pieces
from stt.eval import _compare, score_rows
from stt.lm import BinaryBPELM
from stt.onnx_runner import OnnxMoth

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BINARY_BASELINE = 5_116_464
TOTAL_BASELINE = 19_440_761
PACKAGE_LIMIT = 20_000_000
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}
BASELINE = REPO / "runs/E-030/eval/e028-int8"
SETS = ("t600", "t-s3", "fleurs_it_test", "fleurs_en_test")


def _e022():
    spec = importlib.util.spec_from_file_location("e022eval", REPO / "runs/E-022/eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_int8, module.sets, module.write_jsonl


def _rare_ids() -> set[str]:
    return {
        json.loads(line)["id"]
        for line in (REPO / "runs/E-026/rare47-16k.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _words(text: str) -> int:
    return len(text.split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--noise", action="store_true", help="also decode the E-011 non-speech set")
    args = parser.parse_args()
    args.lm = args.lm.resolve()

    decode_int8, sets, write_jsonl = _e022()
    available = sets()
    lm = BinaryBPELM.load(args.lm)
    moth = OnnxMoth(REPO / "release/model.int8.onnx", REPO / "release/vocab.json")
    pieces = load_pieces(REPO / "release/vocab.json")
    out_dir = OUT / "eval" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "name": args.name,
        "lm": str(args.lm.relative_to(REPO)),
        "lm_bytes": args.lm.stat().st_size,
        "params": PARAMS,
        "sets": {},
    }
    for set_name in SETS:
        index, rows, root = available[set_name]
        hyp_path = out_dir / f"{set_name}.hyp.jsonl"
        metrics_path = out_dir / f"{set_name}.metrics.json"
        if hyp_path.exists() and metrics_path.exists():
            hypotheses = {
                item["id"]: item["hyp"]
                for item in (
                    json.loads(line)
                    for line in hyp_path.read_text(encoding="utf-8").splitlines()
                )
            }
            scored = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"{set_name:15s} reused existing decode", flush=True)
        else:
            hypotheses = decode_int8(moth, pieces, lm, index, rows, root, PARAMS)
            write_jsonl(hyp_path, hypotheses)
            scored = score_rows(rows, hypotheses, bootstrap=1000, seed=0)
            metrics_path.write_text(
                json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        baseline = json.loads((BASELINE / f"{set_name}.metrics.json").read_text(encoding="utf-8"))
        overall = scored["overall"]
        eta_text = "n/a" if overall["eta"] is None else f"{overall['eta'] * 100:6.2f}"
        result["sets"][set_name] = {
            "n": overall["n"],
            "wer": overall["wer"],
            "eta": overall["eta"],
            "vs_0.8.0": _compare(scored, baseline, bootstrap=1000, seed=0),
        }
        print(f"{set_name:15s} wer {overall['wer'] * 100:6.2f} eta {eta_text}", flush=True)

    rare_ids = _rare_ids()
    _, t_rows, _ = available["t600"]
    rare_rows = [row for row in t_rows if row["id"] in rare_ids]
    candidate_all = {
        item["id"]: item["hyp"]
        for item in (
            json.loads(line)
            for line in (out_dir / "t600.hyp.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    baseline_all = {
        item["id"]: item["hyp"]
        for item in (
            json.loads(line)
            for line in (BASELINE / "t600.hyp.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    rare_hyp = {uid: candidate_all[uid] for uid in rare_ids}
    rare_base_hyp = {uid: baseline_all[uid] for uid in rare_ids}
    rare_scored = score_rows(rare_rows, rare_hyp, bootstrap=1000, seed=0)
    rare_base = score_rows(rare_rows, rare_base_hyp, bootstrap=1000, seed=0)
    rare_per = rare_scored["per_utterance"]
    base_per = rare_base["per_utterance"]
    result["rare"] = {
        "n": rare_scored["overall"]["n"],
        "en_ref": sum(item["en_ref"] for item in rare_per.values()),
        "en_hit": sum(item["en_hit"] for item in rare_per.values()),
        "aligned_eta": rare_scored["overall"]["eta"],
        "baseline_en_hit": sum(item["en_hit"] for item in base_per.values()),
        "baseline_aligned_eta": rare_base["overall"]["eta"],
        "vs_0.8.0": _compare(rare_scored, rare_base, bootstrap=1000, seed=0),
    }
    print(json.dumps(result["rare"], ensure_ascii=False), flush=True)

    terms_out = out_dir / "terms.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "runs/E-015/eval_terms.py"),
            "--hyp", str(out_dir / "t600.hyp.jsonl"),
            "--baseline", str(BASELINE / "t600.hyp.jsonl"),
            "--out", str(terms_out),
        ],
        check=True,
        cwd=REPO,
    )
    result["terms"] = json.loads(terms_out.read_text(encoding="utf-8"))

    package_bytes = TOTAL_BASELINE - BINARY_BASELINE + args.lm.stat().st_size
    result["package_bytes"] = package_bytes
    if args.noise:
        candidate_words = baseline_words = 0
        index, rows, root = available["noise"]
        candidate_hyp = decode_int8(moth, pieces, lm, index, rows, root, PARAMS)
        baseline_lm = BinaryBPELM.load(REPO / "release/lm_bpe3.bin")
        baseline_hyp = decode_int8(moth, pieces, baseline_lm, index, rows, root, PARAMS)
        candidate_words = sum(_words(hyp) for hyp in candidate_hyp.values())
        baseline_words = sum(_words(hyp) for hyp in baseline_hyp.values())
        result["noise"] = {
            "clips": len(rows),
            "candidate_words": candidate_words,
            "baseline_words": baseline_words,
        }
        print(json.dumps(result["noise"]), flush=True)

    result["gates"] = {
        "bytes": package_bytes <= PACKAGE_LIMIT,
        "t600_wer": result["sets"]["t600"]["wer"] <= 0.2521,
        "t600_eta": result["sets"]["t600"]["eta"] >= 0.5367,
        "t-s3_wer": result["sets"]["t-s3"]["wer"] <= 0.2571,
        "t-s3_eta": result["sets"]["t-s3"]["eta"] >= 0.4964,
        "fleurs_it": result["sets"]["fleurs_it_test"]["wer"] <= 0.2075,
        "fleurs_en": result["sets"]["fleurs_en_test"]["wer"] <= 0.5031,
        "rare_eta": (result["rare"]["aligned_eta"] or 0.0) >= 0.284,
        "natural": result["terms"]["groups"]["natural"]["new"]["correct"] >= 431,
    }
    (OUT / "gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    print(json.dumps(result["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
