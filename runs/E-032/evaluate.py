"""E-032 — evaluate the new checkpoint against 0.8.0 and against the E-031 LM.

The checkpoint is exported once (`stt.export`, int8) into `runs/E-032/release/`; this script
decodes T-600, T-S3, FLEURS-IT/EN, the 47 rare rows and the natural/gold groups with **two LMs**:
the shipped 0.8.0 binary (the pre-registered E-032 baseline) and the adopted E-031 A3 binary
(the current product candidate). Decoder settings are fixed at 0.55/2.7.

    uv run python -m stt.export --out runs/E-032/release \
        --checkpoint runs/E-032/checkpoints/ckpt_best.pt \
        --vocab release/vocab.json --head-blocks 2 --version 0.8.1-e032
    uv run python runs/E-032/evaluate.py
"""

from __future__ import annotations

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
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}
SETS = ("t600", "t-s3", "fleurs_it_test", "fleurs_en_test")
LMS = {
    "shipped-0.8.0": REPO / "release/lm_bpe3.bin",
    "e031-a3": REPO / "runs/E-031/build/frozen_1_2_terms.bin",
}
BASELINES = {
    "shipped-0.8.0": REPO / "runs/E-030/eval/e028-int8",
    "e031-a3": REPO / "runs/E-031/eval/frozen_1_2_terms",
}


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


def _read_hyp(path: Path) -> dict[str, str]:
    return {
        item["id"]: item["hyp"]
        for item in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def main() -> int:
    decode_int8, sets, write_jsonl = _e022()
    available = sets()
    rare_ids = _rare_ids()
    _, t_rows, _ = available["t600"]
    rare_rows = [row for row in t_rows if row["id"] in rare_ids]
    moth = OnnxMoth(OUT / "release/model.int8.onnx", REPO / "release/vocab.json")
    pieces = load_pieces(REPO / "release/vocab.json")

    result: dict = {"params": PARAMS, "checkpoint": "runs/E-032/checkpoints/ckpt_best.pt", "lms": {}}
    for lm_name, lm_path in LMS.items():
        lm = BinaryBPELM.load(lm_path)
        lm_out = OUT / "eval" / lm_name
        lm_out.mkdir(parents=True, exist_ok=True)
        entry: dict = {"lm": str(lm_path.relative_to(REPO)), "sets": {}}
        for set_name in SETS:
            index, rows, root = available[set_name]
            hypotheses = decode_int8(moth, pieces, lm, index, rows, root, PARAMS)
            write_jsonl(lm_out / f"{set_name}.hyp.jsonl", hypotheses)
            scored = score_rows(rows, hypotheses, bootstrap=1000, seed=0)
            (lm_out / f"{set_name}.metrics.json").write_text(
                json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            baseline = json.loads(
                (BASELINES[lm_name] / f"{set_name}.metrics.json").read_text(encoding="utf-8")
            )
            entry["sets"][set_name] = {
                "wer": scored["overall"]["wer"],
                "eta": scored["overall"]["eta"],
                "vs_baseline": _compare(scored, baseline, bootstrap=1000, seed=0),
            }
            eta_value = scored["overall"]["eta"]
            eta_text = "n/a" if eta_value is None else f"{eta_value * 100:6.2f}"
            print(
                f"{lm_name:14s} {set_name:15s} wer {scored['overall']['wer'] * 100:6.2f} "
                f"eta {eta_text}",
                flush=True,
            )

        t600_hyp = _read_hyp(lm_out / "t600.hyp.jsonl")
        base_hyp = _read_hyp(BASELINES[lm_name] / "t600.hyp.jsonl")
        rare_scored = score_rows(rare_rows, t600_hyp, bootstrap=1000, seed=0)
        rare_base = score_rows(rare_rows, base_hyp, bootstrap=1000, seed=0)
        entry["rare"] = {
            "en_hit": sum(v["en_hit"] for v in rare_scored["per_utterance"].values()),
            "en_ref": sum(v["en_ref"] for v in rare_scored["per_utterance"].values()),
            "aligned_eta": rare_scored["overall"]["eta"],
            "baseline_en_hit": sum(v["en_hit"] for v in rare_base["per_utterance"].values()),
            "vs_baseline": _compare(rare_scored, rare_base, bootstrap=1000, seed=0),
        }
        print(f"{lm_name:14s} rare {entry['rare']['en_hit']}/{entry['rare']['en_ref']}", flush=True)

        terms_out = lm_out / "terms.json"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "runs/E-015/eval_terms.py"),
                "--hyp", str(lm_out / "t600.hyp.jsonl"),
                "--baseline", str(BASELINES[lm_name] / "t600.hyp.jsonl"),
                "--out", str(terms_out),
            ],
            check=True,
            cwd=REPO,
        )
        entry["terms"] = json.loads(terms_out.read_text(encoding="utf-8"))
        result["lms"][lm_name] = entry

    (OUT / "eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
