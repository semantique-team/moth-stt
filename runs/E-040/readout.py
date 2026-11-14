"""E-040 — one product readout per multi-seed replicate, against the committed 0.8.1 baseline.

The replicate's best checkpoint is exported to int8 by the driver, then this script decodes
T-600, T-S3, FLEURS-IT/EN, the rare 47 and noise with the shipped A3 LM and decoder (beam 12 /
top-k 10 / 0.55 / 2.7), and reports per-set metrics against the E-035 baseline hypotheses.

E-040 measures dispersion, not adoption: no gate decides anything here.

    uv run python runs/E-040/readout.py --run runs/E-040/seed1/frozen --name s1-frozen
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from stt.decode import load_pieces
from stt.eval import _compare, score_rows

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASELINE = REPO / "runs/E-035/readout/readout-1/baseline"
LM = REPO / "runs/E-031/build/frozen_1_2_terms.bin"
RARE_IDS = REPO / "runs/E-026/rare47-16k.jsonl"
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}
SETS = ("t600", "fleurs_it_test", "fleurs_en_test", "noise")


def _e022():
    spec = importlib.util.spec_from_file_location("e022eval", REPO / "runs/E-022/eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_int8, module.sets


def load_hyps(path: Path) -> dict[str, str]:
    return {
        item["id"]: item["hyp"]
        for item in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def write_jsonl(path: Path, hypotheses: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
                for key, value in hypotheses.items()),
        encoding="utf-8",
    )


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(baseline: dict, candidate: dict) -> dict:
    return {
        "n": candidate["overall"]["n"],
        "baseline": {"wer": baseline["overall"]["wer"], "eta": baseline["overall"]["eta"]},
        "candidate": {"wer": candidate["overall"]["wer"], "eta": candidate["overall"]["eta"]},
        "vs_baseline": _compare(candidate, baseline, bootstrap=1000, seed=0),
    }


def silence_probe(rows: list[dict], hyps: dict[str, str]) -> dict:
    silent = [row for row in rows if not (row.get("norm") or "").strip()]
    silence = [row for row in silent if row.get("source") == "noise_silence"]
    return {
        "silence_clips": len(silence),
        "silence_non_empty": sum(1 for row in silence if hyps[row["id"]].strip()),
        "silence_x": sum(1 for row in silence if hyps[row["id"]].strip().lower() == "x"),
        "empty_reference_clips": len(silent),
    }


def main() -> int:
    from stt.lm import BinaryBPELM
    from stt.onnx_runner import OnnxMoth

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory with config.yaml")
    parser.add_argument("--name", required=True, help="new output directory name")
    parser.add_argument("--release", type=Path, default=None)
    args = parser.parse_args()
    run_dir = (REPO / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be one directory name")
    release = args.release or run_dir / "release-int8"
    model_path = release / "model.int8.onnx"
    if not model_path.is_file():
        raise SystemExit(f"{model_path} is missing; export the checkpoint first")
    out_dir = OUT / "eval" / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; use a new --name")
    out_dir.mkdir(parents=True)

    decode_int8, sets = _e022()
    available = sets()
    pieces = load_pieces(release / "vocab.json")
    moth = OnnxMoth(model_path, release / "vocab.json")
    lm = BinaryBPELM.load(LM)
    rare_ids = {
        json.loads(line)["id"]
        for line in RARE_IDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    result: dict = {
        "run": str(run_dir.relative_to(REPO)), "name": args.name, "params": PARAMS,
        "release": str(release.relative_to(REPO)),
        "model_bytes": model_path.stat().st_size,
        "baseline": str(BASELINE.relative_to(REPO)),
        "sets": {},
    }
    candidate: dict[str, dict[str, str]] = {}
    candidate_metrics: dict[str, dict] = {}
    for set_name in SETS:
        index, rows, root = available[set_name]
        started = time.time()
        hyps = decode_int8(moth, pieces, lm, index, rows, root, PARAMS)
        write_jsonl(out_dir / "candidate" / f"{set_name}.hyp.jsonl", hyps)
        scored = score_rows(rows, hyps, bootstrap=1000, seed=0)
        (out_dir / "candidate" / f"{set_name}.metrics.json").write_text(
            json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        candidate[set_name] = hyps
        candidate_metrics[set_name] = scored
        result["sets"][set_name] = compare(read_metrics(BASELINE / f"{set_name}.metrics.json"),
                                           scored)
        print(f"{set_name:15s} {time.time() - started:6.1f}s", flush=True)

    _, t_rows, _ = available["t600"]
    baseline_t600 = load_hyps(BASELINE / "t600.hyp.jsonl")
    s3_rows = [row for row in t_rows if row.get("speaker") == "S3"]
    result["sets"]["t-s3"] = compare(
        score_rows(s3_rows, {row["id"]: baseline_t600[row["id"]] for row in s3_rows},
                   bootstrap=1000, seed=0),
        score_rows(s3_rows, {row["id"]: candidate["t600"][row["id"]] for row in s3_rows},
                   bootstrap=1000, seed=0),
    )

    rare_rows = [row for row in t_rows if row["id"] in rare_ids]
    if len(rare_rows) != 47:
        raise SystemExit(f"expected 47 rare rows, found {len(rare_rows)}")
    rare_base = score_rows(rare_rows, {row["id"]: baseline_t600[row["id"]] for row in rare_rows},
                           bootstrap=1000, seed=0)
    rare_cand = score_rows(rare_rows, {row["id"]: candidate["t600"][row["id"]] for row in rare_rows},
                           bootstrap=1000, seed=0)
    result["rare"] = {
        "n": rare_cand["overall"]["n"],
        "baseline": {
            "en_hit": sum(v["en_hit"] for v in rare_base["per_utterance"].values()),
            "en_ref": sum(v["en_ref"] for v in rare_base["per_utterance"].values()),
            "aligned_eta": rare_base["overall"]["eta"],
        },
        "candidate": {
            "en_hit": sum(v["en_hit"] for v in rare_cand["per_utterance"].values()),
            "en_ref": sum(v["en_ref"] for v in rare_cand["per_utterance"].values()),
            "aligned_eta": rare_cand["overall"]["eta"],
        },
        "vs_baseline": _compare(rare_cand, rare_base, bootstrap=1000, seed=0),
    }

    terms_path = out_dir / "terms.json"
    subprocess.run(
        [sys.executable, str(REPO / "runs/E-015/eval_terms.py"),
         "--hyp", str(out_dir / "candidate/t600.hyp.jsonl"),
         "--baseline", str(BASELINE / "t600.hyp.jsonl"),
         "--out", str(terms_path)],
        check=True, cwd=REPO,
    )
    result["terms"] = json.loads(terms_path.read_text(encoding="utf-8"))

    _, noise_rows, _ = available["noise"]
    result["noise"] = {
        "baseline": read_metrics(BASELINE / "noise.metrics.json")["noise"],
        "candidate": candidate_metrics["noise"]["noise"],
        "silence": {
            "baseline": silence_probe(noise_rows, load_hyps(BASELINE / "noise.hyp.jsonl")),
            "candidate": silence_probe(noise_rows, candidate["noise"]),
        },
    }
    result["note"] = "E-040 dispersion evidence; no gate and no adoption"
    (out_dir / "readout.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"t600": result["sets"]["t600"]["candidate"],
                      "rare": result["rare"]["candidate"],
                      "silence": result["noise"]["silence"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
