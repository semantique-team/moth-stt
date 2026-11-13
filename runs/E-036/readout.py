"""E-036 — one product readout per conservative arm, against the committed 0.8.1 baseline.

The candidate checkpoint is exported to int8 first (`stt.export`, version e036-<arm>); this script
decodes T-600, T-S3, FLEURS-IT/EN, the rare 47 and noise with the same A3 LM and decoder as the
0.8.1 baseline, reuses the committed E-035 baseline hypotheses, and applies the gates frozen in
`PLAN.md`.

    uv run python -m stt.export --out runs/E-036/release-frozen \
        --checkpoint runs/E-036/frozen/checkpoints/ckpt_best.pt \
        --vocab release/vocab.json --head-blocks 2 --version e036-frozen
    uv run python runs/E-036/readout.py --arm frozen --name frozen-1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from stt.data.sources import sha256_file
from stt.decode import beam_search, load_pieces
from stt.eval import _compare, score_rows

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASELINE = REPO / "runs/E-035/readout/readout-1/baseline"
BASELINE_TOTAL = 19_706_289
BASELINE_MODEL = 11_910_677
LM = REPO / "runs/E-031/build/frozen_1_2_terms.bin"
RARE_IDS = REPO / "runs/E-026/rare47-16k.jsonl"
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}
SETS = ("t600", "fleurs_it_test", "fleurs_en_test", "noise")
PRIMARY = {
    "t600_wer": ("t600", "wer"),
    "t600_eta": ("t600", "eta"),
    "ts3_wer": ("t-s3", "wer"),
    "ts3_eta": ("t-s3", "eta"),
    "fleurs_it_wer": ("fleurs_it_test", "wer"),
    "fleurs_en_wer": ("fleurs_en_test", "wer"),
    "rare_eta": ("rare", "eta"),
}


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
        "baseline": {"wer": baseline["overall"]["wer"],
                     "eta": baseline["overall"]["eta"]},
        "candidate": {"wer": candidate["overall"]["wer"],
                      "eta": candidate["overall"]["eta"]},
        "vs_baseline": _compare(candidate, baseline, bootstrap=1000, seed=0),
    }


def noise_gates(candidate: dict, baseline: dict) -> dict[str, bool]:
    return {
        f"noise_{group}": candidate[f"{group}_clips"] > 0
        and candidate[f"{group}_clips"] == baseline[f"{group}_clips"]
        and candidate[f"{group}_words"] <= baseline[f"{group}_words"]
        and candidate[f"{group}_non_empty_clips"] <= baseline[f"{group}_non_empty_clips"]
        for group in ("non_music", "music")
    }


def run_parity(out_dir: Path, model_path: Path) -> dict:
    import numpy as np

    from stt.data.build_mel import load_audio_16k
    from stt.features import load_mel_filters, log_mel
    from stt.lm import BinaryBPELM
    from stt.onnx_runner import OnnxMoth

    audio = REPO / "web/tests/fixtures/speech.wav"
    release = model_path.parent
    moth = OnnxMoth(model_path, release / "vocab.json")
    filters = release / "mel_filters.bin"
    if not filters.is_file():
        filters = REPO / "release/mel_filters.bin"
    mel = log_mel(load_audio_16k(audio), filters=load_mel_filters(filters))
    log_probs = np.asarray(moth.log_probs(mel), dtype=np.float32)
    pieces = load_pieces(release / "vocab.json")
    with tempfile.TemporaryDirectory(prefix="e036-parity-") as directory:
        work = Path(directory)
        (work / "logits.bin").write_bytes(log_probs.tobytes())
        (work / "meta.json").write_text(
            json.dumps({"frames": int(log_probs.shape[0]), "vocab": int(log_probs.shape[1]),
                        "params": PARAMS, "audio_sha256": sha256_file(audio)}, indent=2) + "\n",
            encoding="utf-8",
        )
        python_text = beam_search(
            log_probs, log_probs.shape[0], pieces,
            beam_width=PARAMS["beam"], top_k=PARAMS["top_k"],
            lm=BinaryBPELM.load(LM), lm_weight=PARAMS["lm_weight"],
            token_bonus=PARAMS["token_bonus"],
        )
        started = time.perf_counter()
        completed = subprocess.run(
            ["node", str(REPO / "runs/E-031/js_beam.mjs"), str(work / "logits.bin"),
             str(LM), str(release / "vocab.json")],
            check=True, capture_output=True, text=True,
        )
        node_seconds = time.perf_counter() - started
        result = {
            "python_text": python_text,
            "js_text": completed.stdout.strip(),
            "exact_match": python_text == completed.stdout.strip(),
            "scope": "same-logits decoder only; no browser acoustic claim",
            "node_seconds": round(node_seconds, 3),
            "assets": {"model_sha256": sha256_file(model_path),
                       "lm_sha256": sha256_file(LM),
                       "vocab_sha256": sha256_file(release / "vocab.json")},
        }
    (out_dir / "parity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    from build_lm import RARE_TERMS

    from stt.lm import BinaryBPELM
    from stt.onnx_runner import OnnxMoth

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("frozen", "last2"))
    parser.add_argument("--name", required=True, help="new output directory")
    parser.add_argument("--release", type=Path, default=None)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be one directory name")
    release = args.release or OUT / f"release-{args.arm}"
    model_path = release / "model.int8.onnx"
    if not model_path.is_file():
        raise SystemExit(f"{model_path} is missing; export the checkpoint first")
    out_dir = OUT / "eval" / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; use a new --name")
    out_dir.mkdir(parents=True)

    package = BASELINE_TOTAL - BASELINE_MODEL + model_path.stat().st_size
    if package > 20_000_000:
        raise SystemExit(f"package {package} exceeds 20,000,000; reject before decoding")

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
        "arm": args.arm, "name": args.name, "params": PARAMS,
        "release": str(release.relative_to(REPO)),
        "package_bytes": package,
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
        baseline = read_metrics(BASELINE / f"{set_name}.metrics.json")
        result["sets"][set_name] = compare(baseline, scored)
        print(f"{set_name:15s} {time.time() - started:6.1f}s", flush=True)

    _, t_rows, _ = available["t600"]
    baseline_t600 = load_hyps(BASELINE / "t600.hyp.jsonl")
    s3_rows = [row for row in t_rows if row.get("speaker") == "S3"]
    s3_base = score_rows(
        s3_rows, {row["id"]: baseline_t600[row["id"]] for row in s3_rows},
        bootstrap=1000, seed=0,
    )
    s3_cand = score_rows(
        s3_rows, {row["id"]: candidate["t600"][row["id"]] for row in s3_rows},
        bootstrap=1000, seed=0,
    )
    result["sets"]["t-s3"] = compare(s3_base, s3_cand)

    rare_rows = [row for row in t_rows if row["id"] in rare_ids]
    if len(rare_rows) != 47:
        raise SystemExit(f"expected 47 rare rows, found {len(rare_rows)}")
    rare_base = score_rows(
        rare_rows, {row["id"]: baseline_t600[row["id"]] for row in rare_rows},
        bootstrap=1000, seed=0,
    )
    rare_cand = score_rows(
        rare_rows, {row["id"]: candidate["t600"][row["id"]] for row in rare_rows},
        bootstrap=1000, seed=0,
    )
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
    target_rows = [
        {**row, "en_spans": [[i, i + 1] for start, end in row.get("en_spans", [])
                              for i in range(start, end)
                              if row["norm"].split()[i] in set(RARE_TERMS)]}
        for row in rare_rows
    ]
    target_rows = [row for row in target_rows if row["en_spans"]]
    for arm_name, hyps in (("baseline", baseline_t600), ("candidate", candidate["t600"])):
        scored = score_rows(target_rows, hyps, bootstrap=0)
        result["rare"].setdefault("target_only", {})[arm_name] = {
            "en_hit": sum(v["en_hit"] for v in scored["per_utterance"].values()),
            "en_ref": sum(v["en_ref"] for v in scored["per_utterance"].values()),
            "aligned_eta": scored["overall"]["eta"],
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

    result["noise"] = {
        "baseline": read_metrics(BASELINE / "noise.metrics.json")["noise"],
        "candidate": candidate_metrics["noise"]["noise"],
    }
    result["noise"]["gates"] = noise_gates(
        result["noise"]["candidate"], result["noise"]["baseline"]
    )
    result["parity"] = run_parity(out_dir, model_path)

    primary_worse = {}
    primary_better = {}
    for name, (set_name, metric) in PRIMARY.items():
        comparison = (
            result["rare"]["vs_baseline"] if set_name == "rare"
            else result["sets"][set_name]["vs_baseline"]
        )
        if metric == "wer":
            low, high = comparison["wer_diff_ci"]
            primary_worse[name] = low > 0
            primary_better[name] = high < 0
        else:
            low, high = comparison["eta_diff_ci"]
            primary_worse[name] = high < 0
            primary_better[name] = low > 0

    gates = {
        "bytes": package <= 20_000_000,
        "t600_wer": result["sets"]["t600"]["candidate"]["wer"] <= 0.2521,
        "t600_eta": (result["sets"]["t600"]["candidate"]["eta"] or 0.0) >= 0.5367,
        "ts3_wer": result["sets"]["t-s3"]["candidate"]["wer"] <= 0.2571,
        "ts3_eta": (result["sets"]["t-s3"]["candidate"]["eta"] or 0.0) >= 0.4964,
        "fleurs_it": result["sets"]["fleurs_it_test"]["candidate"]["wer"] <= 0.2075,
        "fleurs_en": result["sets"]["fleurs_en_test"]["candidate"]["wer"] <= 0.5031,
        "natural": result["terms"]["groups"]["natural"]["new"]["correct"] >= 431,
        "rare_target": result["rare"]["candidate"]["en_hit"] >= 42,
        "decoder_parity": result["parity"]["exact_match"],
        **result["noise"]["gates"],
    }
    absolute_ok = all(value is True for value in gates.values())
    no_regression = not any(primary_worse.values())
    candidate_ok = absolute_ok and no_regression
    if candidate_ok:
        verdict = "candidate for the independent new-voice readout"
    elif not absolute_ok:
        verdict = "reject: one or more absolute gates failed"
    else:
        verdict = "reject: a primary metric is significantly worse than 0.8.1"

    result["gates"] = gates
    result["primary"] = {"significantly_worse": primary_worse,
                         "significantly_better": primary_better}
    result["absolute_ok"] = absolute_ok
    result["no_significant_regression"] = no_regression
    result["candidate"] = candidate_ok
    result["verdict"] = verdict
    (out_dir / "readout.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gates": gates, "primary": result["primary"],
                      "candidate": candidate_ok, "verdict": verdict}, indent=2))
    return 0 if candidate_ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "runs/E-031"))
    raise SystemExit(main())
