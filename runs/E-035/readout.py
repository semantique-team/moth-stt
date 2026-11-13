"""E-035 — the single product readout, frozen in `READOUT-PLAN.md` before this run.

Decodes T-600, FLEURS-IT, FLEURS-EN and the noise set with the shipped int8 graph, once with the
0.8.1 LM and once with the budgeted candidate, plus the rare 47 rows, the term groups and the
decoder-level parity check. Dev is not read.

    uv run python runs/E-035/readout.py --name readout-1
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
sys.path.insert(0, str(REPO / "runs/E-031"))

from build_lm import RARE_TERMS

RELEASE = REPO / "runs/E-031/release"
RECORD = REPO / "runs/E-031/release-0.8.1.json"
BASELINE_LM = REPO / "runs/E-031/build/frozen_1_2_terms.bin"
CANDIDATE_LM = OUT / "build/B1_budgeted.bin"
BUILD = OUT / "build.json"
RARE_IDS = REPO / "runs/E-026/rare47-16k.jsonl"
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}
PACKAGE_LIMIT = 20_000_000
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
GAIN = ("t600_wer", "fleurs_it_wer", "rare_eta")


def _e022():
    spec = importlib.util.spec_from_file_location("e022eval", REPO / "runs/E-022/eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_int8, module.sets


def verify_assets() -> dict:
    record = json.loads(RECORD.read_text(encoding="utf-8"))["files"]
    for name in ("model.int8.onnx", "vocab.json", "mel_filters.bin", "silero_vad.onnx",
                 "config.json", "NOTICE"):
        if sha256_file(RELEASE / name) != record[name]["sha256"]:
            raise SystemExit(f"{name} does not match the 0.8.1 release record")
    if sha256_file(BASELINE_LM) != record["lm_bpe3.bin"]["sha256"]:
        raise SystemExit("baseline LM does not match the 0.8.1 release record")
    build = json.loads(BUILD.read_text(encoding="utf-8"))["arms"]["B1_budgeted"]
    if sha256_file(CANDIDATE_LM) != build["binary_sha256"]:
        raise SystemExit("candidate LM does not match its build record")
    base_package = sum(item["bytes"] for item in record.values())
    package = base_package - record["lm_bpe3.bin"]["bytes"] + CANDIDATE_LM.stat().st_size
    return {
        "record": str(RECORD.relative_to(REPO)),
        "release": str(RELEASE.relative_to(REPO)),
        "baseline_lm": {"path": str(BASELINE_LM.relative_to(REPO)),
                        "bytes": BASELINE_LM.stat().st_size,
                        "sha256": sha256_file(BASELINE_LM)},
        "candidate_lm": {"path": str(CANDIDATE_LM.relative_to(REPO)),
                         "bytes": CANDIDATE_LM.stat().st_size,
                         "sha256": sha256_file(CANDIDATE_LM)},
        "package": {"baseline_bytes": base_package, "candidate_bytes": package},
    }


def write_jsonl(path: Path, hypotheses: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
                for key, value in hypotheses.items()),
        encoding="utf-8",
    )


def write_metrics(path: Path, rows: list[dict], hypotheses: dict[str, str],
                  lm_sha: str) -> dict:
    scored = score_rows(rows, hypotheses, bootstrap=1000, seed=0)
    payload = {
        **scored,
        "provenance": {
            "lm_sha256": lm_sha,
            "model_sha256": sha256_file(RELEASE / "model.int8.onnx"),
            "vocab_sha256": sha256_file(RELEASE / "vocab.json"),
            "params": PARAMS,
            "scoring": {"bootstrap": 1000, "seed": 0},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return scored


def run_terms(out_dir: Path, candidate_hyp: Path, baseline_hyp: Path) -> dict:
    path = out_dir / "terms.json"
    subprocess.run(
        [sys.executable, str(REPO / "runs/E-015/eval_terms.py"),
         "--hyp", str(candidate_hyp), "--baseline", str(baseline_hyp), "--out", str(path)],
        check=True, cwd=REPO,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def run_parity(out_dir: Path) -> dict:
    """Same-logits JS/Python parity for the candidate LM (no browser acoustic claim)."""
    import numpy as np

    from stt.data.build_mel import load_audio_16k
    from stt.features import load_mel_filters, log_mel
    from stt.lm import BinaryBPELM
    from stt.onnx_runner import OnnxMoth

    audio = REPO / "web/tests/fixtures/speech.wav"
    moth = OnnxMoth(RELEASE / "model.int8.onnx", RELEASE / "vocab.json")
    mel = log_mel(load_audio_16k(audio), filters=load_mel_filters(RELEASE / "mel_filters.bin"))
    log_probs = np.asarray(moth.log_probs(mel), dtype=np.float32)
    pieces = load_pieces(RELEASE / "vocab.json")
    with tempfile.TemporaryDirectory(prefix="e035-parity-") as directory:
        work = Path(directory)
        (work / "logits.bin").write_bytes(log_probs.tobytes())
        logits_sha = sha256_file(work / "logits.bin")
        (work / "meta.json").write_text(
            json.dumps({"frames": int(log_probs.shape[0]), "vocab": int(log_probs.shape[1]),
                        "params": PARAMS,
                        "audio_sha256": sha256_file(audio)}, indent=2) + "\n",
            encoding="utf-8",
        )
        python_text = beam_search(
            log_probs, log_probs.shape[0], pieces,
            beam_width=PARAMS["beam"], top_k=PARAMS["top_k"],
            lm=BinaryBPELM.load(CANDIDATE_LM), lm_weight=PARAMS["lm_weight"],
            token_bonus=PARAMS["token_bonus"],
        )
        started = time.perf_counter()
        completed = subprocess.run(
            ["node", str(REPO / "runs/E-031/js_beam.mjs"), str(work / "logits.bin"),
             str(CANDIDATE_LM), str(RELEASE / "vocab.json")],
            check=True, capture_output=True, text=True,
        )
        node_seconds = time.perf_counter() - started
        js_text = completed.stdout.strip()
        result = {
            "python_text": python_text,
            "js_text": js_text,
            "exact_match": python_text == js_text,
            "scope": "same-logits decoder only; no browser acoustic claim",
            "node_seconds": round(node_seconds, 3),
            "assets": {"model_sha256": sha256_file(RELEASE / "model.int8.onnx"),
                       "lm_sha256": sha256_file(CANDIDATE_LM),
                       "vocab_sha256": sha256_file(RELEASE / "vocab.json")},
            "logits_sha256": logits_sha,
            "code": {name: sha256_file(REPO / name) for name in (
                "stt/decode.py", "stt/lm.py", "web/decode.js", "web/lm.js",
                "runs/E-031/js_beam.mjs")},
            "audio_sha256": sha256_file(audio),
        }
    (out_dir / "parity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def noise_gates(candidate: dict, baseline: dict) -> dict[str, bool]:
    return {
        f"noise_{group}": candidate[f"{group}_clips"] > 0
        and candidate[f"{group}_clips"] == baseline[f"{group}_clips"]
        and candidate[f"{group}_words"] <= baseline[f"{group}_words"]
        and candidate[f"{group}_non_empty_clips"] <= baseline[f"{group}_non_empty_clips"]
        for group in ("non_music", "music")
    }


def target_only(rows: list[dict], hypotheses: dict[str, str], targets: set[str]) -> dict:
    filtered = [
        {**row, "en_spans": [[i, i + 1] for start, end in row.get("en_spans", [])
                              for i in range(start, end)
                              if row["norm"].split()[i] in targets]}
        for row in rows
    ]
    filtered = [row for row in filtered if row["en_spans"]]
    scored = score_rows(filtered, hypotheses, bootstrap=1000, seed=0)
    per = scored["per_utterance"].values()
    return {
        "en_ref": sum(item["en_ref"] for item in per),
        "en_hit": sum(item["en_hit"] for item in per),
        "aligned_eta": scored["overall"]["eta"],
    }


def compare(baseline: dict, candidate: dict) -> dict:
    return {
        "n": candidate["overall"]["n"],
        "baseline": {"wer": baseline["overall"]["wer"], "eta": baseline["overall"]["eta"]},
        "candidate": {"wer": candidate["overall"]["wer"], "eta": candidate["overall"]["eta"]},
        "vs_baseline": _compare(candidate, baseline, bootstrap=1000, seed=0),
    }


def main() -> int:
    from stt.lm import BinaryBPELM
    from stt.onnx_runner import OnnxMoth

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="new output directory")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be one directory name")
    out_dir = OUT / "readout" / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; use a new --name")
    out_dir.mkdir(parents=True)

    provenance = verify_assets()
    decode_int8, sets = _e022()
    available = sets()
    pieces = load_pieces(RELEASE / "vocab.json")
    moth = OnnxMoth(RELEASE / "model.int8.onnx", RELEASE / "vocab.json")
    lms = {"baseline": BinaryBPELM.load(BASELINE_LM),
           "candidate": BinaryBPELM.load(CANDIDATE_LM)}
    rare_ids = {
        json.loads(line)["id"]
        for line in RARE_IDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    result: dict = {"name": args.name, "params": PARAMS, "provenance": provenance,
                    "sets": {}, "rare": {}, "terms": None, "noise": None, "parity": None}
    hypotheses: dict[str, dict[str, dict[str, str]]] = {}
    metrics: dict[str, dict[str, dict]] = {}
    for set_name in SETS:
        index, rows, root = available[set_name]
        hypotheses[set_name], metrics[set_name] = {}, {}
        for arm, lm in lms.items():
            started = time.time()
            hyps = decode_int8(moth, pieces, lm, index, rows, root, PARAMS)
            path = out_dir / arm / f"{set_name}.hyp.jsonl"
            write_jsonl(path, hyps)
            metrics[set_name][arm] = write_metrics(
                out_dir / arm / f"{set_name}.metrics.json", rows, hyps,
                provenance[f"{arm}_lm"]["sha256"],
            )
            hypotheses[set_name][arm] = hyps
            print(f"{set_name:15s} {arm:9s} {time.time() - started:6.1f}s", flush=True)
        result["sets"][set_name] = compare(
            metrics[set_name]["baseline"], metrics[set_name]["candidate"])

    _, t_rows, _ = available["t600"]
    s3_rows = [row for row in t_rows if row.get("speaker") == "S3"]
    s3_metrics = {}
    for arm in ("baseline", "candidate"):
        s3_metrics[arm] = score_rows(
            s3_rows, hypotheses["t600"][arm], bootstrap=1000, seed=0)
    result["sets"]["t-s3"] = compare(s3_metrics["baseline"], s3_metrics["candidate"])

    rare_rows = [row for row in t_rows if row["id"] in rare_ids]
    if len(rare_rows) != 47:
        raise SystemExit(f"expected 47 rare rows, found {len(rare_rows)}")
    rare_metrics = {}
    target = {}
    for arm in ("baseline", "candidate"):
        rare_hyps = {row["id"]: hypotheses["t600"][arm][row["id"]] for row in rare_rows}
        rare_metrics[arm] = score_rows(rare_rows, rare_hyps, bootstrap=1000, seed=0)
        target[arm] = target_only(rare_rows, rare_hyps, set(RARE_TERMS))
    result["rare"] = {
        "n": rare_metrics["candidate"]["overall"]["n"],
        "baseline": {"aligned_eta": rare_metrics["baseline"]["overall"]["eta"],
                     "en_hit": sum(p["en_hit"] for p in rare_metrics["baseline"]
                                   ["per_utterance"].values()),
                     "en_ref": sum(p["en_ref"] for p in rare_metrics["baseline"]
                                   ["per_utterance"].values())},
        "candidate": {"aligned_eta": rare_metrics["candidate"]["overall"]["eta"],
                      "en_hit": sum(p["en_hit"] for p in rare_metrics["candidate"]
                                    ["per_utterance"].values()),
                      "en_ref": sum(p["en_ref"] for p in rare_metrics["candidate"]
                                    ["per_utterance"].values())},
        "vs_baseline": _compare(rare_metrics["candidate"], rare_metrics["baseline"],
                                bootstrap=1000, seed=0),
        "target_only": {"baseline": target["baseline"], "candidate": target["candidate"]},
    }

    result["terms"] = run_terms(
        out_dir,
        out_dir / "candidate/t600.hyp.jsonl",
        out_dir / "baseline/t600.hyp.jsonl",
    )
    result["noise"] = {
        "baseline": metrics["noise"]["baseline"]["noise"],
        "candidate": metrics["noise"]["candidate"]["noise"],
    }
    result["noise"]["gates"] = noise_gates(
        result["noise"]["candidate"], result["noise"]["baseline"])

    result["parity"] = run_parity(out_dir)

    primary_worse = {}
    primary_better = {}
    for name, (set_name, metric) in PRIMARY.items():
        if set_name == "rare":
            comparison = result["rare"]["vs_baseline"]
        else:
            comparison = result["sets"][set_name]["vs_baseline"]
        if metric == "wer":
            low, high = comparison["wer_diff_ci"]
            primary_worse[name] = low > 0
            primary_better[name] = high < 0
        else:
            low, high = comparison["eta_diff_ci"]
            primary_worse[name] = high < 0
            primary_better[name] = low > 0

    gates = {
        "bytes": result["provenance"]["package"]["candidate_bytes"] <= PACKAGE_LIMIT,
        "t600_wer": result["sets"]["t600"]["candidate"]["wer"] <= 0.2521,
        "t600_eta": (result["sets"]["t600"]["candidate"]["eta"] or 0.0) >= 0.5367,
        "ts3_wer": result["sets"]["t-s3"]["candidate"]["wer"] <= 0.2571,
        "ts3_eta": (result["sets"]["t-s3"]["candidate"]["eta"] or 0.0) >= 0.4964,
        "fleurs_it": result["sets"]["fleurs_it_test"]["candidate"]["wer"] <= 0.2075,
        "fleurs_en": result["sets"]["fleurs_en_test"]["candidate"]["wer"] <= 0.5031,
        "rare_eta": (result["rare"]["candidate"]["aligned_eta"] or 0.0) >= 0.284,
        "natural": result["terms"]["groups"]["natural"]["new"]["correct"] >= 431,
        "decoder_parity": result["parity"]["exact_match"],
        **result["noise"]["gates"],
    }
    absolute_ok = all(value is True for value in gates.values())
    no_regression = not any(primary_worse.values())
    significant_gain = any(primary_better[name] for name in GAIN)
    adopt = absolute_ok and no_regression and significant_gain
    if adopt:
        verdict = "adopt as a decode-only 0.8.2"
    elif not absolute_ok:
        verdict = "reject: one or more absolute gates failed"
    elif not no_regression:
        verdict = "reject: a primary metric is significantly worse than 0.8.1"
    else:
        verdict = "product tie: do not ship; keep 0.8.1"

    result["gates"] = gates
    result["primary"] = {"significantly_worse": primary_worse,
                         "significantly_better": primary_better}
    result["absolute_ok"] = absolute_ok
    result["no_significant_regression"] = no_regression
    result["significant_gain"] = significant_gain
    result["adopt"] = adopt
    result["verdict"] = verdict
    (out_dir / "readout.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "primary": result["primary"],
                      "adopt": adopt, "verdict": verdict}, indent=2))
    return 0 if adopt else 1


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "runs/E-031"))
    raise SystemExit(main())
