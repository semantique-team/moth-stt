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
import tempfile
from importlib.metadata import version
from pathlib import Path

from audit import (
    E031_REVISION,
    INPUT_REVISION,
    digest,
    git_bytes,
    json_digest,
    read_pinned_hypotheses,
    release_record,
    require_new,
    verified_parity,
    verify_file,
    verify_source,
)

from stt.data.sources import sha256_file
from stt.decode import load_pieces
from stt.eval import _compare, read_hypotheses, score_rows
from stt.lm import BinaryBPELM

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BINARY_BASELINE = 5_116_464
TOTAL_BASELINE = 19_440_761
PACKAGE_LIMIT = 20_000_000
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}
SETS = ("t600", "t-s3", "fleurs_it_test", "fleurs_en_test")
TERM_INPUTS = (
    "runs/E-015/eval_terms.py", "recordings/manifests/T.jsonl",
    "data/manifests/cv_it_train.jsonl", "data/manifests/mls_it_train.jsonl",
    "data/manifests/spliced_it_en.jsonl", "recordings/manifests/R-S1.jsonl",
    "recordings/manifests/R-S2.jsonl",
)


def _e022():
    spec = importlib.util.spec_from_file_location("e022eval", REPO / "runs/E-022/eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_int8, module.sets, module.write_jsonl


def _rare_ids() -> set[str]:
    return {
        json.loads(line)["id"]
        for line in git_bytes("runs/E-026/rare47-16k.jsonl").decode("utf-8").splitlines()
        if line.strip()
    }


def cache_fingerprint(lm_path, manifest, rows, index, root):
    return {
        "model": sha256_file(REPO / "release/model.int8.onnx"),
        "lm": sha256_file(lm_path),
        "vocab": sha256_file(REPO / "release/vocab.json"),
        "manifest": sha256_file(manifest), "reference": json_digest(rows),
        "index": json_digest(index),
        "shards": {name: sha256_file(root / name)
                   for name in sorted({index[row["id"]]["shard"] for row in rows})},
        "params": PARAMS,
        "code": {name: digest((REPO / name).read_bytes()) for name in (
            "runs/E-031/gate.py", "runs/E-031/audit.py", "runs/E-022/eval.py",
            "stt/decode.py", "stt/lm.py",
            "stt/onnx_runner.py", "stt/eval.py", "stt/text.py")},
        "scoring": {"bootstrap": 1000, "seed": 0},
        "versions": {name: version(name) for name in ("numpy", "num2words", "onnxruntime")},
    }


def verify_term_inputs():
    hashes = {}
    for name in TERM_INPUTS:
        expected = git_bytes(name, INPUT_REVISION)
        actual = (REPO / name).read_bytes().replace(b"\r\n", b"\n")
        if actual != expected:
            raise ValueError(
                f"term input drift: {name}; use a disposable checkout with pinned inputs "
                f"from {INPUT_REVISION}; existing results are preserved"
            )
        hashes[name] = digest(expected)
    return {"revision": INPUT_REVISION, "inputs": hashes}


def score_terms(out_dir):
    provenance = verify_term_inputs()
    pinned_base = out_dir / "baseline-t600.hyp.jsonl"
    baseline = git_bytes("runs/E-030/eval/e028-int8/t600.hyp.jsonl")
    if pinned_base.exists():
        if pinned_base.read_bytes() != baseline:
            raise ValueError("stale pinned baseline; use a new --name; existing file preserved")
    else:
        with pinned_base.open("xb") as handle:
            handle.write(baseline)
    terms_out = out_dir / "terms.json"
    with tempfile.TemporaryDirectory(prefix="e031-terms-") as directory:
        temporary_out = Path(directory) / "terms.json"
        subprocess.run(
            [sys.executable, str(REPO / "runs/E-015/eval_terms.py"),
             "--hyp", str(out_dir / "t600.hyp.jsonl"), "--baseline", str(pinned_base),
             "--out", str(temporary_out)],
            check=True, cwd=REPO,
        )
        result = json.loads(temporary_out.read_bytes())
        if terms_out.exists():
            if json.loads(terms_out.read_bytes()) != result:
                raise ValueError("stale terms result; use a new --name; existing file preserved")
        else:
            with terms_out.open("xb") as handle:
                handle.write(temporary_out.read_bytes())
    return result, provenance


def cached_score(hyp_path, metrics_path, rows, fingerprint, decode):
    if hyp_path.exists() or metrics_path.exists():
        if not (hyp_path.exists() and metrics_path.exists()):
            raise ValueError("partial cache; use a new --name, existing artifacts are preserved")
        hypotheses = read_hypotheses(hyp_path)
        if set(hypotheses) != {str(row["id"]) for row in rows}:
            raise ValueError("cache ids differ from reference; use a new --name")
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        provenance = {"fingerprint": fingerprint, "hyp_sha256": digest(hyp_path.read_bytes())}
        if stored.get("provenance") != provenance:
            raise ValueError("unverified/legacy or mismatched cache; use a new --name")
        scored = score_rows(rows, hypotheses, bootstrap=1000, seed=0)
        if stored != {**scored, "provenance": provenance}:
            raise ValueError("cache metrics disagree with hypotheses; use a new --name")
        return hypotheses, scored
    hypotheses = decode()
    if set(hypotheses) != {str(row["id"]) for row in rows}:
        raise ValueError("decode/cache must cover exactly the reference ids")
    scored = score_rows(rows, hypotheses, bootstrap=1000, seed=0)
    with hyp_path.open("x", encoding="utf-8") as handle:
        handle.write("".join(json.dumps({"id": uid, "hyp": hyp}, ensure_ascii=False) + "\n"
                             for uid, hyp in hypotheses.items()))
    provenance = {"fingerprint": fingerprint, "hyp_sha256": digest(hyp_path.read_bytes())}
    with metrics_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({**scored, "provenance": provenance}, indent=2) + "\n")
    return hypotheses, scored


def target_rare_rows(rows):
    spec = importlib.util.spec_from_file_location("e031build", OUT / "build_lm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    targets = set(module.RARE_TERMS)
    return [{**row, "en_spans": [[i, i + 1] for start, end in row.get("en_spans", [])
                                for i in range(start, end)
                                if row["norm"].split()[i] in targets]} for row in rows]


def noise_gates(candidate, baseline):
    if candidate is None or baseline is None:
        raise ValueError("noise check requires non-speech references")
    return {f"noise_{group}": candidate[f"{group}_clips"] > 0
            and candidate[f"{group}_clips"] == baseline[f"{group}_clips"]
            and candidate[f"{group}_words"] <= baseline[f"{group}_words"]
            and candidate[f"{group}_non_empty_clips"] <= baseline[f"{group}_non_empty_clips"]
            for group in ("non_music", "music")}


def gate_exit(gates):
    return 0 if all(value is True for value in gates.values()) else 1


def main() -> int:
    from stt.onnx_runner import OnnxMoth

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm", type=Path, required=True)
    parser.add_argument("--name", required=True)
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--noise", action="store_true", help="run required E-011 noise check (default)")
    noise.add_argument("--skip-noise", action="store_true", help="diagnostic only; gate incomplete")
    parser.add_argument("--baseline-lm", type=Path, default=REPO / "runs/E-028/release/lm_bpe3.bin")
    parser.add_argument("--parity", type=Path, help="attributed exact decoder parity report")
    args = parser.parse_args()
    args.lm = args.lm.resolve()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be one directory name")
    out_dir = OUT / "eval" / args.name
    gate_path = out_dir / "gate.json"
    require_new(gate_path)
    record = verify_source(REPO / "release")
    if not args.skip_noise:
        verify_file(args.baseline_lm, record, "lm_bpe3.bin")
    package_bytes = TOTAL_BASELINE - BINARY_BASELINE + args.lm.stat().st_size
    if package_bytes > PACKAGE_LIMIT:
        parser.error(f"package {package_bytes} exceeds {PACKAGE_LIMIT}; rejected before decoding")
    verify_term_inputs()

    decode_int8, sets, _ = _e022()
    available = sets()
    lm = BinaryBPELM.load(args.lm)
    moth = OnnxMoth(REPO / "release/model.int8.onnx", REPO / "release/vocab.json")
    pieces = load_pieces(REPO / "release/vocab.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "name": args.name,
        "lm": str(args.lm),
        "lm_bytes": args.lm.stat().st_size,
        "params": PARAMS,
        "sets": {},
        "baseline_provenance": {"revision": E031_REVISION,
                                "release": release_record("0.8.0")},
    }
    for set_name in SETS:
        index, rows, root = available[set_name]
        hyp_path = out_dir / f"{set_name}.hyp.jsonl"
        metrics_path = out_dir / f"{set_name}.metrics.json"
        manifest = REPO / ("recordings/manifests/T.jsonl" if set_name.startswith("t")
                           else f"data/manifests/{set_name}.jsonl")
        _, scored = cached_score(
            hyp_path, metrics_path, rows, cache_fingerprint(args.lm, manifest, rows, index, root),
            lambda index=index, rows=rows, root=root: decode_int8(
                moth, pieces, lm, index, rows, root, PARAMS),
        )
        baseline_hyp = read_pinned_hypotheses(f"runs/E-030/eval/e028-int8/{set_name}.hyp.jsonl")
        baseline = score_rows(rows, baseline_hyp, bootstrap=1000, seed=0)
        historical_manifest = git_bytes(str(manifest.relative_to(REPO)).replace("\\", "/"))
        if manifest.read_bytes().replace(b"\r\n", b"\n") != historical_manifest:
            raise ValueError(f"{set_name}: manifest differs from pinned 0.8.0 comparison inputs")
        retained = json.loads(git_bytes(f"runs/E-030/eval/e028-int8/{set_name}.metrics.json"))
        if baseline["per_utterance"] != retained["per_utterance"]:
            raise ValueError(f"{set_name}: reference/scorer no longer matches pinned 0.8.0 evidence")
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
    candidate_all = read_hypotheses(out_dir / "t600.hyp.jsonl")
    baseline_all = read_pinned_hypotheses("runs/E-030/eval/e028-int8/t600.hyp.jsonl")
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
    target_rows = target_rare_rows(rare_rows)
    target_new = score_rows(target_rows, rare_hyp, bootstrap=1000)
    target_base = score_rows(target_rows, rare_base_hyp, bootstrap=1000)
    result["rare_target_only"] = {
        "gate": False, "definition": "aligned annotated occurrences of the 44 frozen target terms",
        "en_ref": sum(p["en_ref"] for p in target_new["per_utterance"].values()),
        "en_hit": sum(p["en_hit"] for p in target_new["per_utterance"].values()),
        "aligned_eta": target_new["overall"]["eta"],
        "baseline_aligned_eta": target_base["overall"]["eta"],
        "vs_0.8.0": _compare(target_new, target_base, bootstrap=1000, seed=0),
    }

    result["terms"], result["terms_provenance"] = score_terms(out_dir)

    package_bytes = TOTAL_BASELINE - BINARY_BASELINE + args.lm.stat().st_size
    result["package_bytes"] = package_bytes
    if not args.skip_noise:
        index, rows, root = available["noise"]
        candidate_hyp = decode_int8(moth, pieces, lm, index, rows, root, PARAMS)
        baseline_lm = BinaryBPELM.load(args.baseline_lm)
        baseline_hyp = decode_int8(moth, pieces, baseline_lm, index, rows, root, PARAMS)
        result["noise"] = {
            "candidate": score_rows(rows, candidate_hyp, bootstrap=0)["noise"],
            "baseline": score_rows(rows, baseline_hyp, bootstrap=0)["noise"],
            "baseline_lm_sha256": digest(args.baseline_lm.read_bytes()),
            "fingerprint": cache_fingerprint(
                args.lm, REPO / "data/manifests/noise.jsonl", rows, index, root),
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
    result["thresholds"] = {
        "rare_eta": 0.284, "source": "E-031 PLAN, not the E-030 0.30 gate",
        "discrepancy": "0.284 is the rounded historical 23/81 baseline, not a 30% guard",
    }
    result["noise_policy"] = "required; --skip-noise is incomplete diagnostic output only"
    result["gates"].update(
        {"noise_non_music": None, "noise_music": None} if args.skip_noise
        else noise_gates(result["noise"]["candidate"], result["noise"]["baseline"])
    )
    result["gates"]["decoder_parity"] = None
    if args.parity:
        parity = json.loads(args.parity.read_text(encoding="utf-8"))
        expected = {"model": digest((REPO / "release/model.int8.onnx").read_bytes()),
                    "lm": digest(args.lm.read_bytes()),
                    "vocab": digest((REPO / "release/vocab.json").read_bytes())}
        result["gates"]["decoder_parity"] = verified_parity(
            parity, expected, PARAMS,
            {name: digest((REPO / name).read_bytes()) for name in (
                "stt/decode.py", "stt/lm.py", "web/decode.js", "web/lm.js",
                "runs/E-031/decoder_parity.py", "runs/E-031/js_beam.mjs")},
            digest((REPO / "web/tests/fixtures/speech.wav").read_bytes()),
        )
    result["browser_acoustic_parity"] = "not certified by this gate"
    result["gates"]["significant_gain"] = any(
        comparison[key] is not None and (comparison[key][0] > 0 if positive
                                        else comparison[key][1] < 0)
        for comparison, key, positive in (
            (result["rare"]["vs_0.8.0"], "eta_diff_ci", True),
            (result["sets"]["t600"]["vs_0.8.0"], "wer_diff_ci", False),
            (result["sets"]["fleurs_it_test"]["vs_0.8.0"], "wer_diff_ci", False))
    )
    result["passed"] = gate_exit(result["gates"]) == 0
    with gate_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["gates"], indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
