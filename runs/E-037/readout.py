"""E-037 — paired fp32/INT8 readout with the shipped A3 LM.

Decodes the same clips through the fp32 export and the shipped INT8 graph, with the same A3 LM and
decoder, and reports the INT8 cost per set. This is closure evidence: fp32 cannot ship, so no gate
changes the release.

    uv run python runs/E-037/export_fp32.py
    uv run python runs/E-037/readout.py --name paired-1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from stt.data.sources import sha256_file
from stt.decode import load_pieces
from stt.eval import _compare, macro_wer_bootstrap, score_rows
from stt.lm import BinaryBPELM
from stt.onnx_runner import OnnxMoth

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FP32 = OUT / "model.fp32.onnx"
INT8 = REPO / "runs/E-028/release/model.int8.onnx"
VOCAB = REPO / "release/vocab.json"
LM = REPO / "runs/E-031/build/frozen_1_2_terms.bin"
RARE_IDS = REPO / "runs/E-026/rare47-16k.jsonl"
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}
DEV = ("mls_it_dev", "ls_en_dev")
PRODUCT = ("t600", "fleurs_it_test", "fleurs_en_test", "noise")
BASELINE_PACKAGE = 19_706_289
BASELINE_MODEL = 11_910_677


def _e022():
    spec = importlib.util.spec_from_file_location("e022eval", REPO / "runs/E-022/eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_int8, module.sets


def write_jsonl(path: Path, hypotheses: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
                for key, value in hypotheses.items()),
        encoding="utf-8",
    )


def load_hyps(path: Path) -> dict[str, str]:
    return {
        item["id"]: item["hyp"]
        for item in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def compare(fp32: dict, int8: dict) -> dict:
    return {
        "n": int8["overall"]["n"],
        "fp32": {"wer": fp32["overall"]["wer"], "eta": fp32["overall"]["eta"]},
        "int8": {"wer": int8["overall"]["wer"], "eta": int8["overall"]["eta"]},
        "int8_minus_fp32": _compare(int8, fp32, bootstrap=1000, seed=0),
    }


def dev_sources() -> tuple[dict, list[dict], Path]:
    from stt.train import load_sources

    index: dict = {}
    rows: list[dict] = []
    root = None
    for name in DEV:
        part_index, part_rows, part_root = load_sources(
            [{"manifest": f"data/manifests/{name}.jsonl", "shards": f"data/shards/{name}"}]
        )
        index.update(part_index)
        rows.extend(part_rows)
        root = part_root
    return index, rows, root


def logit_gap(fp32: OnnxMoth, int8: OnnxMoth) -> dict:
    import numpy as np

    from stt.data.build_mel import load_audio_16k
    from stt.features import load_mel_filters, log_mel

    audio = REPO / "web/tests/fixtures/speech.wav"
    mel = log_mel(load_audio_16k(audio), filters=load_mel_filters(REPO / "release/mel_filters.bin"))
    a = np.asarray(fp32.log_probs(mel), dtype=np.float32)
    b = np.asarray(int8.log_probs(mel), dtype=np.float32)
    return {
        "audio": str(audio.relative_to(REPO)),
        "frames": int(a.shape[0]),
        "max_abs": float(np.max(np.abs(a - b))),
        "mean_abs": float(np.mean(np.abs(a - b))),
        "top1_agreement": float(np.mean(a.argmax(-1) == b.argmax(-1))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="new output directory")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be one directory name")
    if not FP32.is_file():
        raise SystemExit(f"{FP32} is missing; run export_fp32.py first")
    out_dir = OUT / "readout" / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; use a new --name")
    out_dir.mkdir(parents=True)

    decode_int8, sets = _e022()
    pieces = load_pieces(VOCAB)
    lm = BinaryBPELM.load(LM)
    graphs = {"fp32": OnnxMoth(FP32, VOCAB), "int8": OnnxMoth(INT8, VOCAB)}
    rare_ids = {
        json.loads(line)["id"]
        for line in RARE_IDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    index, rows, root = dev_sources()
    dev_hyps: dict[str, dict[str, str]] = {}
    dev_scored: dict[str, dict] = {}
    dev_summary: dict[str, dict] = {}
    for name, moth in graphs.items():
        started = time.time()
        hyps = decode_int8(moth, pieces, lm, index, rows, root, PARAMS)
        write_jsonl(out_dir / "dev" / name / "dev.hyp.jsonl", hyps)
        scored = score_rows(rows, hyps, bootstrap=1000, seed=0)
        per_language = {}
        for language in ("it", "en"):
            subset = [row for row in rows if row.get("lang") == language]
            per_language[language] = score_rows(
                subset, {row["id"]: hyps[row["id"]] for row in subset}, bootstrap=0
            )["overall"]["wer"]
        dev_hyps[name] = hyps
        dev_scored[name] = scored
        dev_summary[name] = {
            "macro": (per_language["it"] + per_language["en"]) / 2,
            "it": per_language["it"], "en": per_language["en"],
            "wer": scored["overall"]["wer"], "eta": scored["overall"]["eta"],
            "seconds": round(time.time() - started, 1),
        }
        (out_dir / "dev" / name / "dev.metrics.json").write_text(
            json.dumps({**scored, **dev_summary[name]}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"dev {name:5s} {time.time() - started:6.1f}s", flush=True)

    dev_comparison = {
        "macro_delta_pp": (dev_summary["int8"]["macro"] - dev_summary["fp32"]["macro"]) * 100,
        "macro": macro_wer_bootstrap(
            rows, dev_scored["int8"], dev_scored["fp32"], bootstrap=1000, seed=0
        ),
        "pooled": _compare(dev_scored["int8"], dev_scored["fp32"], bootstrap=1000, seed=0),
    }

    available = sets()
    product: dict[str, dict] = {}
    product_hyps: dict[str, dict[str, dict[str, str]]] = {}
    product_metrics: dict[str, dict[str, dict]] = {}
    for set_name in PRODUCT:
        p_index, p_rows, p_root = available[set_name]
        product[set_name] = {"graphs": {}}
        product_hyps[set_name] = {}
        product_metrics[set_name] = {}
        for name, moth in graphs.items():
            started = time.time()
            hyps = decode_int8(moth, pieces, lm, p_index, p_rows, p_root, PARAMS)
            write_jsonl(out_dir / "product" / name / f"{set_name}.hyp.jsonl", hyps)
            scored = score_rows(p_rows, hyps, bootstrap=1000, seed=0)
            product[set_name]["graphs"][name] = scored
            product_hyps[set_name][name] = hyps
            product_metrics[set_name][name] = scored
            (out_dir / "product" / name / f"{set_name}.metrics.json").write_text(
                json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"{set_name:15s} {name:5s} {time.time() - started:6.1f}s", flush=True)
        product[set_name] = compare(
            product[set_name]["graphs"]["fp32"], product[set_name]["graphs"]["int8"]
        )

    _, t_rows, _ = available["t600"]
    s3_rows = [row for row in t_rows if row.get("speaker") == "S3"]
    rare_rows = [row for row in t_rows if row["id"] in rare_ids]
    for key, subset in (("t-s3", s3_rows), ("rare", rare_rows)):
        entry = {"n": len(subset)}
        for name in ("fp32", "int8"):
            hyps = {row["id"]: product_hyps["t600"][name][row["id"]] for row in subset}
            entry[name] = score_rows(subset, hyps, bootstrap=1000, seed=0)
        entry["int8_minus_fp32"] = _compare(entry["int8"], entry["fp32"], bootstrap=1000, seed=0)
        product[key] = {
            "n": entry["n"],
            "fp32": {"wer": entry["fp32"]["overall"]["wer"],
                     "eta": entry["fp32"]["overall"]["eta"]},
            "int8": {"wer": entry["int8"]["overall"]["wer"],
                     "eta": entry["int8"]["overall"]["eta"]},
            "int8_minus_fp32": entry["int8_minus_fp32"],
        }

    terms_path = out_dir / "terms.json"
    subprocess.run(
        [sys.executable, str(REPO / "runs/E-015/eval_terms.py"),
         "--hyp", str(out_dir / "product/int8/t600.hyp.jsonl"),
         "--baseline", str(out_dir / "product/fp32/t600.hyp.jsonl"),
         "--out", str(terms_path)],
        check=True, cwd=REPO,
    )

    noise = {
        "fp32": product_metrics["noise"]["fp32"]["noise"],
        "int8": product_metrics["noise"]["int8"]["noise"],
    }

    fp32_report = json.loads((OUT / "fp32.json").read_text(encoding="utf-8"))
    result = {
        "name": args.name,
        "params": PARAMS,
        "assets": {
            "fp32": fp32_report["fp32"],
            "int8": fp32_report["int8"],
            "lm": {"path": str(LM.relative_to(REPO)), "bytes": LM.stat().st_size,
                   "sha256": sha256_file(LM)},
            "vocab": {"path": str(VOCAB.relative_to(REPO)),
                      "sha256": sha256_file(VOCAB)},
        },
        "package": {
            "fp32_bytes": BASELINE_PACKAGE - BASELINE_MODEL + fp32_report["fp32"]["bytes"],
            "int8_bytes": BASELINE_PACKAGE,
            "limit": 20_000_000,
        },
        "dev": {"fp32": dev_summary["fp32"], "int8": dev_summary["int8"],
                "int8_minus_fp32": dev_comparison},
        "product": product,
        "terms": json.loads(terms_path.read_text(encoding="utf-8")),
        "noise": noise,
        "logits": logit_gap(graphs["fp32"], graphs["int8"]),
        "note": "measurement only: INT8 is the shipped format and no gate changes the release",
    }
    (out_dir / "readout.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dev": result["dev"], "package": result["package"],
                      "logits": result["logits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
