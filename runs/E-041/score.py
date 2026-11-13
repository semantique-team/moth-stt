"""E-041 — score the browser hypotheses and compare them with the Python INT8 path.

The browser uses the pinned 0.8.1 bundle with an exact local runtime snapshot, not immutable
deployed code; the Python hypotheses are the committed E-037 ones
(same INT8 graph, same A3 LM, same decoder). The browser includes shipped VAD guarding and
segmentation; E-037 uses cached ungated mel clips. Differences are an end-to-end deployment
property, not isolated acoustic parity or a model change. Only ``metrics.json`` is written.

    uv run python runs/E-041/score.py --name browser-1
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from stt.data.sources import sha256_file
from stt.eval import _compare, score_rows

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = REPO / "recordings/manifests/T.jsonl"
PYTHON_HYP = REPO / "runs/E-037/readout/paired-1/product/int8/t600.hyp.jsonl"
RELEASE_RECORD = REPO / "runs/E-031/release-0.8.1.json"
PYTHON_RUN = REPO / "runs/E-037/readout/paired-1/readout.json"
PYTHON_METRICS = PYTHON_HYP.with_name("t600.metrics.json")


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate IDs")
    return rows


def validate_provenance(run: dict, reference: dict, expected: dict) -> None:
    provenance = run.get("provenance", {})
    if (provenance.get("version") != "0.8.1"
            or provenance.get("model") != "model.int8.onnx"
            or provenance.get("config_sha256") != expected["files"]["config.json"]["sha256"]
            or provenance.get("vad") != expected["vad"]
            or provenance.get("segmentation") is not True
            or provenance.get("cross_origin_isolated") is not True
            or run.get("failures") != 0):
        raise SystemExit("browser provenance does not match the pinned E-031 0.8.1 bundle")
    for key, value in expected["decode"].items():
        if provenance.get("decode", {}).get(key) != value:
            raise SystemExit(f"browser decoder mismatch: {key}")
    for file in ("model.int8.onnx", "lm_bpe3.bin", "vocab.json", "mel_filters.bin",
                 "silero_vad.onnx"):
        actual = provenance.get("files", {}).get(file, "").removeprefix("sha256:")
        if actual != expected["files"][file]["sha256"]:
            raise SystemExit(f"browser release identity mismatch: {file}")
    for asset, file in (("int8", "model.int8.onnx"), ("lm", "lm_bpe3.bin"),
                        ("vocab", "vocab.json")):
        if reference.get("assets", {}).get(asset, {}).get("sha256") != expected["files"][file]["sha256"]:
            raise SystemExit(f"E-037 identity mismatch: {asset}")
    params = {key: expected["decode"][key] for key in
              ("beam", "top_k", "lm_weight", "token_bonus")}
    if reference.get("params") != params:
        raise SystemExit("E-037 decoder mismatch")


def validate_measurement(run: dict, rows: list[dict], reference: dict,
                         reference_metrics: dict) -> None:
    measurement = run.get("measurement", {})
    manifest = measurement.get("manifest", {})
    if (manifest.get("path") != MANIFEST.relative_to(REPO).as_posix()
            or manifest.get("sha256") != sha256_file(MANIFEST)):
        raise SystemExit("measurement manifest identity mismatch")
    for retained in (reference, reference.get("product", {}).get("t600", {}), reference_metrics):
        for recorded in (retained.get("manifest_sha256"),
                         retained.get("assets", {}).get("manifest", {}).get("sha256")):
            if recorded is not None and recorded != manifest["sha256"]:
                raise SystemExit("E-037 retained manifest identity mismatch")
    audio = measurement.get("audio", [])
    if len(audio) != len(rows) or {item["id"] for item in audio} != {row["id"] for row in rows}:
        raise SystemExit("measurement audio receipts do not cover the manifest exactly")
    audio_paths = {row["id"]: f"recordings/audio/{row['audio']}" for row in rows}
    if any(item.get("path") != audio_paths[item["id"]] for item in audio):
        raise SystemExit("measurement audio path mismatch")
    runtime = measurement.get("runtime", [])
    paths = [item.get("path", "") for item in runtime]
    ort_root = "web/node_modules/onnxruntime-web/dist/"
    required = {"runs/E-041/browser_wer.mjs", "runs/E-041/score.py", "web/server.py",
                "web/ort.js", "web/mel.js", "web/decode.js", "web/lm.js", "web/vad.js",
                f"{ort_root}ort.wasm.min.mjs"}
    if (len(set(paths)) != len(paths) or not required.issubset(paths)
            or not any(file.startswith(ort_root) and file.endswith(".wasm") for file in paths)
            or not any(file.startswith(f"{ort_root}ort-wasm") and file.endswith(".mjs")
                       for file in paths)):
        raise SystemExit("measurement runtime receipts are incomplete or duplicated")
    for receipt in [*audio, *runtime]:
        file = receipt["path"]
        path = (REPO / file).resolve()
        if (not path.is_relative_to(REPO.resolve()) or "\\" in file
                or not path.is_file() or sha256_file(path) != receipt.get("sha256")):
            raise SystemExit(f"measurement local file identity mismatch: {file}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="readout directory produced by browser_wer.mjs")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be one directory name")
    out_dir = OUT / "readout" / args.name
    hyp_path = out_dir / "browser.hyp.jsonl"
    timings_path = out_dir / "browser.timings.jsonl"
    run_path = out_dir / "browser.run.json"
    for path in (hyp_path, timings_path, run_path):
        if not path.is_file():
            raise SystemExit(f"{path} is missing")

    run = json.loads(run_path.read_text(encoding="utf-8"))
    reference = json.loads(PYTHON_RUN.read_text(encoding="utf-8"))
    expected = json.loads(RELEASE_RECORD.read_text(encoding="utf-8"))
    validate_provenance(run, reference, expected)
    rows = read_jsonl(MANIFEST)
    reference_metrics = json.loads(PYTHON_METRICS.read_text(encoding="utf-8"))
    validate_measurement(run, rows, reference, reference_metrics)
    browser = {item["id"]: item["hyp"] for item in read_jsonl(hyp_path)}
    python = {item["id"]: item["hyp"] for item in read_jsonl(PYTHON_HYP)}
    if set(browser) != {str(row["id"]) for row in rows}:
        raise SystemExit("browser hypotheses do not cover the T manifest exactly")
    if set(python) != set(browser):
        raise SystemExit("E-037 hypotheses do not cover the T manifest exactly")

    timings = read_jsonl(timings_path)
    if {item["id"] for item in timings} != set(browser) or run.get("clips") != len(browser):
        raise SystemExit("timings/run clip count do not cover the T manifest exactly")
    rtfx = [item["rtfx"] for item in timings]
    matched = sum(1 for uid in browser if browser[uid] == python[uid])

    browser_metrics = score_rows(rows, browser, bootstrap=1000, seed=0)
    python_metrics = score_rows(rows, python, bootstrap=1000, seed=0)
    comparison = _compare(browser_metrics, python_metrics, bootstrap=1000, seed=0)
    per_speaker = {}
    for speaker in ("S1", "S2", "S3"):
        subset = [row for row in rows if row.get("speaker") == speaker]
        per_speaker[speaker] = {
            "browser": score_rows(subset, {row["id"]: browser[row["id"]] for row in subset},
                                  bootstrap=1000, seed=0)["overall"],
            "python": score_rows(subset, {row["id"]: python[row["id"]] for row in subset},
                                 bootstrap=1000, seed=0)["overall"],
        }

    result = {
        "name": args.name,
        "release": run["release"],
        "provenance": run["provenance"],
        "measurement": run["measurement"],
        "assets": {
            "manifest": {"path": str(MANIFEST.relative_to(REPO)),
                         "sha256": sha256_file(MANIFEST)},
            "browser_hyp": {"path": str(hyp_path.relative_to(REPO)),
                            "sha256": sha256_file(hyp_path)},
            "python_hyp": {"path": str(PYTHON_HYP.relative_to(REPO)),
                            "sha256": sha256_file(PYTHON_HYP)},
            "release_record": {"path": str(RELEASE_RECORD.relative_to(REPO)),
                               "sha256": sha256_file(RELEASE_RECORD)},
            "python_run": {"path": str(PYTHON_RUN.relative_to(REPO)),
                            "sha256": sha256_file(PYTHON_RUN)},
            "python_metrics": {"path": str(PYTHON_METRICS.relative_to(REPO)),
                                "sha256": sha256_file(PYTHON_METRICS)},
        },
        "browser": {"overall": browser_metrics["overall"], "noise": browser_metrics["noise"]},
        "python": {"overall": python_metrics["overall"], "noise": python_metrics["noise"]},
        "browser_minus_python": comparison,
        "identical_hypotheses": matched,
        "differing_hypotheses": len(browser) - matched,
        "per_speaker": per_speaker,
        "rtfx": {
            "median": statistics.median(rtfx),
            "mean": statistics.fmean(rtfx),
            "min": min(rtfx),
            "max": max(rtfx),
            "clips": len(rtfx),
        },
        "wall_seconds": run["wall_seconds"],
        "audio_seconds": run["audio_seconds"],
        "note": "browser shipped VAD guard/segmentation vs E-037 cached ungated mel clips; "
                "exact local runtime snapshot, not immutable deployed code; "
                "end-to-end deployment comparison, not isolated acoustic parity or a model change",
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    overall = result["browser"]["overall"]
    print(
        f"browser  WER {overall['wer'] * 100:.4f}%  ETA {(overall['eta'] or 0) * 100:.4f}%  "
        f"RTFx median {result['rtfx']['median']:.2f}"
    )
    print(
        f"python   WER {result['python']['overall']['wer'] * 100:.4f}%  "
        f"ETA {(result['python']['overall']['eta'] or 0) * 100:.4f}%"
    )
    print(
        f"delta    WER {comparison['wer_diff'] * 100:+.4f} pp "
        f"[{comparison['wer_diff_ci'][0] * 100:+.4f}, {comparison['wer_diff_ci'][1] * 100:+.4f}] "
        f"tie={comparison['wer_tie']}  differing hypotheses {result['differing_hypotheses']}/600"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
