"""E-041 — score the browser hypotheses and compare them with the Python INT8 path.

The browser decode is the deployed artifact; the Python hypotheses are the committed E-037 ones
(same INT8 graph, same A3 LM, same decoder). Differences are a deployment property, not a model
change. This script only measures and writes ``metrics.json``.

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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="readout directory produced by browser_wer.mjs")
    args = parser.parse_args()
    out_dir = OUT / "readout" / args.name
    hyp_path = out_dir / "browser.hyp.jsonl"
    timings_path = out_dir / "browser.timings.jsonl"
    run_path = out_dir / "browser.run.json"
    for path in (hyp_path, timings_path, run_path):
        if not path.is_file():
            raise SystemExit(f"{path} is missing")

    rows = read_jsonl(MANIFEST)
    browser = {item["id"]: item["hyp"] for item in read_jsonl(hyp_path)}
    python = {item["id"]: item["hyp"] for item in read_jsonl(PYTHON_HYP)}
    if set(browser) != {str(row["id"]) for row in rows}:
        raise SystemExit("browser hypotheses do not cover the T manifest exactly")

    timings = read_jsonl(timings_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
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
        "assets": {
            "manifest": {"path": str(MANIFEST.relative_to(REPO)),
                         "sha256": sha256_file(MANIFEST)},
            "browser_hyp": {"path": str(hyp_path.relative_to(REPO)),
                            "sha256": sha256_file(hyp_path)},
            "python_hyp": {"path": str(PYTHON_HYP.relative_to(REPO)),
                           "sha256": sha256_file(PYTHON_HYP)},
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
        "note": "browser vs Python is a deployment comparison, not a model change",
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
