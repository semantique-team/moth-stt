"""E-040 — aggregate the four replicates per arm into one dispersion summary.

Reads the per-replicate product readouts (E-040 runs plus the E-036 seed-0 readouts) and the
training summaries, and writes ``summary.json``: per-arm mean/min/max/range for dev, T-600,
T-S3, FLEURS, rare aligned ETA, silence behaviour and natural terms.

    uv run python runs/E-040/summarize.py
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPLICATES = {
    ("frozen", 0): REPO / "runs/E-036/eval/frozen-1/readout.json",
    ("frozen", 1): OUT / "eval/s1-frozen/readout.json",
    ("frozen", 2): OUT / "eval/s2-frozen/readout.json",
    ("frozen", 3): OUT / "eval/s3-frozen/readout.json",
    ("last2", 0): REPO / "runs/E-036/eval/last2-1/readout.json",
    ("last2", 1): OUT / "eval/s1-last2/readout.json",
    ("last2", 2): OUT / "eval/s2-last2/readout.json",
    ("last2", 3): OUT / "eval/s3-last2/readout.json",
}
TRAIN_SUMMARIES = {
    ("frozen", 0): REPO / "runs/E-036/frozen/train.out",
    ("frozen", 1): OUT / "seed1/frozen/train.out",
    ("frozen", 2): OUT / "seed2/frozen/train.out",
    ("frozen", 3): OUT / "seed3/frozen/train.out",
    ("last2", 0): REPO / "runs/E-036/last2/train.out",
    ("last2", 1): OUT / "seed1/last2/train.out",
    ("last2", 2): OUT / "seed2/last2/train.out",
    ("last2", 3): OUT / "seed3/last2/train.out",
}


def best_dev(path: Path) -> float:
    match = re.search(r'\{\s*"run".*?\n\}', path.read_text(encoding="utf-8"), re.DOTALL)
    if not match:
        raise ValueError(f"no JSON summary in {path}")
    return json.loads(match.group(0))["best_metric"]


def silence_from_hypotheses(path: Path) -> dict:
    """Recompute the silence probe when a historical readout predates it."""
    rows = [
        json.loads(line)
        for line in (REPO / "data/manifests/noise.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    silence = [row for row in rows if row.get("source") == "noise_silence"]
    hyps = {
        item["id"]: item["hyp"]
        for item in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    return {
        "silence_clips": len(silence),
        "silence_non_empty": sum(1 for row in silence if hyps[row["id"]].strip()),
        "silence_x": sum(1 for row in silence if hyps[row["id"]].strip().lower() == "x"),
    }


def metrics(readout: dict, readout_path: Path) -> dict:
    sets = readout["sets"]
    rare = readout["rare"]
    if "silence" in readout["noise"]:
        silence = readout["noise"]["silence"]
        candidate = silence.get("candidate", silence)
    else:
        candidate = silence_from_hypotheses(readout_path.parent / "candidate/noise.hyp.jsonl")
    terms = readout["terms"]["groups"]
    return {
        "t600_wer": sets["t600"]["candidate"]["wer"],
        "t600_eta": sets["t600"]["candidate"]["eta"],
        "t600_wer_diff_pp": sets["t600"]["vs_baseline"]["wer_diff"] * 100,
        "t600_wer_ci_pp": [value * 100 for value in sets["t600"]["vs_baseline"]["wer_diff_ci"]],
        "ts3_wer": sets["t-s3"]["candidate"]["wer"],
        "ts3_eta": sets["t-s3"]["candidate"]["eta"],
        "fleurs_it_wer": sets["fleurs_it_test"]["candidate"]["wer"],
        "fleurs_en_wer": sets["fleurs_en_test"]["candidate"]["wer"],
        "rare_hit": rare["candidate"]["en_hit"],
        "rare_ref": rare["candidate"]["en_ref"],
        "rare_eta": rare["candidate"]["aligned_eta"],
        "natural_correct": terms["natural"]["new"]["correct"],
        "gold_correct": terms["gold"]["new"]["correct"],
        "silence_non_empty": candidate["silence_non_empty"],
        "silence_x": candidate["silence_x"],
        "non_music_words": readout["noise"]["candidate"]["non_music_words"],
        "music_words": readout["noise"]["candidate"]["music_words"],
    }


def spread(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "values": values,
    }


def main() -> int:
    per_run: dict[str, dict[int, dict]] = {"frozen": {}, "last2": {}}
    for (arm, seed), path in REPLICATES.items():
        if not path.is_file():
            raise SystemExit(f"missing readout: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_run[arm][seed] = {
            "dev_best": best_dev(TRAIN_SUMMARIES[(arm, seed)]),
            "readout": str(path.relative_to(REPO)),
            **metrics(payload, path),
        }

    summary: dict = {"replicates": per_run, "dispersion": {}}
    for arm, runs in per_run.items():
        keys = [
            key for key, value in next(iter(runs.values())).items()
            if key != "readout" and isinstance(value, (int, float))
        ]
        summary["dispersion"][arm] = {
            key: spread([runs[seed][key] for seed in sorted(runs)]) for key in keys
        }
    for key in ("dev_best", "t600_wer", "rare_eta", "silence_non_empty", "natural_correct"):
        left = summary["dispersion"]["frozen"][key]
        right = summary["dispersion"]["last2"][key]
        summary["dispersion"].setdefault("arm_gap", {})[key] = {
            "last2_minus_frozen_mean": right["mean"] - left["mean"],
            "frozen_range": left["range"],
            "last2_range": right["range"],
        }
    summary["note"] = (
        "four replicates per arm; ranges are descriptive and do not estimate a variance"
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for arm in ("frozen", "last2"):
        d = summary["dispersion"][arm]
        print(
            f"{arm:7s} dev {d['dev_best']['mean'] * 100:.4f} "
            f"[{d['dev_best']['min'] * 100:.4f}, {d['dev_best']['max'] * 100:.4f}]  "
            f"T-600 {d['t600_wer']['mean'] * 100:.4f} "
            f"[{d['t600_wer']['min'] * 100:.4f}, {d['t600_wer']['max'] * 100:.4f}]  "
            f"rare {d['rare_hit']['mean']:.1f}/81  "
            f"silence {d['silence_non_empty']['mean']:.0f}/120"
        )
    gaps = summary["dispersion"]["arm_gap"]
    print(
        f"gap last2-frozen: dev {gaps['dev_best']['last2_minus_frozen_mean'] * 100:+.4f} pp, "
        f"T-600 {gaps['t600_wer']['last2_minus_frozen_mean'] * 100:+.4f} pp, "
        f"rare {gaps['rare_eta']['last2_minus_frozen_mean'] * 100:+.4f} pp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
