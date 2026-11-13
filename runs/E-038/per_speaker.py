"""E-038 — per-speaker T readout of 0.8.1 from the retained E-037 hypotheses.

Decode-free: reads the reference manifest and the committed hypotheses, then reports WER/ETA for
S1, S2 and S3 separately with per-utterance bootstrap CIs, plus the dispersion across voices.
No decode, no training, no new audio.

    uv run python runs/E-038/per_speaker.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from stt.data.sources import sha256_file
from stt.eval import _bootstrap_ratio, score_rows

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = REPO / "recordings/manifests/T.jsonl"
HYP = REPO / "runs/E-037/readout/paired-1/product/int8/t600.hyp.jsonl"
SPEAKERS = ("S1", "S2", "S3")
BOOTSTRAP = 1000
SEED = 0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def speaker_metrics(rows: list[dict], hyps: dict[str, str]) -> dict:
    scored = score_rows(rows, hyps, bootstrap=0)
    per_utterance = scored["per_utterance"]
    ids = list(per_utterance)
    overall = scored["overall"]
    return {
        **overall,
        "wer_ci": _bootstrap_ratio(
            [per_utterance[uid]["sub"] + per_utterance[uid]["del"] + per_utterance[uid]["ins"]
             for uid in ids],
            [per_utterance[uid]["ref_words"] for uid in ids],
            BOOTSTRAP,
            SEED,
        ),
        "eta_ci": _bootstrap_ratio(
            [per_utterance[uid]["en_hit"] for uid in ids],
            [per_utterance[uid]["en_ref"] for uid in ids],
            BOOTSTRAP,
            SEED,
        ),
        "ci_kind": "per-utterance bootstrap, not speaker-cluster",
    }


def main() -> int:
    rows = read_jsonl(MANIFEST)
    hyps = {item["id"]: item["hyp"] for item in read_jsonl(HYP)}
    if set(hyps) != {str(row["id"]) for row in rows}:
        raise SystemExit("hypotheses do not cover the manifest exactly")

    per_speaker = {}
    for speaker in SPEAKERS:
        subset = [row for row in rows if row.get("speaker") == speaker]
        per_speaker[speaker] = speaker_metrics(subset, hyps)

    pooled = speaker_metrics(rows, hyps)
    wers = [per_speaker[speaker]["wer"] for speaker in SPEAKERS]
    etas = [per_speaker[speaker]["eta"] for speaker in SPEAKERS]
    result = {
        "assets": {
            "manifest": str(MANIFEST.relative_to(REPO)),
            "manifest_sha256": sha256_file(MANIFEST),
            "hyp": str(HYP.relative_to(REPO)),
            "hyp_sha256": sha256_file(HYP),
            "git_sha": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip(),
        },
        "protocol": {"bootstrap": BOOTSTRAP, "seed": SEED,
                     "ci_kind": "per-utterance bootstrap, not speaker-cluster"},
        "roles": {
            "S1": "acoustic training speaker; dev/diagnostic only",
            "S2": "acoustic training speaker; dev/diagnostic only",
            "S3": "never in acoustic training; inspected repeatedly, not an untouched test",
        },
        "per_speaker": per_speaker,
        "pooled": pooled,
        "dispersion": {
            "wer": {"min": min(wers), "max": max(wers), "range_pp": (max(wers) - min(wers)) * 100,
                    "mean": sum(wers) / len(wers)},
            "eta": {"min": min(etas), "max": max(etas), "range_pp": (max(etas) - min(etas)) * 100,
                    "mean": sum(etas) / len(etas)},
            "note": "descriptive only: three voices cannot estimate a between-speaker variance",
        },
        "note": "decode-free re-scoring of committed E-037 hypotheses; T is reused evidence",
    }
    (OUT / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for speaker in SPEAKERS:
        entry = per_speaker[speaker]
        print(
            f"{speaker}  n={entry['n']:3d}  WER {entry['wer'] * 100:7.4f}%  "
            f"CI [{entry['wer_ci'][0] * 100:7.4f}, {entry['wer_ci'][1] * 100:7.4f}]  "
            f"ETA {entry['eta'] * 100:7.4f}%  CI [{entry['eta_ci'][0] * 100:7.4f}, "
            f"{entry['eta_ci'][1] * 100:7.4f}]"
        )
    print(
        f"pooled  n={pooled['n']}  WER {pooled['wer'] * 100:.4f}%  ETA {pooled['eta'] * 100:.4f}%"
    )
    print(
        f"spread  WER {result['dispersion']['wer']['range_pp']:.4f} pp  "
        f"ETA {result['dispersion']['eta']['range_pp']:.4f} pp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
