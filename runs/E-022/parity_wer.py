"""E-022 — browser/Python WER gap at the shipped settings, on a sample large enough to decide.

The 25-clip parity sample is too small: 150 reference words give a CI of ±5 WER points, so the
``+4.0`` browser-vs-Python difference measured there is uninformative. This runs 200 clips through
both sides and reports the paired difference, which is the number the model card needs.

Browser output must exist first:  ``runs/E-022/parity_browser.py --clips 200`` then
``cd web && npx playwright test tests/parity.spec.ts`` then ``--compare``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stt.decode import beam_search, load_pieces
from stt.eval import _compare, score_rows
from stt.lm import BPETrigramLM
from stt.onnx_runner import OnnxMoth

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ARMS = {
    "shipped": {"beam": 5, "top_k": 8, "lm_weight": 0.2, "token_bonus": 0.0},
    "frozen": {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4},
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="E-022/parity_wer.py")
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument(
        "--browser",
        default="web/test-results/parity-browser.json",
        help="browser transcripts for the frozen arm",
    )
    parser.add_argument(
        "--browser-shipped",
        default="web/test-results/parity-browser-shipped.json",
        help="browser transcripts at the shipped decode (optional)",
    )
    args = parser.parse_args()

    all_rows = [
        json.loads(line)
        for line in (REPO / "recordings/manifests/T.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows = all_rows[: args.clips]
    wanted = {r["id"] for r in rows}
    raw = json.loads((REPO / "data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    shard = np.load(REPO / "data/shards/recorded_t/shard-000.npy", mmap_mode="r")
    moth = OnnxMoth(REPO / "release/model.int8.onnx", REPO / "release/vocab.json")
    pieces = load_pieces(REPO / "release/vocab.json")
    lm = BPETrigramLM.load(REPO / "runs/E-008/lm_bpe3_aug.pkl.gz")

    python_arms: dict[str, dict[str, str]] = {}
    for arm, params in ARMS.items():
        hypotheses = {}
        for row in rows:
            entry = raw[row["id"]]
            start = entry["offset"] * 80
            mel = np.asarray(
                shard[start : start + entry["frames"] * 80], dtype=np.float32
            ).reshape(entry["frames"], 80).T
            log_probs = moth.log_probs(mel)
            hypotheses[row["id"]] = beam_search(
                log_probs,
                log_probs.shape[0],
                pieces,
                beam_width=params["beam"],
                top_k=params["top_k"],
                lm=lm,
                lm_weight=params["lm_weight"],
                token_bonus=params["token_bonus"],
            )
        python_arms[arm] = hypotheses
        (OUT / f"parity-{arm}-python.jsonl").write_text(
            "".join(
                json.dumps({"id": k, "hyp": v}, ensure_ascii=False) + "\n"
                for k, v in hypotheses.items()
            ),
            encoding="utf-8",
        )

    def load_browser(path: str) -> dict[str, str] | None:
        candidate = REPO / path
        if not candidate.exists():
            return None
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in wanted}

    report: dict[str, dict] = {}
    for arm, params in ARMS.items():
        browser = load_browser(args.browser if arm == "frozen" else args.browser_shipped)
        if browser is None:
            print(f"{arm}: no browser output at that path, skipped")
            continue
        subset = [r for r in rows if r["id"] in browser]
        identical = sum(1 for uid, text in browser.items() if python_arms[arm].get(uid) == text)
        ours = score_rows(subset, browser, bootstrap=1000, seed=0)
        theirs = score_rows(subset, python_arms[arm], bootstrap=1000, seed=0)
        diff = _compare(ours, theirs, bootstrap=1000, seed=0)
        report[arm] = {
            "clips": len(subset),
            "identical": identical,
            "identical_ratio": identical / len(subset),
            "browser_wer": ours["overall"]["wer"],
            "python_wer": theirs["overall"]["wer"],
            "wer_diff": diff["wer_diff"],
            "wer_ci": diff["wer_diff_ci"],
            "wer_tie": diff["wer_tie"],
            "browser_eta": ours["overall"]["eta"],
            "python_eta": theirs["overall"]["eta"],
        }
        print(
            json.dumps(
                {
                    "arm": arm,
                    "clips": len(subset),
                    "identical": f"{identical}/{len(subset)}",
                    "browser_wer": round(ours["overall"]["wer"], 4),
                    "python_wer": round(theirs["overall"]["wer"], 4),
                    "diff": round(diff["wer_diff"], 4),
                    "ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                    "tie": diff["wer_tie"],
                }
            ),
            flush=True,
        )

    (OUT / "parity-wer.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())