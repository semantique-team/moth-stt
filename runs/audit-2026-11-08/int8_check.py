"""Audit probe 10 — the recalibrated decode on the SHIPPED int8 artifact.

The change is decode-only, so it must survive quantization. Runs ``release/model.int8.onnx``
through ``stt.onnx_runner``-style decoding on 200 T clips (S1+S2+S3 mixed) with the shipped
parameters and with the new ones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import beam_with_bonus

from stt.decode import load_pieces
from stt.eval import _compare, score_rows
from stt.lm import BPETrigramLM
from stt.onnx_runner import OnnxMoth


def main() -> int:
    rows = [
        json.loads(line) for line in (Path("recordings/manifests/T.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    rows = [r for r in rows if r["id"] in {x["id"] for x in rows}][:200]
    index = json.loads(Path("data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    shard = np.load("data/shards/recorded_t/shard-000.npy", mmap_mode="r")

    model = OnnxMoth("release/model.int8.onnx", "release/vocab.json")
    pieces = load_pieces("release/vocab.json")
    lm = BPETrigramLM.load("runs/E-008/lm_bpe3_aug.pkl.gz")

    arms = {"shipped-0.2-g0": (0.2, 0.0), "recal-0.5-g2.4-b8": (0.5, 2.4)}
    results = {}
    for arm, (lm_weight, gamma) in arms.items():
        hyps = {}
        for row in rows:
            entry = index[row["id"]]
            start, frames = entry["offset"] * 80, entry["frames"]
            mel = np.asarray(shard[start : start + frames * 80], dtype=np.float32)
            mel = mel.reshape(frames, 80).T
            log_probs = model.log_probs(mel)
            hyps[row["id"]] = beam_with_bonus(
                log_probs, log_probs.shape[0], pieces,
                beam_width=8, top_k=8, lm=lm, lm_weight=lm_weight, gamma=gamma, beta=0.0,
            )
        scored = score_rows(rows, hyps, bootstrap=1000, seed=0)
        results[arm] = scored
        o = scored["overall"]
        print(
            json.dumps(
                {
                    "arm": arm,
                    "n": len(rows),
                    "wer": round(o["wer"], 4),
                    "eta": round(o["eta"], 4),
                    "ci": [round(x, 4) for x in o["wer_ci"]],
                }
            ),
            flush=True,
        )
    diff = _compare(results["recal-0.5-g2.4-b8"], results["shipped-0.2-g0"], bootstrap=1000, seed=0)
    print(
        json.dumps(
            {
                "int8_wer_diff": round(diff["wer_diff"], 4),
                "ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                "tie": diff["wer_tie"],
                "eta_diff": round(diff["eta_diff"], 4),
                "eta_tie": diff["eta_tie"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())