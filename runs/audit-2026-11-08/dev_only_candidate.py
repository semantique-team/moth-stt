"""Audit probe 16 — the honest, dev-only candidate.

``PROBE 14`` used ``lm_weight = 0.6, gamma = 3.0``, but that cell is **not** in the dev grid: dev
tested gamma in {1.6, 2.0, 2.4, 2.8, 3.2}. The cell was picked after a look at T-S3, which breaks
the repository rule "do not tune against S3" (``docs/plans/2026-11-07-e018-scale-the-gold.md``).

This probe re-reads the **best cell that dev alone selects** (beam 12 / top-k 10 / lm_weight 0.5 /
gamma 2.4, dev WER 0.2659), so the recommendation can be made without touching the held-out speaker.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from final_candidate import SHIPPED, decode_shards
from sweep_token_bonus import beam_with_bonus
from sweep_token_bonus_ext import collect

from stt.decode import load_pieces
from stt.eval import _compare, score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.onnx_runner import OnnxMoth
from stt.tokenizer import load_tokenizer
from stt.train import load_config

DEV_ONLY = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "gamma": 2.4}
CHEAP = {"beam": 8, "top_k": 8, "lm_weight": 0.5, "gamma": 2.4}


def main() -> int:
    cfg = load_config("runs/E-017/config.yaml")
    proc = load_tokenizer(cfg["tokenizer"])
    pieces = {i + 1: proc.id_to_piece(i) for i in range(proc.vocab_size())}
    model = MothSTT(vocab_size=1024, head_blocks=2)
    payload = torch.load(
        "runs/E-017/checkpoints/ckpt_best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model"])
    model.eval()
    lm = BPETrigramLM.load("runs/E-008/lm_bpe3_aug.pkl.gz")

    t_rows = [
        json.loads(line) for line in (Path("recordings/manifests/T.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    raw = json.loads(Path("data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    t_index = {uid: {**e, "shard": f"recorded_t/{e['shard']}"} for uid, e in raw.items()}
    s3_rows = [r for r in t_rows if r.get("speaker") == "S3"]

    sets = {
        "t600": (t_index, t_rows, Path("data/shards")),
        "t-s3": (t_index, s3_rows, Path("data/shards")),
        "fleurs_it": collect([("fleurs_it_test", 0)]),
        "fleurs_en": collect([("fleurs_en_test", 0)]),
        "cv_it_dev": collect([("cv_it_dev", 600)]),
    }

    summary = {}
    for name, (idx, rows, root) in sets.items():
        row = {}
        for label, params in (("shipped", SHIPPED), ("cheap", CHEAP), ("dev-only", DEV_ONLY)):
            started = time.time()
            hyps = decode_shards(model, pieces, lm, idx, rows, root, params)
            scored = score_rows(rows, hyps, bootstrap=1000, seed=0)
            row[label] = scored
            o = scored["overall"]
            audio_s = sum(r["duration"] for r in rows)
            print(
                json.dumps(
                    {
                        "set": name,
                        "arm": label,
                        "wer": round(o["wer"], 4),
                        "eta": None if o["eta"] is None else round(o["eta"], 4),
                        "ci": [round(x, 4) for x in o["wer_ci"]],
                        "rtfx": round(audio_s / (time.time() - started), 1),
                    }
                ),
                flush=True,
            )
            target = Path("runs/audit-2026-11-08/dev-only") / label
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{name}.hyp.jsonl").write_text(
                "".join(
                    json.dumps({"id": k, "hyp": v}, ensure_ascii=False) + "\n"
                    for k, v in hyps.items()
                ),
                encoding="utf-8",
            )
        entry = {"shipped": {"wer": row["shipped"]["overall"]["wer"]}}
        for label in ("cheap", "dev-only"):
            diff = _compare(row[label], row["shipped"], bootstrap=1000, seed=0)
            entry[label] = {
                "wer": row[label]["overall"]["wer"],
                "eta": row[label]["overall"]["eta"],
                "wer_diff": diff["wer_diff"],
                "wer_ci": diff["wer_diff_ci"],
                "wer_tie": diff["wer_tie"],
                "eta_diff": diff["eta_diff"],
                "eta_tie": diff["eta_tie"],
            }
            print(
                json.dumps(
                    {
                        "set": name,
                        f"{label}_minus_shipped": round(diff["wer_diff"], 4),
                        "ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                        "tie": diff["wer_tie"],
                        "eta_diff": None
                        if diff["eta_diff"] is None
                        else round(diff["eta_diff"], 4),
                    }
                ),
                flush=True,
            )
        summary[name] = entry

    # int8 on the S3 slice, for browser parity of the dev-only arm
    index = json.loads(Path("data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    shard = np.load("data/shards/recorded_t/shard-000.npy", mmap_mode="r")
    onnx = OnnxMoth("release/model.int8.onnx", "release/vocab.json")
    int8_pieces = load_pieces("release/vocab.json")
    for label, params in (("shipped", SHIPPED), ("dev-only", DEV_ONLY)):
        hyps = {}
        for r in s3_rows:
            entry = index[r["id"]]
            start = entry["offset"] * 80
            mel = np.asarray(
                shard[start : start + entry["frames"] * 80], dtype=np.float32
            ).reshape(entry["frames"], 80).T
            log_probs = onnx.log_probs(mel)
            hyps[r["id"]] = beam_with_bonus(
                log_probs, log_probs.shape[0], int8_pieces,
                beam_width=params["beam"], top_k=params["top_k"],
                lm=lm, lm_weight=params["lm_weight"], gamma=params["gamma"], beta=0.0,
            )
        o = score_rows(s3_rows, hyps, bootstrap=1000, seed=0)["overall"]
        print(
            json.dumps(
                {"int8_s3": label, "wer": round(o["wer"], 4), "eta": round(o["eta"], 4)}
            ),
            flush=True,
        )

    Path("runs/audit-2026-11-08/dev-only-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())