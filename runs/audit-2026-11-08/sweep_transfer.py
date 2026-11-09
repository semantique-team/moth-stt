"""Audit probe 6 — does the best dev cell transfer? Reported per set with the honest caveat.

The dev grid (probe 5) put its optimum on the corner (``lm_weight`` 0.4, ``gamma`` 2.0), so the
grid is extended once more on dev, and the leading cells are then read on T-600, FLEURS IT/EN and
CV-IT dev. T is **not** used for selection here: the decision rule stays "pick the best dev cell".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import decode_set
from sweep_token_bonus_ext import collect

from stt.eval import score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import load_config


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

    dev_idx, dev_rows, dev_root = collect([("mls_it_dev", 200), ("ls_en_dev", 200)])

    print("--- dev extension", flush=True)
    for lm_weight in (0.4, 0.5, 0.6, 0.8):
        for gamma in (1.6, 2.0, 2.4, 2.8, 3.2):
            hyps = decode_set(
                model, pieces, dev_rows, dev_idx, dev_root, cfg,
                lm=lm, lm_weight=lm_weight, gamma=gamma, beta=0.0,
            )
            o = score_rows(dev_rows, hyps, bootstrap=0, seed=0)["overall"]
            print(
                json.dumps(
                    {
                        "lm_weight": lm_weight,
                        "gamma": gamma,
                        "wer": round(o["wer"], 4),
                        "del": round(o["del"] / o["ref_words"], 4),
                        "ins": round(o["ins"] / o["ref_words"], 4),
                    }
                ),
                flush=True,
            )

    t_rows = [
        json.loads(line) for line in (Path("recordings/manifests/T.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    raw = json.loads(Path("data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    t_index = {uid: {**e, "shard": f"recorded_t/{e['shard']}"} for uid, e in raw.items()}
    sets = {
        "t600": (t_index, t_rows, Path("data/shards")),
        "fleurs_it": collect([("fleurs_it_test", 0)]),
        "fleurs_en": collect([("fleurs_en_test", 0)]),
        "cv_it_dev": collect([("cv_it_dev", 600)]),
    }

    print("--- transfer check", flush=True)
    cells = [(0.0, 0.0), (0.2, 0.0), (0.3, 1.6), (0.4, 2.0), (0.5, 2.4), (0.6, 2.8)]
    out = []
    for lm_weight, gamma in cells:
        record = {"lm_weight": lm_weight, "gamma": gamma}
        for name, (idx, rows, root) in sets.items():
            hyps = decode_set(
                model, pieces, rows, idx, root, cfg,
                lm=lm, lm_weight=lm_weight, gamma=gamma, beta=0.0,
            )
            o = score_rows(rows, hyps, bootstrap=0, seed=0)["overall"]
            record[name] = {
                "wer": round(o["wer"], 4),
                "eta": None if o["eta"] is None else round(o["eta"], 4),
                "del": round(o["del"] / o["ref_words"], 4),
                "ins": round(o["ins"] / o["ref_words"], 4),
            }
        out.append(record)
        print(json.dumps(record), flush=True)

    Path("runs/audit-2026-11-08/lm-gamma-transfer.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())