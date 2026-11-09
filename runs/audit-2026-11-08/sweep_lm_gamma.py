"""Audit probe 5 — 2-D sweep on the two knobs that currently do one job each way.

``lm_weight`` (shallow fusion) also acts as a per-token length penalty, because every emitted
token pays ``lm_weight * log P(token)``. The token bonus ``gamma`` pays it back. Selection is on
the dev subset only (400 clips: ``mls_it_dev`` + ``ls_en_dev``), never on T.
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
    index, rows, root = collect([("mls_it_dev", 200), ("ls_en_dev", 200)])

    grid = []
    for lm_weight in (0.0, 0.1, 0.2, 0.3, 0.4):
        for gamma in (0.0, 0.4, 0.8, 1.2, 1.6, 2.0):
            hyps = decode_set(
                model, pieces, rows, index, root, cfg,
                lm=lm, lm_weight=lm_weight, gamma=gamma, beta=0.0,
            )
            o = score_rows(rows, hyps, bootstrap=0, seed=0)["overall"]
            row = {
                "lm_weight": lm_weight,
                "gamma": gamma,
                "wer": round(o["wer"], 4),
                "del": round(o["del"] / o["ref_words"], 4),
                "sub": round(o["sub"] / o["ref_words"], 4),
                "ins": round(o["ins"] / o["ref_words"], 4),
            }
            grid.append(row)
            print(json.dumps(row), flush=True)

    out = Path("runs/audit-2026-11-08/lm-gamma-grid.json")
    out.write_text(json.dumps(grid, indent=2) + "\n", encoding="utf-8")
    best = min(grid, key=lambda r: r["wer"])
    print("best on dev:", json.dumps(best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())