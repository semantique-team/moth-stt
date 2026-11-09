"""Audit probe 3 — the missing ``gamma = 0`` baseline on the same subsets as probe 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import decode_set
from sweep_token_bonus_ext import SETS, collect

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

    record = {"gamma": 0.0}
    for name in ("fleurs_it_test", "fleurs_en_test", "cv_it_dev"):
        idx, rs, rt = collect(SETS[name])
        hyps = decode_set(
            model, pieces, rs, idx, rt, cfg, lm=lm, lm_weight=0.2, gamma=0.0, beta=0.0
        )
        o = score_rows(rs, hyps, bootstrap=0, seed=0)["overall"]
        record[name] = {
            "wer": round(o["wer"], 4),
            "eta": None if o["eta"] is None else round(o["eta"], 4),
            "del": round(o["del"] / o["ref_words"], 4),
            "sub": round(o["sub"] / o["ref_words"], 4),
            "ins": round(o["ins"] / o["ref_words"], 4),
        }
    print(json.dumps(record), flush=True)
    Path("runs/audit-2026-11-08/token-bonus-gamma0.json").write_text(
        json.dumps([record], indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())