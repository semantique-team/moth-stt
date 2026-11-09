"""Audit probe 13 — complete the beam-8 row on the two sets probe 9 did not cover."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import beam_with_bonus
from sweep_token_bonus_ext import collect

from stt.data.dataset import MelDataset, plan_batches
from stt.eval import score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import _prepare_batch, load_config


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

    for name, source in (("fleurs_en", [("fleurs_en_test", 0)]), ("cv_it_dev", [("cv_it_dev", 600)])):
        idx, rows, root = collect(source)
        ds = MelDataset(idx, root, rows)
        batches = plan_batches(
            [float(r["duration"]) for r in rows], max_seconds=192, max_clips=32, seed=0, epoch=0
        )
        hyps = {}
        started = time.time()
        with torch.inference_mode():
            for positions in batches:
                items = [ds[p] for p in positions]
                tensor, lengths, brows = _prepare_batch(items, "truncated")
                log_probs, out_lengths = model(tensor, lengths)
                for offset, row in enumerate(brows):
                    hyps[row["id"]] = beam_with_bonus(
                        log_probs[offset], int(out_lengths[offset]), pieces,
                        beam_width=8, top_k=8, lm=lm, lm_weight=0.5, gamma=2.4, beta=0.0,
                    )
        o = score_rows(rows, hyps, bootstrap=1000, seed=0)["overall"]
        print(
            json.dumps(
                {
                    "set": name,
                    "wer": round(o["wer"], 4),
                    "eta": None if o["eta"] is None else round(o["eta"], 4),
                    "ci": [round(x, 4) for x in o["wer_ci"]],
                    "decode_s": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())