"""Audit probe 8 — beam width, and whether the token bonus is stable across it."""

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
    index, rows, root = collect([("mls_it_dev", 200), ("ls_en_dev", 200)])

    ds = MelDataset(index, root, rows)
    durations = [float(r["duration"]) for r in rows]
    batches = plan_batches(durations, max_seconds=192, max_clips=32, seed=0, epoch=0)

    # Cache log_probs once; the decoders are then compared on the same acoustic output.
    cached = []
    with torch.inference_mode():
        for positions in batches:
            items = [ds[p] for p in positions]
            tensor, lengths, brows = _prepare_batch(items, "truncated")
            log_probs, out_lengths = model(tensor, lengths)
            for offset, row in enumerate(brows):
                cached.append((log_probs[offset], int(out_lengths[offset]), row["id"]))

    for beam_width, top_k, gamma, lm_weight in [
        (5, 8, 2.0, 0.4),
        (8, 8, 2.0, 0.4),
        (12, 10, 2.0, 0.4),
        (8, 16, 2.0, 0.4),
        (8, 8, 2.4, 0.5),
    ]:
        started = time.time()
        hyps = {}
        for log_probs, length, uid in cached:
            hyps[uid] = beam_with_bonus(
                log_probs, length, pieces,
                beam_width=beam_width, top_k=top_k,
                lm=lm, lm_weight=lm_weight, gamma=gamma, beta=0.0,
            )
        elapsed = time.time() - started
        o = score_rows(rows, hyps, bootstrap=0, seed=0)["overall"]
        print(
            json.dumps(
                {
                    "beam": beam_width,
                    "top_k": top_k,
                    "gamma": gamma,
                    "lm_weight": lm_weight,
                    "wer": round(o["wer"], 4),
                    "decode_s": round(elapsed, 1),
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())