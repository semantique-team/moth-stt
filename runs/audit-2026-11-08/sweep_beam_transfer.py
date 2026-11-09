"""Audit probe 9 — beam 8 vs 5 at the current optimum, on the held-out-ish sets.

Beam 8 is only interesting if it pays outside dev. Also reports the decode cost, since the browser
budget (RTFx >= 10 at 1 thread) is the constraint on raising it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import beam_with_bonus
from sweep_token_bonus_ext import collect

from stt.eval import _compare, score_rows
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

    t_rows = [
        json.loads(line) for line in (Path("recordings/manifests/T.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    raw = json.loads(Path("data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    t_index = {uid: {**e, "shard": f"recorded_t/{e['shard']}"} for uid, e in raw.items()}
    t_ctx = (t_index, t_rows, Path("data/shards"))
    s3 = [r for r in t_rows if r.get("speaker") == "S3"]
    s3_ctx = (t_index, s3, Path("data/shards"))
    fleurs = collect([("fleurs_it_test", 0)])

    def run(ctx, beam, top_k, gamma, lm_weight):
        idx, rows, root = ctx
        from stt.data.dataset import MelDataset, plan_batches

        ds = MelDataset(idx, root, rows)
        batches = plan_batches(
            [float(r["duration"]) for r in rows], max_seconds=192, max_clips=32, seed=0, epoch=0
        )
        hyps = {}
        started = time.time()
        with torch.inference_mode():
            for positions in batches:
                items = [ds[p] for p in positions]
                tensor, lengths, brows = _prepare_batch_(items, "truncated")
                log_probs, out_lengths = model(tensor, lengths)
                for offset, row in enumerate(brows):
                    hyps[row["id"]] = beam_with_bonus(
                        log_probs[offset], int(out_lengths[offset]), pieces,
                        beam_width=beam, top_k=top_k, lm=lm, lm_weight=lm_weight,
                        gamma=gamma, beta=0.0,
                    )
        return hyps, time.time() - started

    from stt.train import _prepare_batch as _prepare_batch_

    for name, ctx in (("t600", t_ctx), ("t-s3", s3_ctx), ("fleurs_it", fleurs)):
        results = {}
        for beam, top_k in ((5, 8), (8, 8)):
            hyps, seconds = run(ctx, beam, top_k, 2.4, 0.5)
            scored = score_rows(ctx[1], hyps, bootstrap=1000, seed=0)
            results[beam] = scored
            o = scored["overall"]
            print(
                json.dumps(
                    {
                        "set": name,
                        "beam": beam,
                        "wer": round(o["wer"], 4),
                        "eta": None if o["eta"] is None else round(o["eta"], 4),
                        "decode_s": round(seconds, 1),
                        "audio_s": round(sum(r["duration"] for r in ctx[1]), 1),
                    }
                ),
                flush=True,
            )
        diff = _compare(results[8], results[5], bootstrap=1000, seed=0)
        print(
            json.dumps(
                {
                    "set": name,
                    "beam8_minus_beam5": round(diff["wer_diff"], 4),
                    "ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                    "tie": diff["wer_tie"],
                    "eta_diff": None if diff["eta_diff"] is None else round(diff["eta_diff"], 4),
                    "eta_tie": diff["eta_tie"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())