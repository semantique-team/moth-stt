"""Audit probe 14 — the final candidate, everywhere, including the shipped int8 graph.

Candidate chosen on **dev only** (probe 13 extension): beam 12 / top-k 10 / ``lm_weight`` 0.6 /
``gamma`` 3.0, dev WER 0.2655 vs 0.2690 for the cheaper beam-8 cell. Everything below is a read of
that one frozen choice, not a search.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import beam_with_bonus
from sweep_token_bonus_ext import collect

from stt.data.dataset import MelDataset, plan_batches
from stt.decode import load_pieces
from stt.eval import _compare, score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.onnx_runner import OnnxMoth
from stt.tokenizer import load_tokenizer
from stt.train import _prepare_batch, load_config

FINAL = {"beam": 12, "top_k": 10, "lm_weight": 0.6, "gamma": 3.0}
SHIPPED = {"beam": 5, "top_k": 8, "lm_weight": 0.2, "gamma": 0.0}


def decode_shards(model, pieces, lm, idx, rows, root, params):
    ds = MelDataset(idx, root, rows)
    batches = plan_batches(
        [float(r["duration"]) for r in rows], max_seconds=192, max_clips=32, seed=0, epoch=0
    )
    hyps = {}
    with torch.inference_mode():
        for positions in batches:
            items = [ds[p] for p in positions]
            tensor, lengths, brows = _prepare_batch(items, "truncated")
            log_probs, out_lengths = model(tensor, lengths)
            for offset, row in enumerate(brows):
                hyps[row["id"]] = beam_with_bonus(
                    log_probs[offset], int(out_lengths[offset]), pieces,
                    beam_width=params["beam"], top_k=params["top_k"],
                    lm=lm, lm_weight=params["lm_weight"], gamma=params["gamma"], beta=0.0,
                )
    return hyps


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
    s4_rows = [r for r in t_rows if r.get("speaker") == "S4"]

    sets = {
        "t600": (t_index, t_rows, Path("data/shards")),
        "t-s3": (t_index, s3_rows, Path("data/shards")),
        "t-s1s2": (
            t_index,
            [r for r in t_rows if r.get("speaker") in ("S1", "S2")],
            Path("data/shards"),
        ),
        "fleurs_it": collect([("fleurs_it_test", 0)]),
        "fleurs_en": collect([("fleurs_en_test", 0)]),
        "cv_it_dev": collect([("cv_it_dev", 600)]),
    }
    if s4_rows:
        sets["t-s4"] = (t_index, s4_rows, Path("data/shards"))

    out = {}
    for name, (idx, rows, root) in sets.items():
        row = {}
        for label, params in (("shipped", SHIPPED), ("final", FINAL)):
            started = time.time()
            hyps = decode_shards(model, pieces, lm, idx, rows, root, params)
            scored = score_rows(rows, hyps, bootstrap=1000, seed=0)
            row[label] = scored
            o = scored["overall"]
            print(
                json.dumps(
                    {
                        "set": name,
                        "arm": label,
                        "wer": round(o["wer"], 4),
                        "eta": None if o["eta"] is None else round(o["eta"], 4),
                        "ci": [round(x, 4) for x in o["wer_ci"]],
                        "decode_s": round(time.time() - started, 1),
                        "audio_s": round(sum(r["duration"] for r in rows), 1),
                    }
                ),
                flush=True,
            )
            target = Path("runs/audit-2026-11-08/final") / label
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{name}.hyp.jsonl").write_text(
                "".join(
                    json.dumps({"id": k, "hyp": v}, ensure_ascii=False) + "\n"
                    for k, v in hyps.items()
                ),
                encoding="utf-8",
            )
        diff = _compare(row["final"], row["shipped"], bootstrap=1000, seed=0)
        out[name] = {
            "shipped": {
                "wer": row["shipped"]["overall"]["wer"],
                "eta": row["shipped"]["overall"]["eta"],
            },
            "final": {
                "wer": row["final"]["overall"]["wer"],
                "eta": row["final"]["overall"]["eta"],
            },
            "wer_diff": diff["wer_diff"],
            "wer_ci": diff["wer_diff_ci"],
            "wer_tie": diff["wer_tie"],
            "eta_diff": diff["eta_diff"],
            "eta_ci": diff["eta_diff_ci"],
            "eta_tie": diff["eta_tie"],
        }
        print(
            json.dumps(
                {
                    "set": name,
                    "final_minus_shipped": round(diff["wer_diff"], 4),
                    "wer_ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                    "wer_tie": diff["wer_tie"],
                    "eta_diff": None if diff["eta_diff"] is None else round(diff["eta_diff"], 4),
                    "eta_tie": diff["eta_tie"],
                }
            ),
            flush=True,
        )

    # --- the shipped int8 artifact, same two arms, 200 T clips
    print("--- int8", flush=True)
    index = json.loads(Path("data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    shard = np.load("data/shards/recorded_t/shard-000.npy", mmap_mode="r")
    subsets = {"S1": t_rows[:200], "S2b": t_rows[200:400], "S3b": s3_rows[:200]}
    onnx = OnnxMoth("release/model.int8.onnx", "release/vocab.json")
    int8_pieces = load_pieces("release/vocab.json")
    for subset_name, rows in subsets.items():
        row = {}
        for label, params in (("shipped", SHIPPED), ("final", FINAL)):
            hyps = {}
            for r in rows:
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
            row[label] = score_rows(rows, hyps, bootstrap=1000, seed=0)
        diff = _compare(row["final"], row["shipped"], bootstrap=1000, seed=0)
        print(
            json.dumps(
                {
                    "int8_set": subset_name,
                    "shipped_wer": round(row["shipped"]["overall"]["wer"], 4),
                    "final_wer": round(row["final"]["overall"]["wer"], 4),
                    "wer_diff": round(diff["wer_diff"], 4),
                    "ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                    "tie": diff["wer_tie"],
                }
            ),
            flush=True,
        )

    Path("runs/audit-2026-11-08/final-summary.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())