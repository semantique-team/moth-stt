"""Audit probe 2 (2026-11-08) — where does the per-token bonus stop paying, and does it hold
outside T? Selection is on dev (MLS-IT + LS-EN); FLEURS/CV are reported as an external check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import decode_set

from stt.eval import score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import load_config, load_sources

SETS = {
    "dev": [("mls_it_dev", 200), ("ls_en_dev", 200)],
    "fleurs_it_test": [("fleurs_it_test", 0)],
    "fleurs_en_test": [("fleurs_en_test", 0)],
    "cv_it_dev": [("cv_it_dev", 600)],
}


def collect(names):
    index, rows, root = {}, [], None
    for name, limit in names:
        idx, rs, rt = load_sources(
            [{"manifest": f"data/manifests/{name}.jsonl", "shards": f"data/shards/{name}"}]
        )
        if limit:
            rs = rs[:limit]
            keep = {r["id"] for r in rs}
            idx = {uid: entry for uid, entry in idx.items() if uid in keep}
        rows.extend(rs)
        root = rt
        index.update(idx)
    return index, rows, root


def main() -> int:
    cfg = load_config("runs/E-017/config.yaml")
    proc = load_tokenizer(cfg["tokenizer"])
    pieces = {i + 1: proc.id_to_piece(i) for i in range(proc.vocab_size())}
    model = MothSTT(vocab_size=1024, head_blocks=2)
    payload = torch.load("runs/E-017/checkpoints/ckpt_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    lm = BPETrigramLM.load("runs/E-008/lm_bpe3_aug.pkl.gz")

    cache = {name: collect(names) for name, names in SETS.items()}
    t_rows = [
        json.loads(line) for line in (Path("recordings/manifests/T.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    raw_index = json.loads(Path("data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    t_index = {
        uid: {**entry, "shard": f"recorded_t/{entry['shard']}"} for uid, entry in raw_index.items()
    }
    cache["recorded_t"] = (t_index, t_rows, Path("data/shards"))

    gammas = [1.0, 1.2, 1.5, 1.8, 2.2, 2.6]
    out = []
    for gamma in gammas:
        record = {"gamma": gamma}
        for name in ("dev", "recorded_t", "fleurs_it_test", "fleurs_en_test", "cv_it_dev"):
            idx, rs, rt = cache[name]
            hyps = decode_set(
                model, pieces, rs, idx, rt, cfg, lm=lm, lm_weight=0.2, gamma=gamma, beta=0.0
            )
            o = score_rows(rs, hyps, bootstrap=0, seed=0)["overall"]
            record[name] = {
                "wer": round(o["wer"], 4),
                "eta": None if o["eta"] is None else round(o["eta"], 4),
                "del": round(o["del"] / o["ref_words"], 4),
                "sub": round(o["sub"] / o["ref_words"], 4),
                "ins": round(o["ins"] / o["ref_words"], 4),
            }
        out.append(record)
        print(json.dumps(record), flush=True)

    Path("runs/audit-2026-11-08/token-bonus-ext.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())