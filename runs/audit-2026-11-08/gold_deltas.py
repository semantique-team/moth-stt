"""Audit probe 18 — were E-016/E-017's gold gains partly a decode artefact?

E-014 (no gold) → E-016 (S1 gold) → E-017 (S1+S2 gold) were all measured with the shipped decode.
If the deletion bias was masked by gold data (more tokens seen for those words), the gold deltas
would shrink once the decoder is fixed — which would change how much E-018 is worth.

Decodes each checkpoint at **both** decodes on T-600 and reports the paired deltas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from final_candidate import SHIPPED, decode_shards

from stt.eval import _compare, score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import load_config

FIXED = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "gamma": 2.4}
CHECKPOINTS = {
    "E-014": "runs/E-014/checkpoints/ckpt_best.pt",
    "E-016": "runs/E-016/checkpoints/ckpt_best.pt",
    "E-017": "runs/E-017/checkpoints/ckpt_best.pt",
}


def main() -> int:
    cfg = load_config("runs/E-017/config.yaml")
    proc = load_tokenizer(cfg["tokenizer"])
    pieces = {i + 1: proc.id_to_piece(i) for i in range(proc.vocab_size())}
    lm = BPETrigramLM.load("runs/E-008/lm_bpe3_aug.pkl.gz")

    rows = [
        json.loads(line)
        for line in Path("recordings/manifests/T.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    raw = json.loads(Path("data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    index = {uid: {**e, "shard": f"recorded_t/{e['shard']}"} for uid, e in raw.items()}
    root = Path("data/shards")

    results: dict[str, dict[str, dict]] = {}
    for name, path in CHECKPOINTS.items():
        model = MothSTT(vocab_size=1024, head_blocks=2)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        results[name] = {}
        for label, params in (("shipped", SHIPPED), ("fixed", FIXED)):
            hyps = decode_shards(model, pieces, lm, index, rows, root, params)
            scored = score_rows(rows, hyps, bootstrap=1000, seed=0)
            results[name][label] = scored
            o = scored["overall"]
            print(
                json.dumps(
                    {
                        "checkpoint": name,
                        "decode": label,
                        "wer": round(o["wer"], 4),
                        "eta": round(o["eta"], 4),
                    }
                ),
                flush=True,
            )
        del model

    print("--- gold deltas, both decodes", flush=True)
    for older, newer in (("E-014", "E-016"), ("E-016", "E-017"), ("E-014", "E-017")):
        for label in ("shipped", "fixed"):
            diff = _compare(results[newer][label], results[older][label], bootstrap=1000, seed=0)
            print(
                json.dumps(
                    {
                        "delta": f"{older}->{newer}",
                        "decode": label,
                        "wer_diff": round(diff["wer_diff"], 4),
                        "wer_ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                        "eta_diff": round(diff["eta_diff"], 4),
                        "eta_ci": [round(x, 4) for x in diff["eta_diff_ci"]],
                    }
                ),
                flush=True,
            )

    Path("runs/audit-2026-11-08/gold-deltas.json").write_text(
        json.dumps(
            {
                name: {
                    label: {
                        "wer": r["overall"]["wer"],
                        "eta": r["overall"]["eta"],
                    }
                    for label, r in arms.items()
                }
                for name, arms in results.items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())