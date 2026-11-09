"""Audit probe 7 — final validation of the LM-weight × token-bonus recalibration.

Arms: the shipped decode (``lm_weight`` 0.2, no bonus) versus the two best dev cells. Paired
bootstrap (1,000 resamples, seed 0) on T-600, the S3 slice (the only speaker held out from
training), FLEURS IT/EN, CV-IT dev. Writes hypothesis + metrics files under ``eval/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import decode_set
from sweep_token_bonus_ext import collect

from stt.eval import _compare, score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import load_config

OUT = Path("runs/audit-2026-11-08/eval")
ARMS = {
    "shipped-0.2-g0": (0.2, 0.0),
    "dev-0.4-g2.0": (0.4, 2.0),
    "dev-0.5-g2.4": (0.5, 2.4),
}


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
    s3_rows = [r for r in t_rows if r.get("speaker") == "S3"]
    s3_ctx = (t_index, s3_rows, Path("data/shards"))

    sets = {
        "t600": t_ctx,
        "t-s3": s3_ctx,
        "fleurs_it": collect([("fleurs_it_test", 0)]),
        "fleurs_en": collect([("fleurs_en_test", 0)]),
        "cv_it_dev": collect([("cv_it_dev", 600)]),
    }

    results: dict[str, dict[str, dict]] = {}
    for arm, (lm_weight, gamma) in ARMS.items():
        results[arm] = {}
        for name, (idx, rows, root) in sets.items():
            hyps = decode_set(
                model, pieces, rows, idx, root, cfg,
                lm=lm, lm_weight=lm_weight, gamma=gamma, beta=0.0,
            )
            scored = score_rows(rows, hyps, bootstrap=1000, seed=0)
            results[arm][name] = scored
            target = OUT / arm
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{name}.hyp.jsonl").write_text(
                "".join(
                    json.dumps({"id": k, "hyp": v}, ensure_ascii=False) + "\n"
                    for k, v in hyps.items()
                ),
                encoding="utf-8",
            )
            (target / f"{name}.metrics.json").write_text(
                json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            o = scored["overall"]
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "set": name,
                        "wer": round(o["wer"], 4),
                        "eta": None if o["eta"] is None else round(o["eta"], 4),
                        "ci": [round(x, 4) for x in (o["wer_ci"] or [])],
                    }
                ),
                flush=True,
            )

    print("--- paired vs shipped", flush=True)
    base_arm = "shipped-0.2-g0"
    for arm in ("dev-0.4-g2.0", "dev-0.5-g2.4"):
        for name in sets:
            diff = _compare(results[arm][name], results[base_arm][name], bootstrap=1000, seed=0)
            print(
                json.dumps(
                    {
                        "set": name,
                        "arm": arm,
                        "wer_diff": round(diff["wer_diff"], 4),
                        "wer_ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                        "wer_tie": diff["wer_tie"],
                        "eta_diff": None if diff["eta_diff"] is None else round(diff["eta_diff"], 4),
                        "eta_ci": None
                        if diff["eta_diff_ci"] is None
                        else [round(x, 4) for x in diff["eta_diff_ci"]],
                        "eta_tie": diff["eta_tie"],
                    }
                ),
                flush=True,
            )

    Path("runs/audit-2026-11-08/final-arm-summary.json").write_text(
        json.dumps(
            {
                arm: {
                    name: {
                        "wer": r["overall"]["wer"],
                        "eta": r["overall"]["eta"],
                        "wer_ci": r["overall"]["wer_ci"],
                        "del": r["overall"]["del"] / r["overall"]["ref_words"],
                        "ins": r["overall"]["ins"] / r["overall"]["ref_words"],
                    }
                    for name, r in per_set.items()
                }
                for arm, per_set in results.items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())