"""E-035 — dev-only decode of the budgeted LM arm, control first.

Reuses the retained E-031 dev harness (same checkpoint, decoder, batches, scorer) so both arms in
this run share session and code. T, T-S3, FLEURS, the natural set and noise are not read.

    uv run python runs/E-035/dev_sweep.py --name budgeted-1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "runs/E-031"))

from audit import require_new
from dev_sweep import PARAMS, decode_arm, load_dev, write_jsonl

from stt.eval import _compare, macro_wer_bootstrap, score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer

CHECKPOINT = REPO / "runs/E-028/checkpoints/ckpt_best.pt"
CONFIG = REPO / "runs/E-028/config.yaml"
TOKENIZER = REPO / "data/tokenizer/moth_bpe.model"
REFERENCE = REPO / "runs/E-034/dev/dev-only/A3_shipped/dev.hyp.jsonl"
ARMS = {
    "A3_shipped": OUT / "build/A3_shipped.pkl.gz",
    "B1_budgeted": OUT / "build/B1_budgeted.pkl.gz",
}


def main() -> int:
    from stt.train import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="new output directory")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be one directory name")
    output = OUT / "dev" / args.name
    require_new(output)

    cfg = load_config(CONFIG)
    processor = load_tokenizer(TOKENIZER)
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    model = MothSTT(vocab_size=1024, head_blocks=2)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    index, rows, root = load_dev(cfg)
    print(
        f"full dev: {len(rows)} clips, {sum(float(r['duration']) for r in rows) / 3600:.2f} h",
        flush=True,
    )

    results: dict[str, dict] = {}
    control_metrics = None
    control_check = None
    for name, lm_path in ARMS.items():
        started = time.time()
        print(f"--- {name}: {lm_path.relative_to(REPO)}", flush=True)
        lm = BPETrigramLM.load(lm_path)
        hypotheses = decode_arm(model, pieces, lm, index, rows, root)
        out_dir = output / name
        write_jsonl(out_dir / "dev.hyp.jsonl", hypotheses)
        scored = score_rows(rows, hypotheses, bootstrap=1000, seed=0)
        scored["hyp"] = str((out_dir / "dev.hyp.jsonl").relative_to(REPO))
        (out_dir / "dev.metrics.json").write_text(
            json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        per_language = {}
        for language in ("it", "en"):
            subset = [row for row in rows if row.get("lang") == language]
            subset_hyp = {row["id"]: hypotheses[row["id"]] for row in subset}
            per_language[language] = score_rows(
                subset, subset_hyp, bootstrap=0, seed=0
            )["overall"]["wer"]
        row = {
            "lm": str(lm_path.relative_to(REPO)),
            "lm_bytes_pickle": lm_path.stat().st_size,
            "overall_wer": scored["overall"]["wer"],
            "overall_eta": scored["overall"]["eta"],
            "it": per_language["it"],
            "en": per_language["en"],
            "macro": (per_language["it"] + per_language["en"]) / 2,
            "macro_bootstrap": macro_wer_bootstrap(rows, scored),
            "seconds": round(time.time() - started, 1),
        }
        if control_metrics is not None:
            row["vs_control"] = _compare(scored, control_metrics, bootstrap=1000, seed=0)
            row["macro_vs_control"] = macro_wer_bootstrap(rows, scored, control_metrics)
        results[name] = row
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if control_metrics is None:
            control_metrics = scored
            retained = {
                item["id"]: item["hyp"]
                for item in (
                    json.loads(line)
                    for line in REFERENCE.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            differing = sum(
                1 for key, value in hypotheses.items() if retained.get(key) != value
            )
            control_check = {
                "reference": str(REFERENCE.relative_to(REPO)),
                "identical": differing == 0 and len(retained) == len(hypotheses),
                "differing": differing,
                "clips": len(hypotheses),
            }
            print(f"control check: {json.dumps(control_check)}", flush=True)

    (output / "sweep.json").write_text(
        json.dumps(
            {"params": PARAMS, "control_check": control_check, "arms": results},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    ordered = sorted(results.items(), key=lambda item: item[1]["macro"])
    print("ranking by dev macro:", [(name, round(row["macro"], 4)) for name, row in ordered])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
