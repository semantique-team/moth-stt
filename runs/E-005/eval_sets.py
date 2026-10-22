"""E-005 post-run evaluation: checkpoint averaging + hypotheses/metrics for each held-out set.

Inference only (no training). Run from the repository root:

    uv run python runs/E-005/eval_sets.py --checkpoint runs/E-005/checkpoints/ckpt_best.pt \
        --sets mls_it_dev,ls_en_dev --out runs/E-005/eval/best
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stt.data.dataset import MelDataset, plan_batches
from stt.decode import greedy
from stt.eval import score_rows
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import _prepare_batch, load_config, load_sources

SETS = {
    "mls_it_dev": ("data/manifests/mls_it_dev.jsonl", "data/shards/mls_it_dev"),
    "ls_en_dev": ("data/manifests/ls_en_dev.jsonl", "data/shards/ls_en_dev"),
    "fleurs_it_test": ("data/manifests/fleurs_it_test.jsonl", "data/shards/fleurs_it_test"),
    "fleurs_en_test": ("data/manifests/fleurs_en_test.jsonl", "data/shards/fleurs_en_test"),
    "cv_it_dev": ("data/manifests/cv_it_dev.jsonl", "data/shards/cv_it_dev"),
    "cv_it_test": ("data/manifests/cv_it_test.jsonl", "data/shards/cv_it_test"),
    "recorded_t": ("recordings/manifests/T.jsonl", "data/shards/recorded_t"),
}


def decode(ckpt: str, names: list[str], out_dir: Path, config_path: str) -> dict:
    config = load_config(config_path)
    batch = config["batch"]
    input_mode = config["train"]["input"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    processor = load_tokenizer(config["tokenizer"])
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}

    model = MothSTT(
        vocab_size=int(config["model"]["vocab_size"]),
        head_blocks=int(config["model"].get("head_blocks", 0)),
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    print(f"checkpoint {ckpt} (step {payload.get('step')}) on {device}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        manifest, shards = SETS[name]
        index, rows, root = load_sources([{"manifest": manifest, "shards": shards}])
        dataset = MelDataset(index, root, rows)
        batches = plan_batches(
            [float(row["duration"]) for row in rows],
            max_seconds=float(batch["max_seconds"]),
            max_clips=int(batch["max_clips"]),
            seed=0,
            epoch=0,
        )
        hyps: dict[str, str] = {}
        with torch.inference_mode():
            for positions in batches:
                tensor, lengths, brows = _prepare_batch(
                    [dataset[position] for position in positions], input_mode
                )
                log_probs, out_lengths = model(tensor.to(device), lengths.to(device))
                for offset, row in enumerate(brows):
                    hyps[row["id"]] = greedy(log_probs[offset], int(out_lengths[offset]), pieces)
        (out_dir / f"{name}.hyp.jsonl").write_text(
            "".join(json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
                    for key, value in hyps.items()),
            encoding="utf-8",
        )
        scored = score_rows(rows, hyps, bootstrap=1000, seed=0)
        metrics = {
            "checkpoint": ckpt,
            "set": name,
            "n_utterances": scored["n_utterances"],
            "overall": scored["overall"],
            "breakdown": scored["breakdown"],
            "noise": scored["noise"],
            "per_utterance": scored["per_utterance"],
        }
        (out_dir / f"{name}.metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        results[name] = scored["overall"]
        print(f"{name}: WER {scored['overall']['wer']:.4f} "
              f"CI {scored['overall'].get('wer_ci')}", flush=True)
        if device == "mps":
            torch.mps.empty_cache()
    return results


def average_checkpoints(paths: list[str], out: str) -> None:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    averaged = {}
    for key, first in payloads[0]["model"].items():
        tensors = [payload["model"][key] for payload in payloads]
        averaged[key] = (
            torch.stack([tensor.float() for tensor in tensors]).mean(0).to(first.dtype)
            if first.is_floating_point()
            else first
        )
    base = dict(payloads[-1])
    base["model"] = averaged
    base["averaged_from"] = paths
    torch.save(base, out)
    print(f"averaged {len(paths)} checkpoints -> {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="runs/E-005/config.yaml")
    parser.add_argument("--average", default=None, help="comma list of checkpoints to average first")
    args = parser.parse_args()

    if args.average:
        average_checkpoints([path.strip() for path in args.average.split(",")], args.checkpoint)
    decode(
        args.checkpoint,
        [name.strip() for name in args.sets.split(",")],
        Path(args.out),
        args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
