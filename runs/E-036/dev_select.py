"""E-036 — dev readout for the two conservative fine-tunes and the E-028 initial checkpoint.

Decodes all three checkpoints with the trainer's own greedy dev path (3,939 clips, no LM) in one
session and reports macro WER plus paired within-language bootstrap intervals. Dev is a sanity
guard here, not the selection metric: the pre-registered decision is the product readout in
`PLAN.md`.

    uv run python runs/E-036/dev_select.py --name select-1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from stt.data.dataset import MelDataset, plan_batches
from stt.decode import greedy_ids
from stt.eval import _compare, macro_wer_bootstrap, score_rows
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DEV = (("mls_it_dev", "it"), ("ls_en_dev", "en"))
ARMS = {
    "E-028_initial": REPO / "runs/E-028/checkpoints/ckpt_best.pt",
    "E-036-frozen": OUT / "frozen/checkpoints/ckpt_best.pt",
    "E-036-last2": OUT / "last2/checkpoints/ckpt_best.pt",
}
CHUNK = 192.0
CLIPS = 32


def dev_rows() -> tuple[dict, list[dict], Path]:
    from stt.train import load_sources

    index: dict = {}
    rows: list[dict] = []
    root = None
    for name, _ in DEV:
        part_index, part_rows, part_root = load_sources(
            [{"manifest": f"data/manifests/{name}.jsonl", "shards": f"data/shards/{name}"}]
        )
        index.update(part_index)
        rows.extend(part_rows)
        root = part_root
    return index, rows, root


def decode(model, dataset, pieces, device) -> dict[str, str]:
    from stt.train import _prepare_batch

    durations = [float(row["duration"]) for row in dataset.rows]
    batches = plan_batches(durations, max_seconds=CHUNK, max_clips=CLIPS, seed=0, epoch=0)
    hyps: dict[str, str] = {}
    model.eval()
    with torch.inference_mode():
        for positions in batches:
            items = [dataset[position] for position in positions]
            tensor, lengths, rows = _prepare_batch(items, "truncated")
            log_probs, out_lengths = model(tensor.to(device), lengths.to(device))
            ids = log_probs.argmax(dim=-1)
            for offset, row in enumerate(rows):
                hyps[row["id"]] = greedy_ids(
                    ids[offset, : int(out_lengths[offset])].tolist(), pieces
                )
    return hyps


def write_jsonl(path: Path, hypotheses: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
                for key, value in hypotheses.items()),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="new output directory")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be one directory name")
    output = OUT / "dev" / args.name
    if output.exists():
        raise SystemExit(f"{output} exists; use a new --name")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = load_tokenizer(REPO / "data/tokenizer/moth_bpe.model")
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    index, rows, root = dev_rows()
    print(f"dev: {len(rows)} clips", flush=True)

    scored: dict[str, dict] = {}
    for name, checkpoint in ARMS.items():
        if not checkpoint.is_file():
            raise SystemExit(f"{checkpoint} is missing; train the arm first")
        started = time.time()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = MothSTT(vocab_size=1024, head_blocks=2)
        model.load_state_dict(payload["model"])
        model.to(device)
        hyps = decode(model, MelDataset(index, root, rows), pieces, device)
        write_jsonl(output / name / "dev.hyp.jsonl", hyps)
        scored_result = score_rows(rows, hyps, bootstrap=1000, seed=0)
        per_language = {}
        for language in ("it", "en"):
            subset = [row for row in rows if row.get("lang") == language]
            per_language[language] = score_rows(
                subset, {row["id"]: hyps[row["id"]] for row in subset}, bootstrap=0
            )["overall"]["wer"]
        row = {
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "step": int(payload["step"]),
            "macro": (per_language["it"] + per_language["en"]) / 2,
            "it": per_language["it"],
            "en": per_language["en"],
            "wer": scored_result["overall"]["wer"],
            "eta": scored_result["overall"]["eta"],
            "seconds": round(time.time() - started, 1),
        }
        (output / name / "dev.metrics.json").write_text(
            json.dumps({**scored_result, **row}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        scored[name] = row
        print(name, json.dumps({"step": row["step"], "macro": row["macro"]}), flush=True)

    initial = scored["E-028_initial"]["macro"]
    comparisons = {}
    for name in ARMS:
        if name == "E-028_initial":
            continue
        arm_hyps = {
            item["id"]: item["hyp"]
            for item in (
                json.loads(line)
                for line in (output / name / "dev.hyp.jsonl").read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
        }
        base_hyps = {
            item["id"]: item["hyp"]
            for item in (
                json.loads(line)
                for line in (output / "E-028_initial" / "dev.hyp.jsonl")
                .read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        arm_scored = score_rows(rows, arm_hyps, bootstrap=1000, seed=0)
        base_scored = score_rows(rows, base_hyps, bootstrap=1000, seed=0)
        comparisons[name] = {
            "macro_delta_pp": (scored[name]["macro"] - initial) * 100,
            "macro_vs_initial": macro_wer_bootstrap(
                rows, arm_scored, base_scored, bootstrap=1000, seed=0
            ),
            "pooled_vs_initial": _compare(arm_scored, base_scored, bootstrap=1000, seed=0),
            "dev_guard": scored[name]["macro"] <= initial + 0.005,
        }

    selection = {
        "selection": "greedy dev, no LM, 3,939 clips, paired within-language macro WER",
        "initial": "E-028_initial",
        "arms": scored,
        "comparisons": comparisons,
        "dev_guard": "an arm with macro > initial + 0.50 pp is broken and gets no product readout",
        "decision": "product readout in PLAN.md; dev is a guard, not the selection metric",
    }
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparisons, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
