"""E-033 — paired dev selection between the control and the InterCTC arm.

The trainer logs aggregate dev WER only, so the pre-registered paired comparison needs the
per-utterance hypotheses. This decodes both best checkpoints with the trainer's own greedy dev
path (same batches, same pieces, no LM) on the 3,939 dev clips, then applies
`macro_wer_bootstrap` within language and `_compare` on pooled WER.

    uv run python runs/E-033/select.py
"""

from __future__ import annotations

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
ARMS = {"control": OUT / "control", "interctc": OUT / "interctc"}
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


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = load_tokenizer(REPO / "data/tokenizer/moth_bpe.model")
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    index, rows, root = dev_rows()
    print(f"dev: {len(rows)} clips", flush=True)

    scored: dict[str, dict] = {}
    for name, arm_dir in ARMS.items():
        checkpoint = arm_dir / "checkpoints/ckpt_best.pt"
        started = time.time()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = MothSTT(vocab_size=1024, head_blocks=2)
        model.load_state_dict(payload["model"])
        model.to(device)
        dataset = MelDataset(index, root, rows)
        hyps = decode(model, dataset, pieces, device)
        out = arm_dir / "dev"
        out.mkdir(parents=True, exist_ok=True)
        (out / "dev.hyp.jsonl").write_text(
            "".join(json.dumps({"id": k, "hyp": v}, ensure_ascii=False) + "\n"
                    for k, v in hyps.items()),
            encoding="utf-8",
        )
        result = score_rows(rows, hyps, bootstrap=1000, seed=0)
        per_language = {}
        for language in ("it", "en"):
            subset = [row for row in rows if row.get("lang") == language]
            per_language[language] = score_rows(
                subset, {row["id"]: hyps[row["id"]] for row in subset}, bootstrap=0
            )["overall"]["wer"]
        result["per_language"] = per_language
        result["macro"] = (per_language["it"] + per_language["en"]) / 2
        result["wer"] = result["overall"]["wer"]
        result["checkpoint"] = str(checkpoint.relative_to(REPO))
        result["step"] = int(payload["step"])
        result["seconds"] = round(time.time() - started, 1)
        scored[name] = result
        (out / "dev.metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(name, json.dumps({k: result[k] for k in ("step", "macro", "wer")}),
              flush=True)

    control, arm = scored["control"], scored["interctc"]
    macro_delta = macro_wer_bootstrap(rows, arm, control, bootstrap=1000, seed=0)
    init = REPO / "runs/E-024/checkpoints/ckpt_best.pt"
    initial_note = (
        "the E-024 initial checkpoint is also a candidate; its greedy dev macro is recorded in "
        "E-028 (31.01 after fine-tuning, 30.81 before) and it was not re-decoded here"
        if init.is_file() else "E-024 checkpoint absent"
    )
    result = {
        "selection": "paired within-language macro WER on 3,939 dev clips, greedy, no LM",
        "control": {k: control[k] for k in ("step", "macro", "wer", "per_language")},
        "arm": {k: arm[k] for k in ("step", "macro", "wer", "per_language")},
        "arm_vs_control": {
            "pooled_wer": _compare(arm, control, bootstrap=1000, seed=0),
            "macro": macro_delta,
        },
        "advance_rule": "macro point lower by >= 0.05 pp AND paired CI upper < 0",
        "initial_checkpoint": initial_note,
    }
    delta_pp = macro_delta["point"] * 100
    rule = (delta_pp <= -0.05 and macro_delta["ci"][1] < 0)
    result["arm_advances"] = bool(rule)
    result["decision"] = (
        "advance to the product readout" if rule else
        "no dev benefit at weight 0.3; close the option for this recipe"
    )
    (OUT / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"macro_delta_pp": delta_pp, "ci_pp": [v * 100 for v in macro_delta["ci"]],
                      "advances": rule, "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
