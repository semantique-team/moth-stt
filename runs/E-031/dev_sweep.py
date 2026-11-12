"""E-031 — dev selection on the full IT/EN dev sets, one decode per LM arm.

Selection only: `mls_it_dev` (1,248) + `ls_en_dev` (2,691), macro WER as in E-008/E-023, decoder
frozen at beam 12 / top-k 10 / LM 0.55 / token bonus 2.7 (E-030). T, T-S3 and FLEURS are not read
here; the advancing arm gets one readout in `gate.py`.

    uv run python runs/E-031/dev_sweep.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from stt.data.dataset import MelDataset, plan_batches
from stt.decode import beam_search
from stt.eval import _compare, score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import _prepare_batch, load_config, load_sources

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CHECKPOINT = REPO / "runs/E-028/checkpoints/ckpt_best.pt"
CONFIG = REPO / "runs/E-028/config.yaml"
TOKENIZER = REPO / "data/tokenizer/moth_bpe.model"
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}
ARMS = {
    "A0_shipped": REPO / "runs/E-008/lm_bpe3_aug.pkl.gz",
    "A1_frozen_2_2": OUT / "build/frozen_2_2.pkl.gz",
    "A2_frozen_1_2": OUT / "build/frozen_1_2.pkl.gz",
    "A3_frozen_1_2_terms": OUT / "build/frozen_1_2_terms.pkl.gz",
}


def load_dev(cfg) -> tuple[dict, list[dict], Path]:
    index: dict = {}
    rows: list[dict] = []
    root = None
    for name in ("mls_it_dev", "ls_en_dev"):
        part_index, part_rows, part_root = load_sources(
            [{"manifest": f"data/manifests/{name}.jsonl", "shards": f"data/shards/{name}"}]
        )
        index.update(part_index)
        rows.extend(part_rows)
        root = part_root
    return index, rows, root


def decode_arm(model, pieces, lm, index, rows, root) -> dict[str, str]:
    dataset = MelDataset(index, root, rows)
    batches = plan_batches(
        [float(row["duration"]) for row in rows],
        max_seconds=192,
        max_clips=32,
        seed=0,
        epoch=0,
    )
    hypotheses: dict[str, str] = {}
    with torch.inference_mode():
        for number, positions in enumerate(batches, start=1):
            items = [dataset[position] for position in positions]
            tensor, lengths, batch_rows = _prepare_batch(items, "truncated")
            log_probs, out_lengths = model(tensor, lengths)
            for offset, row in enumerate(batch_rows):
                hypotheses[row["id"]] = beam_search(
                    log_probs[offset],
                    int(out_lengths[offset]),
                    pieces,
                    beam_width=PARAMS["beam"],
                    top_k=PARAMS["top_k"],
                    lm=lm,
                    lm_weight=PARAMS["lm_weight"],
                    token_bonus=PARAMS["token_bonus"],
                )
            if number % 50 == 0:
                print(f"    batch {number}/{len(batches)}", flush=True)
    return hypotheses


def write_jsonl(path: Path, hypotheses: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
            for key, value in hypotheses.items()
        ),
        encoding="utf-8",
    )


def main() -> int:
    cfg = load_config(CONFIG)
    processor = load_tokenizer(TOKENIZER)
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    model = MothSTT(vocab_size=1024, head_blocks=2)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    index, rows, root = load_dev(cfg)
    print(f"full dev: {len(rows)} clips, {sum(float(r['duration']) for r in rows) / 3600:.2f} h",
          flush=True)

    results: dict[str, dict] = {}
    base_metrics = None
    for name, lm_path in ARMS.items():
        started = time.time()
        print(f"--- {name}: {lm_path.relative_to(REPO)}", flush=True)
        lm = BPETrigramLM.load(lm_path)
        hypotheses = decode_arm(model, pieces, lm, index, rows, root)
        out_dir = OUT / "dev" / name
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
            "seconds": round(time.time() - started, 1),
        }
        if base_metrics is not None:
            row["vs_A0"] = _compare(scored, base_metrics, bootstrap=1000, seed=0)
        results[name] = row
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if base_metrics is None:
            base_metrics = scored

    (OUT / "sweep.json").write_text(
        json.dumps({"params": PARAMS, "arms": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ordered = sorted(results.items(), key=lambda item: item[1]["macro"])
    print("ranking by dev macro:", [(name, round(row["macro"], 4)) for name, row in ordered])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
