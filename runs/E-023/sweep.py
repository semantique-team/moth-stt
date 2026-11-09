"""E-023 — re-tune the LM now that the decoder is calibrated.

Three phases, selection on **full dev** only (`mls_it_dev` + `ls_en_dev`, macro WER — the metric
E-008's tuning used). The gate sets are read once, at the end, for the winning arm:
`runs/E-023/gate.py`.

    uv run python runs/E-023/sweep.py --phase 1     # lm_weight x gamma on full dev
    uv run python runs/E-023/sweep.py --phase 2     # backoff
    uv run python runs/E-023/sweep.py --phase 3     # count pruning (rebuilds the LM)
    uv run python runs/E-023/sweep.py --phase all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from stt.data.dataset import MelDataset, plan_batches
from stt.decode import beam_search
from stt.eval import score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import _prepare_batch, load_config, load_sources

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
CHECKPOINT = REPO / "runs/E-017/checkpoints/ckpt_best.pt"
TOKENIZER = REPO / "data/tokenizer/moth_bpe.model"
CURRENT_LM = REPO / "runs/E-008/lm_bpe3_aug.pkl.gz"

#: Frozen with the plan: beam 12 / top-k 10 (E-022's arm).
BEAM, TOP_K = 12, 10

PHASE_1 = [(w, g) for w in (0.3, 0.4, 0.5, 0.6, 0.7) for g in (1.6, 2.0, 2.4, 2.8)]
PHASE_2 = [0.2, 0.3, 0.4, 0.5, 0.6]
PHASE_3 = [(1, 1), (2, 2), (3, 3), (2, 3)]


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


def build_lm(min_bigram: int, min_trigram: int, backoff: float, out: Path) -> Path:
    """Rebuild the LM with the same corpus, different knobs (`runs/E-008/build_lm_augmented.py`)."""
    if out.exists() and out.stat().st_size > 0:
        return out
    started = time.time()
    subprocess.run(
        [
            sys.executable,
            str(REPO / "runs/E-008/build_lm_augmented.py"),
            "--out",
            str(out),
            "--min-bigram-count",
            str(min_bigram),
            "--min-trigram-count",
            str(min_trigram),
            "--backoff",
            str(backoff),
        ],
        check=True,
        cwd=REPO,
    )
    print(f"  built {out.name} in {time.time() - started:.0f} s", flush=True)
    return out


def dev_wer(model, pieces, lm, index, rows, root, lm_weight: float, gamma: float) -> dict:
    """Macro WER over the two dev groups (IT and EN), the metric E-008's tuning used."""
    per_group = {}
    for language in ("it", "en"):
        subset = [r for r in rows if r.get("lang") == language]
        dataset = MelDataset(index, root, subset)
        batches = plan_batches(
            [float(r["duration"]) for r in subset],
            max_seconds=192,
            max_clips=32,
            seed=0,
            epoch=0,
        )
        hypotheses: dict[str, str] = {}
        with torch.inference_mode():
            for positions in batches:
                items = [dataset[p] for p in positions]
                tensor, lengths, batch_rows = _prepare_batch(items, "truncated")
                log_probs, out_lengths = model(tensor, lengths)
                for offset, row in enumerate(batch_rows):
                    hypotheses[row["id"]] = beam_search(
                        log_probs[offset],
                        int(out_lengths[offset]),
                        pieces,
                        beam_width=BEAM,
                        top_k=TOP_K,
                        lm=lm,
                        lm_weight=lm_weight,
                        token_bonus=gamma,
                    )
        per_group[language] = score_rows(subset, hypotheses, bootstrap=0, seed=0)["overall"]["wer"]
    macro = sum(per_group.values()) / len(per_group)
    return {"it": per_group["it"], "en": per_group["en"], "macro": macro}


def main() -> int:
    parser = argparse.ArgumentParser(prog="E-023/sweep.py")
    parser.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(REPO / "runs/E-017/config.yaml")
    processor = load_tokenizer(TOKENIZER)
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    model = MothSTT(vocab_size=1024, head_blocks=2)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    index, rows, root = load_dev(cfg)
    durations = [float(r["duration"]) for r in rows]
    audio_seconds = sum(durations)
    print(f"full dev: {len(rows)} clips, {audio_seconds / 3600:.2f} h", flush=True)

    results = {"phase1": [], "phase2": [], "phase3": []}

    if args.phase in ("1", "all"):
        lm = BPETrigramLM.load(CURRENT_LM)
        print(f"--- phase 1: lm_weight x gamma on full dev ({len(PHASE_1)} configs)", flush=True)
        for lm_weight, gamma in PHASE_1:
            started = time.time()
            metrics = dev_wer(model, pieces, lm, index, rows, root, lm_weight, gamma)
            metrics |= {"lm_weight": lm_weight, "gamma": gamma, "seconds": round(time.time() - started, 1)}
            results["phase1"].append(metrics)
            print(json.dumps(metrics), flush=True)

    if args.phase in ("2", "all"):
        best = min(results["phase1"], key=lambda r: r["macro"])
        print(f"--- phase 2: backoff at (w={best['lm_weight']}, g={best['gamma']})", flush=True)
        for backoff in PHASE_2:
            path = OUT / f"lm-backoff-{backoff}.pkl.gz"
            build_lm(2, 2, backoff, path)
            lm = BPETrigramLM.load(path)
            started = time.time()
            metrics = dev_wer(model, pieces, lm, index, rows, root, best["lm_weight"], best["gamma"])
            metrics |= {
                "backoff": backoff,
                "lm_weight": best["lm_weight"],
                "gamma": best["gamma"],
                "lm_bytes": path.stat().st_size,
                "seconds": round(time.time() - started, 1),
            }
            results["phase2"].append(metrics)
            print(json.dumps(metrics), flush=True)

    if args.phase in ("3", "all"):
        best1 = min(results["phase1"], key=lambda r: r["macro"])
        best2 = min(results["phase2"], key=lambda r: r["macro"]) if results["phase2"] else None
        backoff = best2["backoff"] if best2 else 0.4
        print(f"--- phase 3: pruning at backoff={backoff}", flush=True)
        for min_bigram, min_trigram in PHASE_3:
            tag = f"prune-{min_bigram}-{min_trigram}"
            path = OUT / f"lm-{tag}.pkl.gz"
            build_lm(min_bigram, min_trigram, backoff, path)
            lm = BPETrigramLM.load(path)
            started = time.time()
            metrics = dev_wer(
                model, pieces, lm, index, rows, root, best1["lm_weight"], best1["gamma"]
            )
            metrics |= {
                "min_bigram": min_bigram,
                "min_trigram": min_trigram,
                "lm_bytes": path.stat().st_size,
                "seconds": round(time.time() - started, 1),
            }
            results["phase3"].append(metrics)
            print(json.dumps(metrics), flush=True)

    (OUT / "sweep.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for phase, rows_ in results.items():
        if rows_:
            best = min(rows_, key=lambda r: r["macro"])
            print(f"best {phase}: {json.dumps(best)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())