"""E-023 — the gate read, run once for the winning arm.

Selection happened in `sweep.py` on full dev. This script reads the **gate sets** once for the
winner and once for the current shipped settings (the control), then writes the paired
differences. Run it only after the sweep has picked a winner:

    uv run python runs/E-023/gate.py --lm-weight W --gamma G \
        [--lm runs/E-023/lm-prune-3-3.pkl.gz] [--backoff 0.3]

The control is always the shipped 0.6.0 configuration, so `--vs` style comparisons are meaningful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stt.data.build_mel import load_audio_16k
from stt.data.dataset import MelDataset, plan_batches
from stt.decode import beam_search
from stt.eval import _compare, score_rows
from stt.features import log_mel
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import _prepare_batch, load_config, load_sources
from stt.vad import has_speech

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
CHECKPOINT = REPO / "runs/E-017/checkpoints/ckpt_best.pt"
SHIPPED = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4}


def decode_torch(model, pieces, lm, index, rows, root, params) -> dict[str, str]:
    dataset = MelDataset(index, root, rows)
    batches = plan_batches(
        [float(r["duration"]) for r in rows], max_seconds=192, max_clips=32, seed=0, epoch=0
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
                    beam_width=params["beam"],
                    top_k=params["top_k"],
                    lm=lm,
                    lm_weight=params["lm_weight"],
                    token_bonus=params["token_bonus"],
                )
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
    parser = argparse.ArgumentParser(prog="E-023/gate.py")
    parser.add_argument("--lm-weight", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument(
        "--lm",
        default="runs/E-008/lm_bpe3_aug.pkl.gz",
        help="the candidate LM; the control arm always uses the shipped one",
    )
    parser.add_argument("--arm", default="winner")
    args = parser.parse_args()

    params = {
        "beam": 12,
        "top_k": 10,
        "lm_weight": args.lm_weight,
        "token_bonus": args.gamma,
    }
    out_dir = OUT / "eval" / args.arm
    config = load_config(REPO / "runs/E-017/config.yaml")
    processor = load_tokenizer(REPO / config["tokenizer"])
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    #: Two LMs, deliberately: the candidate under test and the one 0.6.0 actually ships. Loading a
    #: single LM and reusing it for both arms is the bug this script shipped with on its first run —
    #: it made the two arms identical and the comparison meaningless.
    candidate_lm = BPETrigramLM.load(REPO / args.lm)
    shipped_lm = BPETrigramLM.load(REPO / "runs/E-008/lm_bpe3_aug.pkl.gz")

    model = MothSTT(vocab_size=1024, head_blocks=2)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    t_rows = [
        json.loads(line)
        for line in (REPO / "recordings/manifests/T.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    raw = json.loads((REPO / "data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    t_index = {uid: {**e, "shard": f"recorded_t/{e['shard']}"} for uid, e in raw.items()}
    root = REPO / "data/shards"

    sets = {
        "t600": (t_index, t_rows, root),
        "t-s3": (t_index, [r for r in t_rows if r.get("speaker") == "S3"], root),
        "fleurs_it_test": load_sources(
            [{"manifest": "data/manifests/fleurs_it_test.jsonl", "shards": "data/shards/fleurs_it_test"}]
        ),
        "fleurs_en_test": load_sources(
            [{"manifest": "data/manifests/fleurs_en_test.jsonl", "shards": "data/shards/fleurs_en_test"}]
        ),
        "cv_it_dev": load_sources(
            [{"manifest": "data/manifests/cv_it_dev.jsonl", "shards": "data/shards/cv_it_dev"}]
        ),
    }

    arms = {
        "shipped-0.6.0": {"lm": shipped_lm, **SHIPPED},
        args.arm: {"lm": candidate_lm, **params},
    }
    decoded: dict[str, dict[str, dict]] = {arm: {} for arm in arms}
    for arm, spec in arms.items():
        arm_lm = spec["lm"]
        arm_params = {k: v for k, v in spec.items() if k != "lm"}
        for name, (index, rows, set_root) in sets.items():
            hypotheses = decode_torch(model, pieces, arm_lm, index, rows, set_root, arm_params)
            decoded[arm][name] = score_rows(rows, hypotheses, bootstrap=1000, seed=0)
            write_jsonl(out_dir / arm / f"{name}.hyp.jsonl", hypotheses)
            (out_dir / arm / f"{name}.metrics.json").write_text(
                json.dumps(decoded[arm][name], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            overall = decoded[arm][name]["overall"]
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "set": name,
                        "wer": round(overall["wer"], 4),
                        "eta": None if overall["eta"] is None else round(overall["eta"], 4),
                    }
                ),
                flush=True,
            )

    # noise with the VAD guard
    index, noise_rows, _ = load_sources(
        [{"manifest": "data/manifests/noise.jsonl", "shards": "data/shards/noise"}]
    )
    audio_root = REPO / "data/raw/noise"
    for arm, spec in arms.items():
        arm_lm = spec["lm"]
        arm_params = {k: v for k, v in spec.items() if k != "lm"}
        hypotheses = {row["id"]: "" for row in noise_rows}
        for row in noise_rows:
            audio = load_audio_16k(audio_root / row["audio"])
            if not has_speech(audio):
                continue
            mel = log_mel(audio)
            with torch.inference_mode():
                log_probs, out_lengths = model(
                    torch.from_numpy(mel[None]), torch.tensor([mel.shape[1]])
                )
            hypotheses[row["id"]] = beam_search(
                log_probs[0],
                int(out_lengths[0]),
                pieces,
                beam_width=arm_params["beam"],
                top_k=arm_params["top_k"],
                lm=arm_lm,
                lm_weight=arm_params["lm_weight"],
                token_bonus=arm_params["token_bonus"],
            )
        decoded[arm]["noise"] = score_rows(noise_rows, hypotheses, bootstrap=0, seed=0)
        write_jsonl(out_dir / arm / "noise.hyp.jsonl", hypotheses)
        print(
            json.dumps(
                {"arm": arm, "set": "noise", "non_music_words": decoded[arm]["noise"]["noise"]["non_music_words"]}
            ),
            flush=True,
        )

    rows = []
    for name in ("t600", "t-s3", "fleurs_it_test", "fleurs_en_test", "cv_it_dev"):
        diff = _compare(
            decoded[args.arm][name], decoded["shipped-0.6.0"][name], bootstrap=1000, seed=0
        )
        record = {
            "set": name,
            "wer_diff": diff["wer_diff"],
            "wer_ci": diff["wer_diff_ci"],
            "wer_tie": diff["wer_tie"],
            "eta_diff": diff["eta_diff"],
            "eta_tie": diff["eta_tie"],
        }
        rows.append(record)
        print(
            json.dumps(
                {
                    "set": name,
                    "delta": round(record["wer_diff"], 4),
                    "ci": [round(x, 4) for x in record["wer_ci"]],
                    "tie": record["wer_tie"],
                }
            ),
            flush=True,
        )
    (OUT / "gate-diffs.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())