"""Does recording several sentences in one pass cost accuracy? (probe, 2026-11-09)

Motivated by a demo recording: 18.1 s of continuous speech containing ~6 sentences, which produced
a poor transcript. Two hypotheses were tested and only one survived on existing artifacts:

- **clip duration** — falsified. On FLEURS-IT, WER by duration quartile is 17.8 / 19.9 / 23.1 /
  20.0: not monotonic, and the longest quartile is better than the third.
- **English terms** — confirmed. On T-600, sentences with 0 English terms score 14.2% WER, with
  one term 26.7%, with two or more 25.3%. Presence of a term nearly doubles WER; the count does not
  matter further.

This probe tests the remaining question, which is specific to the demo: if several sentences are
spoken into one clip, does decoding the whole thing lose accuracy against decoding each sentence
separately? The boundaries here are the *true* ones, so the result is an upper bound on what an
automatic segmenter could recover.

    uv run python runs/audit-2026-11-09/long_clip_probe.py [--groups 40]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from stt.data.dataset import MelDataset
from stt.decode import beam_search
from stt.eval import _compare, score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import load_config, load_sources

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4}
GAP_FRAMES = 20  # 200 ms of silence between concatenated sentences


def main() -> int:
    parser = argparse.ArgumentParser(prog="long_clip_probe.py")
    parser.add_argument("--groups", type=int, default=40)
    parser.add_argument("--per-group", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(REPO / "runs/E-017/config.yaml")
    processor = load_tokenizer(REPO / cfg["tokenizer"])
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    model = MothSTT(vocab_size=1024, head_blocks=2)
    payload = torch.load(
        REPO / "runs/E-017/checkpoints/ckpt_best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model"])
    model.eval()
    lm = BPETrigramLM.load(REPO / "runs/E-008/lm_bpe3_aug.pkl.gz")

    index, rows, root = load_sources(
        [{"manifest": "recordings/manifests/T.jsonl", "shards": "data/shards/recorded_t"}]
    )
    dataset = MelDataset(index, root, rows)
    by_id = {r["id"]: position for position, r in enumerate(rows)}

    # S3 only: the held-out speaker, so this is not a speech we trained on.
    s3 = [r for r in rows if r.get("speaker") == "S3"]
    needed = args.groups * args.per_group
    s3 = s3[:needed]
    groups = [s3[i : i + args.per_group] for i in range(0, len(s3) - args.per_group + 1, args.per_group)]
    print(f"{len(groups)} groups of {args.per_group} clips (S3, held out)", flush=True)

    def decode_mel(mel: np.ndarray) -> str:
        tensor = torch.from_numpy(mel[None])
        with torch.inference_mode():
            log_probs, out_lengths = model(tensor, torch.tensor([mel.shape[1]]))
        return beam_search(
            log_probs[0],
            int(out_lengths[0]),
            pieces,
            beam_width=PARAMS["beam"],
            top_k=PARAMS["top_k"],
            lm=lm,
            lm_weight=PARAMS["lm_weight"],
            token_bonus=PARAMS["token_bonus"],
        )

    whole_hyps: dict[str, str] = {}
    split_hyps: dict[str, str] = {}
    whole_rows: list[dict] = []
    started = time.time()

    for group_index, group in enumerate(groups):
        mels = [dataset.raw_mel(by_id[r["id"]]) for r in group]
        concatenated = np.concatenate(
            [
                np.concatenate([mel, np.zeros((80, GAP_FRAMES), dtype=mel.dtype)], axis=1)
                for mel in mels
            ],
            axis=1,
        )
        uid = f"group-{group_index:03d}"
        dur = sum(float(r["duration"]) for r in group) + GAP_FRAMES * len(group) / 100
        norm = " ".join(r["norm"] for r in group)
        en_spans: list[list[int]] = []
        cursor = 0
        for r in group:
            offset = len(r["norm"].split())
            for start, end in r.get("en_spans") or []:
                en_spans.append([start + cursor, end + cursor])
            cursor += offset
        whole_rows.append({"id": uid, "norm": norm, "duration": dur, "en_spans": en_spans})

        whole_hyps[uid] = decode_mel(concatenated)
        split_hyps[uid] = " ".join(decode_mel(mel) for mel in mels)
        if (group_index + 1) % 10 == 0:
            print(f"  {group_index + 1}/{len(groups)} groups ({time.time() - started:.0f}s)", flush=True)

    whole = score_rows(whole_rows, whole_hyps, bootstrap=1000, seed=0)
    split = score_rows(whole_rows, split_hyps, bootstrap=1000, seed=0)
    diff = _compare(split, whole, bootstrap=1000, seed=0)

    summary = {
        "groups": len(groups),
        "clips_per_group": args.per_group,
        "mean_seconds": sum(r["duration"] for r in whole_rows) / len(whole_rows),
        "whole_wer": whole["overall"]["wer"],
        "split_wer": split["overall"]["wer"],
        "whole_eta": whole["overall"]["eta"],
        "split_eta": split["overall"]["eta"],
        "split_minus_whole": diff["wer_diff"],
        "ci": diff["wer_diff_ci"],
        "tie": diff["wer_tie"],
    }
    print(json.dumps(summary, indent=1), flush=True)
    (OUT / "long_clip_probe.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Also report what the same clips score when decoded one by one as themselves (the reference
    # point: this is T-S3's own number, 26.66).
    singles = score_rows(
        [r for group in groups for r in group],
        {r["id"]: decode_mel(dataset.raw_mel(by_id[r["id"]])) for group in groups for r in group},
        bootstrap=0,
        seed=0,
    )
    print(json.dumps({"single_clips_wer": singles["overall"]["wer"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())