"""E-020 — the WER cost of the shipped int8 model, measured end to end.

The release ships ``model.int8.onnx``, but its WER had never been measured on our own test
sets: ``runs/E-017/release-0.5.0.json`` carries only top-1 logit agreement and the count of
identical transcripts on 100 FLEURS-IT clips, neither of which is a WER. This script answers
the question with the same manifest, the same text rule and the same decoders as every other
number in the repo.

Decode-only: nothing is trained, nothing is tuned. The checkpoint is the one the 0.5.0 release
was built from, so a difference against ``runs/E-017/eval/`` is the quantization and nothing else.

    uv run python runs/E-020/eval_onnx.py --model release/model.int8.onnx \
        --vocab release/vocab.json --sets recorded_t --out runs/E-020/int8
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from stt.data.build_mel import read_clip
from stt.data.dataset import plan_batches
from stt.eval import score_rows
from stt.lm import BPETrigramLM
from stt.onnx_runner import OnnxMoth
from stt.train import load_sources

#: ``name -> (manifest, shards)`` for the sets the release is judged on.
SETS = {
    "recorded_t": ("recordings/manifests/T.jsonl", "data/shards/recorded_t"),
    "fleurs_it_test": ("data/manifests/fleurs_it_test.jsonl", "data/shards/fleurs_it_test"),
    "fleurs_en_test": ("data/manifests/fleurs_en_test.jsonl", "data/shards/fleurs_en_test"),
}

#: Batch shape for ordering only: ONNX runs one clip at a time, but the clip *order* must match
#: the PyTorch runs so a paired bootstrap compares the same clips in the same sequence.
BATCH_SECONDS = 192
BATCH_CLIPS = 32


def decode_set(
    model: OnnxMoth,
    set_name: str,
    out_dir: Path,
    *,
    decoder: str,
    beam_width: int,
    top_k: int,
    lm: BPETrigramLM | None,
    lm_weight: float,
    limit: int,
) -> dict:
    manifest, shards = SETS[set_name]
    index, rows, root = load_sources([{"manifest": manifest, "shards": shards}])
    if limit and limit < len(rows):
        rows = random.Random(0).sample(rows, limit)
        rows.sort(key=lambda row: row["id"])
    all_rows = rows

    # Same batching plan as runs/E-005/eval_sets.py, so the clip order matches the PyTorch runs.
    batches = plan_batches(
        [float(row["duration"]) for row in rows],
        max_seconds=BATCH_SECONDS,
        max_clips=BATCH_CLIPS,
        seed=0,
        epoch=0,
    )

    hyps: dict[str, str] = {}
    cache: dict[str, np.ndarray] = {}
    decode_seconds = 0.0
    audio_seconds = 0.0
    started = time.perf_counter()
    for positions in batches:
        for position in positions:
            row = rows[position]
            mel = read_clip(index, root, row["id"], cache)
            text, seconds = model.transcribe(
                mel,
                decoder=decoder,
                beam_width=beam_width,
                top_k=top_k,
                lm=lm,
                lm_weight=lm_weight,
            )
            hyps[row["id"]] = text
            decode_seconds += seconds
            audio_seconds += float(row["duration"])
    wall_seconds = time.perf_counter() - started

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{set_name}.hyp.jsonl").write_text(
        "".join(
            json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
            for key, value in hyps.items()
        ),
        encoding="utf-8",
    )
    scored = score_rows(all_rows, hyps, bootstrap=1000, seed=0)
    metrics = {
        "model": str(model.model_path),
        "provider": model.provider,
        "set": set_name,
        "n_utterances": scored["n_utterances"],
        "decode": {
            "method": decoder,
            "beam_width": beam_width if decoder == "beam" else None,
            "top_k": top_k if decoder == "beam" else None,
            "lm_weight": lm_weight if decoder == "beam" else None,
            "limit": limit,
        },
        "timing": {
            "audio_seconds": round(audio_seconds, 2),
            "decode_seconds": round(decode_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "rtfx": round(audio_seconds / wall_seconds, 1) if wall_seconds else None,
        },
        "overall": scored["overall"],
        "per_utterance": scored["per_utterance"],
    }
    (out_dir / f"{set_name}.metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{set_name}: WER {scored['overall']['wer']:.4f} "
        f"CI {scored['overall'].get('wer_ci')}  ETA {scored['overall'].get('eta')}  "
        f"RTFx {metrics['timing']['rtfx']}",
        flush=True,
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="release/model.int8.onnx")
    parser.add_argument("--vocab", default="release/vocab.json")
    parser.add_argument("--sets", default="recorded_t")
    parser.add_argument("--out", required=True)
    parser.add_argument("--decoder", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lm", default=None)
    parser.add_argument("--lm-weight", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    if args.decoder == "beam" and args.lm and args.lm_weight == 0.0:
        parser.error("--lm needs --lm-weight > 0")

    model = OnnxMoth(args.model, args.vocab)
    print(
        f"model {args.model}  provider {model.provider}  vocab {model.vocab_size} ids", flush=True
    )
    lm = BPETrigramLM.load(args.lm) if args.lm else None
    if lm is not None:
        print(f"lm {args.lm} (weight {args.lm_weight})", flush=True)

    out_dir = Path(args.out)
    for name in (value.strip() for value in args.sets.split(",") if value.strip()):
        if name not in SETS:
            parser.error(f"unknown set {name!r} (known: {', '.join(sorted(SETS))})")
        decode_set(
            model,
            name,
            out_dir,
            decoder=args.decoder,
            beam_width=args.beam_width,
            top_k=args.top_k,
            lm=lm,
            lm_weight=args.lm_weight,
            limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
