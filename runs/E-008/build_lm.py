"""Build the E-008 BPE trigram LM from the training transcripts only.

The E-006 checkpoint supplies the acoustic model; this script never reads dev/test audio or
transcripts. Output is a gzip pickle consumed by ``--lm`` in ``runs/E-005/eval_sets.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

from stt.lm import BPETrigramLM
from stt.tokenizer import load_tokenizer

TRAIN_MANIFESTS = [
    "data/manifests/cv_it_train.jsonl",
    "data/manifests/mls_it_train.jsonl",
    "data/manifests/ls_en_train.jsonl",
]


def iter_sequences(processor, limit: int = 0) -> Iterator[list[int]]:
    seen = 0
    for manifest in TRAIN_MANIFESTS:
        with Path(manifest).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = row.get("norm") or row["text"]
                yield processor.encode(text, out_type=int)
                seen += 1
                if limit and seen >= limit:
                    return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/E-008/lm_bpe3.pkl.gz")
    parser.add_argument("--tokenizer", default="data/tokenizer/moth_bpe.model")
    parser.add_argument("--min-bigram-count", type=int, default=2)
    parser.add_argument("--min-trigram-count", type=int, default=2)
    parser.add_argument("--backoff", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=0, help="sentences (smoke only)")
    args = parser.parse_args()

    started = time.time()
    processor = load_tokenizer(args.tokenizer)
    lm = BPETrigramLM.from_sequences(
        iter_sequences(processor, limit=args.limit),
        min_bigram_count=args.min_bigram_count,
        min_trigram_count=args.min_trigram_count,
        backoff=args.backoff,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lm.save(out)
    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "out": str(out),
                "size_mb": round(out.stat().st_size / 1e6, 2),
                "seconds": round(elapsed, 1),
                **lm.summary(),
                "limit": args.limit,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
