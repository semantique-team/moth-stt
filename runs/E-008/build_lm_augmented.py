"""E-008 addendum — LM rebuilt with the training-side mixed sentences.

The E-008 LM reads CV/MLS/LS transcripts only, so the English terms that the recorded gold
teaches (`standup`, `backlog`, `deadline`, …) have **no prior**. On a real demo clip the LM
preferred `st meeting` over `standup meeting` by +15.9 log-prob.

This builder adds the **training-side** mixed scripts to the LM text: R-S1, R-S2 and the pool.
T and A are never read (T is the test set, A is a subset of T), so there is no leakage.

Usage: python runs/E-008/build_lm_augmented.py
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
TRAIN_SCRIPTS = [
    "recordings/scripts/R-S1.jsonl",
    "recordings/scripts/R-S2.jsonl",
    "recordings/scripts/pool.jsonl",
]


def iter_sequences(processor) -> Iterator[list[int]]:
    for manifest in TRAIN_MANIFESTS:
        with Path(manifest).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield processor.encode(row.get("norm") or row["text"], out_type=int)
    for script in TRAIN_SCRIPTS:
        with Path(script).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield processor.encode(json.loads(line)["text"], out_type=int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/E-008/lm_bpe3_aug.pkl.gz")
    parser.add_argument("--tokenizer", default="data/tokenizer/moth_bpe.model")
    parser.add_argument("--min-bigram-count", type=int, default=2)
    parser.add_argument("--min-trigram-count", type=int, default=2)
    parser.add_argument("--backoff", type=float, default=0.4)
    args = parser.parse_args()

    started = time.time()
    processor = load_tokenizer(args.tokenizer)
    lm = BPETrigramLM.from_sequences(
        iter_sequences(processor),
        min_bigram_count=args.min_bigram_count,
        min_trigram_count=args.min_trigram_count,
        backoff=args.backoff,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lm.save(out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
