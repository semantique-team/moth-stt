"""E-031 probe — measure browser-LM bytes for intermediate count-pruning arms.

Byte gate first (guide `docs/guides/mixed-dev-and-quality-experiments.md`, "Intermediate Trigram
LM"): a candidate is rejected before quality work if its exported browser binary does not leave
the package <= 20,000,000 bytes. The shipped LM is 5,116,464 B; with every other 0.8.0 asset
unchanged the theoretical maximum is 5,675,704 B (engineering target <= 5.5 MB).

This probe rebuilds the **same text** as the historical augmented builder
(`runs/E-008/build_lm_augmented.py`: CV/MLS/LS train transcripts + R-S1/R-S2/pool, never T/A)
and exports `save_browser` binaries for several `(min_bigram, min_trigram)` arms. It does not
train, decode or modify the release; it only answers "does it fit?".

    uv run python runs/E-031/probe_lm_bytes.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from stt.lm import BPETrigramLM
from stt.tokenizer import load_tokenizer

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "probe"
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
ARMS = [(2, 2), (1, 2), (2, 1), (1, 1)]


def load_sequences(processor) -> list[list[int]]:
    sequences: list[list[int]] = []
    for manifest in TRAIN_MANIFESTS:
        with (REPO / manifest).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sequences.append(
                    processor.encode(row.get("norm") or row["text"], out_type=int)
                )
    for script in TRAIN_SCRIPTS:
        with (REPO / script).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sequences.append(processor.encode(json.loads(line)["text"], out_type=int))
    return sequences


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    processor = load_tokenizer(REPO / "data/tokenizer/moth_bpe.model")
    sequences = load_sequences(processor)
    tokens = sum(len(sequence) for sequence in sequences)
    print(f"corpus: {len(sequences)} sentences, {tokens} tokens", flush=True)
    print(f"tokenize: {time.time() - started:.1f}s", flush=True)

    report = {"sentences": len(sequences), "tokens": tokens, "arms": {}}
    shipped_bytes = 5_116_464
    max_bytes = 5_675_704
    for min_bigram, min_trigram in ARMS:
        arm_started = time.time()
        lm = BPETrigramLM.from_sequences(
            sequences,
            min_bigram_count=min_bigram,
            min_trigram_count=min_trigram,
            backoff=0.4,
        )
        path = OUT / f"lm_{min_bigram}_{min_trigram}.bin"
        lm.save_browser(path)
        n_bigrams = sum(len(row) for row in lm._bigram_counts.values())
        n_trigrams = sum(len(row) for row in lm._trigram_counts.values())
        row = {
            "bytes": path.stat().st_size,
            "fits_20mb": path.stat().st_size <= max_bytes,
            "delta_vs_shipped": path.stat().st_size - shipped_bytes,
            "bigram_entries": n_bigrams,
            "trigram_entries": n_trigrams,
            "seconds": round(time.time() - arm_started, 1),
        }
        report["arms"][f"{min_bigram}_{min_trigram}"] = row
        print(
            f"({min_bigram},{min_trigram})  {row['bytes']:>10,} B  "
            f"bigrams {n_bigrams:>9,}  trigrams {n_trigrams:>10,}  "
            f"fits {row['fits_20mb']}  {row['seconds']:.0f}s",
            flush=True,
        )
    (OUT / "probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"total {time.time() - started:.0f}s -> {OUT / 'probe.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
