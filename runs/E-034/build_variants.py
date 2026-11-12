"""E-034 — build the LM coverage/weighting variants from the frozen E-031 corpus.

Same corpus, receipt, tokenizer, backoff and builder code path as E-031; the shipped LM is the
control and is not rebuilt. The browser binary is what the byte budget counts, the gzip pickle is
what the PyTorch dev sweep decodes with.

    uv run python runs/E-034/build_variants.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "runs/E-031"))

from build_lm import TERM_RE, sha256, weighted_sequences

from stt.lm import BPETrigramLM
from stt.tokenizer import load_tokenizer

CORPUS = REPO / "runs/E-031/frozen/lm-corpus.jsonl"
RECEIPT = REPO / "runs/E-031/frozen/lm-corpus.receipt.json"
TOKENIZER = REPO / "data/tokenizer/moth_bpe.model"
BUILD = OUT / "build"

VARIANTS = (
    {"name": "N1_2_1", "min_bigram": 2, "min_trigram": 1, "term_weight": 1},
    {"name": "N2_1_1", "min_bigram": 1, "min_trigram": 1, "term_weight": 1},
    {"name": "N3_1_2_terms5", "min_bigram": 1, "min_trigram": 2, "term_weight": 5},
)


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if sha256(CORPUS) != receipt["corpus_sha256"]:
        raise SystemExit("frozen corpus hash does not match its receipt")

    processor = load_tokenizer(TOKENIZER)
    rows = [
        json.loads(line)
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tagged = [(row["text"], processor.encode(row["text"], out_type=int)) for row in rows]
    term_rows = [text for text, _ in tagged if TERM_RE.search(text)]
    print(
        f"corpus {len(rows)} sentences, {sum(len(t) for _, t in tagged)} tokens, "
        f"term sentences {len(term_rows)}",
        flush=True,
    )

    report = {
        "corpus": str(CORPUS.relative_to(REPO)),
        "corpus_sha256": sha256(CORPUS),
        "corpus_receipt": str(RECEIPT.relative_to(REPO)),
        "tokenizer": str(TOKENIZER.relative_to(REPO)),
        "tokenizer_sha256": sha256(TOKENIZER),
        "builder_sha256": sha256(Path(__file__)),
        "term_sentences": len(term_rows),
        "variants": {},
    }
    for variant in VARIANTS:
        started = time.time()
        lm = BPETrigramLM.from_sequences(
            weighted_sequences(tagged, variant["term_weight"]),
            min_bigram_count=variant["min_bigram"],
            min_trigram_count=variant["min_trigram"],
            backoff=0.4,
        )
        pickle_path = BUILD / f"{variant['name']}.pkl.gz"
        binary_path = BUILD / f"{variant['name']}.bin"
        lm.save(pickle_path)
        lm.save_browser(binary_path)
        row = {
            **variant,
            "pickle_bytes": pickle_path.stat().st_size,
            "binary_bytes": binary_path.stat().st_size,
            "pickle_sha256": sha256(pickle_path),
            "binary_sha256": sha256(binary_path),
            "bigram_entries": sum(len(r) for r in lm._bigram_counts.values()),
            "trigram_entries": sum(len(r) for r in lm._trigram_counts.values()),
            "seconds": round(time.time() - started, 1),
        }
        report["variants"][variant["name"]] = row
        print(
            f"{variant['name']:16s} binary {row['binary_bytes']:>10,} B  "
            f"pickle {row['pickle_bytes']:>10,} B  {row['seconds']:.0f}s",
            flush=True,
        )
    (OUT / "build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
