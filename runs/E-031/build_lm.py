"""E-031 — build the frozen-corpus LM arms and export their browser binaries.

The corpus is `runs/E-031/frozen/lm-corpus.jsonl` (receipt committed next to it). The shipped LM
is the control and is not rebuilt here. Arms, pre-registered in `PLAN.md`:

* A1 `frozen_2_2`      — frozen corpus, bigrams >= 2, trigrams >= 2;
* A2 `frozen_1_2`      — frozen corpus, all bigrams, trigrams >= 2;
* A3 `frozen_1_2_terms`— A2 text with every sentence containing one of the 44 rare terms x3.

Every arm writes a gzip pickle (PyTorch decode) and the browser binary (byte gate + int8 decode).
Build hashes are recorded in `build.json`; no release file is modified.

    uv run python runs/E-031/build_lm.py
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable
from pathlib import Path

from stt.lm import BPETrigramLM
from stt.tokenizer import load_tokenizer

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CORPUS = OUT / "frozen/lm-corpus.jsonl"
RECEIPT = OUT / "frozen/lm-corpus.receipt.json"
TOKENIZER = REPO / "data/tokenizer/moth_bpe.model"
BUILD = OUT / "build"

#: The 44 terms from `runs/E-015/PLAN.md` that appear in no training text at ship time.
RARE_TERMS = (
    "airpods", "baskin", "bitbucket", "cadbury", "charger", "checkout", "chromecast",
    "delivery", "domain", "dyson", "exam", "folder", "hosting", "invision", "ipad", "kfc",
    "logout", "mondays", "oral", "popup", "prototype", "reset", "restart", "roomba",
    "scaleup", "screenshot", "selfie", "sharing", "shutdown", "shutterstock", "skills",
    "skyscanner", "smartwatch", "solving", "starbucks", "startup", "subway", "swipe",
    "tripadvisor", "twinings", "unfollow", "upload", "wallpaper", "website",
)
TERM_RE = re.compile(r"\b(" + "|".join(RARE_TERMS) + r")\b", re.IGNORECASE)

ARMS = (
    {"name": "frozen_2_2", "min_bigram": 2, "min_trigram": 2, "term_weight": 1},
    {"name": "frozen_1_2", "min_bigram": 1, "min_trigram": 2, "term_weight": 1},
    {"name": "frozen_1_2_terms", "min_bigram": 1, "min_trigram": 2, "term_weight": 3},
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_sequences(
    tagged: list[tuple[str, list[int]]], weight: int
) -> Iterable[list[int]]:
    for text, tokens in tagged:
        repeat = weight if weight > 1 and TERM_RE.search(text) else 1
        for _ in range(repeat):
            yield tokens


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
        "arms": {},
    }
    for arm in ARMS:
        started = time.time()
        weight = arm["term_weight"]

        lm = BPETrigramLM.from_sequences(
            weighted_sequences(tagged, weight),
            min_bigram_count=arm["min_bigram"],
            min_trigram_count=arm["min_trigram"],
            backoff=0.4,
        )
        pickle_path = BUILD / f"{arm['name']}.pkl.gz"
        binary_path = BUILD / f"{arm['name']}.bin"
        lm.save(pickle_path)
        lm.save_browser(binary_path)
        row = {
            **arm,
            "pickle_bytes": pickle_path.stat().st_size,
            "binary_bytes": binary_path.stat().st_size,
            "pickle_sha256": sha256(pickle_path),
            "binary_sha256": sha256(binary_path),
            "bigram_entries": sum(len(r) for r in lm._bigram_counts.values()),
            "trigram_entries": sum(len(r) for r in lm._trigram_counts.values()),
            "seconds": round(time.time() - started, 1),
        }
        report["arms"][arm["name"]] = row
        print(
            f"{arm['name']:18s} binary {row['binary_bytes']:>10,} B  "
            f"pickle {row['pickle_bytes']:>10,} B  {row['seconds']:.0f}s",
            flush=True,
        )
    (OUT / "build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
