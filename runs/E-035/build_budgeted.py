"""E-035 — build the budgeted LM arm from the frozen E-031 corpus.

Keeps the shipped `(1,2)` + term-×3 LM and adds single-count trigrams ranked by the occurrence
count of their preceding bigram until the browser binary reaches the 20 MB package budget. The
control is rebuilt from the same code path and must reproduce the shipped binary byte for byte;
the selection rule is frozen in `PLAN.md`.

    uv run python runs/E-035/build_budgeted.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "runs/E-031"))

from build_lm import sha256, weighted_sequences

from stt.lm import BPETrigramLM
from stt.tokenizer import load_tokenizer

CORPUS = REPO / "runs/E-031/frozen/lm-corpus.jsonl"
RECEIPT = REPO / "runs/E-031/frozen/lm-corpus.receipt.json"
TOKENIZER = REPO / "data/tokenizer/moth_bpe.model"
BUILD = OUT / "build"

TERM_WEIGHT = 3
BACKOFF = 0.4
PACKAGE_LIMIT = 20_000_000
SHIPPED_PACKAGE = 19_706_289
SHIPPED_BYTES = 5_381_992
SHIPPED_SHA256 = "3be1177b0ca9f18343efb8d00f0d96365226595c9de32d992720aa4e0b0f16f1"
NON_LM_BYTES = SHIPPED_PACKAGE - SHIPPED_BYTES
BINARY_LIMIT = PACKAGE_LIMIT - NON_LM_BYTES


def full_counts(tagged: list[tuple[str, list[int]]]) -> BPETrigramLM:
    """Unpruned counts and the canonical unigram/bigram tables of the shipped recipe."""
    return BPETrigramLM.from_sequences(
        weighted_sequences(tagged, TERM_WEIGHT),
        min_bigram_count=1,
        min_trigram_count=1,
        backoff=BACKOFF,
    )


def shipped_trigrams(full: BPETrigramLM) -> dict[tuple[int, int], dict[int, int]]:
    """The shipped `(1,2)` trigram set: counts >= 2, empty contexts dropped."""
    kept = {
        context: {token: count for token, count in row.items() if count >= 2}
        for context, row in full._trigram_counts.items()
    }
    return {context: row for context, row in kept.items() if row}


def rank_single_count(
    full: BPETrigramLM, base: dict[tuple[int, int], dict[int, int]]
) -> list[tuple[int, int, int, int, bool]]:
    """Rank count-1 trigrams by context frequency, then by ids for reproducibility."""
    ranked: list[tuple[int, int, int, int, bool]] = []
    for context, row in full._trigram_counts.items():
        context_count = sum(row.values())
        has_base = context in base
        for token, count in row.items():
            if count == 1:
                ranked.append((-context_count, context[0], context[1], token, has_base))
    ranked.sort()
    return ranked


def select_entries(
    ranked: list[tuple[int, int, int, int, bool]], base_bytes: int
) -> tuple[list[tuple[tuple[int, int], int]], int]:
    """Greedily fill the byte budget; a new context costs its key and offset as well."""
    size = base_bytes
    selected: list[tuple[tuple[int, int], int]] = []
    opened: set[tuple[int, int]] = set()
    for neg_count, prev2, prev1, token, has_base in ranked:
        context = (prev2, prev1)
        creates_context = not has_base and context not in opened
        cost = 4 + (8 if creates_context else 0)
        if size + cost > BINARY_LIMIT:
            continue
        selected.append((context, token))
        if creates_context:
            opened.add(context)
        size += cost
        if BINARY_LIMIT - size < 4:
            break
    return selected, size


def merge(
    base: dict[tuple[int, int], dict[int, int]],
    selected: list[tuple[tuple[int, int], int]],
) -> dict[tuple[int, int], dict[int, int]]:
    trigrams = {context: dict(row) for context, row in base.items()}
    for context, token in selected:
        trigrams.setdefault(context, {})[token] = 1
    return trigrams


def build(full: BPETrigramLM, trigrams: dict[tuple[int, int], dict[int, int]]) -> BPETrigramLM:
    return BPETrigramLM(
        full._unigram_counts,
        full._bigram_counts,
        trigrams,
        backoff=BACKOFF,
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
    print(
        f"corpus {len(rows)} sentences, {sum(len(t) for _, t in tagged)} tokens, "
        f"term weight x{TERM_WEIGHT}",
        flush=True,
    )

    started = time.time()
    full = full_counts(tagged)
    base_trigrams = shipped_trigrams(full)
    base_lm = build(full, base_trigrams)
    base_pickle = BUILD / "A3_shipped.pkl.gz"
    base_binary = BUILD / "A3_shipped.bin"
    base_lm.save(base_pickle)
    base_lm.save_browser(base_binary)
    base_bytes = base_binary.stat().st_size
    base_sha = sha256(base_binary)
    if (base_bytes, base_sha) != (SHIPPED_BYTES, SHIPPED_SHA256):
        raise SystemExit(
            "control does not reproduce the shipped LM: "
            f"{base_bytes} B sha256 {base_sha}; expected {SHIPPED_BYTES} B {SHIPPED_SHA256}"
        )
    print(f"control reproduces the shipped binary: {base_bytes:,} B {base_sha[:12]}…", flush=True)

    ranked = rank_single_count(full, base_trigrams)
    selected, predicted = select_entries(ranked, base_bytes)
    while True:
        opened = {context for context, _ in selected if context not in base_trigrams}
        predicted = base_bytes + 4 * len(selected) + 8 * len(opened)
        arm_lm = build(full, merge(base_trigrams, selected))
        arm_pickle = BUILD / "B1_budgeted.pkl.gz"
        arm_binary = BUILD / "B1_budgeted.bin"
        arm_lm.save(arm_pickle)
        arm_lm.save_browser(arm_binary)
        arm_bytes = arm_binary.stat().st_size
        package = NON_LM_BYTES + arm_bytes
        if arm_bytes <= BINARY_LIMIT and package <= PACKAGE_LIMIT:
            break
        excess = max(arm_bytes - BINARY_LIMIT, package - PACKAGE_LIMIT)
        drop = excess // 4 + 1
        if drop >= len(selected):
            raise SystemExit("cannot bring the budgeted arm under the byte cap")
        print(f"actual binary {arm_bytes:,} B over the cap; dropping {drop} entries", flush=True)
        selected = selected[:-drop]

    cutoff = (
        min(sum(full._trigram_counts[context].values()) for context, _ in selected)
        if selected
        else 0
    )
    summary = arm_lm.summary()
    report = {
        "corpus": str(CORPUS.relative_to(REPO)),
        "corpus_sha256": sha256(CORPUS),
        "corpus_receipt": str(RECEIPT.relative_to(REPO)),
        "tokenizer": str(TOKENIZER.relative_to(REPO)),
        "tokenizer_sha256": sha256(TOKENIZER),
        "builder_sha256": sha256(Path(__file__)),
        "term_weight": TERM_WEIGHT,
        "backoff": BACKOFF,
        "package_limit": PACKAGE_LIMIT,
        "non_lm_bytes": NON_LM_BYTES,
        "binary_limit": BINARY_LIMIT,
        "control": {
            "pickle": str(base_pickle.relative_to(REPO)),
            "binary": str(base_binary.relative_to(REPO)),
            "pickle_bytes": base_pickle.stat().st_size,
            "binary_bytes": base_bytes,
            "binary_sha256": base_sha,
            **base_lm.summary(),
        },
        "arms": {
            "B1_budgeted": {
                "ranking": "context_count_desc, then (prev2, prev1, token) asc",
                "entries_added": len(selected),
                "new_contexts": len(opened),
                "cutoff_context_count": cutoff,
                "candidate_entries": len(ranked),
                "predicted_binary_bytes": predicted,
                "binary_bytes": arm_bytes,
                "package_bytes": package,
                "headroom_bytes": PACKAGE_LIMIT - package,
                "binary_sha256": sha256(arm_binary),
                "pickle_bytes": arm_pickle.stat().st_size,
                "seconds": round(time.time() - started, 1),
                **summary,
            }
        },
    }
    (OUT / "build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"B1_budgeted: +{len(selected):,} trigrams, {arm_bytes:,} B binary, "
        f"package {package:,} B, headroom {PACKAGE_LIMIT - package:,} B, "
        f"cutoff context count {cutoff}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
