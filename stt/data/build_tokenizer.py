"""Build the shared IT+EN BPE tokenizer (task 2.3; design: ``docs/02`` §5, ``docs/03`` §6).

Corpus, in the order fixed by the design:

1. the **training transcripts** of the training sets — MLS IT and LibriSpeech EN, Italian-weighted
   to ``--it-share`` (0.7 by default) *by text*, i.e. by word count: the Italian side is duplicated
   with a seeded sample until it reaches the requested share (SentencePiece has no sample weights);
2. the **mixed pool** (``recordings/scripts/pool.jsonl``, task 1.1) as it is;
3. the curated **English terms** (``data/en_terms.txt``), up-weighted to ``--terms-share`` of the
   final corpus so BPE prefers pieces that spell them in few tokens.

Test sentences never enter the corpus: script ``T`` is not read here, and the sources are the same
manifests ``stt/train.py`` trains on. The tokenizer itself is already normalized text.

Outputs: ``<out>.model`` and ``<out>.vocab`` (SentencePiece) + ``<out>.vocab.json`` (pieces by CTC
id), and the statistics merged into the data build report (tokens/word, tokens/second on the real
transcripts, share of English terms within 3 pieces).

Deterministic given the corpus and ``--seed``: the corpus assembly is seeded, and SentencePiece BPE
yields the same piece list and the same ids — encodings are stable across runs (measured on 6,080
texts). The serialized ``.model`` bytes still differ in float-score noise, so the report records the
sha256 of the artifact actually built instead of claiming byte reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from stt.manifest import load_manifest
from stt.tokenizer import load_tokenizer, tokenizer_stats, train_tokenizer, write_vocab

__all__ = [
    "build_corpus",
    "main",
    "merge_report",
    "oversample_to_words",
    "read_pool",
    "read_terms",
    "read_transcripts",
]

DEFAULT_REPORT = "data/manifests/build_report.json"


def read_transcripts(manifest_path: Path | str) -> list[tuple[str, float]]:
    """``[(norm_text, duration_s), ...]`` from a manifest (validated by ``load_manifest``)."""
    return [(row["norm"], float(row.get("duration") or 0.0)) for row in load_manifest(manifest_path)]


def oversample_to_words(
    texts: list[str], target_words: int, seed: int = 0
) -> tuple[list[str], int]:
    """Return ``(texts + duplicates, added)`` until the word count reaches ``target_words``.

    Lines are drawn by cycling a seeded shuffle of the indices, so the choice is deterministic and
    spread over the whole set instead of duplicating only the first lines.
    """
    words = sum(len(text.split()) for text in texts)
    if words >= target_words or not texts:
        return list(texts), 0
    order: list[int] = []
    rng = random.Random(seed)
    while len(order) < len(texts) * 2:
        block = list(range(len(texts)))
        rng.shuffle(block)
        order.extend(block)
    added: list[str] = []
    for index in order:
        if words >= target_words:
            break
        text = texts[index]
        added.append(text)
        words += len(text.split())
    return list(texts) + added, len(added)


def build_corpus(
    it: list[str],
    en: list[str],
    pool: list[str],
    it_share: float = 0.7,
    seed: int = 0,
) -> tuple[list[str], dict]:
    """Italian-weighted transcript lines + the pool; returns ``(lines, provenance_stats)``."""
    en_words = sum(len(text.split()) for text in en)
    target = round(it_share / (1.0 - it_share) * en_words)
    it_dup, added = oversample_to_words(it, target, seed=seed)
    lines = it_dup + list(en) + list(pool)
    it_words = sum(len(text.split()) for text in it_dup)
    pool_words = sum(len(text.split()) for text in pool)
    transcript_words = it_words + en_words
    stats = {
        "it_lines": len(it),
        "it_lines_duplicated": added,
        "en_lines": len(en),
        "pool_lines": len(pool),
        "transcript_words": transcript_words,
        "pool_words": pool_words,
        "it_share_words": it_words / transcript_words if transcript_words else 0.0,
        "pool_share_words": pool_words / (transcript_words + pool_words or 1),
    }
    return lines, stats


def read_pool(path: Path | str) -> list[str]:
    """The mixed pool's ``text`` fields (already normalized by the script generator)."""
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row["text"] for row in rows]


def read_terms(path: Path | str) -> list[str]:
    """One English term per line, ``#`` comments and blanks ignored."""
    terms = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def merge_report(path: Path | str, section: dict) -> dict:
    """Merge ``{"tokenizer": section}`` into the JSON build report, keeping every other key."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report["tokenizer"] = section
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--it", default="data/manifests/mls_it_train.jsonl", help="IT transcripts")
    parser.add_argument("--en", default="data/manifests/ls_en_train.jsonl", help="EN transcripts")
    parser.add_argument("--pool", default="recordings/scripts/pool.jsonl", help="mixed pool JSONL")
    parser.add_argument("--terms", default="data/en_terms.txt", help="curated English terms")
    parser.add_argument("--out", default="data/tokenizer/moth_bpe", help="output prefix (no suffix)")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--it-share", type=float, default=0.7, help="Italian share by words")
    parser.add_argument("--terms-share", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default=DEFAULT_REPORT, help="build report to update")
    parser.add_argument("--no-report", action="store_true", help="do not touch the build report")
    args = parser.parse_args(argv)

    it_rows = read_transcripts(args.it)
    en_rows = read_transcripts(args.en)
    pool = read_pool(args.pool)
    terms = read_terms(args.terms)
    lines, provenance = build_corpus(
        [text for text, _ in it_rows],
        [text for text, _ in en_rows],
        pool,
        it_share=args.it_share,
        seed=args.seed,
    )

    out_prefix = Path(args.out)
    model_path = train_tokenizer(
        lines,
        out_prefix,
        vocab_size=args.vocab_size,
        extra_weight_terms=terms,
        terms_share=args.terms_share,
    )
    processor = load_tokenizer(model_path)
    vocab_path = out_prefix.with_suffix(".vocab.json")
    write_vocab(processor, vocab_path)

    # Statistics on the *unique* transcripts (real speech, real durations) and on the pool.
    transcript_texts = [text for text, _ in it_rows + en_rows]
    durations = [duration for _, duration in it_rows + en_rows]
    stats = tokenizer_stats(processor, transcript_texts, terms=terms, durations=durations)
    pool_stats = tokenizer_stats(processor, pool)

    print(
        f"vocab {processor.vocab_size()} pieces | corpus {len(lines)} lines "
        f"({provenance['transcript_words']} transcript words, IT share "
        f"{provenance['it_share_words']:.3f}, pool {provenance['pool_lines']})"
    )
    print(
        f"tokens/word {stats['tokens_per_word']:.3f} | tokens/s {stats['tokens_per_second']:.2f} "
        f"| terms within 3 pieces {stats['terms_within_3_pieces']:.3f} "
        f"| pool tokens/word {pool_stats['tokens_per_word']:.3f}"
    )
    print(f"wrote {model_path} and {vocab_path}")

    if not args.no_report:
        section = {
            "model": str(model_path),
            "vocab_json": str(vocab_path),
            "vocab_size": processor.vocab_size(),
            "it_share_requested": args.it_share,
            "terms_share": args.terms_share,
            "seed": args.seed,
            "terms": len(terms),
            "corpus": provenance,
            "stats": stats,
            "pool_stats": pool_stats,
            "reproducibility": (
                "corpus assembly seeded; piece list and ids stable across runs (encodings stable, "
                "measured); serialized model bytes differ in float-score noise — sha256 identifies "
                "the built artifact"
            ),
            "sha256": {str(model_path): _sha256(model_path), str(vocab_path): _sha256(vocab_path)},
        }
        merge_report(args.report, section)
        print(f"stats written to {args.report} (key 'tokenizer')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
