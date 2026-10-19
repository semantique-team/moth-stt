"""Shared IT+EN BPE tokenizer (design: ``docs/02-architecture.md`` §5).

One SentencePiece **BPE** vocabulary covers both languages (Latin script, so no language switch
is needed to spell "meeting" or "deployato"). Training text: the (normalized) transcripts —
Italian-weighted ≈ 70/30 — the mixed-sentence pool, and the curated English-terms list
up-weighted. Test sentences never enter the training text.

CTC mapping: ``blank = 0``, ``ctc_id = sp_id + 1``, output size ``V_ctc = V_sp + 1``.
``vocab_json`` exports pieces by CTC id (blank excluded: the decoder owns it).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import sentencepiece as spm

__all__ = [
    "BLANK_ID",
    "TERMS_SHARE",
    "decode",
    "encode",
    "load_tokenizer",
    "tokenizer_stats",
    "train_tokenizer",
    "vocab_json",
    "weighted_term_lines",
]

#: CTC blank id; SentencePiece ids shift by +1 (``unk`` is 0, so it maps to 1).
BLANK_ID = 0
_DEFAULT_VOCAB_SIZE = 1024

#: Target share of the **final** corpus made of English-term lines (``docs/02`` §5: "multiple
#: times (up-weight)" — the number is ours, recorded in the build report).
TERMS_SHARE = 0.05


def weighted_term_lines(terms: Iterable[str], n_lines: int, share: float = TERMS_SHARE) -> list[str]:
    """Repeat ``terms`` so they make up ``share`` of the final corpus (base + term lines).

    The old rule repeated the whole list ``n_lines // 5`` times, which on a ~100k-line corpus
    means ~100M lines: the terms are up-weighted by *share*, not by a multiple of the corpus.
    """
    terms = [term for term in terms if term.strip()]
    if not terms or n_lines <= 0 or share <= 0.0:
        return []
    target = round(n_lines * share / (1.0 - share))
    repeat = max(1, round(target / len(terms)))
    return terms * repeat


def train_tokenizer(
    texts: Iterable[str],
    out_prefix: Path | str,
    vocab_size: int = _DEFAULT_VOCAB_SIZE,
    extra_weight_terms: Iterable[str] = (),
    terms_share: float = TERMS_SHARE,
) -> Path:
    """Train the shared BPE model and return the ``.model`` path.

    ``extra_weight_terms`` (the English-terms list) is appended so that term lines are
    ``terms_share`` of the final corpus (see :func:`weighted_term_lines`), so BPE prefers pieces
    that spell those words in few tokens.
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    lines = [text for text in texts if text.strip()]
    lines.extend(weighted_term_lines(extra_weight_terms, len(lines), terms_share))

    corpus = out_prefix.with_suffix(".txt")
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        normalization_rule_name="identity",  # the text is already normalized
        remove_extra_whitespaces=False,
        hard_vocab_limit=False,
    )
    corpus.unlink()
    return out_prefix.with_suffix(".model")


def load_tokenizer(model_path: Path | str) -> spm.SentencePieceProcessor:
    processor = spm.SentencePieceProcessor()
    processor.load(str(model_path))
    return processor


def encode(processor: spm.SentencePieceProcessor, text: str) -> list[int]:
    """Canonical text → CTC ids (never ``BLANK_ID``)."""
    return [piece_id + 1 for piece_id in processor.encode(text, out_type=int)]


def decode(processor: spm.SentencePieceProcessor, ids: Iterable[int]) -> str:
    """CTC ids → text (blank ids are dropped; ``▁`` decodes to spaces)."""
    pieces = [identifier - 1 for identifier in ids if identifier != BLANK_ID]
    return processor.decode(pieces, out_type=str)


def vocab_json(processor: spm.SentencePieceProcessor) -> dict[str, str]:
    """Pieces by CTC id as a JSON-serializable dict (blank excluded)."""
    return {str(piece_id + 1): processor.id_to_piece(piece_id) for piece_id in range(processor.vocab_size())}


def write_vocab(processor: spm.SentencePieceProcessor, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab_json(processor), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def tokenizer_stats(
    processor: spm.SentencePieceProcessor,
    texts: Iterable[str],
    terms: Iterable[str] = (),
    durations: list[float] | None = None,
) -> dict[str, Any]:
    """Reported statistics: tokens per word, per second, and English terms within 3 pieces."""
    texts = [text for text in texts if text.strip()]
    words = tokens = 0
    for text in texts:
        words += len(text.split())
        tokens += len(processor.encode(text, out_type=int))
    seconds = sum(durations or [])
    term_list = [term for term in terms if term.strip()]
    within = sum(1 for term in term_list if len(processor.encode(term, out_type=int)) <= 3)
    return {
        "texts": len(texts),
        "tokens_per_word": tokens / words if words else 0.0,
        "tokens_per_second": tokens / seconds if seconds else 0.0,
        "terms": len(term_list),
        "terms_within_3_pieces": within / len(term_list) if term_list else 0.0,
    }
