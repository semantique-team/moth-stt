import random
from pathlib import Path

from stt.tokenizer import (
    BLANK_ID,
    decode,
    encode,
    load_tokenizer,
    tokenizer_stats,
    train_tokenizer,
    vocab_json,
)

#: Every charset character appears here, so ``character_coverage=1.0`` learns the full alphabet.
BASE_WORDS = ["il", "meeting", "è", "pronto", "per", "il", "deploy", "di", "domani", "con", "il", "team", "e", "la", "call", "delle", "nove", "zero", "uno", "jeans", "kayak", "wiki", "xenofobo", "yogurt", "zaino", "quota", "perché", "città", "così", "più", "l'ho", "un'idea", "sprint", "ticket", "kpi", "brand", "test", "video", "server", "newsletter", "rollout"]


def _corpus(count: int = 300) -> list[str]:
    rng = random.Random(0)
    return [
        " ".join(rng.choices(BASE_WORDS, k=rng.randint(4, 8)))
        for _ in range(count)
    ]


def _tiny(tmp_path: Path):
    model = train_tokenizer(_corpus(), tmp_path / "moth", vocab_size=64)
    return load_tokenizer(model)


def test_train_and_roundtrip(tmp_path: Path) -> None:
    processor = _tiny(tmp_path)
    text = "il meeting è pronto perché l'ho spostato"
    ids = encode(processor, text)
    assert ids  # non-empty
    assert all(identifier > BLANK_ID for identifier in ids)  # never the blank id
    assert decode(processor, ids) == text


def test_vocab_json_uses_ctc_ids(tmp_path: Path) -> None:
    processor = _tiny(tmp_path)
    vocab = vocab_json(processor)
    keys = sorted(int(key) for key in vocab)
    assert keys == list(range(1, len(vocab) + 1))
    assert all(piece for piece in vocab.values())


def test_tokenizer_stats(tmp_path: Path) -> None:
    processor = _tiny(tmp_path)
    stats = tokenizer_stats(
        processor,
        _corpus(50),
        terms=["meeting", "deploy", "kpi"],
        durations=[3.0] * 50,
    )
    assert stats["tokens_per_word"] > 0
    assert stats["tokens_per_second"] > 0
    assert 0.0 <= stats["terms_within_3_pieces"] <= 1.0
