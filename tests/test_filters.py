from pathlib import Path

import numpy as np
import soundfile as sf

from stt.data.filters import (
    FilterConfig,
    apply_filters,
    build_report,
    ctc_feasible,
    filter_row,
    speaking_rate_ok,
)


def _row(uid: str = "r1", text: str = "ciao come stai", duration: float = 3.0, **extra) -> dict:
    from stt.text import normalize

    row = {
        "id": uid,
        "audio": "a.wav",
        "text": text,
        "norm": normalize(text),
        "duration": duration,
        "source": "test",
        "split": "train",
        "speaker": "s1",
        "lang": "it",
        "category": None,
        "en_spans": [],
    }
    row.update(extra)
    return row


def test_duration_bounds() -> None:
    config = FilterConfig(min_chars_per_second=0.0, max_chars_per_second=10_000.0)
    assert filter_row(_row(duration=0.5), config) == "duration"
    assert filter_row(_row(duration=31.0), config) == "duration"
    assert filter_row(_row(duration=2.0), config) is None
    assert filter_row(_row(duration=25.0), config) is None


def test_digits_rejected_for_training() -> None:
    assert filter_row(_row(text="ho 2 meeting"), FilterConfig()) == "digits"
    assert filter_row(_row(text="alle 15:30"), FilterConfig()) == "digits"


def test_charset_violation_rejected() -> None:
    assert filter_row(_row(text="ciao 😀"), FilterConfig()) == "charset"


def test_empty_text_rejected() -> None:
    assert filter_row(_row(text="!!!"), FilterConfig()) == "empty"


def test_speaking_rate_bounds() -> None:
    config = FilterConfig(min_chars_per_second=4.0, max_chars_per_second=30.0)
    assert speaking_rate_ok("ciao", 10.0, config) is False  # 0.4 chars/s
    assert speaking_rate_ok("a" * 100, 1.0, config) is False  # 100 chars/s
    assert speaking_rate_ok("ciao come stai", 2.0, config) is True


def test_ctc_feasibility() -> None:
    # 50 frames/s at the encoder: 1 s -> 50 frames; a 3-token label fits easily.
    assert ctc_feasible("ciao come stai", 1.0, vocab_size=1024) is True
    # A very short clip with a long transcript does not fit.
    assert ctc_feasible("a b c d e f g h i l m n o p q r s t u v", 0.02, vocab_size=1024) is False


def test_corrupt_and_clipping_audio(tmp_path: Path) -> None:
    config = FilterConfig()
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(16000, dtype="float32"), 16000)
    assert filter_row(_row(), config, audio_path=silent) == "silent"

    clipped = tmp_path / "clipped.wav"
    sf.write(clipped, np.ones(16000, dtype="float32"), 16000)
    assert filter_row(_row(), config, audio_path=clipped) == "clipping"

    missing = tmp_path / "missing.wav"
    assert filter_row(_row(), config, audio_path=missing) == "unreadable"


def test_overlap_with_eval_texts() -> None:
    eval_hashes = {"deadbeef"}
    assert filter_row(_row(), FilterConfig(), eval_hashes=eval_hashes) is None
    from stt.manifest import norm_hash

    row = _row(text="frase di test")
    assert filter_row(row, FilterConfig(), eval_hashes={norm_hash(row["text"])}) == "overlap"


def test_apply_filters_counts_reasons_and_keeps_order() -> None:
    config = FilterConfig(min_chars_per_second=0.0, max_chars_per_second=10_000.0)
    rows = [
        _row("a", duration=3.0),
        _row("b", duration=0.1),
        _row("c", text="ho 2 gatti"),
        _row("d", duration=4.0),
    ]
    kept, report = apply_filters(rows, config)
    assert [row["id"] for row in kept] == ["a", "d"]
    assert report["kept"] == 2
    assert report["dropped"] == {"duration": 1, "digits": 1}
    assert report["total"] == 4


def test_build_report_renders_hours_and_reasons() -> None:
    kept = [_row("a", duration=3600.0), _row("b", duration=1800.0)]
    report = {"kept": 2, "total": 3, "dropped": {"duration": 1}, "seconds": 5400.0}
    text = build_report("cv_it_27", report, kept)
    assert "cv_it_27" in text
    assert "1.50 h" in text
    assert "duration: 1" in text


def test_duration_warning_for_long_clips() -> None:
    config = FilterConfig(min_chars_per_second=0.0, max_chars_per_second=10_000.0)
    row = _row(duration=25.0)
    assert filter_row(row, config) is None
    assert filter_row(row, FilterConfig(warn_above_seconds=20.0, min_chars_per_second=0.0, max_chars_per_second=10_000.0)) is None
