"""Tests for the ONNX runtime decoder used to measure the shipped int8 model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stt.model import MothSTT
from stt.onnx_runner import OnnxMoth, decode_clip

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from stt.export import export_fp32

VOCAB_SIZE = 16


def _vocab(path: Path) -> Path:
    pieces = {index: f"▁p{index}" for index in range(1, VOCAB_SIZE + 1)}
    path.write_text(json.dumps({str(k): v for k, v in pieces.items()}), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> tuple[MothSTT, Path, Path]:
    tmp = tmp_path_factory.mktemp("onnx-runner")
    torch.manual_seed(0)
    model = MothSTT(vocab_size=VOCAB_SIZE, head_blocks=0).eval()
    path = export_fp32(model, tmp / "model.fp32.onnx")
    return model, path, _vocab(tmp / "vocab.json")


def _mel(frames: int = 220) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal((80, frames)) * 0.5).astype(np.float32)


def test_vocab_size_is_pieces_plus_blank(exported) -> None:
    _, path, vocab = exported
    runner = OnnxMoth(path, vocab)
    assert runner.vocab_size == VOCAB_SIZE + 1
    assert runner.input_name == "mel"


def test_log_probs_matches_the_torch_graph(exported) -> None:
    _, path, vocab = exported
    runner = OnnxMoth(path, vocab)
    mel = _mel()
    got = runner.log_probs(mel)
    assert got.ndim == 2
    assert got.shape[0] == pytest.approx(110, abs=2)  # 50 Hz mel -> 50 Hz frames after /2
    assert got.shape[1] == VOCAB_SIZE + 1
    assert np.isfinite(got).all()
    # The graph returns log-softmax, so exponentiating must sum to 1 per frame.
    assert np.allclose(np.exp(got).sum(axis=-1), 1.0, atol=1e-4)


def test_accepts_batched_and_unbatched_input(exported) -> None:
    _, path, vocab = exported
    runner = OnnxMoth(path, vocab)
    mel = _mel()
    assert np.allclose(runner.log_probs(mel), runner.log_probs(mel[None]))


def test_rejects_wrong_shapes(exported) -> None:
    _, path, vocab = exported
    runner = OnnxMoth(path, vocab)
    with pytest.raises(ValueError):
        runner.log_probs(np.zeros((2, 80, 100), np.float32))
    with pytest.raises(ValueError):
        runner.log_probs(np.zeros((80, 100, 1), np.float32))


def test_transcribe_decodes_and_times(exported) -> None:
    _, path, vocab = exported
    runner = OnnxMoth(path, vocab)
    text, seconds = runner.transcribe(_mel())
    assert isinstance(text, str)
    assert seconds >= 0.0


def test_greedy_matches_the_torch_path_on_the_same_frames(exported) -> None:
    """The ONNX decoder must agree with the reference decoder given the same log-probs."""
    from stt.decode import greedy

    _, path, vocab = exported
    runner = OnnxMoth(path, vocab)
    mel = _mel()
    log_probs = runner.log_probs(mel)
    assert runner.transcribe(mel)[0] == greedy(log_probs, log_probs.shape[0], runner.pieces)
    assert decode_clip(runner, mel) == greedy(log_probs, log_probs.shape[0], runner.pieces)


def test_unknown_decoder_is_rejected(exported) -> None:
    _, path, vocab = exported
    runner = OnnxMoth(path, vocab)
    with pytest.raises(ValueError):
        runner.transcribe(_mel(), decoder="nope")


def test_missing_model_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OnnxMoth(tmp_path / "absent.onnx", _vocab(tmp_path / "vocab.json"))
