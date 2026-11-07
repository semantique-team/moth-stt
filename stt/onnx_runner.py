"""Decode the shipped ONNX model in Python (design: ``docs/06-release.md`` §3).

The release ships ``model.int8.onnx``, but until now nothing in the repository could *run* it:
the browser path is JavaScript and ``stt/export.py`` only uses ONNX Runtime to verify logits.
This module closes that gap so the shipped artifact can be measured with ``stt.eval``, on the
same manifests and the same text rule as the PyTorch checkpoint.

The graph takes ``mel`` ``[1, 80, T]`` (dynamic time) and returns ``log_probs`` ``[1, frames, V]``
for a **batch of one**. Clips are therefore decoded one at a time; the CTC decoders in
:mod:`stt.decode` accept the numpy output unchanged, so the only difference from a PyTorch run is
the quantized acoustic model.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from stt.decode import beam_search, greedy, load_pieces
from stt.lm import BPETrigramLM

__all__ = ["OnnxMoth", "decode_clip"]


class OnnxMoth:
    """ONNX Runtime wrapper around a Moth export (int8 by default, fp32 works too)."""

    def __init__(
        self,
        model_path: Path | str,
        vocab_path: Path | str | None = None,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ) -> None:
        import onnxruntime as ort  # heavy import, kept local

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"{self.model_path} is missing")
        self.session = ort.InferenceSession(str(self.model_path), providers=list(providers))
        self.provider = self.session.get_providers()[0]
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.vocab_path = (
            Path(vocab_path) if vocab_path else self.model_path.with_name("vocab.json")
        )
        self.pieces = load_pieces(self.vocab_path)

    @property
    def vocab_size(self) -> int:
        """Number of CTC outputs, including the blank at id 0."""
        return len(self.pieces) + 1

    def log_probs(self, mel: np.ndarray) -> np.ndarray:
        """Run the encoder on one ``[80, T]`` (or ``[1, 80, T]``) mel and return ``[frames, V]``."""
        array = np.asarray(mel, dtype=np.float32)
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(f"expected one clip [1, 80, T], got {array.shape}")
        out = self.session.run([self.output_name], {self.input_name: array})[0]
        return np.asarray(out[0], dtype=np.float32)

    def transcribe(
        self,
        mel: np.ndarray,
        *,
        decoder: str = "greedy",
        beam_width: int = 5,
        top_k: int = 8,
        lm: BPETrigramLM | None = None,
        lm_weight: float = 0.0,
    ) -> tuple[str, float]:
        """Decode one clip; returns ``(text, wall_seconds)``."""
        started = time.perf_counter()
        log_probs = self.log_probs(mel)
        if decoder == "beam":
            text = beam_search(
                log_probs,
                log_probs.shape[0],
                self.pieces,
                beam_width=beam_width,
                top_k=top_k,
                lm=lm,
                lm_weight=lm_weight,
            )
        elif decoder == "greedy":
            text = greedy(log_probs, log_probs.shape[0], self.pieces)
        else:
            raise ValueError(f"unknown decoder {decoder!r}")
        return text, time.perf_counter() - started


def decode_clip(
    model: OnnxMoth,
    mel: np.ndarray,
    *,
    decoder: str = "greedy",
    beam_width: int = 5,
    top_k: int = 8,
    lm: BPETrigramLM | None = None,
    lm_weight: float = 0.0,
) -> str:
    """Thin function form of :meth:`OnnxMoth.transcribe` for callers that already hold a model."""
    text, _ = model.transcribe(
        mel,
        decoder=decoder,
        beam_width=beam_width,
        top_k=top_k,
        lm=lm,
        lm_weight=lm_weight,
    )
    return text
