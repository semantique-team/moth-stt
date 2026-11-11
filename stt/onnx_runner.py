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
from stt.features import load_mel_filters, log_mel
from stt.hotwords import ContextualBias
from stt.lm import BinaryBPELM, BPETrigramLM
from stt.release import validate_release
from stt.vad import WINDOW_THRESHOLD, segment_bounds, speech_probabilities

__all__ = ["OnnxMoth", "ReleaseMoth", "decode_clip"]


def _validate_contextual(contextual, pieces, decoder):
    if contextual is None:
        return
    if not isinstance(contextual, ContextualBias):
        raise TypeError("contextual must be a ContextualBias or None")
    if contextual.pieces != pieces:
        raise ValueError("contextual vocabulary does not match model pieces")
    if decoder != "beam":
        raise ValueError("contextual bias requires the beam decoder")


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
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("expected one ONNX input and one output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.vocab_path = (
            Path(vocab_path) if vocab_path else self.model_path.with_name("vocab.json")
        )
        self.pieces = load_pieces(self.vocab_path)
        if (
            not self.pieces
            or set(self.pieces) != set(range(1, self.vocab_size))
            or any(not isinstance(piece, str) for piece in self.pieces.values())
        ):
            raise ValueError("vocab must contain string pieces at contiguous CTC IDs starting at 1")
        for node, expected in (
            (inputs[0], (1, 80, None)),
            (outputs[0], (1, None, self.vocab_size)),
        ):
            if (
                node.type != "tensor(float)"
                or len(node.shape) != 3
                or any(
                    isinstance(actual, int) and wanted is not None and actual != wanted
                    for actual, wanted in zip(node.shape, expected, strict=True)
                )
            ):
                raise ValueError(f"incompatible ONNX tensor metadata: {node.name} {node.shape}")

    @property
    def vocab_size(self) -> int:
        """Number of CTC outputs, including the blank at id 0."""
        return len(self.pieces) + 1

    def log_probs(self, mel: np.ndarray) -> np.ndarray:
        """Run the encoder on one ``[80, T]`` (or ``[1, 80, T]``) mel and return ``[frames, V]``."""
        if np.iscomplexobj(mel):
            raise ValueError("mel must be real")
        array = np.asarray(mel, dtype=np.float32)
        if array.ndim == 2:
            array = array[None]
        if (
            array.ndim != 3
            or array.shape[:2] != (1, 80)
            or not 1 <= array.shape[2] <= 3000
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"expected one clip [1, 80, T], got {array.shape}")
        out = self.session.run([self.output_name], {self.input_name: array})[0]
        out = np.asarray(out, dtype=np.float32)
        if (
            out.ndim != 3
            or out.shape[0] != 1
            or out.shape[1] < 1
            or out.shape[2] != self.vocab_size
        ):
            raise ValueError(
                f"expected ONNX output [1, frames, {self.vocab_size}], got {out.shape}"
            )
        if np.isnan(out).any() or np.isposinf(out).any():
            raise ValueError("ONNX log probabilities contain NaN or positive infinity")
        return out[0]

    def transcribe(
        self,
        mel: np.ndarray,
        *,
        decoder: str = "greedy",
        beam_width: int = 5,
        top_k: int = 8,
        lm: BPETrigramLM | BinaryBPELM | None = None,
        lm_weight: float = 0.0,
        token_bonus: float = 0.0,
        contextual: ContextualBias | None = None,
    ) -> tuple[str, float]:
        """Decode one clip; returns ``(text, wall_seconds)``."""
        started = time.perf_counter()
        _validate_contextual(contextual, self.pieces, decoder)
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
                token_bonus=token_bonus,
                **({"contextual": contextual} if contextual is not None else {}),
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
    lm: BPETrigramLM | BinaryBPELM | None = None,
    lm_weight: float = 0.0,
    token_bonus: float = 0.0,
    contextual: ContextualBias | None = None,
) -> str:
    """Thin function form of :meth:`OnnxMoth.transcribe` for callers that already hold a model."""
    text, _ = model.transcribe(
        mel,
        decoder=decoder,
        beam_width=beam_width,
        top_k=top_k,
        lm=lm,
        lm_weight=lm_weight,
        token_bonus=token_bonus,
        **({"contextual": contextual} if contextual is not None else {}),
    )
    return text


class ReleaseMoth:
    """Validated CPU release runtime; sessions, filters and binary LM load once.

    ``transcribe`` accepts normalized mono PCM in [-1, 1], at exactly 16 kHz,
    and returns ``(text, wall_seconds)`` including VAD, features and decoding, not load.
    Segmentation defaults to ``config.decode.segment`` (True when absent) and is
    used only for >=2 speech runs. After the no-speech gate, each selected span
    must be 560 samples to 30 seconds; longer spans raise instead of truncating.
    ``config`` and ``report`` expose the validated settings and asset size/hash report.

    Opt in per call with ``contextual=ContextualBias(sequences, runtime.model.pieces)``.
    Sequences must be CTC IDs from the release's frozen tokenizer, not guessed from
    text. The vocabulary snapshot must match exactly; bias requires beam decoding.
    No contextual settings are read from release config.
    """

    def __init__(self, path: Path | str, *, budget_bytes: int = 20_000_000) -> None:
        self.config, self.report = validate_release(path, budget_bytes=budget_bytes)
        root = Path(self.report["release"])
        self.filters = load_mel_filters(root / self.config["features"]["mel_filters"])
        if not np.isfinite(self.filters).all():
            raise ValueError("release mel filters must be finite")
        self.lm = BinaryBPELM.load(root / self.config["decode"]["lm"])
        self.model = OnnxMoth(
            root / self.config["model"]["file"], root / self.config["tokenizer"]["vocab"]
        )
        if (
            self.model.input_name != self.config["model"]["input"]
            or self.model.output_name != self.config["model"]["output"]
        ):
            raise ValueError("ONNX input/output names do not match release config")
        import onnxruntime as ort

        self.vad_session = ort.InferenceSession(
            str(root / self.config["vad"]["file"]), providers=["CPUExecutionProvider"]
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        *,
        segment: bool | None = None,
        contextual: ContextualBias | None = None,
    ) -> tuple[str, float]:
        """Decode PCM without resampling or gain adjustment; see the class contract."""
        started = time.perf_counter()
        _validate_contextual(contextual, self.model.pieces, self.config["decode"]["default"])
        if segment is None:
            segment = self.config["decode"].get("segment", True)
        if type(segment) is not bool:
            raise ValueError("segment must be a boolean or None to use the release default")
        if sample_rate != 16000:
            raise ValueError(f"expected 16000 Hz, got {sample_rate}; resampling is not provided")
        samples = np.asarray(audio)
        if (
            samples.ndim != 1
            or samples.size < 560
            or samples.dtype.kind not in "fiu"
            or not np.isfinite(samples).all()
            or np.any(np.abs(samples.astype(np.float64)) > 1)
        ):
            raise ValueError("expected finite mono PCM in [-1, 1], at least 560 samples")
        samples = np.asarray(samples, dtype=np.float32)
        max_samples = int(self.config["model"]["max_seconds"] * sample_rate)
        probabilities = speech_probabilities(samples, session=self.vad_session)
        fraction = float((probabilities > WINDOW_THRESHOLD).mean())
        if fraction <= self.config["vad"]["threshold"]:
            return "", time.perf_counter() - started
        bounds = segment_bounds(samples, probabilities=probabilities) if segment else []
        if len(bounds) < 2:
            bounds = [(0, samples.size)]
        if any(not 560 <= end - start <= max_samples for start, end in bounds):
            raise ValueError("each audio span must contain 560 samples to 30 seconds")
        settings = self.config["decode"]
        texts = []
        for start, end in bounds:
            text, _ = self.model.transcribe(
                log_mel(samples[start:end], filters=self.filters),
                decoder=settings["default"],
                beam_width=settings["beam"],
                top_k=settings["top_k"],
                lm=self.lm,
                lm_weight=settings["lm_weight"],
                token_bonus=settings["token_bonus"],
                **({"contextual": contextual} if contextual is not None else {}),
            )
            if text:
                texts.append(text)
        return " ".join(texts), time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    """Run ``python -m stt.onnx_runner RELEASE AUDIO.wav`` (16 kHz mono only)."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("audio", type=Path)
    args = parser.parse_args(argv)
    import soundfile as sf

    try:
        runtime = ReleaseMoth(args.release)
        audio, rate = sf.read(args.audio, dtype="float32")
        text, elapsed = runtime.transcribe(audio, sample_rate=rate)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(text)
    print(f"Elapsed: {elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
