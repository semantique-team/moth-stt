import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from stt.export import (
    FP32_GATE,
    FP32_GATE_WORST,
    ExportError,
    ExportModel,
    build_notice,
    export_fp32,
    quantize_int8,
    verify_fp32,
    verify_int8,
    write_release,
)
from stt.model import EncoderConfig, MothSTT

GATE_SECONDS = (1.0, 5.0, 10.0, 29.0)

# Integrity bounds for the int8 smoke checks — NOT the product gate. Measured 2026-10-19 with the
# same seeded weights and the byte-identical fp32 graph: onnxruntime's ARM64 int8 path gives
# max|Δ log_probs| ≈ 0.02 (top1 ≥ 0.98) while the x86_64 MLAS path gives ≈ 0.85 (top1 ≥ 0.28) on
# random-weight tiny models. Broken quantization (garbage/zeroed weights) would land near
# max|Δ| ≫ 2 and top1 ≈ 1/vocab (0.015–0.03). The real int8 accept/reject decision is Task 7.1's,
# measured on the trained model per platform (`docs/06` §4).
INT8_MAX_ABS = 2.0
INT8_MIN_TOP1 = 0.15


def _mel(seconds: float, seed: int = 0, frames: int | None = None) -> np.ndarray:
    count = frames if frames is not None else int(seconds * 100)
    return (np.random.default_rng(seed).standard_normal((1, 80, count)) * 0.1).astype(np.float32)


def _session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def fp32_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[MothSTT, Path]:
    torch.manual_seed(0)  # fixed weights: int8 noise must not depend on the random draw
    model = MothSTT(vocab_size=64, encoder_config=EncoderConfig(n_layers=2)).eval()
    path = tmp_path_factory.mktemp("export") / "model.fp32.onnx"
    export_fp32(model, path)
    return model, path


def test_fp32_parity_on_gate_clips(fp32_export: tuple[MothSTT, Path]) -> None:
    model, path = fp32_export
    session = _session(path)
    reference = ExportModel(model).eval()
    for seconds in GATE_SECONDS:
        mel = _mel(seconds)
        with torch.no_grad():
            expected = reference(torch.from_numpy(mel)).numpy()
        got = session.run(None, {"mel": mel})[0]
        assert got.shape == (1, int(seconds) * 50, 65)
        assert float(np.abs(expected - got).max()) <= FP32_GATE


def test_verify_fp32_is_a_sweep_not_one_draw(fp32_export: tuple[MothSTT, Path]) -> None:
    """E-021: the gate reports a distribution, so one unlucky draw cannot decide it."""
    model, path = fp32_export
    report = verify_fp32(model, path, seconds=(1.0,), seeds=5, seed=0)
    assert report["draws"] == 5
    assert set(report["sweep"]["1s"]) == {"median", "max", "over_gate", "draws"}
    assert report["median_abs"] <= report["max_abs"]
    assert report["median_abs"] <= FP32_GATE
    assert report["max_abs"] <= FP32_GATE_WORST


def test_verify_fp32_single_seed_keeps_the_old_shape(fp32_export: tuple[MothSTT, Path]) -> None:
    model, path = fp32_export
    report = verify_fp32(model, path, seconds=(1.0, 5.0), seeds=1)
    assert "sweep" not in report
    assert report["max_abs"] <= FP32_GATE
    assert report["top1_agreement"] > 0.99


def test_verify_fp32_gate_accepts_an_unlucky_draw(
    fp32_export: tuple[MothSTT, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single draw above 1e-3 no longer fails the gate; only the median and the worst do."""
    model, path = fp32_export
    fake = {
        "per_clip": {"29s": 1.28e-3},
        "max_abs": 1.28e-3,
        "top1_agreement": 1.0,
        "sweep": {"29s": {"median": 1.8e-4, "max": 1.28e-3, "over_gate": 1, "draws": 20}},
        "median_abs": 1.8e-4,
        "over_gate": 1,
        "draws": 20,
    }
    monkeypatch.setattr("stt.export._logit_gap", lambda *args, **kwargs: fake)
    assert verify_fp32(model, path)["max_abs"] == 1.28e-3


def test_verify_fp32_gate_rejects_a_bad_median(
    fp32_export: tuple[MothSTT, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, path = fp32_export
    fake = {
        "per_clip": {"1s": 3e-2},
        "max_abs": 3e-2,
        "top1_agreement": 0.5,
        "median_abs": 3e-2,
        "draws": 20,
    }
    monkeypatch.setattr("stt.export._logit_gap", lambda *args, **kwargs: fake)
    with pytest.raises(ExportError, match="median"):
        verify_fp32(model, path)


def test_verify_fp32_gate_rejects_a_catastrophic_draw(
    fp32_export: tuple[MothSTT, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, path = fp32_export
    fake = {
        "per_clip": {"1s": 5e-2},
        "max_abs": 5e-2,
        "top1_agreement": 0.1,
        "median_abs": 1e-4,
        "draws": 20,
    }
    monkeypatch.setattr("stt.export._logit_gap", lambda *args, **kwargs: fake)
    with pytest.raises(ExportError, match="worst"):
        verify_fp32(model, path)


def test_export_model_includes_extra_head_blocks() -> None:
    model = MothSTT(
        vocab_size=16,
        encoder_config=EncoderConfig(n_layers=1),
        head_blocks=2,
    ).eval()
    export_model = ExportModel(model).eval()
    with torch.no_grad():
        log_probs = export_model(torch.from_numpy(_mel(1.0)))

    assert log_probs.shape == (1, 50, 17)
    assert len(model.head_blocks) == 2


def test_graph_contract_and_in_graph_sinusoids(fp32_export: tuple[MothSTT, Path]) -> None:
    _, path = fp32_export
    graph = onnx.load(str(path)).graph
    operations = {node.op_type for node in graph.node}
    assert {"Range", "Sin", "Cos"} <= operations  # positions follow the dynamic length
    assert all(int(np.prod(initializer.dims)) != 1500 * 384 for initializer in graph.initializer)
    assert graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1  # batch 1
    assert graph.input[0].type.tensor_type.shape.dim[2].dim_param  # dynamic time
    assert graph.output[0].type.tensor_type.shape.dim[1].dim_param  # frames follow the input
    assert onnx.load(str(path)).opset_import[0].version >= 17
    assert [node.op_type for node in graph.node][-1] in ("LogSoftmax", "Softmax")


def test_int8_export_integrity_bounds(fp32_export: tuple[MothSTT, Path], tmp_path: Path) -> None:
    """int8 smoke: runs, is smaller, and degrades within the platform bounds (see constants)."""
    model, fp32 = fp32_export
    int8 = quantize_int8(fp32, tmp_path / "model.int8.onnx")
    assert int8.stat().st_size < fp32.stat().st_size
    report = verify_int8(model, int8, seconds=(1.0, 5.0), seed=0)
    assert report["max_abs"] < INT8_MAX_ABS
    assert report["top1_agreement"] >= INT8_MIN_TOP1


def test_release_writer_packages_verified_artifacts(tmp_path: Path) -> None:
    pieces = {index: f"▁piece{index}" for index in range(1, 33)}
    torch.manual_seed(0)  # fixed weights: int8 noise must not depend on the random draw
    model = MothSTT(vocab_size=len(pieces), encoder_config=EncoderConfig(n_layers=1)).eval()
    out = tmp_path / "release"

    report = write_release(out, model, pieces, version="0.1.0")

    assert {path.name for path in out.iterdir()} == {
        "model.int8.onnx",
        "vocab.json",
        "mel_filters.bin",
        "silero_vad.onnx",
        "config.json",
        "NOTICE",
    }
    assert report["verification"]["fp32"]["max_abs"] <= FP32_GATE
    assert report["verification"]["int8"]["max_abs"] < INT8_MAX_ABS  # platform bounds, see top
    assert report["verification"]["int8"]["top1_agreement"] >= INT8_MIN_TOP1
    assert report["total_bytes"] == sum(entry["bytes"] for entry in report["files"].values())
    assert report["total_bytes"] < 20_000_000  # the ship ceiling, on the tiny test model

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["name"] == "moth-stt"
    assert config["sample_rate"] == 16000
    assert config["features"]["drop_last_frame"] is True
    assert config["model"]["file"] == "model.int8.onnx"
    assert config["model"]["frame_rate_hz"] == 50
    assert config["tokenizer"]["blank_id"] == 0
    assert config["decode"] == {"default": "greedy", "beam": 8, "hotword_bonus": 2.0}
    assert config["vad"] == {
        "file": "silero_vad.onnx",
        "window": 512,
        "context": 64,
        "threshold": 0.10,
    }

    from stt.data.sources import sha256_file

    for name, digest in config["files"].items():
        assert digest == f"sha256:{sha256_file(out / name)}"
    assert set(config["files"]) == {
        "model.int8.onnx",
        "vocab.json",
        "mel_filters.bin",
        "silero_vad.onnx",
    }

    vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))
    assert sorted(int(key) for key in vocab) == list(range(1, len(pieces) + 1))

    notice = (out / "NOTICE").read_text(encoding="utf-8")
    assert "MIT License" in notice and "Copyright (c) 2022 OpenAI" in notice
    assert "Silero VAD" in notice and "Copyright (c) 2020-present Silero Team" in notice
    assert "LibriSpeech" in notice and "FLEURS" in notice
    assert "MUSAN" in notice and "CC-BY 4.0" in notice  # verified 2026-10-23

    session = _session(out / "model.int8.onnx")
    log_probs = session.run(None, {"mel": _mel(1.0)})[0]
    assert log_probs.shape == (1, 50, 33)  # vocab + blank


def test_release_rejects_vocab_head_mismatch(tmp_path: Path) -> None:
    model = MothSTT(vocab_size=8, encoder_config=EncoderConfig(n_layers=1))
    with pytest.raises(ExportError, match="head expects"):
        write_release(tmp_path / "release", model, {1: "a", 2: "b"})


def test_notice_skips_unverified_licenses() -> None:
    notice = build_notice()
    assert "Mozilla Common Voice" in notice
    assert "verify before use" not in notice


def test_notice_lists_each_dataset_once() -> None:
    """FLEURS is two registry entries (IT and EN) but one dataset: one attribution line."""
    lines = [line for line in build_notice().splitlines() if line.startswith("- ")]
    assert len(lines) == len(set(lines))
    assert sum(1 for line in lines if "FLEURS" in line) == 1
    assert sum(1 for line in lines if "MUSAN" in line) == 1
