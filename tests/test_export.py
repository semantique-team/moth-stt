import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from stt.export import (
    FP32_GATE,
    ExportError,
    ExportModel,
    build_notice,
    export_fp32,
    quantize_int8,
    verify_int8,
    write_release,
)
from stt.model import EncoderConfig, MothSTT

GATE_SECONDS = (1.0, 5.0, 10.0, 29.0)


def _mel(seconds: float, seed: int = 0, frames: int | None = None) -> np.ndarray:
    count = frames if frames is not None else int(seconds * 100)
    return (np.random.default_rng(seed).standard_normal((1, 80, count)) * 0.1).astype(np.float32)


def _session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def fp32_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[MothSTT, Path]:
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


def test_int8_stays_close_to_pytorch(fp32_export: tuple[MothSTT, Path], tmp_path: Path) -> None:
    model, fp32 = fp32_export
    int8 = quantize_int8(fp32, tmp_path / "model.int8.onnx")
    assert int8.stat().st_size < fp32.stat().st_size
    report = verify_int8(model, int8, seconds=(1.0, 5.0), seed=0)
    assert report["max_abs"] < 0.1
    assert report["top1_agreement"] >= 0.95


def test_release_writer_packages_verified_artifacts(tmp_path: Path) -> None:
    pieces = {index: f"▁piece{index}" for index in range(1, 33)}
    model = MothSTT(vocab_size=len(pieces), encoder_config=EncoderConfig(n_layers=1)).eval()
    out = tmp_path / "release"

    report = write_release(out, model, pieces, version="0.1.0")

    assert {path.name for path in out.iterdir()} == {
        "model.int8.onnx",
        "vocab.json",
        "mel_filters.bin",
        "config.json",
        "NOTICE",
    }
    assert report["verification"]["fp32"]["max_abs"] <= FP32_GATE
    assert report["verification"]["int8"]["top1_agreement"] >= 0.95
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

    from stt.data.sources import sha256_file

    for name, digest in config["files"].items():
        assert digest == f"sha256:{sha256_file(out / name)}"
    assert set(config["files"]) == {"model.int8.onnx", "vocab.json", "mel_filters.bin"}

    vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))
    assert sorted(int(key) for key in vocab) == list(range(1, len(pieces) + 1))

    notice = (out / "NOTICE").read_text(encoding="utf-8")
    assert "MIT License" in notice and "Copyright (c) 2022 OpenAI" in notice
    assert "LibriSpeech" in notice and "FLEURS" in notice
    assert "MUSAN" not in notice  # license still unverified: never attributed as used

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
