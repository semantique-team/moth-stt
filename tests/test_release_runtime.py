"""CPU runtime integration with tiny ONNX graphs, real binary LM and shipped filters."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stt.decode import beam_search
from stt.features import DEFAULT_FILTERS, log_mel
from stt.lm import BPETrigramLM
from stt.onnx_runner import OnnxMoth, ReleaseMoth, decode_clip, main
from stt.vad import ASSET_PATH, STATE_SHAPE, WINDOW, segment_bounds, speech_probabilities


@pytest.fixture
def release(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    h, t = onnx.helper, onnx.TensorProto
    logits = np.log(np.array([[[0.6, 0.3, 0.1], [0.6, 0.1, 0.3]]], np.float32))
    graph = h.make_graph(
        [h.make_node("Constant", [], ["log_probs"], value=onnx.numpy_helper.from_array(logits))],
        "tiny",
        [h.make_tensor_value_info("mel", t.FLOAT, [1, 80, "time"])],
        [h.make_tensor_value_info("log_probs", t.FLOAT, [1, 2, 3])],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, tmp_path / "model.onnx")
    (tmp_path / "vocab.json").write_text('{"1":"hello","2":" world"}', encoding="utf-8")
    (tmp_path / "filters.bin").write_bytes(DEFAULT_FILTERS.read_bytes())
    BPETrigramLM.from_sequences([[0, 1]]).save_browser(tmp_path / "lm.bin")
    (tmp_path / "vad.onnx").write_bytes(ASSET_PATH.read_bytes())
    (tmp_path / "NOTICE").write_text("Synthetic integration fixture", encoding="utf-8")
    config = {
        "name": "moth-stt",
        "version": "test",
        "sample_rate": 16000,
        "features": {
            "n_fft": 400,
            "hop": 160,
            "n_mels": 80,
            "window": "hann_periodic",
            "center": True,
            "pad_mode": "reflect",
            "drop_last_frame": True,
            "floor_range": 8.0,
            "scale": "(x+4)/4",
            "mel_filters": "filters.bin",
        },
        "model": {
            "file": "model.onnx",
            "input": "mel",
            "output": "log_probs",
            "frame_rate_hz": 50,
            "max_seconds": 30,
        },
        "tokenizer": {"vocab": "vocab.json", "blank_id": 0, "space_marker": "\u2581"},
        "decode": {
            "default": "beam",
            "beam": 12,
            "top_k": 10,
            "lm": "lm.bin",
            "lm_weight": 0.5,
            "token_bonus": 2.4,
        },
        "vad": {"file": "vad.onnx", "window": 512, "context": 64, "threshold": 0.1},
        "files": {},
    }
    for name in ["model.onnx", "vocab.json", "filters.bin", "lm.bin", "vad.onnx"]:
        config["files"][name] = (
            "sha256:" + hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_validation_precedes_session_creation(release, monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **k: pytest.fail("created session"))
    (release / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        ReleaseMoth(release)


def test_sessions_and_assets_load_once(release, monkeypatch):
    import onnxruntime as ort

    original = ort.InferenceSession
    sessions = []

    def create(path, **kwargs):
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        sessions.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", create)
    runtime = ReleaseMoth(release)
    assert len(sessions) == 2
    monkeypatch.setattr("stt.vad._session", lambda: pytest.fail("used installed VAD"))
    for _ in range(2):
        text, elapsed = runtime.transcribe(np.zeros(16000, np.float32))
        assert text == "" and elapsed >= 0
    assert len(sessions) == 2
    assert runtime.report["total_bytes"] < 20_000_000
    assert np.array_equal(
        runtime.filters, np.fromfile(release / "filters.bin", np.float32).reshape(80, 201)
    )


def test_token_bonus_matches_direct_decoder(release):
    runner = OnnxMoth(release / "model.onnx", release / "vocab.json")
    mel = np.zeros((80, 4), np.float32)
    logits = runner.log_probs(mel)
    assert decode_clip(runner, mel, decoder="beam") == ""
    expected = beam_search(logits, 2, runner.pieces, token_bonus=2.4)
    assert expected == "hello world"
    assert runner.transcribe(mel, decoder="beam", token_bonus=2.4)[0] == expected
    assert decode_clip(runner, mel, decoder="beam", token_bonus=2.4) == expected


@pytest.mark.parametrize(
    "mel",
    [
        np.zeros((79, 10)),
        np.zeros((80, 0)),
        np.zeros((80, 3001)),
        np.zeros((2, 80, 10)),
        np.full((80, 10), np.nan),
        np.full((80, 10), np.inf),
    ],
)
def test_low_level_rejects_bad_mel(release, mel):
    runner = OnnxMoth(release / "model.onnx", release / "vocab.json")
    with pytest.raises(ValueError):
        runner.log_probs(mel)


@pytest.mark.parametrize("shape", [(2, 3), (2, 2, 3), (1, 0, 3), (1, 2, 4)])
def test_output_shape_guard(release, shape):
    runner = OnnxMoth(release / "model.onnx", release / "vocab.json")
    runner.session = SimpleNamespace(run=lambda *a: [np.zeros(shape, np.float32)])
    with pytest.raises(ValueError, match="output"):
        runner.log_probs(np.zeros((80, 4)))


def test_model_metadata_guard(release):
    (release / "vocab.json").write_text('{"1":"only"}', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        OnnxMoth(release / "model.onnx", release / "vocab.json")


@pytest.mark.parametrize(
    "audio,rate",
    [
        (np.zeros((560, 2)), 16000),
        (np.zeros(559), 16000),
        (np.full(560, np.nan), 16000),
        (np.full(560, np.inf), 16000),
        (np.full(560, 1.01), 16000),
        (np.zeros(560), 48000),
        (np.full(560, 100, np.int16), 16000),
        (np.zeros(560, complex), 16000),
    ],
)
def test_invalid_pcm_rejected(release, monkeypatch, audio, rate):
    runtime = ReleaseMoth(release)
    monkeypatch.setattr("stt.onnx_runner.speech_probabilities", lambda *a, **k: pytest.fail("VAD"))
    with pytest.raises(ValueError):
        runtime.transcribe(audio, sample_rate=rate)


@pytest.mark.parametrize("runs", [0, 1, 2])
def test_vad_reused_and_only_multiple_runs_segment(release, monkeypatch, runs):
    runtime = ReleaseMoth(release)
    audio = np.zeros(WINDOW * 100, np.float32)
    probabilities = np.zeros(100, np.float32)
    probabilities[10:30] = 0.9
    if runs == 2:
        probabilities[60:80] = 0.9
    elif runs == 0:
        # Speech gate passes, but the run is shorter than min_segment_s.
        probabilities[:] = 0
        probabilities[:11] = 0.9
    calls = []

    def vad(samples, session):
        assert session is runtime.vad_session
        calls.append("vad")
        return probabilities

    def transcribe(mel, **settings):
        assert settings == {
            "decoder": "beam",
            "beam_width": 12,
            "top_k": 10,
            "lm": runtime.lm,
            "lm_weight": 0.5,
            "token_bonus": 2.4,
        }
        calls.append(mel.shape[1])
        return "text", 0.0

    monkeypatch.setattr("stt.onnx_runner.speech_probabilities", vad)
    monkeypatch.setattr(runtime.model, "transcribe", transcribe)
    text, elapsed = runtime.transcribe(audio)
    bounds = segment_bounds(audio, probabilities=probabilities)
    lengths = [(end - start) // 160 for start, end in bounds] if runs == 2 else [320]
    assert calls == ["vad", *lengths]
    assert text == ("text text" if runs == 2 else "text") and elapsed >= 0


def test_duration_checked_per_selected_span(release, monkeypatch):
    runtime = ReleaseMoth(release)
    audio = np.zeros(40 * 16000, np.float32)
    probabilities = np.ones((audio.size + 511) // 512, np.float32)
    monkeypatch.setattr("stt.onnx_runner.speech_probabilities", lambda *a, **k: probabilities)
    for segment in [False, True]:
        with pytest.raises(ValueError, match="30 seconds"):
            runtime.transcribe(audio, segment=segment)
    probabilities[500:750] = 0
    monkeypatch.setattr(runtime.model, "transcribe", lambda *a, **k: ("ok", 0.0))
    assert runtime.transcribe(audio)[0] == "ok ok"


@pytest.mark.parametrize("config_segment", [None, False, True])
@pytest.mark.parametrize("segment", [None, False, True])
def test_long_nonspeech_gated_before_span_checks(release, monkeypatch, config_segment, segment):
    if config_segment is not None:
        path = release / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["decode"]["segment"] = config_segment
        path.write_text(json.dumps(config), encoding="utf-8")
    runtime = ReleaseMoth(release)
    calls = []

    def vad(audio, **kwargs):
        calls.append("vad")
        return np.zeros((audio.size + 511) // 512, np.float32)

    monkeypatch.setattr("stt.onnx_runner.speech_probabilities", vad)
    monkeypatch.setattr("stt.onnx_runner.segment_bounds", lambda *a, **k: pytest.fail("segmented"))
    monkeypatch.setattr(runtime.model, "transcribe", lambda *a, **k: pytest.fail("decoded"))
    kwargs = {} if segment is None else {"segment": segment}
    text, elapsed = runtime.transcribe(np.zeros(40 * 16000, np.float32), **kwargs)
    assert text == "" and elapsed >= 0
    assert calls == ["vad"]


@pytest.mark.parametrize("config_segment", [None, False, True])
@pytest.mark.parametrize("segment", [None, False, True])
def test_config_segment_default_and_caller_override(release, monkeypatch, config_segment, segment):
    if config_segment is not None:
        path = release / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["decode"]["segment"] = config_segment
        path.write_text(json.dumps(config), encoding="utf-8")
    runtime = ReleaseMoth(release)
    audio = np.zeros(40 * 16000, np.float32)
    probabilities = np.ones((audio.size + 511) // 512, np.float32)
    probabilities[500:750] = 0
    monkeypatch.setattr("stt.onnx_runner.speech_probabilities", lambda *a, **k: probabilities)
    monkeypatch.setattr(runtime.model, "transcribe", lambda *a, **k: ("ok", 0.0))
    effective = segment if segment is not None else config_segment is not False
    kwargs = {} if segment is None else {"segment": segment}
    if effective:
        assert runtime.transcribe(audio, **kwargs)[0] == "ok ok"
    else:
        with pytest.raises(ValueError, match="30 seconds"):
            runtime.transcribe(audio, **kwargs)


@pytest.mark.parametrize("segment", [0, 1, "false", [], np.bool_(True)])
def test_invalid_caller_segment_rejected(release, monkeypatch, segment):
    runtime = ReleaseMoth(release)
    monkeypatch.setattr("stt.onnx_runner.speech_probabilities", lambda *a, **k: pytest.fail("VAD"))
    with pytest.raises(ValueError, match="segment must be a boolean"):
        runtime.transcribe(np.zeros(560), segment=segment)


def test_release_decode_matches_direct_binary_lm_and_uses_release_filters(release, monkeypatch):
    runtime = ReleaseMoth(release)
    runtime.filters[:] = 0
    audio = np.full(16000, 0.01, np.float32)
    probabilities = np.ones((audio.size + 511) // 512, np.float32)
    monkeypatch.setattr("stt.onnx_runner.speech_probabilities", lambda *a, **k: probabilities)
    original = runtime.model.log_probs
    seen = []

    def logits(mel):
        seen.append(mel)
        return original(mel)

    monkeypatch.setattr(runtime.model, "log_probs", logits)
    expected_mel = log_mel(audio, filters=runtime.filters)
    expected = beam_search(
        original(expected_mel),
        2,
        runtime.model.pieces,
        beam_width=12,
        top_k=10,
        lm=runtime.lm,
        lm_weight=0.5,
        token_bonus=2.4,
    )
    assert runtime.transcribe(audio, segment=False)[0] == expected
    assert np.array_equal(seen[0], expected_mel)


def test_exact_pcm_duration_boundaries(release, monkeypatch):
    runtime = ReleaseMoth(release)
    monkeypatch.setattr(
        "stt.onnx_runner.speech_probabilities",
        lambda audio, **k: np.ones((audio.size + 511) // 512, np.float32),
    )
    monkeypatch.setattr(runtime.model, "transcribe", lambda *a, **k: ("ok", 0.0))
    for size in [560, 30 * 16000]:
        assert runtime.transcribe(np.zeros(size), segment=False)[0] == "ok"
    with pytest.raises(ValueError, match="30 seconds"):
        runtime.transcribe(np.zeros(30 * 16000 + 1), segment=False)


def test_gate_threshold_is_strict(release, monkeypatch):
    runtime = ReleaseMoth(release)
    probabilities = np.zeros(20, np.float32)
    probabilities[:2] = 0.9
    monkeypatch.setattr("stt.onnx_runner.speech_probabilities", lambda *a, **k: probabilities)
    monkeypatch.setattr(runtime.model, "transcribe", lambda *a, **k: pytest.fail("decoded"))
    assert runtime.transcribe(np.zeros(20 * WINDOW))[0] == ""


def test_explicit_vad_session_and_probabilities(monkeypatch):
    calls = []

    def run(outputs, feeds):
        calls.append(feeds["state"].copy())
        return np.array([[0.75]], np.float32), feeds["state"] + 1

    monkeypatch.setattr("stt.vad._session", lambda: pytest.fail("default session"))
    audio = np.zeros(WINDOW * 20, np.float32)
    session = SimpleNamespace(run=run)
    probabilities = speech_probabilities(audio, session=session)
    assert len(calls) == 20 and calls[0].shape == STATE_SHAPE
    assert segment_bounds(audio, probabilities=probabilities) == [(0, audio.size)]
    speech_probabilities(audio[:560], session=session)
    assert np.all(calls[20] == 0)  # Recurrent state is reset for each clip.
    with pytest.raises(ValueError, match="probabilities"):
        segment_bounds(audio, probabilities=np.array([np.nan]))


def test_cli(release, tmp_path, capsys):
    sf = pytest.importorskip("soundfile")
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(560), 16000)
    assert main([str(release), str(audio)]) == 0
    assert "Elapsed:" in capsys.readouterr().out
    sf.write(audio, np.zeros(560), 8000)
    with pytest.raises(SystemExit) as exc:
        main([str(release), str(audio)])
    assert exc.value.code == 1


def test_runtime_import_does_not_load_training_dependencies():
    code = (
        "import sys; import stt.onnx_runner; "
        "assert not any(m in sys.modules for m in ['torch', 'onnx', 'sentencepiece', 'soundfile'])"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_optional_shipped_release():
    root = os.environ.get("MOTH_RELEASE_DIR")
    if not root:
        pytest.skip("set MOTH_RELEASE_DIR to validate and run a complete shipped release")
    pytest.importorskip("onnxruntime")
    runtime = ReleaseMoth(Path(root))
    assert runtime.transcribe(np.zeros(16000, np.float32))[0] == ""
