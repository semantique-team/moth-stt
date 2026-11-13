"""E-041 local validation only: no browser inference or release assets needed."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e041_score", REPO / "runs/E-041/score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


@pytest.fixture
def provenance():
    expected = json.loads(score.RELEASE_RECORD.read_text())
    reference = json.loads(score.PYTHON_RUN.read_text())
    run = {"failures": 0, "provenance": {
        "version": "0.8.1", "model": "model.int8.onnx", "decode": copy.deepcopy(expected["decode"]),
        "vad": copy.deepcopy(expected["vad"]), "segmentation": True, "cross_origin_isolated": True,
        "config_sha256": expected["files"]["config.json"]["sha256"],
        "files": {file: f"sha256:{record['sha256']}"
                  for file, record in expected["files"].items()},
    }}
    return copy.deepcopy((run, reference, expected))


def test_committed_e037_matches_expected_release(provenance):
    score.validate_provenance(*provenance)


@pytest.mark.parametrize("field,value", [
    ("version", "0.8.0"), ("model", "other.onnx"), ("config_sha256", "0" * 64),
    ("segmentation", False), ("cross_origin_isolated", False), ("vad", {}),
])
def test_wrong_browser_provenance_rejected(provenance, field, value):
    run, reference, expected = provenance
    run["provenance"][field] = value
    with pytest.raises(SystemExit, match="provenance"):
        score.validate_provenance(run, reference, expected)


@pytest.mark.parametrize("file", ["model.int8.onnx", "lm_bpe3.bin", "vocab.json",
                                  "mel_filters.bin", "silero_vad.onnx"])
def test_self_consistent_wrong_release_hash_rejected(provenance, file):
    run, reference, expected = provenance
    run["provenance"]["files"][file] = "sha256:" + "0" * 64
    with pytest.raises(SystemExit, match="identity mismatch"):
        score.validate_provenance(run, reference, expected)


@pytest.mark.parametrize("side", ["browser", "reference"])
def test_wrong_decoder_rejected(provenance, side):
    run, reference, expected = provenance
    if side == "browser":
        run["provenance"]["decode"]["lm_weight"] = 0.1
    else:
        reference["params"]["beam"] = 1
    with pytest.raises(SystemExit, match="decoder"):
        score.validate_provenance(run, reference, expected)


@pytest.mark.parametrize("asset", ["int8", "lm", "vocab"])
def test_wrong_reference_asset_rejected(provenance, asset):
    run, reference, expected = provenance
    reference["assets"][asset]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="E-037 identity"):
        score.validate_provenance(run, reference, expected)


def test_missing_provenance_rejected(provenance):
    _, reference, expected = provenance
    with pytest.raises(SystemExit, match="provenance"):
        score.validate_provenance({}, reference, expected)


def test_failed_run_rejected(provenance):
    run, reference, expected = provenance
    run["failures"] = 1
    with pytest.raises(SystemExit, match="provenance"):
        score.validate_provenance(run, reference, expected)


def test_duplicate_hypothesis_ids_rejected(tmp_path):
    path = tmp_path / "hyp.jsonl"
    path.write_text('{"id":"T-0001","hyp":"a"}\n{"id":"T-0001","hyp":"b"}\n')
    with pytest.raises(SystemExit, match="duplicate IDs"):
        score.read_jsonl(path)


def test_main_rejects_provenance_before_scoring_or_output(tmp_path, monkeypatch):
    out = tmp_path / "readout" / "bad"
    out.mkdir(parents=True)
    (out / "browser.run.json").write_text("{}")
    for file in ("browser.hyp.jsonl", "browser.timings.jsonl"):
        (out / file).write_text("")
    monkeypatch.setattr(score, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["score.py", "--name", "bad"])
    monkeypatch.setattr(score, "score_rows", lambda *a, **k: pytest.fail("must not score"))
    with pytest.raises(SystemExit, match="provenance"):
        score.main()
    assert not (out / "metrics.json").exists()


@pytest.fixture
def measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "REPO", tmp_path)
    manifest = tmp_path / "recordings/manifests/T.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = [{"id": "T-0001", "audio": "S1/T-0001.wav", "text": "original"}]
    manifest.write_text(json.dumps(rows[0]) + "\n")
    monkeypatch.setattr(score, "MANIFEST", manifest)
    audio = tmp_path / "recordings/audio/S1/T-0001.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"original audio bytes")
    paths = ["runs/E-041/browser_wer.mjs", "runs/E-041/score.py", "web/server.py",
             "web/ort.js", "web/mel.js", "web/decode.js", "web/lm.js", "web/vad.js",
             "web/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs",
             "web/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
             "web/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"]
    for file in paths:
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.encode())
    run = {"measurement": {
        "manifest": {"path": manifest.relative_to(tmp_path).as_posix(),
                     "sha256": score.sha256_file(manifest)},
        "audio": [{"id": rows[0]["id"], "path": audio.relative_to(tmp_path).as_posix(),
                   "sha256": score.sha256_file(audio)}],
        "runtime": [{"path": file, "sha256": score.sha256_file(tmp_path / file)}
                    for file in paths],
    }}
    return run, rows


def test_measurement_receipts_match_local_files(measurement):
    run, rows = measurement
    score.validate_measurement(run, rows, {}, {})


def test_same_id_manifest_text_substitution_rejected(measurement):
    run, rows = measurement
    rows[0]["text"] = "substituted"
    score.MANIFEST.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(SystemExit, match="manifest identity"):
        score.validate_measurement(run, rows, {}, {})


def test_raw_manifest_bytes_change_rejected(measurement):
    run, rows = measurement
    with score.MANIFEST.open("ab") as file:
        file.write(b"\n")
    with pytest.raises(SystemExit, match="manifest identity"):
        score.validate_measurement(run, rows, {}, {})


def test_same_id_audio_substitution_rejected(measurement):
    run, rows = measurement
    (score.REPO / run["measurement"]["audio"][0]["path"]).write_bytes(b"different audio")
    with pytest.raises(SystemExit, match="local file identity.*T-0001.wav"):
        score.validate_measurement(run, rows, {}, {})


@pytest.mark.parametrize("file", ["web/ort.js", "web/mel.js", "web/decode.js", "web/lm.js",
                                  "web/vad.js", "runs/E-041/score.py",
                                  "web/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs",
                                  "web/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"])
def test_changed_local_runtime_rejected(measurement, file):
    run, rows = measurement
    (score.REPO / file).write_bytes(b"changed code")
    with pytest.raises(SystemExit, match="local file identity"):
        score.validate_measurement(run, rows, {}, {})


@pytest.mark.parametrize("location", ["run", "product", "metrics"])
def test_retained_e037_manifest_mismatch_rejected(measurement, location):
    run, rows = measurement
    reference, metrics = {}, {}
    retained = {"assets": {"manifest": {"sha256": "0" * 64}}}
    if location == "run":
        reference = retained
    elif location == "product":
        reference = {"product": {"t600": retained}}
    else:
        metrics = retained
    with pytest.raises(SystemExit, match="E-037 retained manifest"):
        score.validate_measurement(run, rows, reference, metrics)


def test_matching_retained_e037_manifest_allowed(measurement):
    run, rows = measurement
    metrics = {"manifest_sha256": run["measurement"]["manifest"]["sha256"]}
    score.validate_measurement(run, rows, {}, metrics)


@pytest.mark.parametrize("receipt", ["audio", "runtime"])
def test_missing_measurement_receipts_rejected(measurement, receipt):
    run, rows = measurement
    run["measurement"][receipt] = []
    with pytest.raises(SystemExit, match=f"{receipt} receipts"):
        score.validate_measurement(run, rows, {}, {})


def test_audio_path_substitution_rejected(measurement):
    run, rows = measurement
    run["measurement"]["audio"][0]["path"] = "recordings/audio/other.wav"
    with pytest.raises(SystemExit, match="audio path mismatch"):
        score.validate_measurement(run, rows, {}, {})


@pytest.mark.parametrize("changed", ["manifest", "audio"])
def test_main_rejects_substitution_before_scoring(measurement, provenance, monkeypatch, changed):
    run, rows = measurement
    run.update(provenance[0])
    out = score.REPO / "readout" / "changed"
    out.mkdir(parents=True)
    (out / "browser.run.json").write_text(json.dumps(run))
    for file in ("browser.hyp.jsonl", "browser.timings.jsonl"):
        (out / file).write_text("")
    if changed == "manifest":
        rows[0]["text"] = "substituted"
        score.MANIFEST.write_text(json.dumps(rows[0]) + "\n")
    else:
        (score.REPO / run["measurement"]["audio"][0]["path"]).write_bytes(b"substituted")
    monkeypatch.setattr(score, "OUT", score.REPO)
    monkeypatch.setattr(sys, "argv", ["score.py", "--name", "changed"])
    monkeypatch.setattr(score, "score_rows", lambda *a, **k: pytest.fail("must not score"))
    with pytest.raises(SystemExit, match="identity mismatch"):
        score.main()
    assert not (out / "metrics.json").exists()
