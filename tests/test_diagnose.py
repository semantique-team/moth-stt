"""Exact tiny CTC checks and synthetic CLI checks; no acoustic model or training."""

import hashlib
import itertools
import json

import numpy as np
import pytest

from stt.decode import beam_search
from stt.diagnose import (
    ctc_forward,
    diagnose_utterance,
    log_softmax,
    main,
    reference_support,
    top_k_mask,
    verify_tokenizer,
)
from stt.lm import BPETrigramLM


def brute_force(array, labels, mask=None):
    scores = []
    for path in itertools.product(range(array.shape[1]), repeat=len(array)):
        collapsed = [v for i, v in enumerate(path) if v and (i == 0 or v != path[i - 1])]
        if collapsed == labels and (mask is None or all(mask[t, v] for t, v in enumerate(path))):
            scores.append(sum(array[t, v] for t, v in enumerate(path)))
    return float(np.logaddexp.reduce(scores)) if scores else -np.inf


@pytest.mark.parametrize("frames", range(5))
@pytest.mark.parametrize("labels", [[], [1], [2], [1, 1], [1, 2], [2, 1, 2]])
def test_forward_matches_exhaustive_paths(frames, labels):
    array = log_softmax(np.random.default_rng(frames).normal(size=(frames, 3)))
    for mask in (None, top_k_mask(array, 1)):
        assert ctc_forward(array, labels, mask) == pytest.approx(brute_force(array, labels, mask))


def test_repeated_labels_need_blank_and_enough_frames():
    array = log_softmax(np.zeros((2, 3)))
    assert ctc_forward(array, [1, 1]) == -np.inf
    array = np.array([[-np.inf, 0, -np.inf], [0, -np.inf, -np.inf], [-np.inf, 0, -np.inf]])
    assert ctc_forward(array, [1, 1]) == 0
    assert ctc_forward(array, [2]) == -np.inf
    assert ctc_forward(np.empty((0, 3)), []) == 0
    assert ctc_forward(np.empty((0, 3)), [1]) == -np.inf


@pytest.mark.parametrize("labels", [[0], [-1], [3], [True], [1.0]])
def test_invalid_reference_ids(labels):
    with pytest.raises(ValueError, match="nonblank IDs"):
        ctc_forward(log_softmax(np.zeros((2, 3))), labels)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros(3),
        np.zeros((2, 0)),
        [[np.nan, 0]],
        [[np.inf, 0]],
        [[-np.inf, -np.inf]],
        [[1j, 0]],
        [["a", "b"]],
    ],
)
def test_invalid_scores(array):
    with pytest.raises(ValueError):
        log_softmax(array)
    with pytest.raises(ValueError):
        ctc_forward(array, [])


def test_log_softmax_stable_with_impossible_labels():
    result = log_softmax(np.array([[10000, 10001, -np.inf], [-np.inf, -10000, -10000]]))
    assert not np.isnan(result).any()
    assert np.exp(result).sum(axis=1) == pytest.approx([1, 1])
    assert np.isneginf(result[0, 2])
    with pytest.raises(ValueError, match="normalized"):
        ctc_forward(np.zeros((1, 3)), [1])


def test_candidate_ties_match_decoder():
    array = log_softmax(np.array([[-100, 0, 0, 0], [-100, 0, 1, 1]]))
    mask = top_k_mask(array, 1)
    assert mask.tolist() == [[True, True, False, False], [True, False, True, False]]
    pieces = {1: "a", 2: "b", 3: "c"}
    assert beam_search(array, 2, pieces, top_k=1) == "ab"
    assert top_k_mask(array, 99).all()
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            top_k_mask(array, value)


def test_support_removes_mass_without_renormalizing_or_nan():
    array = log_softmax(np.array([[0, 2, 1], [0, 1, 2], [0, 2, 1]]))
    result = reference_support(array, [1], 1)
    assert result["full_reachable"] and result["top_k_reachable"]
    assert 0 < result["retained_probability_fraction"] < 1
    assert result["log_probability_loss"] > 0
    pruned = reference_support(array[:1], [2], 1)
    assert pruned["all_reference_mass_pruned"]
    assert pruned["retained_probability_fraction"] == 0
    impossible = reference_support(array[:1], [1, 1], 1)
    assert not impossible["full_reachable"]
    assert impossible["retained_probability_fraction"] is None
    json.dumps([result, pruned, impossible], allow_nan=False)
    for mask in (np.ones((1, 1), dtype=bool), np.ones(array.shape)):
        with pytest.raises(ValueError, match="mask"):
            ctc_forward(array, [1], mask)


class FakeTokenizer:
    def get_piece_size(self):
        return 1024

    def id_to_piece(self, identifier):
        return f"p{identifier}"

    def unk_id(self):
        return 0

    def encode(self, text, out_type):
        return [1]

    def decode(self, ids):
        return "meeting"


def test_tokenizer_checks_every_id_including_last_and_unk():
    processor = FakeTokenizer()
    pieces = {i + 1: processor.id_to_piece(i) for i in range(1024)}
    verify_tokenizer(processor, pieces)
    for identifier in (1, 512, 1024):
        changed = pieces | {identifier: "wrong"}
        with pytest.raises(ValueError, match="all 1024"):
            verify_tokenizer(processor, changed)
    with pytest.raises(ValueError):
        verify_tokenizer(processor, {1: "p0"})


def test_optional_reference_scoring_and_unknown_ref(monkeypatch):
    processor = FakeTokenizer()
    pieces = {i + 1: processor.id_to_piece(i) for i in range(1024)}
    row = {"id": "u", "norm": "meeting", "lang": "mix", "en_spans": [[0, 1]]}
    array = log_softmax(np.zeros((2, 1025)))
    settings = {"beam": 2, "top_k": 2, "lm_weight": 0, "token_bonus": 0}
    _, detail = diagnose_utterance(array, row, pieces, settings, None, processor)
    assert detail["forced_reference"]["available"]
    assert detail["forced_reference"]["ctc_ids"] == [2]
    monkeypatch.setattr(processor, "encode", lambda *a, **k: [0])
    _, detail = diagnose_utterance(array, row, pieces, settings, None, processor)
    assert "unk" in detail["forced_reference"]["reason"]
    monkeypatch.setattr(processor, "encode", lambda *a, **k: [1])
    monkeypatch.setattr(processor, "decode", lambda ids: "different")
    _, detail = diagnose_utterance(array, row, pieces, settings, None, processor)
    assert "round-trip" in detail["forced_reference"]["reason"]


@pytest.mark.parametrize("frames,vocab", [(0, 3), (1501, 3), (1, 4)])
def test_utterance_frame_and_vocab_limits(frames, vocab):
    row = {"id": "u", "norm": "", "lang": "mix", "en_spans": []}
    with pytest.raises(ValueError, match="1..1500 frames"):
        diagnose_utterance(log_softmax(np.zeros((frames, vocab))), row, {1: "a", 2: "b"}, {}, None)


@pytest.fixture
def audit(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    pieces = {str(i): f"p{i}" for i in range(1, 1025)}
    pieces["2"] = "\u2581meeting"
    for name in ("model.onnx", "vad.onnx", "filters.bin", "NOTICE"):
        (root / name).write_bytes(b"synthetic, never executed")
    (root / "vocab.json").write_text(json.dumps(pieces), encoding="utf-8")
    BPETrigramLM({1: 2}, {}, {}).save_browser(root / "lm.bin")
    config = {
        "name": "moth-stt",
        "version": "0.7.0",
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
        "files": {
            p.name: "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.iterdir()
            if p.name != "NOTICE"
        },
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    manifest = tmp_path / "refs.jsonl"
    row = {
        "id": "u",
        "text": "meeting meeting",
        "norm": "meeting meeting",
        "duration": 1,
        "source": "synthetic",
        "split": "dev",
        "lang": "mix",
        "en_spans": [[0, 2]],
        "audio": "missing.wav",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    logits = tmp_path / "scores.npz"
    array = np.full((1, 1025), -np.inf)
    array[0, 2] = 0
    np.savez(logits, u=array)
    out = tmp_path / "out"
    args = [str(root), "--manifest", str(manifest), "--logits", str(logits), "--out", str(out)]
    return root, manifest, logits, out, args


def test_cli_synthetic_report_and_official_metrics(audit, capsys):
    _root, manifest, logits, out, args = audit
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    report = json.loads((out / "report.json").read_text())
    assert report["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert report["input"]["sha256"] == hashlib.sha256(logits.read_bytes()).hexdigest()
    assert report["release"]["files"]["model.onnx"]["hash_verified"]
    assert not report["utterances"][0]["forced_reference"]["available"]
    assert report["decode"]["beam_acoustic0"]["token_bonus"] == 0
    assert report["decode"]["beam_lm"]["token_bonus"] == 2.4
    for name in ("greedy", "beam_acoustic0", "beam_lm"):
        assert report["metrics"][name]["overall"]["wer"] == 0.5
        assert report["metrics"][name]["overall"]["eta"] == 0.5
        assert (
            report["utterances"][0]["english_word_presence_not_eta"][name]["present_anywhere"] == 2
        )
        assert json.loads((out / f"{name}.hyp.jsonl").read_text()) == {"id": "u", "hyp": "meeting"}
    assert main(args) == 1
    assert "refusing to overwrite" in capsys.readouterr().out


@pytest.mark.parametrize(
    "problem", ["hash", "path", "missing_manifest", "spans", "duration", "keys", "nan", "audio"]
)
def test_cli_rejects_bad_inputs_before_output(audit, problem, capsys):
    root, manifest, logits, out, args = audit
    if problem == "hash":
        (root / "model.onnx").write_bytes(b"tampered")
    elif problem == "path":
        config_path = root / "config.json"
        config = json.loads(config_path.read_text())
        config["model"]["file"] = "../outside.onnx"
        config_path.write_text(json.dumps(config), encoding="utf-8")
    elif problem == "missing_manifest":
        manifest.unlink()
    elif problem in ("spans", "duration"):
        row = json.loads(manifest.read_text())
        if problem == "spans":
            del row["en_spans"]
        else:
            row["duration"] = float("nan")
        manifest.write_text(json.dumps(row), encoding="utf-8")
    elif problem == "keys":
        np.savez(logits, other=np.zeros((1, 1025)))
    elif problem == "nan":
        np.savez(logits, u=np.full((1, 1025), np.nan))
    else:
        args = args[:-4] + args[-2:]  # remove --logits: missing explicit audio must fail
    assert main(args) == 1
    assert not out.exists()
    assert json.loads(capsys.readouterr().out)["ok"] is False
