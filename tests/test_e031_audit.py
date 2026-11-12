"""Local evidence guards, without weights, training, or network."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stt.eval import macro_wer_bootstrap

RUN = Path(__file__).resolve().parents[1] / "runs/E-031"


def load(name):
    spec = importlib.util.spec_from_file_location(name, RUN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = load("audit")


@pytest.fixture
def pinned_history():
    for revision in (audit.E031_REVISION, audit.INPUT_REVISION, audit.RELEASE_REVISION):
        completed = subprocess.run(
            ["git", "-C", str(audit.REPO), "cat-file", "-e", f"{revision}^{{commit}}"],
            capture_output=True, check=False,
        )
        if completed.returncode:
            if os.environ.get("CI"):
                pytest.fail(f"CI requires pinned Git history: {revision}; use fetch-depth: 0")
            pytest.skip(f"missing pinned Git history: {revision}; requires full history "
                        "(actions/checkout fetch-depth: 0); no network fetch performed")


def test_all_original_inputs_match_pre_e032_revision(pinned_history):
    receipt = json.loads(audit.git_bytes("runs/E-031/frozen/lm-corpus.receipt.json"))
    for item in receipt["inputs"]:
        relative = item["path"].split("/moth-stt/", 1)[1]
        assert audit.digest(audit.git_bytes(relative, audit.INPUT_REVISION)) == item["sha256"]
    old = audit.git_bytes("recordings/scripts/R-S1.jsonl", audit.INPUT_REVISION)
    assert len(old.splitlines()) == 691
    assert old != audit.git_bytes("recordings/scripts/R-S1.jsonl", "HEAD")


def test_immutable_release_records(pinned_history):
    old = audit.release_record("0.8.0")
    new = audit.release_record("0.8.1")
    assert old["decode"] == new["decode"]
    assert old["files"]["model.int8.onnx"] == new["files"]["model.int8.onnx"]
    assert old["files"]["lm_bpe3.bin"] != new["files"]["lm_bpe3.bin"]


def test_source_checks_before_mutating_stage(tmp_path, monkeypatch):
    old = {"release": "0.8.0", "files": {}, "decode": {"beam": 12}, "vad": {"threshold": 0.1}}
    names = ("model.int8.onnx", "vocab.json", "mel_filters.bin", "silero_vad.onnx", "NOTICE")
    for name in names:
        content = name.encode()
        (tmp_path / name).write_bytes(content)
        old["files"][name] = {"sha256": audit.digest(content)}
    config = {"version": "0.8.1", "decode": old["decode"], "vad": old["vad"]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(audit, "release_record", lambda _: old)
    assert audit.verify_source(tmp_path) == old  # Already staged 0.8.1 is a valid source.
    config["decode"] = {"beam": 5}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="decoder/VAD"):
        audit.verify_source(tmp_path)
    (tmp_path / "model.int8.onnx").write_bytes(b"wrong graph")
    with pytest.raises(ValueError, match="verified 0.8.0"):
        audit.verify_source(tmp_path)


def test_lm_identity_and_preservation(tmp_path):
    path = tmp_path / "lm.bin"
    path.write_bytes(b"new LM")
    record = {"release": "0.8.0", "files": {"lm_bpe3.bin": {"sha256": audit.digest(b"old LM")}}}
    with pytest.raises(ValueError, match="verified 0.8.0"):
        audit.verify_file(path, record, "lm_bpe3.bin")
    with pytest.raises(FileExistsError, match="new --name"):
        audit.require_new(path)
    assert path.read_bytes() == b"new LM"


def test_gate_cache_attribution_consistency_and_legacy(tmp_path):
    pytest.importorskip("num2words")
    gate = load("gate")
    rows = [{"id": "a", "norm": "a"}]
    hyp = tmp_path / "hyp.jsonl"
    metrics = tmp_path / "metrics.json"
    fingerprint = {"model": "model-a", "lm": "lm-a"}
    hypotheses, scored = gate.cached_score(hyp, metrics, rows, fingerprint, lambda: {"a": "a"})
    assert gate.cached_score(hyp, metrics, rows, fingerprint, lambda: pytest.fail("redecoded")) \
        == (hypotheses, scored)
    preserved = metrics.read_bytes()
    with pytest.raises(ValueError, match="new --name"):
        gate.cached_score(hyp, metrics, rows, {**fingerprint, "lm": "lm-b"}, dict)
    assert metrics.read_bytes() == preserved
    stored = json.loads(preserved)
    stored["overall"]["wer"] = 1
    metrics.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="disagree"):
        gate.cached_score(hyp, metrics, rows, fingerprint, dict)
    metrics.write_text(json.dumps(scored))
    with pytest.raises(ValueError, match="legacy"):
        gate.cached_score(hyp, metrics, rows, fingerprint, dict)


def test_target_only_alignment_excludes_other_span_words():
    pytest.importorskip("num2words")
    gate = load("gate")
    from stt.eval import score_rows

    rows = [{"id": "a", "norm": "startup meeting", "en_spans": [[0, 2]]}]
    target_rows = gate.target_rare_rows(rows)
    assert rows[0]["en_spans"] == [[0, 2]]
    assert target_rows[0]["en_spans"] == [[0, 1]]
    assert score_rows(rows, {"a": "startup"})["overall"]["eta"] == 0.5
    assert score_rows(target_rows, {"a": "startup"})["overall"]["eta"] == 1


def test_reproducer_original_hash_and_receipt_preserved(tmp_path, pinned_history):
    pytest.importorskip("num2words")
    reproducer = load("reproduce_corpus")
    receipt = RUN / "frozen/lm-corpus.receipt.json"
    before = receipt.read_bytes()
    out = tmp_path / "corpus.jsonl"
    result = reproducer.reproduce(out)
    assert audit.digest(out.read_bytes()) == result["corpus_sha256"]
    assert out.stat().st_size == 24756041
    assert reproducer.reproduce(out) == result
    assert receipt.read_bytes() == before
    out.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="choose a new"):
        reproducer.reproduce(out)


def test_noise_music_separate_and_failed_or_incomplete_exit():
    pytest.importorskip("num2words")
    gate = load("gate")
    from stt.eval import score_rows

    rows = [{"id": "noise", "norm": ""}, {"id": "music", "norm": "", "category": "music"}]
    base = score_rows(rows, {"noise": "", "music": "a b"})["noise"]
    new = score_rows(rows, {"noise": "a", "music": ""})["noise"]
    assert gate.noise_gates(new, base) == {"noise_non_music": False, "noise_music": True}
    assert gate.gate_exit({"noise": False}) == 1
    assert gate.gate_exit({"noise": None}) == 1
    assert gate.gate_exit({"noise": True}) == 0


def test_changed_python_syntax_and_plan_threshold(pinned_history):
    for path in [*RUN.glob("*.py"), RUN.parents[1] / "stt/eval.py"]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert b"rare aligned ETA >= 28.4%" in audit.git_bytes("runs/E-031/PLAN.md")
    assert '"rare_eta": 0.284' in (RUN / "gate.py").read_text(encoding="utf-8")


def test_parity_does_not_certify_legacy_mismatch_or_other_assets():
    assets = {"model": "model", "lm": "lm", "vocab": "vocab"}
    params = {"beam": 12}
    code = {"decoder": "sha"}
    report = {"assets": assets, "params": params, "code": code, "audio_sha256": "audio",
              "exact_match": True, "python_text": "a", "js_text": "a", "frames": 4,
              "scope": "same-logits decoder only", "logits_sha256": "a" * 64}
    assert audit.verified_parity(report, assets, params, code, "audio")
    for changed in ({"js_text": "b"}, {"exact_match": False}, {"assets": {}},
                    {"code": {}}, {"scope": "browser end-to-end"}, {"frames": 0},
                    {"audio_sha256": "other"}, {"logits_sha256": None}):
        assert not audit.verified_parity({**report, **changed}, assets, params, code, "audio")
    assert not audit.verified_parity({"exact_match": True}, assets, params, code, "audio")


def test_macro_count_math():
    rows = [{"id": "it", "lang": "it"}] + [{"id": f"en{i}", "lang": "en"} for i in range(9)]
    perfect = {"ref_words": 1, "sub": 0, "del": 0, "ins": 0, "en_ref": 0, "en_hit": 0}
    base = {"per_utterance": {row["id"]: perfect for row in rows}}
    new = {"per_utterance": {**base["per_utterance"], "it": {**perfect, "sub": 1}}}
    result = macro_wer_bootstrap(rows, new, base, bootstrap=100)
    assert result["point"] == 0.5
    assert result["ci"] == [0.5, 0.5]
    assert result["strata"] == {"it": 1, "en": 9}
    assert result["tie"] is False


def test_receipt_comparison_uses_relative_identity_not_order(tmp_path):
    reproducer = load("reproduce_corpus")
    originals = [
        {"path": f"/Users/old/moth-stt/{name}", "role": "exclusion", "sha256": name, "rows": 1}
        for name in ("T.jsonl", "T-S2.jsonl")
    ]
    generated = [{**item, "path": str(tmp_path / Path(item["path"]).name)}
                 for item in reversed(originals)]
    assert reproducer.receipt_inputs(generated, tmp_path) == reproducer.receipt_inputs(originals)
    generated[0]["sha256"] = "wrong"
    assert reproducer.receipt_inputs(generated, tmp_path) != reproducer.receipt_inputs(originals)
    with pytest.raises(ValueError, match="duplicate receipt"):
        reproducer.receipt_inputs(originals * 2)


def test_missing_history_skips_explicitly(monkeypatch, request):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 1, stdout=b"", stderr=b"missing commit"))
    with pytest.raises(pytest.skip.Exception, match="missing pinned Git history.*full history"):
        request.getfixturevalue("pinned_history")
    with pytest.raises(RuntimeError, match="full Git history is required"):
        audit.git_bytes("runs/E-031/PLAN.md")


def test_missing_history_fails_in_ci(monkeypatch, request):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 1, stdout=b"", stderr=b"missing commit"))
    with pytest.raises(pytest.fail.Exception, match="CI requires pinned Git history.*fetch-depth: 0"):
        request.getfixturevalue("pinned_history")


def test_terms_retry_reuses_matching_artifacts_and_preserves_stale(tmp_path, monkeypatch):
    gate = load("gate")
    baseline = b'{"id":"a","hyp":"a"}\n'
    provenance = {"revision": "pinned", "inputs": {}}
    result = {"groups": {"natural": {"new": {"correct": 431}}}}
    monkeypatch.setattr(gate, "verify_term_inputs", lambda: provenance)
    monkeypatch.setattr(gate, "git_bytes", lambda _: baseline)
    calls = []

    def run(command, **kwargs):
        destination = Path(command[command.index("--out") + 1])
        assert destination.parent != tmp_path
        calls.append(destination)
        destination.write_text(json.dumps(result))

    monkeypatch.setattr(gate.subprocess, "run", run)
    assert gate.score_terms(tmp_path) == (result, provenance)
    terms = tmp_path / "terms.json"
    pinned = tmp_path / "baseline-t600.hyp.jsonl"
    before = (terms.read_bytes(), pinned.read_bytes(), terms.stat().st_mtime_ns)
    # Noise may fail after this step; the same name can safely resume here.
    assert gate.score_terms(tmp_path) == (result, provenance)
    assert (terms.read_bytes(), pinned.read_bytes(), terms.stat().st_mtime_ns) == before
    assert len(calls) == 2

    def failed_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(gate.subprocess, "run", failed_run)
    with pytest.raises(subprocess.CalledProcessError):
        gate.score_terms(tmp_path)
    assert (terms.read_bytes(), pinned.read_bytes(), terms.stat().st_mtime_ns) == before
    monkeypatch.setattr(gate.subprocess, "run", run)
    terms.write_text('{"stale":true}')
    stale = terms.read_bytes()
    with pytest.raises(ValueError, match="stale terms.*new --name"):
        gate.score_terms(tmp_path)
    assert terms.read_bytes() == stale
    pinned.write_bytes(b"wrong baseline")
    with pytest.raises(ValueError, match="stale pinned baseline"):
        gate.score_terms(tmp_path)
    assert pinned.read_bytes() == b"wrong baseline"


def test_term_input_drift_rejected_before_decode(tmp_path, monkeypatch):
    gate = load("gate")
    monkeypatch.setattr(gate, "REPO", tmp_path)
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "TERM_INPUTS", ("R-S1.jsonl",))
    monkeypatch.setattr(gate, "git_bytes", lambda *args: b"historical\n")
    path = tmp_path / "R-S1.jsonl"
    path.write_bytes(b"historical\r\n")
    assert gate.verify_term_inputs()["inputs"]["R-S1.jsonl"] == audit.digest(b"historical\n")
    path.write_bytes(b"historical\nE032 addition\n")
    monkeypatch.setattr(gate, "verify_source", lambda _: {})
    monkeypatch.setattr(gate, "_e022", lambda: pytest.fail("loaded decoder before drift check"))
    monkeypatch.setattr(sys, "argv", ["gate.py", "--lm", str(path), "--name", "retry",
                                      "--skip-noise"])
    with pytest.raises(ValueError, match="term input drift.*R-S1"):
        gate.main()
    assert not (tmp_path / "eval/retry").exists()


def test_fingerprint_streams_shards_and_detects_changed_contents(tmp_path, monkeypatch):
    gate = load("gate")
    monkeypatch.setattr(gate, "REPO", tmp_path)
    monkeypatch.setattr(gate, "version", lambda _: "test")
    for name in ("model.int8.onnx", "vocab.json"):
        path = tmp_path / "release" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"asset")
    lm = tmp_path / "lm.bin"
    manifest = tmp_path / "manifest.jsonl"
    shard = tmp_path / "shard.npy"
    lm.write_bytes(b"LM")
    manifest.write_bytes(b"manifest")
    shard.write_bytes(b"first")
    original_read = Path.read_bytes

    def read_bytes(path):
        if path == shard:
            pytest.fail("whole shard read into memory")
        return original_read(path) if path.exists() else b"code"

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    rows = [{"id": "a"}]
    index = {"a": {"shard": shard.name}}
    first = gate.cache_fingerprint(lm, manifest, rows, index, tmp_path)
    assert first["shards"][shard.name] == audit.digest(b"first")
    shard.write_bytes(b"other")
    second = gate.cache_fingerprint(lm, manifest, rows, index, tmp_path)
    assert second["shards"][shard.name] == audit.digest(b"other")
    assert first != second


def test_audit_readout_small_count_invariants():
    readout = load("audit_readout")
    rows = [{"id": "a", "norm": "startup meeting", "lang": "it", "en_spans": [[0, 2]]}]
    rare = readout.rare_readout(rows, {"a": "startup"}, {"a": "meeting"}, bootstrap=100)
    assert rare["historical_all_span_tokens"] == 2
    assert rare["candidate"]["en_ref"] == rare["baseline_0.8.0"]["en_ref"] == 1
    assert rare["candidate"]["en_hit"] == 1
    assert rare["baseline_0.8.0"]["en_hit"] == 0
    assert rare["vs_0.8.0"]["eta_diff_ci"] == [1, 1]
    from stt.eval import score_rows

    perfect = score_rows(rows, {"a": "startup meeting"}, bootstrap=0)
    wrong = score_rows(rows, {"a": "startup"}, bootstrap=0)
    metrics = {"A0_shipped": perfect, "A1": wrong}
    macro = readout.macro_readout(rows, metrics, bootstrap=100)
    assert macro["A1"]["vs_A0"]["point"] == 0.5
    assert macro["A1"]["vs_A0"]["ci"] == [0.5, 0.5]
    wrong["per_utterance"]["a"]["ref_words"] = 1
    with pytest.raises(ValueError, match="word count mismatch"):
        readout.macro_readout(rows, metrics)


def test_retained_audit_readout_identity_and_counts(pinned_history):
    readout = load("audit_readout")
    artifact = RUN / "audit-readout.json"
    data = json.loads(artifact.read_bytes())
    for path, expected in data["inputs_sha256"].items():
        assert audit.digest(audit.git_bytes(path, data["revision"])) == expected
    for path, expected in data["code_sha256"].items():
        assert readout.code_digest((audit.REPO / path).read_bytes()) == expected
    assert data["code_hash_method"] == readout.CODE_HASH_METHOD
    assert data["bootstrap"]["resamples"] == 1000
    assert data["bootstrap"]["seed"] == 0
    rare = data["rare_target_only"]
    assert rare["n_rows"] == rare["candidate"]["en_ref"] == 47
    assert rare["historical_all_span_tokens"] == 81
    assert rare["candidate"]["en_hit"] == 24
    assert rare["baseline_0.8.0"]["en_hit"] == 11
    assert rare["gate"] is False
    for arm in data["dev"].values():
        assert arm["macro_wer"]["strata"] == {"it": 1248, "en": 2691}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit.require_new(artifact)


def test_audit_code_hash_is_checkout_newline_independent():
    readout = load("audit_readout")
    lf = b"first line\nsecond line\n"
    assert readout.code_digest(lf.replace(b"\n", b"\r\n")) == readout.code_digest(lf)
    assert readout.code_digest(lf) == audit.digest(lf)
    assert readout.code_digest(lf + b"changed\n") != readout.code_digest(lf)
