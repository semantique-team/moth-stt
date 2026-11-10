"""Finalist infrastructure tests: tiny fakes only, no training or real benchmark assets."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from stt import training_eval as te
from stt.eval import score_rows
from stt.lm import BinaryBPELM, BPETrigramLM


def row(uid="dev-1", norm="un meeting", **kwargs):
    return dict(
        id=uid,
        norm=norm,
        text=norm,
        duration=0.04,
        source="natural_mixed_dev",
        split="dev",
        lang="mix",
        en_spans=[[1, 2]],
        **kwargs,
    )


PIECES = {1: "un ", 2: "meeting", 3: "errore"}


def outputs(ids=(1, 2)):
    array = np.full((len(ids), 4), -30.0, dtype=np.float32)
    for i, token in enumerate(ids):
        array[i, token] = 0
    return array


def mixed_settings():
    return te.EvaluationConfig(selection_metric="mixed_wer")


def fake_tokenizer(tmp_path, monkeypatch):
    from stt import tokenizer

    class Processor:
        def vocab_size(self):
            return len(PIECES)

        def id_to_piece(self, i):
            return PIECES[i + 1]

    monkeypatch.setattr(tokenizer, "load_tokenizer", lambda path: Processor())
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"tokenizer original mapping")
    return str(path)


def test_config_defaults_and_opt_in():
    assert te.load_evaluation_config({}) == te.EvaluationConfig()
    config = {"evaluation": {"decoder": {"lm_path": "lm.bin", "lm_sha256": "a" * 64}}}
    settings = te.load_evaluation_config(config)
    assert settings.decoder == te.DecoderConfig(mode="beam", lm_path="lm.bin", lm_sha256="a" * 64)
    assert settings.wer_margin == 0.005
    with pytest.raises(ValueError, match="lm_path"):
        te.load_evaluation_config({"evaluation": {}})


@pytest.mark.parametrize(
    "key,value",
    [
        ("beam_width", True),
        ("beam_width", 1.5),
        ("top_k", 0),
        ("top_k", "10"),
        ("lm_weight", float("nan")),
        ("token_bonus", float("inf")),
        ("lm_weight", -1),
        ("token_bonus", False),
        ("mode", "unknown"),
    ],
)
def test_decoder_types_and_finiteness(key, value):
    with pytest.raises(ValueError):
        te.load_evaluation_config({"evaluation": {"decoder": {"mode": "greedy", key: value}}})


@pytest.mark.parametrize(
    "evaluation",
    [
        None,
        {"selection_metric": "wer"},
        {"typo": 1},
        {"decoder": []},
        {"decoder": {"mode": "greedy", "typo": 2}},
        {"decoder": {"mode": "greedy"}, "min_eta": 1.1},
    ],
)
def test_invalid_config(evaluation):
    with pytest.raises((ValueError, TypeError)):
        te.load_evaluation_config({"evaluation": evaluation})


def test_binary_lm_hash_and_not_pickle(tmp_path):
    path = tmp_path / "lm.bin"
    BPETrigramLM.from_sequences([[0, 1], [1, 0]]).save_browser(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    settings = te.load_evaluation_config(
        {"evaluation": {"decoder": {"lm_path": str(path), "lm_sha256": digest}}}
    )
    loaded = te.load_lm(settings)
    assert isinstance(loaded, BinaryBPELM)
    assert loaded.sha256 == digest
    path.write_bytes(b"not a binary LM")
    with pytest.raises(ValueError, match="sha256"):
        te.load_lm(settings)
    settings = replace(
        settings,
        decoder=replace(settings.decoder, lm_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    with pytest.raises(ValueError, match="header|magic"):
        te.load_lm(settings)


@pytest.mark.parametrize(
    "changes",
    [
        {"split": "test"},
        {"split": "train"},
        {"role": "test"},
        {"source": "recorded_t"},
        {"source": "recorded_a"},
        {"source": "recorded_r_s1"},
        {"id": "T-0001"},
        {"id": "A-S1-0001"},
        {"id": "R-S1-0001"},
        {"speaker": "S3"},
    ],
)
def test_role_guard_even_with_relabelled_split(changes):
    with pytest.raises(ValueError, match="forbidden"):
        te.validate_dev_rows([row() | changes], [], mixed=True)


@pytest.mark.parametrize(
    "trained", [[{"id": "dev-1", "norm": "altro"}], [{"id": "train-1", "norm": "UN MEETING!"}]]
)
def test_exact_train_overlap(trained):
    with pytest.raises(ValueError, match="overlap"):
        te.validate_dev_rows([row()], trained, mixed=True)


@pytest.mark.parametrize(
    "spans", [None, [], [[1, 1]], [[0, 2], [1, 2]], [[True, 2]], [[1, 3]], [[-1, 1]]]
)
def test_mixed_annotations_required_and_valid(spans):
    with pytest.raises(ValueError, match="ETA"):
        te.validate_dev_rows([row() | {"en_spans": spans}], [], mixed=True)


def test_controls_and_no_eta():
    control = row("control", "ciao mondo", category="control_it") | {"en_spans": []}
    te.validate_dev_rows([row(), control], [], mixed=True)
    with pytest.raises(ValueError, match="no annotated"):
        te.validate_dev_rows([control], [], mixed=True)
    with pytest.raises(ValueError, match="duplicate"):
        te.validate_dev_rows([row(), row()], [])


def write_source(tmp_path, name, rows):
    manifest = tmp_path / f"{name}.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    shard = tmp_path / name
    shard.mkdir()
    np.save(shard / "mel.npy", np.zeros((80, 4), dtype=np.float32))
    (shard / "index.json").write_text(
        json.dumps({r["id"]: {"shard": "mel.npy", "offset": 0, "frames": 4} for r in rows}),
        encoding="utf-8",
    )
    return {"manifest": str(manifest), "shards": str(shard)}


def test_build_sources_and_optional_mixed(tmp_path):
    train = write_source(tmp_path, "train", [row("train-1", "altro meeting") | {"split": "train"}])
    dev = write_source(tmp_path, "dev", [row()])
    config = {
        "data": {"train": [train], "dev_mixed": [dev]},
        "evaluation": {"selection_metric": "mixed_wer", "decoder": {"mode": "greedy"}},
    }
    sets = te.build_dev_sets(config)
    assert sets[0][0] == "mixed"
    assert sets[0][1].augment_config is None
    assert Path(sets[0][1].index["dev-1"]["shard"]).is_absolute()
    assert sets[0][1][0]["mel"].shape == (80, 4)
    del config["data"]["dev_mixed"]
    with pytest.raises(ValueError):
        te.build_dev_sets(config)
    config["data"]["dev_mixed"] = [dev]
    config["data"]["train"] = []
    with pytest.raises(ValueError, match="inventory"):
        te.build_dev_sets(config)


def test_cached_outputs_canonical_full_metrics():
    rows = [row()]
    result = te.evaluate_log_probs(
        {"mixed": rows}, {"mixed": {"dev-1": outputs((1, 3))}}, PIECES, mixed_settings()
    )
    expected = score_rows(rows, {"dev-1": "un errore"}, bootstrap=0)
    # Exact canonical alignment, not a term-presence score.
    assert result["sets"]["mixed"]["overall"]["eta"] == 0
    assert result["sets"]["mixed"]["overall"]["sub"] == expected["overall"]["sub"] == 1
    assert result["wer"] == result["mixed_wer"] == 0.5
    assert result["sets"]["mixed"]["per_utterance"]["dev-1"]["en_ref"] == 1
    assert "eta_errors" in result["sets"]["mixed"]["overall"]


def test_official_beam_forwarded(tmp_path, monkeypatch):
    path = tmp_path / "lm.bin"
    BPETrigramLM.from_sequences([[0, 1]]).save_browser(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    settings = replace(
        mixed_settings(), decoder=te.DecoderConfig(mode="beam", lm_path=str(path), lm_sha256=digest)
    )
    sentinel = te.load_lm(settings)

    def beam(array, length, pieces, **kwargs):
        assert length == 2 and pieces == PIECES
        assert kwargs == {
            "beam_width": 12,
            "top_k": 10,
            "lm": sentinel,
            "lm_weight": 0.5,
            "token_bonus": 2.4,
        }
        return "un meeting"

    monkeypatch.setattr(te, "beam_search", beam)
    result = te.evaluate_log_probs(
        {"mixed": [row()]}, {"mixed": {"dev-1": outputs()}}, PIECES, settings, lm=sentinel
    )
    assert result["wer"] == 0
    assert result["sets"]["mixed"]["overall"]["eta"] == 1
    assert result["protocol"]["lm_sha256"] == digest
    assert set(result["protocol"]["code_sha256"]) >= {"training_eval", "decode", "eval", "lm"}
    with pytest.raises(ValueError, match="verified binary LM"):
        te.evaluate_log_probs(
            {"mixed": [row()]},
            {"mixed": {"dev-1": outputs()}},
            PIECES,
            settings,
            lm=BinaryBPELM(path.read_bytes()),
        )


@pytest.mark.parametrize(
    "cache",
    [
        {},
        {"extra": outputs()},
        {"dev-1": np.zeros((0, 4))},
        {"dev-1": np.full((2, 4), np.nan)},
        {"dev-1": np.zeros((2, 5))},
    ],
)
def test_bad_cache_fails(cache):
    with pytest.raises(ValueError):
        te.evaluate_log_probs({"mixed": [row()]}, {"mixed": cache}, PIECES, mixed_settings())


class FakeDataset:
    augment_config = None

    def __init__(self, frames=4):
        self.rows = [row()]
        self.frames = frames

    def __getitem__(self, index):
        return {"row": self.rows[index], "mel": np.zeros((80, self.frames), dtype=np.float32)}


class FakeModel(torch.nn.Module):
    def forward(self, tensor, lengths):
        assert not self.training and not torch.is_grad_enabled()
        assert lengths.tolist() == [4]
        # A spurious third frame must not be decoded.
        return torch.tensor(outputs((1, 2, 3)))[None], torch.tensor([2])


def test_model_inference_lengths_and_restore():
    model = FakeModel().train()
    result = te.evaluate_model(model, [("mixed", FakeDataset())], PIECES, mixed_settings())
    assert model.training and result["wer"] == 0
    assert result["sets"]["mixed"]["hypotheses"] == {"dev-1": "un meeting"}
    with pytest.raises(ValueError, match="refusing to crop"):
        te.evaluate_model(model, [("mixed", FakeDataset(3001))], PIECES, mixed_settings())
    assert model.training


def test_padded_forward_error_restores_eval_mode():
    class BrokenModel(torch.nn.Module):
        def forward(self, tensor, lengths):
            assert tensor.shape == (1, 80, 3000)
            assert lengths.tolist() == [4]
            raise RuntimeError("forward failed")

    model = BrokenModel().eval()
    with pytest.raises(RuntimeError, match="forward failed"):
        te.evaluate_model(
            model, [("mixed", FakeDataset())], PIECES, mixed_settings(), input_mode="padded30"
        )
    assert not model.training


def test_protocol_detects_changed_dev_decoder_and_input():
    def evaluate(rows, settings=None, **kwargs):
        settings = mixed_settings() if settings is None else settings
        return te.evaluate_log_probs(
            {"mixed": rows}, {"mixed": {"dev-1": outputs()}}, PIECES, settings, **kwargs
        )["protocol_sha256"]

    original = evaluate([row()])
    assert original == evaluate([row()])
    assert original != evaluate([row() | {"speaker": "new-voice"}])
    assert original != evaluate([row()], input_mode="padded30")
    assert original != evaluate([row()], replace(mixed_settings(), min_eta=0.1))


def metrics(wer=0.2, it=0.2, en=0.3, eta=0.6, insertions=1):
    return {
        "protocol_sha256": "same",
        "acoustic_provenance_verified": True,
        "selection_metric": "mixed_wer",
        "wer": wer,
        "sets": {
            "it": {"wer": it},
            "en": {"wer": en},
            "mixed": {"wer": wer, "overall": {"eta": eta, "insertion_rate": insertions}},
        },
    }


def test_baseline_ties_and_deterministic_candidate_selection():
    baseline = metrics()
    result = te.rank_candidates({"a": metrics(0.2 - 1e-14)}, mixed_settings(), baseline=baseline)
    assert result["selected"] == "baseline"
    for candidates in (
        {"z": metrics(0.1), "a": metrics(0.1)},
        {"a": metrics(0.1), "z": metrics(0.1)},
    ):
        assert (
            te.rank_candidates(candidates, mixed_settings(), baseline=baseline)["selected"] == "a"
        )


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"it": 0.206}, "it WER"),
        ({"en": 0.306}, "en WER"),
        ({"eta": 0.59}, "ETA regression"),
        ({"insertions": 1.1}, "insertion regression"),
    ],
)
def test_baseline_regression_gates(changes, reason):
    result = te.rank_candidates(
        {"candidate": metrics(0.1, **changes)}, mixed_settings(), baseline=metrics()
    )
    assert result["selected"] == "baseline"
    assert reason in " ".join(result["ranking"][0]["reasons"])


def test_boundaries_min_eta_and_mismatched_protocol():
    assert (
        te.rank_candidates(
            {"a": metrics(0.1, it=0.205, en=0.305)}, mixed_settings(), baseline=metrics()
        )["selected"]
        == "a"
    )
    assert (
        te.rank_candidates(
            {"a": metrics(0.1)}, replace(mixed_settings(), min_eta=0.7), baseline=metrics()
        )["selected"]
        == "baseline"
    )
    for change in ({"protocol_sha256": "other"}, {"wer": float("nan")}):
        with pytest.raises(ValueError):
            te.rank_candidates({"a": metrics() | change}, mixed_settings(), baseline=metrics())
    candidate = copy.deepcopy(metrics())
    candidate["sets"]["mixed"]["overall"]["eta"] = None
    with pytest.raises(ValueError, match="ETA"):
        te.rank_candidates({"a": candidate}, mixed_settings(), baseline=metrics())


def test_macro_does_not_include_mixed():
    rows = {
        "it": [row("it") | {"lang": "it", "en_spans": []}],
        "en": [row("en") | {"lang": "en", "en_spans": []}],
        "mixed": [row()],
    }
    cache = {
        "it": {"it": outputs()},
        "en": {"en": outputs((1, 3))},
        "mixed": {"dev-1": outputs((3,))},
    }
    result = te.evaluate_log_probs(rows, cache, PIECES, te.EvaluationConfig())
    assert result["wer"] == result["macro_wer"] == 0.25
    assert result["mixed_wer"] == 1


def test_checkpoint_strict_no_aux_and_architecture(tmp_path, monkeypatch):
    from stt import model as model_module

    monkeypatch.setattr(model_module, "MothSTT", lambda **kwargs: torch.nn.Linear(1, 1))
    path = tmp_path / "checkpoint.pt"
    config = {
        "model": {"vocab_size": 3, "head_blocks": 0},
        "tokenizer": fake_tokenizer(tmp_path, monkeypatch),
    }
    payload = {"model": torch.nn.Linear(1, 1).state_dict(), "config": config}
    torch.save(payload, path)
    model, declared = te.load_checkpoint(path, config)
    assert declared == config and not hasattr(model, "aux")
    payload["model"]["aux.weight"] = torch.zeros(1)
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        te.load_checkpoint(path, config)
    with pytest.raises(ValueError, match="architecture"):
        te.load_checkpoint(path, {"model": {"vocab_size": 4}})


def test_cli_one_checkpoint_never_writes_weights(tmp_path, monkeypatch):
    path = tmp_path / "candidate.pt"
    path.write_bytes(b"fake checkpoint")
    config_path = tmp_path / "run.json"
    config = {
        "model": {"vocab_size": 3},
        "tokenizer": fake_tokenizer(tmp_path, monkeypatch),
        "vocab": str(tmp_path / "vocab.json"),
        "data": {"train": [{"manifest": str(tmp_path / "train.jsonl")}]},
        "evaluation": {"selection_metric": "mixed_wer", "decoder": {"mode": "greedy"}},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "vocab.json").write_text(json.dumps(PIECES), encoding="utf-8")
    (tmp_path / "train.jsonl").write_text(
        json.dumps(row("train", "altro testo") | {"split": "train"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(te, "build_dev_sets", lambda config: [("mixed", FakeDataset())])
    monkeypatch.setattr(te, "load_checkpoint", lambda *args: (FakeModel(), config))
    monkeypatch.setattr(te, "evaluate_model", lambda *args, **kwargs: metrics())
    out = tmp_path / "metrics.json"
    assert (
        te.main(["--config", str(config_path), "--checkpoint", str(path), "--out", str(out)]) == 0
    )
    assert (
        json.loads(out.read_text())["checkpoint_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert path.read_bytes() == b"fake checkpoint"
    assert not list(tmp_path.glob("ckpt_best*"))
    with pytest.raises(ValueError, match="new path"):
        te.main(["--config", str(config_path), "--checkpoint", str(path), "--out", str(path)])
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(metrics()), encoding="utf-8")
    ranked_path = tmp_path / "ranked.json"
    assert (
        te.main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(path),
                "--baseline-metrics",
                str(baseline_path),
                "--out",
                str(ranked_path),
            ]
        )
        == 0
    )
    ranked = json.loads(ranked_path.read_text())
    assert ranked["baseline_metrics"] == metrics()
    assert ranked["baseline_source"] == {
        "path": str(baseline_path),
        "sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    }


def test_batch_plan_fingerprint_and_rank_rejection():
    rows = {
        "it": [row("it") | {"lang": "it", "en_spans": []}],
        "en": [row("en") | {"lang": "en", "en_spans": []}],
        "mixed": [row()],
    }
    cache = {name: {r["id"]: outputs() for r in group} for name, group in rows.items()}
    first = te.evaluate_log_probs(
        rows,
        cache,
        PIECES,
        mixed_settings(),
        acoustic_provenance={
            "producer": "cached",
            "plans": [["it", "en"]],
            "acoustic_provenance_verified": True,
            "audio_sha256": "fixture",
        },
    )
    other = te.evaluate_log_probs(
        rows,
        cache,
        PIECES,
        mixed_settings(),
        acoustic_provenance={
            "producer": "cached",
            "plans": [["it"], ["en"]],
            "acoustic_provenance_verified": True,
            "audio_sha256": "fixture",
        },
    )
    assert first["protocol_sha256"] != other["protocol_sha256"]
    with pytest.raises(ValueError, match="different"):
        te.rank_candidates({"candidate": other}, mixed_settings(), baseline=first)
    default = te.evaluate_log_probs(rows, cache, PIECES, mixed_settings())
    assert default["protocol"]["acoustic_provenance"] == {
        "producer": "external_cache",
        "batch_plan": "unspecified",
    }
    assert default["acoustic_provenance_verified"] is False
    with pytest.raises(ValueError, match="verified acoustic provenance"):
        te.rank_candidates({"candidate": default}, mixed_settings(), baseline=default)
    with pytest.raises(ValueError, match="verified acoustic provenance"):
        te.rank_candidates({"candidate": first}, mixed_settings(), baseline=default)


def test_model_batch_settings_and_actual_plan_fingerprinted():
    first = te.evaluate_model(FakeModel(), [("mixed", FakeDataset())], PIECES, mixed_settings())
    other = te.evaluate_model(
        FakeModel(),
        [("mixed", FakeDataset())],
        PIECES,
        mixed_settings(),
        max_seconds=100,
        max_clips=1,
    )
    assert first["protocol_sha256"] != other["protocol_sha256"]
    assert first["protocol"]["acoustic_provenance"]["plans"] == {
        "mixed": [{"ids": ["dev-1"], "mel_lengths": [4], "padded_frames": 4}]
    }


def test_model_mel_content_changes_fingerprint_with_same_rows_shape_and_outputs():
    class ChangedDataset(FakeDataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            item["mel"][0, -1] = 0.25
            return item

    original = te.evaluate_model(FakeModel(), [("mixed", FakeDataset())], PIECES, mixed_settings())
    changed = te.evaluate_model(
        FakeModel(), [("mixed", ChangedDataset())], PIECES, mixed_settings()
    )
    assert original["sets"] == changed["sets"]
    assert original["protocol_sha256"] != changed["protocol_sha256"]
    assert original["acoustic_provenance_verified"] is True
    content = original["protocol"]["acoustic_provenance"]["mel_inputs"]["mixed"]["dev-1"]
    assert content == {
        "shape": [80, 4],
        "dtype": "<f4",
        "sha256": hashlib.sha256(np.zeros((80, 4), dtype="<f4").tobytes()).hexdigest(),
    }
    padded = te.evaluate_model(
        FakeModel(), [("mixed", FakeDataset())], PIECES, mixed_settings(), input_mode="padded30"
    )
    assert padded["protocol"]["acoustic_provenance"]["mel_inputs"]["mixed"]["dev-1"] == content
    assert (
        original["protocol"]["acoustic_provenance"]["plans"]
        == changed["protocol"]["acoustic_provenance"]["plans"]
    )
    with pytest.raises(ValueError, match="different"):
        te.rank_candidates({"original": original, "changed": changed}, mixed_settings())


def test_cached_provenance_requires_explicit_attestation_not_just_metadata():
    result = te.evaluate_log_probs(
        {"mixed": [row()]},
        {"mixed": {"dev-1": outputs()}},
        PIECES,
        mixed_settings(),
        acoustic_provenance={"audio_sha256": "caller-supplied", "batch_plan": [["dev-1"]]},
    )
    assert result["acoustic_provenance_verified"] is False
    with pytest.raises(ValueError, match="verified acoustic provenance"):
        te.rank_candidates({"candidate": result}, mixed_settings())


@pytest.mark.parametrize("pinned", [True, False])
def test_checkpoint_tokenizer_identity_not_just_size(tmp_path, monkeypatch, pinned):
    from stt import model as model_module

    def forbidden(**kwargs):
        pytest.fail("must reject tokenizer mismatch before constructing model")

    monkeypatch.setattr(model_module, "MothSTT", forbidden)
    original = tmp_path / "original.model"
    changed = tmp_path / "changed.model"
    original.write_bytes(b"same size different token order A")
    changed.write_bytes(b"same size different token order B")
    config = {"model": {"vocab_size": 3}, "tokenizer": str(original)}
    payload = {"config": config, "model": {}}
    if pinned:
        payload["tokenizer_sha256"] = te._sha256(original)
        original.unlink()  # A saved identity must not require the old source file.
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="tokenizer sha256 mismatch"):
        te.load_checkpoint(path, config | {"tokenizer": str(changed)})
    if not pinned:
        original.unlink()
        with pytest.raises(FileNotFoundError):
            te.load_checkpoint(path, config | {"tokenizer": str(changed)})


def test_saved_tokenizer_hash_allows_missing_historical_file(tmp_path, monkeypatch):
    from stt import model as model_module

    monkeypatch.setattr(model_module, "MothSTT", lambda **kwargs: torch.nn.Linear(1, 1))
    tokenizer = fake_tokenizer(tmp_path, monkeypatch)
    config = {"model": {"vocab_size": 3}, "tokenizer": tokenizer}
    payload = {
        "config": config | {"tokenizer": "absent-historical.model"},
        "tokenizer_sha256": te._sha256(tokenizer),
        "model": torch.nn.Linear(1, 1).state_dict(),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    assert isinstance(te.load_checkpoint(path, config)[0], torch.nn.Linear)


def test_tokenizer_vocab_mapping_and_lm_compatibility(tmp_path, monkeypatch):
    tokenizer = fake_tokenizer(tmp_path, monkeypatch)
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps(PIECES | {1: PIECES[2], 2: PIECES[1]}), encoding="utf-8")
    config = {"model": {"vocab_size": 3}, "tokenizer": tokenizer, "vocab": str(vocab)}
    with pytest.raises(ValueError, match="mapping"):
        te.load_evaluation_tokenizer(config)
    vocab.write_text(json.dumps(PIECES), encoding="utf-8")
    assert te.load_evaluation_tokenizer(config)[0] == PIECES
    path = tmp_path / "lm.bin"
    BPETrigramLM.from_sequences([[0, 1]]).save_browser(path)
    with pytest.raises(ValueError, match="LM/tokenizer vocabulary"):
        te.load_evaluation_tokenizer(config, BinaryBPELM(path.read_bytes()))
    with pytest.raises(ValueError, match="tokenizer source"):
        te.load_evaluation_tokenizer({"model": {"vocab_size": 3}, "vocab": str(vocab)})


def test_check_config_cli_no_checkpoint_model_or_forward(tmp_path, monkeypatch):
    from stt import model as model_module

    def forbidden(*args, **kwargs):
        pytest.fail("preflight must not load checkpoints, construct models or run forward")

    monkeypatch.setattr(model_module, "MothSTT", forbidden)
    monkeypatch.setattr(te, "load_checkpoint", forbidden)
    monkeypatch.setattr(te, "evaluate_model", forbidden)
    train = write_source(tmp_path, "train", [row("train", "altro testo") | {"split": "train"}])
    dev = write_source(tmp_path, "dev", [row()])
    config = {
        "model": {"vocab_size": 3},
        "tokenizer": fake_tokenizer(tmp_path, monkeypatch),
        "data": {"train": [train], "dev_mixed": [dev]},
        "evaluation": {"selection_metric": "mixed_wer", "decoder": {"mode": "greedy"}},
    }
    source = tmp_path / "config.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "preflight.json"
    assert te.main(["--config", str(source), "--check-config", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["checked"] and result["sets"] == {"mixed": 1}
    assert result["files_sha256"][str(source)] == te._sha256(source)
    assert str(Path(dev["shards"]) / "mel.npy") in result["files_sha256"]
    Path(dev["shards"], "mel.npy").unlink()
    with pytest.raises(FileNotFoundError):
        te.check_config(config)
    bad = row() | {"split": "test"}
    Path(dev["manifest"]).write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        te.check_config(config)
