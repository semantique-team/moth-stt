"""Synthetic frozen inputs only: never load a real tokenizer or rebuild a real corpus."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pytest

from stt import build_lm as builder
from stt import text as text_module
from stt.data.freeze_lm_corpus import freeze_lm_corpus
from stt.lm import BinaryBPELM, BPETrigramLM


class FakeProcessor:
    def vocab_size(self):
        return 4

    def id_to_piece(self, index):
        return ["<unk>", "ciao", "hello", "world"][index]

    def unk_id(self):
        return 0

    def encode(self, text, out_type):
        assert out_type is int
        return {"ciao world": [1, 3], "hello world": [2, 3]}[text]


def write_corpus(args, rows):
    content = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    args["corpus"].write_bytes(content)
    receipt = {
        "format_version": 1,
        "inputs": [
            {
                "role": "train_script",
                "path": "absent/train.jsonl",
                "sha256": hashlib.sha256(content).hexdigest(),
                "rows": len(rows),
            },
            {
                "role": "exclusion",
                "path": "absent/exclusion.jsonl",
                "sha256": hashlib.sha256(b"synthetic exclusion").hexdigest(),
                "rows": 1,
            },
        ],
        "corpus_sha256": hashlib.sha256(content).hexdigest(),
        "corpus_bytes": len(content),
        "sentences": len(rows),
        "train_rows": len(rows),
        "empty_train_rows": 0,
        "duplicate_train_rows": 0,
        "excluded_sentences": 1,
        "normalizer": "stt.text.normalize",
        "normalizer_sha256": hashlib.sha256(Path(text_module.__file__).read_bytes()).hexdigest(),
        "num2words_version": version("num2words"),
    }
    args["receipt"].write_text(json.dumps(receipt), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    args = {
        "corpus": tmp_path / "frozen.jsonl",
        "receipt": tmp_path / "receipt.json",
        "tokenizer": tmp_path / "tokenizer.model",
        "tokenizer_sha256": hashlib.sha256(b"synthetic model").hexdigest(),
        "min_bigram_count": 1,
        "min_trigram_count": 2,
        "backoff": 0.4,
        "out": tmp_path / "output.bin",
        "report": tmp_path / "report.json",
    }
    args["tokenizer"].write_bytes(b"synthetic model")
    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"text":"ciao world","lang":"mix"}\n'
        '{"text":"hello world","lang":"en"}\n'
        '{"text":"ciao world","lang":"it"}\n'
        '{"text":""}\n',
        encoding="utf-8",
    )
    exclusion = tmp_path / "exclusion.jsonl"
    exclusion.write_text('{"text":"excluded sentence"}\n', encoding="utf-8")
    freeze_lm_corpus(
        train_manifests=[],
        train_scripts=[train],
        exclusions=[exclusion],
        out=args["corpus"],
        report=args["receipt"],
    )
    monkeypatch.setattr(builder, "load_tokenizer", lambda path: FakeProcessor())
    return args


@pytest.mark.parametrize("counts", [(1, 2), (2, 1)])
def test_build_matches_direct_raw_ids(inputs, counts, monkeypatch):
    inputs.update(min_bigram_count=counts[0], min_trigram_count=counts[1])
    before = {name: inputs[name].read_bytes() for name in ("corpus", "receipt", "tokenizer")}
    monkeypatch.setattr(BPETrigramLM, "save", lambda *a: pytest.fail("pickle writer used"))
    result = builder.build_lm(**inputs)
    binary = inputs["out"].read_bytes()
    assert binary.startswith(b"MOTHLMB1")
    assert BinaryBPELM(binary).score(1024, 1024, 1) < 0
    expected = inputs["out"].with_name("expected.bin")
    BPETrigramLM.from_sequences([[1, 3], [2, 3]], *counts, backoff=0.4).save_browser(expected)
    assert binary == expected.read_bytes()
    assert result["output"]["sha256"] == hashlib.sha256(binary).hexdigest()
    assert result["size"]["lm_bytes"] == len(binary)
    assert result["size"]["candidate_asset_bytes"] == len(binary)
    assert result["corpus_stats"]["tokens"] == 4
    assert result["corpus_stats"]["languages"] == {"it": 1, "en": 1, "mix": 0}
    assert json.loads(inputs["report"].read_bytes()) == result
    assert before == {name: inputs[name].read_bytes() for name in before}


def test_check_inputs_never_counts_or_writes(inputs, monkeypatch):
    monkeypatch.setattr(BPETrigramLM, "from_sequences", lambda *a, **k: pytest.fail("counted"))
    result = builder.build_lm(**inputs, check_inputs=True)
    assert result["mode"] == "check-inputs"
    assert "lm_stats" not in result and "output" not in result and "size" not in result
    assert not inputs["out"].exists() and not inputs["report"].exists()


@pytest.mark.parametrize("check_inputs", [True, False])
def test_sources_not_reread_or_authenticated(inputs, monkeypatch, check_inputs):
    receipt = json.loads(inputs["receipt"].read_bytes())
    sources = {Path(entry["path"]) for entry in receipt["inputs"]}
    for source in sources:
        source.unlink()
    original = Path.read_bytes

    def read_bytes(path):
        assert path not in sources, "original sources must not be reread"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    result = builder.build_lm(**inputs, check_inputs=check_inputs)
    validation = result["receipt_validation"]
    assert validation["self_consistent"]
    assert not validation["independently_authenticated"]
    assert not validation["source_files_reread"]
    assert "Operator must trust" in validation["trust_boundary"]


@pytest.mark.parametrize(
    "field",
    [
        "format_version",
        "inputs",
        "corpus_sha256",
        "corpus_bytes",
        "sentences",
        "train_rows",
        "empty_train_rows",
        "duplicate_train_rows",
        "excluded_sentences",
        "normalizer",
        "normalizer_sha256",
        "num2words_version",
    ],
)
def test_requires_complete_receipt(inputs, field):
    receipt = json.loads(inputs["receipt"].read_bytes())
    del receipt[field]
    inputs["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        builder.build_lm(**inputs, check_inputs=True)


@pytest.mark.parametrize("entries", [None, {}, [], [None], ["train_script"]])
def test_invalid_receipt_inputs_list(inputs, entries):
    receipt = json.loads(inputs["receipt"].read_bytes())
    receipt["inputs"] = entries
    inputs["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        builder.build_lm(**inputs, check_inputs=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", None),
        ("role", []),
        ("role", "test"),
        ("path", None),
        ("path", ""),
        ("path", " "),
        ("path", 4),
        ("sha256", None),
        ("sha256", "bad"),
        ("sha256", "A" * 64),
        ("rows", None),
        ("rows", True),
        ("rows", -1),
        ("rows", 1.0),
        ("rows", "1"),
    ],
)
def test_invalid_receipt_input_fields(inputs, field, value):
    receipt = json.loads(inputs["receipt"].read_bytes())
    receipt["inputs"][0][field] = value
    inputs["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        builder.build_lm(**inputs, check_inputs=True)


@pytest.mark.parametrize("role", ["exclusion", "train_script"])
def test_requires_both_training_and_exclusion_roles(inputs, role):
    receipt = json.loads(inputs["receipt"].read_bytes())
    receipt["inputs"] = [entry for entry in receipt["inputs"] if entry["role"] == role]
    inputs["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="training and exclusion"):
        builder.build_lm(**inputs, check_inputs=True)


@pytest.mark.parametrize(
    "field",
    [
        "corpus_bytes",
        "sentences",
        "train_rows",
        "empty_train_rows",
        "duplicate_train_rows",
        "excluded_sentences",
    ],
)
@pytest.mark.parametrize("value", [-1, True, 1.0, "1", None])
def test_receipt_counts_are_nonnegative_integers(inputs, field, value):
    receipt = json.loads(inputs["receipt"].read_bytes())
    receipt[field] = value
    inputs["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="nonnegative integer"):
        builder.build_lm(**inputs, check_inputs=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_rows", 5),
        ("empty_train_rows", 0),
        ("duplicate_train_rows", 0),
        ("excluded_sentences", 0),
        ("excluded_sentences", 2),
    ],
)
def test_receipt_count_consistency(inputs, field, value):
    receipt = json.loads(inputs["receipt"].read_bytes())
    receipt[field] = value
    inputs["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="counts are inconsistent"):
        builder.build_lm(**inputs, check_inputs=True)


def test_training_manifest_and_script_row_totals(inputs):
    receipt = json.loads(inputs["receipt"].read_bytes())
    training = next(entry for entry in receipt["inputs"] if entry["role"] == "train_script")
    training["rows"] -= 1
    receipt["inputs"].append({**training, "role": "train_manifest", "rows": 1})
    inputs["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    assert builder.build_lm(**inputs, check_inputs=True)["receipt_validation"]["self_consistent"]
    training["rows"] += 1
    inputs["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="counts are inconsistent"):
        builder.build_lm(**inputs, check_inputs=True)


@pytest.mark.parametrize("field", ["corpus", "tokenizer"])
def test_hash_mismatch_fails_before_load(inputs, field, monkeypatch):
    inputs[field].write_bytes(inputs[field].read_bytes() + b" ")
    monkeypatch.setattr(builder, "load_tokenizer", lambda *a: pytest.fail("loaded"))
    with pytest.raises(ValueError, match="SHA-256"):
        builder.build_lm(**inputs)
    assert not inputs["out"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("sentences", 8),
        ("corpus_bytes", 8),
        ("format_version", 2),
        ("normalizer_sha256", "0" * 64),
        ("num2words_version", "wrong"),
    ],
)
def test_receipt_mismatch(inputs, field, value):
    receipt = json.loads(inputs["receipt"].read_bytes())
    receipt[field] = value
    inputs["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        builder.build_lm(**inputs, check_inputs=True)


@pytest.mark.parametrize(
    "rows",
    [
        [{"text": "HELLO!", "lang": "en"}],
        [{"text": "hello world", "lang": "fr"}],
        [{"text": "hello world", "lang": "mix", "lang_hint": "fr"}],
        [{"text": "", "lang": "it"}],
        [{"text": "hello world", "lang": "en"}, {"text": "ciao world", "lang": "it"}],
        [{"text": "hello world", "lang": "en"}, {"text": "hello world", "lang": "mix"}],
        [],
    ],
)
def test_rejects_noncanonical_rows(inputs, rows):
    write_corpus(inputs, rows)
    with pytest.raises(ValueError):
        builder.build_lm(**inputs, check_inputs=True)


def test_normalization_uses_language_hint(inputs, monkeypatch):
    hints = []
    original = builder.normalize

    def normalize(text, hint):
        hints.append(hint)
        return original(text, hint)

    monkeypatch.setattr(builder, "normalize", normalize)
    builder.build_lm(**inputs, check_inputs=True)
    assert hints == ["it", "en"]


@pytest.mark.parametrize("tokens", [[], [4], [-1], [1024], [True]])
def test_invalid_tokenizer_output(inputs, tokens, monkeypatch):
    monkeypatch.setattr(FakeProcessor, "encode", lambda *a, **k: tokens)
    with pytest.raises(ValueError, match="tokenizer output"):
        builder.build_lm(**inputs, check_inputs=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_bigram_count", 0),
        ("min_trigram_count", -1),
        ("backoff", float("nan")),
        ("backoff", 0),
        ("backoff", 1.1),
        ("budget_bytes", 25_000_001),
        ("tokenizer_sha256", "not pinned"),
    ],
)
def test_invalid_parameters(inputs, field, value):
    inputs[field] = value
    with pytest.raises(ValueError):
        builder.build_lm(**inputs)


@pytest.mark.parametrize("target", ["out", "report"])
def test_no_overwrite(inputs, target):
    inputs[target].write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        builder.build_lm(**inputs)
    assert inputs[target].read_bytes() == b"keep"


def test_collisions_and_nonbinary_output(inputs):
    with pytest.raises(ValueError, match="distinct"):
        builder.build_lm(**{**inputs, "report": inputs["corpus"]})
    with pytest.raises(ValueError, match="distinct"):
        builder.build_lm(**{**inputs, "report": inputs["out"]})
    with pytest.raises(ValueError, match=".bin"):
        builder.build_lm(**{**inputs, "out": inputs["out"].with_suffix(".pkl")})


def test_budget_failure_leaves_no_outputs(inputs):
    with pytest.raises(ValueError, match="exceeds budget"):
        builder.build_lm(**inputs, budget_bytes=1)
    assert not inputs["out"].exists() and not inputs["report"].exists()


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    assets = {
        "model.onnx": b"fake model",
        "filters.bin": b"fake filters",
        "vad.onnx": b"fake vad",
        "lm.bin": b"MOTHLMB1old",
        "vocab.json": json.dumps(builder.vocab_json(FakeProcessor())).encode(),
        "NOTICE": b"synthetic only",
    }
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
        "files": {
            name: "sha256:" + hashlib.sha256(data).hexdigest() for name, data in assets.items()
        },
    }
    assets["config.json"] = json.dumps(config).encode()
    for name, data in assets.items():
        (root / name).write_bytes(data)
    return root


def test_release_substitutes_lm_across_all_seven_files(inputs, release):
    before = {p.name: p.read_bytes() for p in release.iterdir()}
    result = builder.build_lm(**inputs, release=release)
    size = result["size"]
    expected = sum(map(len, before.values())) - len(before["lm.bin"]) + inputs["out"].stat().st_size
    assert len(result["release"]["files"]) == 7
    assert size["candidate_asset_bytes"] == expected
    assert not size["final_packaging_verified"]
    assert before == {p.name: p.read_bytes() for p in release.iterdir()}


def test_release_budget_checks_substitution_not_lm_alone(inputs, release):
    builder.build_lm(**inputs)
    lm_bytes = inputs["out"].stat().st_size
    inputs["out"] = inputs["out"].with_name("second.bin")
    inputs["report"] = inputs["report"].with_name("second.json")
    with pytest.raises(ValueError, match="exceeds budget"):
        builder.build_lm(**inputs, release=release, budget_bytes=lm_bytes)
    assert not inputs["out"].exists() and not inputs["report"].exists()


def test_release_checks_every_piece_and_original_hashes(inputs, release, monkeypatch):
    monkeypatch.setattr(FakeProcessor, "id_to_piece", lambda self, i: "wrong" if i == 0 else str(i))
    with pytest.raises(ValueError, match="pieces"):
        builder.build_lm(**inputs, release=release, check_inputs=True)
    (release / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        builder.build_lm(**inputs, release=release, check_inputs=True)


def test_no_writes_inside_release(inputs, release):
    with pytest.raises(ValueError, match="immutable release"):
        builder.build_lm(**{**inputs, "out": release / "new.bin"}, release=release)


def test_cli_relative_paths_check_and_build(inputs, monkeypatch, capsys):
    monkeypatch.chdir(inputs["corpus"].parent)
    argv = [
        "--corpus",
        "frozen.jsonl",
        "--receipt",
        "receipt.json",
        "--tokenizer",
        "tokenizer.model",
        "--tokenizer-sha256",
        inputs["tokenizer_sha256"],
        "--min-bigram-count",
        "2",
        "--min-trigram-count",
        "1",
        "--backoff",
        "0.4",
    ]
    assert builder.main([*argv, "--check-inputs"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "check-inputs"
    with pytest.raises(SystemExit) as exc:
        builder.main(argv)
    assert exc.value.code == 2
    assert builder.main([*argv, "--out", "output.bin", "--report", "report.json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["inputs"]["corpus"]["path"] == "frozen.jsonl"
    assert result["parameters"]["min_trigram_count"] == 1
