import json
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.data.build_manifests import (
    SetSpec,
    build_manifests,
    default_sets,
    existing_eval_hashes,
)


def _wav(path: Path, seconds: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(48000 * seconds)) / 48000
    sf.write(path, (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), 48000)


def _flac(path: Path, seconds: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(16000 * seconds)) / 16000
    sf.write(path, (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), 16000, subtype="PCM_16")


def _tiny_raw_tree(root: Path) -> None:
    # FLEURS it/en: headerless tsv + audio per split (official 7-column layout)
    for lang in ("it_it", "en_us"):
        for split in ("dev", "test"):
            lang_dir = root / "fleurs" / lang
            (lang_dir / "audio" / split).mkdir(parents=True, exist_ok=True)
            _wav(lang_dir / "audio" / split / f"{lang}-{split}.wav")
            (lang_dir / f"{split}.tsv").write_text(
                f"1\t{lang}-{split}.wav\tCiao a tutti\tciao a tutti\tc | i | a\t16000\tFEMALE\n",
                encoding="utf-8",
            )

    # MLS: transcripts.txt + audio/<speaker>/<book>/<uid>.flac; train repeats the dev sentence
    mls = root / "mls_it" / "mls_italian"
    for split in ("train", "dev", "test"):
        split_dir = mls / split
        split_dir.mkdir(parents=True, exist_ok=True)
        uid = f"6001_1001_{split}"
        sentence = "frase ripetuta tra train e dev" if split in ("train", "dev") else "frase di test"
        (split_dir / "transcripts.txt").write_text(f"{uid}\t{sentence}\n", encoding="utf-8")
        _flac(split_dir / "audio" / "6001" / "1001" / f"{uid}.flac")

    # LibriSpeech: rglob over the split root
    ls = root / "librispeech" / "LibriSpeech"
    for split, sentence in (("train-clean-100", "hello world"), ("dev-clean", "good morning")):
        chapter = ls / split / "19" / "198"
        chapter.mkdir(parents=True, exist_ok=True)
        (chapter / "19-198.trans.txt").write_text(f"19-198-0001 {sentence}\n", encoding="utf-8")
        _flac(chapter / "19-198-0001.flac")


def _specs() -> list[SetSpec]:
    return [
        SetSpec("fleurs_it_dev", "fleurs_it", "fleurs/it_it", "dev"),
        SetSpec("fleurs_it_test", "fleurs_it", "fleurs/it_it", "test"),
        SetSpec("fleurs_en_dev", "fleurs_en", "fleurs/en_us", "dev"),
        SetSpec("fleurs_en_test", "fleurs_en", "fleurs/en_us", "test"),
        SetSpec("mls_it_dev", "mls_it", "mls_it/mls_italian/dev", "dev"),
        SetSpec("mls_it_test", "mls_it", "mls_it/mls_italian/test", "test"),
        SetSpec("ls_en_dev", "librispeech", "librispeech/LibriSpeech/dev-clean", "dev"),
        SetSpec("mls_it_train", "mls_it", "mls_it/mls_italian/train", "train", 1.0),
        SetSpec("ls_en_train", "librispeech", "librispeech/LibriSpeech/train-clean-100", "train", 1.0),
    ]


def test_build_writes_manifests_and_drops_overlap(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _tiny_raw_tree(raw)
    out = tmp_path / "manifests"
    report = build_manifests(raw, out, _specs())

    for name in (
        "fleurs_it_dev",
        "fleurs_it_test",
        "fleurs_en_dev",
        "fleurs_en_test",
        "mls_it_dev",
        "mls_it_test",
        "ls_en_dev",
        "mls_it_train",
        "ls_en_train",
    ):
        assert (out / f"{name}.jsonl").exists(), name

    # the train sentence identical to a dev sentence must be dropped as overlap
    train_rows = [json.loads(line) for line in (out / "mls_it_train.jsonl").read_text().splitlines()]
    assert train_rows == []
    assert report["sets"]["mls_it_train"]["dropped"].get("overlap") == 1

    kept_dev = [json.loads(line) for line in (out / "mls_it_dev.jsonl").read_text().splitlines()]
    assert kept_dev[0]["text"] == "frase ripetuta tra train e dev"
    assert kept_dev[0]["split"] == "dev"
    assert kept_dev[0]["source"] == "mls_it"
    assert kept_dev[0]["audio"].startswith("mls_it/mls_italian/dev/audio/")


def test_incremental_build_merges_report_and_guards_train(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _tiny_raw_tree(raw)
    out = tmp_path / "manifests"
    build_manifests(raw, out, [spec for spec in _specs() if spec.split != "train"])

    # Rebuild only the train set: the dev manifest must still provide the overlap guard.
    train = [spec for spec in _specs() if spec.name == "mls_it_train"]
    report = build_manifests(
        raw,
        out,
        train,
        extra_eval_hashes=existing_eval_hashes(out, exclude={"mls_it_train"}),
        merge_report=True,
    )

    assert report["sets"]["mls_it_train"]["dropped"].get("overlap") == 1
    assert report["sets"]["fleurs_it_dev"]["kept"] == 1  # preserved from the first build
    written = json.loads((out / "build_report.json").read_text())
    assert written["sets"]["mls_it_train"]["kept"] == 0


def test_build_report_written(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _tiny_raw_tree(raw)
    out = tmp_path / "manifests"
    build_manifests(raw, out, _specs())
    report = json.loads((out / "build_report.json").read_text())
    assert report["sets"]["fleurs_it_test"]["kept"] == 1
    assert report["sets"]["fleurs_it_test"]["rows_after_sampling"] == 1
    assert "sampling" in report["sets"]["ls_en_train"]


def test_default_sets_can_add_train_clean_360() -> None:
    assert "ls_en_train360" not in {spec.name for spec in default_sets()}
    specs = {spec.name: spec for spec in default_sets(ls360_hours=200.0)}
    spec = specs["ls_en_train360"]
    assert spec.relative_root.endswith("train-clean-360")
    assert spec.target_hours == 200.0
    assert spec.split == "train"
