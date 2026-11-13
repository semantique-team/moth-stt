"""Small synthetic E-039 regressions; no corpus audits, weights or network."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np


def load(name):
    path = Path(__file__).resolve().parents[1] / "runs/E-039" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"e039_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load("audit")
audit2 = load("audit2")


def write_rows(root, relative, rows):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_signature_uses_independent_minima_and_is_deterministic():
    prime = (1 << 32) - 5
    rng = np.random.default_rng(0)
    permutations = np.stack([
        rng.integers(1, prime, size=audit2.PERMUTATIONS, dtype=np.uint64),
        rng.integers(0, prime, size=audit2.PERMUTATIONS, dtype=np.uint64),
    ])
    text = " ".join(f"word{i}" for i in range(60))
    signature = audit2._signature(text, permutations, prime)
    np.testing.assert_array_equal(signature, audit2._signature(text, permutations, prime))
    winners = {
        min(audit2._shingles(text), key=lambda h: (h * int(a) + int(b)) % prime)
        for a, b in permutations.T
    }
    assert len(winners) > 10


def test_signature_matches_python_integer_reference_at_uint32_limits(monkeypatch):
    prime = (1 << 32) - 5
    hashes = [0, prime - 1, (1 << 32) - 1]
    permutations = np.array([[prime - 1, prime - 2], [prime - 1, 0]], dtype=np.uint64)
    monkeypatch.setattr(audit2, "_shingles", lambda _: hashes)
    expected = [min((h * int(a) + int(b)) % prime for h in hashes)
                for a, b in permutations.T]
    np.testing.assert_array_equal(audit2._signature("unused", permutations, prime), expected)


def test_lsh_recalls_different_minimum_pairs_against_small_exhaustive_sample(
    tmp_path, monkeypatch,
):
    rows = []
    for family in range(3):
        words = [f"family{family}word{i}" for i in range(60)]
        original = " ".join(words)
        hashes = audit2._shingles(original)
        words[hashes.index(min(hashes))] = f"replacement{family}"
        changed = " ".join(words)
        assert min(hashes) != min(audit2._shingles(changed))
        old_prime = (1 << 61) - 1
        old_seeds = np.random.default_rng(0).integers(
            1, old_prime, size=audit2.PERMUTATIONS, dtype=np.int64,
        )
        old_signatures = [
            ((np.array(audit2._shingles(text), dtype=np.int64)[:, None] * 31
              + old_seeds[None, :]) % old_prime).min(axis=0)
            for text in (original, changed)
        ]
        # The retained implementation misses every band for these high-overlap pairs.
        assert not np.any(
            (old_signatures[0] == old_signatures[1]).reshape(audit2.BANDS, -1).all(axis=1)
        )
        rows.extend([{"id": f"{family}a", "norm": original},
                     {"id": f"{family}b", "norm": changed}])
    rows.append({"id": "unrelated", "norm": "no shared trigrams here"})
    sources = {}
    for row in rows:
        sources[row["id"]] = f"sample/{row['id']}.jsonl"
        write_rows(tmp_path, sources[row["id"]], [row])
    monkeypatch.setattr(audit2, "REPO", tmp_path)
    monkeypatch.setattr(audit2, "TRAIN_SOURCES", sources)
    monkeypatch.setattr(audit2, "EVAL_SOURCES", {})
    monkeypatch.setattr(audit2, "OWN_SOURCES", {})
    exhaustive = set()
    for left, right in combinations(rows, 2):
        a, b = set(audit2._shingles(left["norm"])), set(audit2._shingles(right["norm"]))
        if len(a & b) / len(a | b) > audit2.NEAR_THRESHOLD:
            exhaustive.add("|".join(sorted((left["id"], right["id"]))))
    result = audit2.text_pass()
    assert len(exhaustive) == 3
    assert set(result["by_pair"]) == exhaustive
    assert result["total_pairs"] == 3
    assert result["candidate_pairs"] >= len(exhaustive)
    assert result["oversized_buckets_skipped"] == 0


def test_term_provenance_includes_every_natural_source_and_addendum_boundaries(
    tmp_path, monkeypatch,
):
    terms = ["cvterm", "mlsterm", "lsterm", "spliceterm", "goldterm", "addedterm", "laterterm"]
    assert audit.NATURAL_TRAIN_MANIFESTS == (
        "data/manifests/cv_it_train.jsonl", "data/manifests/mls_it_train.jsonl",
        "data/manifests/ls_en_train.jsonl", "data/manifests/spliced_it_en.jsonl",
    )
    for path, term in zip(audit.NATURAL_TRAIN_MANIFESTS, terms[:4], strict=True):
        # LS-EN also exercises the text fallback and shared normalization rule.
        write_rows(tmp_path, path, [{"text": term.upper()}])
    write_rows(tmp_path, "recordings/manifests/T.jsonl", [
        {"id": "T-1", "norm": " ".join(terms), "en_spans": [[0, len(terms)]]},
    ])
    write_rows(tmp_path, "recordings/manifests/R-S1.jsonl", [
        {"id": "R-S1-0641", "text": "goldterm"},
        {"id": "R-S1-0642", "text": "addedterm lsterm"},
        {"id": "R-S1-0692", "text": "laterterm"},
    ])
    write_rows(tmp_path, "recordings/manifests/R-S2.jsonl", [])
    monkeypatch.setattr(audit, "REPO", tmp_path)
    result = audit.term_provenance()
    assert result["natural_manifests"] == list(audit.NATURAL_TRAIN_MANIFESTS)
    assert result["pre_addendum"]["groups"]["natural"]["terms"] == sorted(terms[:4])
    assert result["counts"]["lsterm"] == {
        "natural": 1, "gold_pre": 0, "addendum_028": 1, "terms3": 0,
    }
    assert result["pre_addendum"]["groups"]["never"]["terms"] == ["addedterm", "laterterm"]
    assert result["shipped"]["groups"]["never"]["terms"] == ["laterterm"]
    assert result["terms3_only"] == ["laterterm"]


def test_audio_scope_is_loose_file_bytes_and_path_labels_not_manifest_splits(
    tmp_path, monkeypatch,
):
    root = tmp_path / "data/raw/librispeech/LibriSpeech"
    for split in ("train-clean-100", "dev-clean"):
        path = root / split / "clip.flac"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"same encoded bytes")
    (root / "archive.tar").write_bytes(b"not inspected")
    outside = tmp_path / "recordings/audio/clip.wav"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"same encoded bytes")
    monkeypatch.setattr(audit2, "REPO", tmp_path)
    monkeypatch.setattr(audit2, "ProcessPoolExecutor", ThreadPoolExecutor)
    result = audit2.audio_pass()
    assert result["files"] == 2
    assert result["repeated_hashes"] == 1
    assert result["cross_source_collisions"] == {}
    assert set(result["sources"]) == {"librispeech/LibriSpeech"}
    assert "not manifest train/dev/test splits" in result["note"]
