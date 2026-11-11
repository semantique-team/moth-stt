"""Build a browser LM from freeze_lm_corpus JSONL and its immutable receipt.

Run ``python -m stt.build_lm --help`` for explicit inputs and pruning settings.
``--check-inputs`` validates and tokenizes only: no n-gram counting or output files.
Release size is an LM-substitution estimate, not final packaging certification.
Receipt checks establish self-consistency, not independent authentication: the operator
must trust the receipt and source selection. Original source files are not reread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from importlib.metadata import version
from pathlib import Path

from stt import text as text_module
from stt.lm import BOS_ID, BPETrigramLM
from stt.release import validate_release
from stt.text import normalize
from stt.tokenizer import load_tokenizer, vocab_json


def build_lm(
    *,
    corpus: Path,
    receipt: Path,
    tokenizer: Path,
    tokenizer_sha256: str,
    min_bigram_count: int,
    min_trigram_count: int,
    backoff: float,
    out: Path | None = None,
    report: Path | None = None,
    budget_bytes: int = 20_000_000,
    release: Path | None = None,
    check_inputs: bool = False,
) -> dict:
    """Use raw SentencePiece IDs, never CTC-shifted IDs or historical corpus builders."""
    if not re.fullmatch(r"[0-9a-f]{64}", tokenizer_sha256):
        raise ValueError("tokenizer_sha256 must be an explicit lowercase SHA-256")
    if any(type(n) is not int or n < 1 for n in (min_bigram_count, min_trigram_count)):
        raise ValueError("minimum n-gram counts must be positive integers")
    if not math.isfinite(backoff) or not 0 < backoff <= 1:
        raise ValueError("backoff must be finite and in (0, 1]")
    if type(budget_bytes) is not int or not 0 < budget_bytes <= 25_000_000:
        raise ValueError("budget_bytes must be a positive integer <= 25000000")
    if not check_inputs and (out is None or report is None):
        raise ValueError("building requires --out and --report")
    inputs = {p.resolve() for p in (corpus, receipt, tokenizer)}
    outputs = [p for p in (out, report) if p is not None]
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError("output and report must be distinct")
    if out is not None and out.suffix != ".bin":
        raise ValueError("output must be a .bin browser LM, not pickle")
    for path in outputs:
        if path.resolve() in inputs:
            raise ValueError("outputs must be distinct from inputs")
        if release is not None and path.resolve().is_relative_to(release.resolve()):
            raise ValueError("outputs must be outside the immutable release")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {path}")
        if not path.parent.is_dir():
            raise ValueError(f"output parent must exist: {path.parent}")

    content = corpus.read_bytes()
    receipt_bytes = receipt.read_bytes()
    frozen = json.loads(receipt_bytes)
    if not isinstance(frozen, dict) or type(frozen.get("format_version")) is not int:
        raise ValueError("invalid freeze receipt")
    if frozen["format_version"] != 1:
        raise ValueError("unsupported freeze receipt version")
    source_inputs = frozen.get("inputs")
    if not isinstance(source_inputs, list) or not source_inputs:
        raise ValueError("freeze receipt requires an inputs list")
    source_rows = {"train_manifest": 0, "train_script": 0, "exclusion": 0}
    roles = set()
    for entry in source_inputs:
        if not isinstance(entry, dict):
            raise ValueError("freeze receipt input must be an object")  # noqa: TRY004 - invalid JSON
        role = entry.get("role")
        if not isinstance(role, str) or role not in source_rows:
            raise ValueError("freeze receipt input has invalid role")
        if not isinstance(entry.get("path"), str) or not entry["path"].strip():
            raise ValueError("freeze receipt input requires a nonempty path")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", entry["sha256"]
        ):
            raise ValueError("freeze receipt input requires a lowercase SHA-256")
        if type(entry.get("rows")) is not int or entry["rows"] < 0:
            raise ValueError("freeze receipt input rows must be a nonnegative integer")
        roles.add(role)
        source_rows[role] += entry["rows"]
    if "exclusion" not in roles or not roles.intersection(("train_manifest", "train_script")):
        raise ValueError("freeze receipt requires training and exclusion inputs")
    for field in (
        "corpus_bytes",
        "sentences",
        "train_rows",
        "empty_train_rows",
        "duplicate_train_rows",
        "excluded_sentences",
    ):
        if type(frozen.get(field)) is not int or frozen[field] < 0:
            raise ValueError(f"freeze receipt {field} must be a nonnegative integer")
    if (
        frozen["train_rows"] != source_rows["train_manifest"] + source_rows["train_script"]
        or frozen["train_rows"]
        != (frozen["sentences"] + frozen["empty_train_rows"] + frozen["duplicate_train_rows"])
        or not 0 < frozen["excluded_sentences"] <= source_rows["exclusion"]
    ):
        raise ValueError("freeze receipt counts are inconsistent")
    corpus_sha = hashlib.sha256(content).hexdigest()
    if frozen.get("corpus_sha256") != corpus_sha:
        raise ValueError("corpus SHA-256 does not match freeze receipt")
    if frozen.get("corpus_bytes") != len(content):
        raise ValueError("corpus bytes do not match freeze receipt")
    normalizer_sha = hashlib.sha256(Path(text_module.__file__).read_bytes()).hexdigest()
    if (
        frozen.get("normalizer") != "stt.text.normalize"
        or frozen.get("normalizer_sha256") != normalizer_sha
        or frozen.get("num2words_version") != version("num2words")
    ):
        raise ValueError("normalizer provenance does not match freeze receipt")
    tokenizer_bytes = tokenizer.read_bytes()
    if hashlib.sha256(tokenizer_bytes).hexdigest() != tokenizer_sha256:
        raise ValueError("tokenizer SHA-256 does not match explicit pin")
    processor = load_tokenizer(tokenizer)
    if tokenizer.read_bytes() != tokenizer_bytes:
        raise ValueError("tokenizer changed while loading")
    vocab_size = processor.vocab_size()
    if not 0 < vocab_size <= BOS_ID:
        raise ValueError("tokenizer vocabulary exceeds LM token range")

    release_config = release_report = None
    if release is not None:
        release_config, release_report = validate_release(release, budget_bytes)
        vocab = json.loads((release / release_config["tokenizer"]["vocab"]).read_bytes())
        if vocab != vocab_json(processor):
            raise ValueError("tokenizer pieces do not match release vocabulary")

    sequences = []
    languages = {"it": 0, "en": 0, "mix": 0}
    previous = None
    seen = set()
    words = tokens = unknown_tokens = 0
    for line_no, line in enumerate(content.decode("utf-8").splitlines(), 1):
        row = json.loads(line)
        location = f"{corpus}:{line_no}"
        if not isinstance(row, dict) or set(row) != {"text", "lang"}:
            raise ValueError(f"{location}: expected frozen text/lang object")
        text, lang = row["text"], row["lang"]
        if not isinstance(lang, str) or lang not in languages:
            raise ValueError(f"{location}: lang must be it, en or mix")
        hint = "en" if lang == "en" else "it"
        if not isinstance(text, str) or not text or normalize(text, hint) != text:
            raise ValueError(f"{location}: text must be nonempty and canonical")
        key = (text, lang)
        if (previous is not None and key <= previous) or text in seen:
            raise ValueError(f"{location}: frozen rows must be sorted and deduplicated")
        previous = key
        seen.add(text)
        sequence = processor.encode(text, out_type=int)
        if not sequence or any(type(t) is not int or not 0 <= t < vocab_size for t in sequence):
            raise ValueError(f"{location}: empty or out-of-range tokenizer output")
        sequences.append(sequence)
        languages[lang] += 1
        words += len(text.split())
        tokens += len(sequence)
        unknown_tokens += sequence.count(processor.unk_id())
    if not sequences or frozen.get("sentences") != len(sequences):
        raise ValueError("nonempty corpus row count must match freeze receipt")

    result = {
        "format_version": 1,
        "mode": "check-inputs" if check_inputs else "build",
        "inputs": {
            "corpus": {"path": str(corpus), "sha256": corpus_sha, "bytes": len(content)},
            "receipt": {"path": str(receipt), "sha256": hashlib.sha256(receipt_bytes).hexdigest()},
            "tokenizer": {"path": str(tokenizer), "sha256": tokenizer_sha256},
        },
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalizer_sha256": normalizer_sha,
        "num2words_version": version("num2words"),
        "sentencepiece_version": version("sentencepiece"),
        "parameters": {
            "min_bigram_count": min_bigram_count,
            "min_trigram_count": min_trigram_count,
            "backoff": backoff,
        },
        "corpus_stats": {
            "rows": len(sequences),
            "languages": languages,
            "words": words,
            "tokens": tokens,
            "unknown_tokens": unknown_tokens,
            "tokenizer_vocab_size": vocab_size,
        },
        "budget_bytes": budget_bytes,
        "receipt_validation": {
            "self_consistent": True,
            "independently_authenticated": False,
            "source_files_reread": False,
            "trust_boundary": "Operator must trust the receipt and training/exclusion selection; "
            "source files may be absent. Hashes do not authenticate provenance or exclusion coverage.",
        },
        "quality_claim": "None. No evaluation or acoustic training performed.",
    }
    if release_report is not None:
        result["release"] = release_report
    if check_inputs:
        return result

    lm = BPETrigramLM.from_sequences(
        sequences,
        min_bigram_count=min_bigram_count,
        min_trigram_count=min_trigram_count,
        backoff=backoff,
    )
    # save_browser owns serialization; stage privately so budget failures leave no outputs.
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / "lm.bin"
        lm.save_browser(staged)
        binary = staged.read_bytes()
    actual_bytes = len(binary)
    estimated_bytes = actual_bytes
    if release_report is not None:
        old_lm = release_config["decode"]["lm"]
        estimated_bytes += release_report["total_bytes"] - release_report["files"][old_lm]["bytes"]
    result["size"] = {
        "lm_bytes": actual_bytes,
        "candidate_asset_bytes": estimated_bytes,
        "remaining_bytes": budget_bytes - estimated_bytes,
        "scope": "LM only" if release is None else "validated release assets with LM substituted",
        "final_packaging_verified": False,
        "caveat": "Future config size may change; revalidate after packaging. Browser ORT/WASM excluded.",
    }
    if estimated_bytes > budget_bytes:
        raise ValueError(
            f"candidate size {estimated_bytes} bytes exceeds budget {budget_bytes} bytes"
        )
    result["lm_stats"] = lm.summary()
    result["output"] = {
        "path": str(out),
        "format": "MOTHLMB1",
        "sha256": hashlib.sha256(binary).hexdigest(),
    }
    encoded_report = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with out.open("xb") as output_handle, report.open("xb") as report_handle:
        output_handle.write(binary)
        report_handle.write(encoded_report)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece .model")
    parser.add_argument("--tokenizer-sha256", required=True, help="trusted lowercase hex pin")
    parser.add_argument("--min-bigram-count", type=int, required=True)
    parser.add_argument("--min-trigram-count", type=int, required=True)
    parser.add_argument("--backoff", type=float, required=True)
    parser.add_argument("--out", type=Path, help="new .bin path; parent must exist")
    parser.add_argument("--report", type=Path, help="new JSON report; parent must exist")
    parser.add_argument("--budget-bytes", type=int, default=20_000_000)
    parser.add_argument(
        "--release", type=Path, help="validate original assets and all tokenizer pieces"
    )
    parser.add_argument(
        "--check-inputs", action="store_true", help="no counting, build or output writes"
    )
    args = parser.parse_args(argv)
    try:
        result = build_lm(**vars(args))
    except (ValueError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
