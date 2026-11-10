"""Remove empty MOTHLMB1 trigram rows without rebuilding or requantizing scores.

Run with ``python -m stt.compact_lm --input old.bin --output new.bin --report report.json``.
Output paths must not exist. Release copies/configuration are deliberately left untouched;
a compacted binary has a new hash and must not silently replace a release artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from stt.lm import BinaryBPELM


def compact_lm(input_path: Path | str, output_path: Path | str, report_path: Path | str) -> dict:
    """Validate, compact and verify every retained row, then write new files exclusively."""
    source, output, report = map(Path, (input_path, output_path, report_path))
    if len({path.resolve() for path in (source, output, report)}) != 3:
        raise ValueError("input, output and report paths must be distinct")
    for path in (output, report):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing path: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"output parent must exist: {path.parent}")

    data = source.read_bytes()
    before = BinaryBPELM(data)
    keep = before.trigram_offsets[1:] > before.trigram_offsets[:-1]
    contexts = int(np.count_nonzero(keep))
    trigram_start = (
        28
        + before.unigram.nbytes
        + 4
        + before.bigram_offsets.nbytes
        + before.bigram_tokens.nbytes
        + before.bigram_logq.nbytes
    )
    trigram_start = (trigram_start + 3) // 4 * 4
    payload_start = trigram_start + 4 + before.trigram_keys.nbytes + before.trigram_offsets.nbytes
    keys = before.trigram_keys[keep].tobytes()
    offsets = before.trigram_offsets[:-1][keep].tobytes() + before.trigram_offsets[-1:].tobytes()
    # Empty rows own no entries: keep the complete token/logq payload verbatim.
    compacted = (
        data[:trigram_start] + struct.pack("<I", contexts) + keys + offsets + data[payload_start:]
    )
    after = BinaryBPELM(compacted)
    new_payload_start = trigram_start + 4 + after.trigram_keys.nbytes + after.trigram_offsets.nbytes
    if not (
        compacted[:trigram_start] == data[:trigram_start]
        and after.trigram_keys.tobytes() == keys
        and after.trigram_offsets.tobytes() == offsets
        and compacted[new_payload_start:] == data[payload_start:]
        and np.all(after.trigram_offsets[1:] > after.trigram_offsets[:-1])
        and len(data) - len(compacted) == 8 * (len(keep) - contexts)
    ):
        raise ValueError("lossless LM verification failed")

    result = {
        "format": "MOTHLMB1",
        "input": str(source.resolve()),
        "output": str(output.resolve()),
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "output_sha256": hashlib.sha256(compacted).hexdigest(),
        "input_bytes": len(data),
        "output_bytes": len(compacted),
        "saved_bytes": len(data) - len(compacted),
        "saved_percent": 100 * (len(data) - len(compacted)) / len(data),
        "trigram_contexts_before": len(keep),
        "trigram_contexts_after": contexts,
        "empty_trigram_rows_removed": len(keep) - contexts,
        "bigram_entries": len(before.bigram_tokens),
        "trigram_entries": len(before.trigram_tokens),
        "verification": {
            "all_retained_rows_verified": True,
            "retained_rows_verified": contexts,
            "header_unigram_bigram_bytes_identical": True,
            "trigram_token_logq_bytes_identical": True,
            "scores_exact": True,
            "method": "Every retained key/offset and all score-bearing bytes preserved; "
            "only empty rows removed, so missing-token fallback is unchanged.",
        },
        "quality_claim": "None. No WER gain claimed or measured; no training or corpus rebuild.",
        "release_config_modified": False,
    }
    encoded_report = json.dumps(result, indent=2) + "\n"
    with output.open("xb") as binary, report.open("x", encoding="utf-8") as handle:
        binary.write(compacted)
        handle.write(encoded_report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compact_lm(args.input, args.output, args.report)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
