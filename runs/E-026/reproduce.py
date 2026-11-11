"""Rebuild the 47-row 16 kHz diagnostic manifest from T and the pre-E-015 vocabulary.

The committed ``rare47-16k.jsonl`` is the exact manifest used by E-026; this script recreates
it plus the resampled audio in a caller-provided temporary directory. It does not write audio or
manifest data into the repository by default.

    uv run python runs/E-026/reproduce.py --out /tmp/e026
    uv run python -m stt.diagnose release \
        --manifest /tmp/e026/rare47-16k.jsonl \
        --audio-root /tmp/e026/audio \
        --tokenizer data/tokenizer/moth_bpe.model \
        --out /tmp/e026/diagnose
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from stt.data.build_mel import load_audio_16k
from stt.text import normalize

REPO = Path(__file__).resolve().parents[2]
T_MANIFEST = REPO / "recordings/manifests/T.jsonl"
NATURAL = (
    "data/manifests/cv_it_train.jsonl",
    "data/manifests/mls_it_train.jsonl",
    "data/manifests/spliced_it_en.jsonl",
)
GOLD = ("recordings/manifests/R-S1.jsonl", "recordings/manifests/R-S2.jsonl")
EXPECTED_SHA256 = "e8b4b05d1a424d176871fede79d816612593ebaccd2c29acb5e4f6827f49c7b3"


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def _vocabulary(paths: tuple[str, ...]) -> Counter:
    counts: Counter = Counter()
    for path in paths:
        for line in (REPO / path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # R-S1-0642 and R-S2-0624 onward are the E-015/E-028 addenda; the diagnostic
            # describes what 0.7.0 saw before either arm.
            if path.endswith("R-S1.jsonl") and int(row["id"].rsplit("-", 1)[1]) >= 642:
                continue
            if path.endswith("R-S2.jsonl") and int(row["id"].rsplit("-", 1)[1]) >= 624:
                continue
            counts.update(_tokens(row.get("text") or row.get("norm") or ""))
    return counts


def _term_occurrences(row: dict) -> list[str]:
    words = _tokens(row["text"])
    return [word for start, end in (row.get("en_spans") or []) for word in words[start:end]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new working directory")
    args = parser.parse_args()

    natural = _vocabulary(NATURAL)
    gold = _vocabulary(GOLD)
    rows = [
        json.loads(line)
        for line in T_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rare = []
    for row in rows:
        terms = [
            term
            for term in _term_occurrences(row)
            if term not in natural and term not in gold
        ]
        if terms:
            rare.append(row)

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    for row in rare:
        audio = load_audio_16k(REPO / "recordings/audio" / row["audio"])
        name = f"{row['id']}.wav"
        sf.write(audio_dir / name, audio, 16_000, subtype="PCM_16")
        item = dict(row)
        item["audio"] = name
        item["duration"] = round(len(audio) / 16_000, 3)
        manifest_rows.append(item)

    manifest = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows
    ).encode("utf-8")
    digest = hashlib.sha256(manifest).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"manifest hash {digest} != expected {EXPECTED_SHA256}")
    (args.out / "rare47-16k.jsonl").write_bytes(manifest)
    print(f"{len(manifest_rows)} rows, {len(manifest_rows)} rare terms, manifest {digest}")
    print(f"wrote {args.out / 'rare47-16k.jsonl'} and {audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
