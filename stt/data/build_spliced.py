"""H3b: utterance-level IT+EN splicing on the mel cache (design: ``docs/03-data.md`` §10).

Rago's two options are utterance splicing ("concatenate an Italian and an English clip;
cheap, gives clause-level switches") and word splicing ("only if H3 shows gold + utterance
splicing is not enough"). This module implements the first one **on the cached log-mel**, so
no raw audio is needed: MLS/LibriSpeech audio was deleted after the mel spot check, and the
two shard sets are already computed with the same mel parameters.

A spliced row concatenates ``mls_it_train`` (Italian) and ``ls_en_train`` (English) clips and
its transcript is the concatenation of the two normalized transcripts; ``lang = "mix"`` with
``en_spans`` covering the English words, exactly like the recorded mixed rows. Pair
generation is deterministic per seed, and a length cap keeps every clip within the encoder
window (30 s = 3000 frames) even after tempo-stretch augmentation.

The reason this is a data experiment and not a model one: E-012 (more English) and E-013
(larger BPE) both left the mixed-set ETA at 21.49, so the hypothesis under test is that the
model has never seen a **switch inside one utterance**.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from stt.data.build_mel import load_index, read_clip, write_index
from stt.manifest import load_manifest
from stt.text import normalize

__all__ = ["build_spliced", "main", "plan_pairs"]

DEFAULT_SHARD_BYTES = 1_500_000_000


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _load_rows_many(value: str) -> list[dict]:
    rows: list[dict] = []
    for path in _split(value):
        rows.extend(_load_rows(path))
    return rows


def _sources(manifests: str, shards: str) -> list[tuple[Path, dict]]:
    manifest_paths, shard_dirs = _split(manifests), _split(shards)
    if len(manifest_paths) != len(shard_dirs):
        raise SystemExit("--*-manifest and --*-shards must list the same number of entries")
    return [(Path(directory), load_index(Path(directory) / "index.json")) for directory in shard_dirs]


def _load_rows(path: Path | str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row.get("norm") and float(row.get("duration", 0.0)) > 0.0]


def plan_pairs(
    it_rows: list[dict],
    en_rows: list[dict],
    *,
    hours: float,
    seed: int = 0,
    max_seconds: float = 27.0,
    min_seconds: float = 2.0,
) -> list[dict]:
    """Deterministically pair Italian and English rows up to ``hours`` of spliced audio.

    ``max_seconds`` defaults to 27: the 30 s encoder window leaves room for a 1.1× tempo
    stretch without cropping (the E-012 crash was a 29.7 s clip stretched past 30 s).
    """
    if not it_rows or not en_rows:
        raise ValueError("both an Italian and an English pool are required")
    rng = random.Random(seed)
    target = hours * 3600.0
    rows: list[dict] = []
    seconds = 0.0
    index = 0
    rejects = 0
    while seconds < target:
        if rejects > 10_000:
            raise ValueError("no pair satisfies the duration bounds; check max_seconds")
        it_row = it_rows[rng.randrange(len(it_rows))]
        en_row = en_rows[rng.randrange(len(en_rows))]
        it_duration = float(it_row["duration"])
        en_duration = float(en_row["duration"])
        duration = it_duration + en_duration
        if duration > max_seconds or duration < min_seconds:
            rejects += 1
            continue
        it_norm = " ".join(it_row["norm"].split())
        en_norm = " ".join(en_row["norm"].split())
        if not it_norm or not en_norm:
            continue
        it_words = len(it_norm.split())
        en_words = len(en_norm.split())
        text = f"{' '.join(it_row['text'].split())} {' '.join(en_row['text'].split())}"
        # An Italian elision cut at the seam ("po'" + "blushed") re-normalizes into one word,
        # which would corrupt the transcript: drop such pairs instead of mangling them.
        if normalize(f"{it_norm} {en_norm}") != f"{it_norm} {en_norm}":
            rejects += 1
            continue
        rows.append(
            {
                "id": f"splice-{index:06d}",
                "audio": f"{it_row['id']}+{en_row['id']}",
                "text": text,
                "norm": f"{it_norm} {en_norm}",
                "duration": round(duration, 3),
                "source": "spliced_it_en",
                "split": "train",
                "speaker": None,
                "lang": "mix",
                "category": "switch",
                "en_spans": [[it_words, it_words + en_words]],
            }
        )
        seconds += duration
        index += 1
        rejects = 0
    return rows


def _read(
    shards_dir: Path,
    index: dict[str, dict],
    uid: str,
    caches: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """``read_clip`` with one memmap cache per shard directory."""
    cache = caches.setdefault(str(shards_dir), {})
    return read_clip(index, shards_dir, uid, cache)


def build_spliced(
    it_sources: list[tuple[Path | str, dict[str, dict]]],
    en_sources: list[tuple[Path | str, dict[str, dict]]],
    rows: list[dict],
    out_dir: Path | str,
    *,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> dict:
    """Concatenate the two cached mels per row and write float16 shards + ``index.json``.

    ``*_sources`` are ``(shards_dir, index)`` pairs: the Italian pool spans Common Voice and
    MLS, the English one LibriSpeech, so a clip can live in any of them. Shard format is
    byte-compatible with ``stt/data/build_mel.py`` (``T × 80`` float16 per clip, contiguous)
    so ``stt.data.dataset.MelDataset`` and ``read_clip`` work unchanged.
    """
    origins = {"it": {}, "en": {}}
    for side, sources in (("it", it_sources), ("en", en_sources)):
        for shards_dir, index in sources:
            for uid in index:
                origins[side][uid] = (Path(shards_dir), index)
    caches: dict[str, dict[str, np.ndarray]] = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    buffer: list[np.ndarray] = []
    buffered = 0
    shard_id = 0
    offset = 0
    seconds = 0.0
    errors: list[dict] = []

    def flush() -> None:
        nonlocal buffer, buffered, shard_id, offset
        if not buffer:
            return
        flat = np.concatenate([mel.T.reshape(-1) for mel in buffer])
        np.save(out_dir / f"shard-{shard_id:03d}.npy", flat)
        buffer, buffered, shard_id, offset = [], 0, shard_id + 1, 0

    for row in rows:
        it_id, en_id = row["audio"].split("+", 1)
        try:
            it_dir, it_index = origins["it"][it_id]
            en_dir, en_index = origins["en"][en_id]
        except KeyError as missing:
            errors.append({"id": row["id"], "error": f"missing source clip {missing}"})
            continue
        it_mel = _read(it_dir, it_index, it_id, caches)
        en_mel = _read(en_dir, en_index, en_id, caches)
        # float16 on disk, like build_mel; read_clip casts back to float32 on load.
        mel = np.concatenate([it_mel, en_mel], axis=1).astype(np.float16)
        frames = int(mel.shape[1])
        clip_bytes = int(mel.size) * 2
        if buffer and buffered + clip_bytes > shard_bytes:
            flush()
        index[row["id"]] = {
            "shard": f"shard-{shard_id:03d}.npy",
            "offset": offset,
            "frames": frames,
            "duration": round(frames / 100.0, 3),
        }
        buffer.append(mel)
        buffered += clip_bytes
        offset += frames
        seconds += frames / 100.0
    flush()
    write_index(index, out_dir)

    report = {
        "clips": len(index),
        "shards": shard_id,
        "seconds": round(seconds, 1),
        "hours": round(seconds / 3600.0, 3),
        "errors": errors,
        "index_path": str(out_dir / "index.json"),
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_spliced.py", description="Build the H3b set")
    parser.add_argument(
        "--it-manifest",
        default="data/manifests/cv_it_train.jsonl,data/manifests/mls_it_train.jsonl",
        help="comma list; the Italian pool spans Common Voice and MLS",
    )
    parser.add_argument(
        "--it-shards",
        default="data/shards/cv_it_train,data/shards/mls_it_train",
        help="comma list, one directory per manifest",
    )
    parser.add_argument("--en-manifest", default="data/manifests/ls_en_train.jsonl")
    parser.add_argument("--en-shards", default="data/shards/ls_en_train")
    parser.add_argument("--hours", type=float, default=110.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=27.0)
    parser.add_argument("--out-dir", default="data/shards/spliced_it_en")
    parser.add_argument("--manifest", default="data/manifests/spliced_it_en.jsonl")
    parser.add_argument("--report", default="data/manifests/spliced_it_en_report.json")
    args = parser.parse_args(argv)

    rows = plan_pairs(
        _load_rows_many(args.it_manifest),
        _load_rows_many(args.en_manifest),
        hours=args.hours,
        seed=args.seed,
        max_seconds=args.max_seconds,
    )
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    load_manifest(manifest)  # raises ManifestError on any invalid row

    report = build_spliced(
        _sources(args.it_manifest, args.it_shards),
        _sources(args.en_manifest, args.en_shards),
        rows,
        args.out_dir,
    )
    report["manifest"] = str(manifest)
    report["seed"] = args.seed
    report["max_seconds"] = args.max_seconds
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"spliced set: {report['clips']} clips, {report['hours']} h, "
        f"{report['shards']} shards, {len(report['errors'])} errors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
