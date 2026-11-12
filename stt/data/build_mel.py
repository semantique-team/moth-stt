"""Mel cache: float16 shards + index (design: ``docs/03-data.md`` §5).

Features are computed once, not per epoch. Each shard is a flat float16 ``.npy`` (frame-major:
``T × 80`` per clip, so a clip is one contiguous block) plus ``index.json`` mapping ``id`` →
``{shard, offset (frames), frames, duration}``. The dataset loader memory-maps the shard and
slices the block; raw audio is deleted after a spot check (``--delete-audio``).

Size: 100 frames/s × 80 bins × 2 B = 57.6 MB per audio hour (≈ 26 GB for 450 h).
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.features import log_mel
from stt.manifest import load_manifest

__all__ = [
    "DTYPE",
    "MEL_BINS",
    "TARGET_RATE",
    "build_shards",
    "delete_audio",
    "load_audio_16k",
    "load_index",
    "main",
    "read_clip",
    "resample_to_target",
    "verify_sample",
    "write_index",
]

TARGET_RATE = 16000
MEL_BINS = 80
DTYPE = np.float16
DEFAULT_SHARD_BYTES = 1_500_000_000  # ≈ 1.5 GB


def resample_to_target(
    signal: np.ndarray, sample_rate: int, target_rate: int = TARGET_RATE
) -> np.ndarray:
    """FFT resampling (no scipy in the base deps); identity when the rates already match."""
    signal = np.asarray(signal, dtype=np.float32)
    if sample_rate == target_rate or len(signal) == 0:
        return signal
    n = len(signal)
    n_target = round(n * target_rate / sample_rate)
    spectrum = np.fft.rfft(signal)
    if n_target < n:
        spectrum = spectrum[: n_target // 2 + 1]
    else:
        extended = np.zeros(n_target // 2 + 1, dtype=complex)
        extended[: len(spectrum)] = spectrum
        spectrum = extended
    out = np.fft.irfft(spectrum, n=n_target)
    return (out * (n_target / n)).astype(np.float32)


def load_audio_16k(path: Path | str) -> np.ndarray:
    """Mono float32 at 16 kHz (stereo is averaged, higher rates are FFT-resampled)."""
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sample_rate != TARGET_RATE:
        mono = resample_to_target(mono, sample_rate)
    return np.ascontiguousarray(mono)


def _safe_mel_job(args: tuple[dict, str]) -> dict:
    """Worker: one manifest row → float16 mel (errors become report entries, never crashes)."""
    row, audio_root = args
    try:
        wav = load_audio_16k(Path(audio_root) / row["audio"])
        mel = log_mel(wav).astype(DTYPE)
        return {
            "id": row["id"],
            "mel": mel,
            "frames": int(mel.shape[1]),
            "duration": round(len(wav) / TARGET_RATE, 3),
        }
    except Exception as exc:  # noqa: BLE001 - reported per clip
        return {"id": row["id"], "error": str(exc)}


def build_shards(
    rows: list[dict],
    audio_root: Path | str,
    out_dir: Path | str,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    workers: int = 1,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    batch_size: int = 1024,
) -> dict:
    """Write the float16 shards and ``index.json``; returns the build report.

    Results are consumed in bounded batches (``batch_size`` jobs in flight) so the parent
    process holds ≈ ``batch_size`` mels at a time — a full-trainset materialization would be
    several GB of float16 on a 16 GB machine.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    rows = rows[:limit] if limit is not None else rows
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate ids in mel build rows")
    audio_root, out_dir = Path(audio_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    shard_names: list[str] = []
    errors: list[dict] = []
    buffer: list[np.ndarray] = []
    buffered = 0
    shard_id = 0
    offset = 0
    seconds = 0.0
    done = 0

    def flush() -> None:
        nonlocal buffer, buffered, shard_id, offset
        if not buffer:
            return
        flat = np.concatenate([mel.T.reshape(-1) for mel in buffer])
        name = f"shard-{shard_id:03d}.npy"
        np.save(out_dir / name, flat)
        shard_names.append(name)
        buffer, buffered, shard_id, offset = [], 0, shard_id + 1, 0

    jobs = [(row, str(audio_root)) for row in rows]
    total = len(jobs)

    def consume(result: dict) -> None:
        nonlocal buffered, offset, seconds, done
        done += 1
        if "error" in result:
            errors.append({"id": result["id"], "error": result["error"]})
        else:
            mel = result["mel"]
            clip_bytes = int(mel.size) * 2
            if buffer and buffered + clip_bytes > shard_bytes:
                flush()
            index[result["id"]] = {
                "shard": f"shard-{shard_id:03d}.npy",
                "offset": offset,
                "frames": result["frames"],
                "duration": result["duration"],
            }
            buffer.append(mel)
            buffered += clip_bytes
            offset += result["frames"]
            seconds += result["duration"]
        if progress is not None:
            progress(done, total)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for start in range(0, total, batch_size):
                chunk = jobs[start : start + batch_size]
                for result in executor.map(_safe_mel_job, chunk, chunksize=8):
                    consume(result)
    else:
        for job in jobs:
            consume(_safe_mel_job(job))
    flush()
    write_index(index, out_dir)

    report = {
        "clips": len(index),
        "shards": len(shard_names),
        "seconds": round(seconds, 1),
        "errors": errors,
        "index_path": str(out_dir / "index.json"),
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def write_index(index: dict[str, dict], out_dir: Path | str) -> Path:
    path = Path(out_dir) / "index.json"
    path.write_text(json.dumps(index, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_index(path: Path | str) -> dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_clip(
    index: dict[str, dict],
    shards_dir: Path | str,
    uid: str,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """One clip as float32 ``[80, T]`` from the memory-mapped shard.

    ``cache`` (shard name → memmap) avoids re-opening and re-reading the header on every clip;
    the returned array is always a copy, so callers may modify it.
    """
    entry = index[uid]
    shard = entry["shard"]
    if cache is not None:
        array = cache.get(shard)
        if array is None:
            array = np.load(Path(shards_dir) / shard, mmap_mode="r")
            cache[shard] = array
    else:
        array = np.load(Path(shards_dir) / shard, mmap_mode="r")
    start = entry["offset"] * MEL_BINS
    block = array[start : start + entry["frames"] * MEL_BINS]
    return np.asarray(block, dtype=np.float32).reshape(entry["frames"], MEL_BINS).T


def verify_sample(
    index: dict[str, dict],
    shards_dir: Path | str,
    rows_by_id: dict[str, dict],
    audio_root: Path | str,
    sample_ids: list[str] | None = None,
    n: int = 3,
    seed: int = 0,
) -> bool:
    """Spot-check ``n`` clips against a fresh computation; False when anything differs.

    With no verifiable clips (audio already gone, empty index) it returns False — the caller
    must not delete audio on an unverified cache.
    """
    candidates = [
        uid
        for uid in index
        if uid in rows_by_id and (Path(audio_root) / rows_by_id[uid]["audio"]).exists()
    ]
    if not candidates or n <= 0:
        return False
    if sample_ids:
        candidate_ids = set(candidates)
        if any(uid not in candidate_ids for uid in sample_ids):
            return False
        candidates = list(sample_ids)
    rng = np.random.default_rng(seed)
    chosen = list(candidates)
    rng.shuffle(chosen)
    for uid in chosen[:n]:
        expected = log_mel(load_audio_16k(Path(audio_root) / rows_by_id[uid]["audio"])).astype(DTYPE)
        got = read_clip(index, shards_dir, uid)
        if got.shape != (MEL_BINS, expected.shape[1]) or not np.array_equal(
            got, expected.astype(np.float32)
        ):
            return False
    return True


def delete_audio(rows: list[dict], audio_root: Path | str) -> int:
    """Delete the raw audio of the given rows (after a successful spot check). Returns the count."""
    root = Path(audio_root).resolve()
    paths = []
    for row in rows:
        path = (root / row["audio"]).resolve()
        if not path.is_relative_to(root) or path == root or (path.exists() and not path.is_file()):
            raise ValueError(f"unsafe audio deletion candidate: {row['audio']!r}")
        paths.append(path)
    removed = 0
    for path in paths:
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_mel.py", description="Build the float16 mel shards and index."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size-mb", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delete-audio", action="store_true")
    args = parser.parse_args(argv)

    rows = load_manifest(args.manifest)

    def _progress(done: int, total: int) -> None:
        if done % 200 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    report = build_shards(
        rows,
        args.audio_root,
        args.out_dir,
        shard_bytes=args.shard_size_mb * 1_000_000,
        workers=args.workers,
        limit=args.limit,
        progress=_progress,
    )
    print(
        f"shards: {report['clips']} clips, {report['shards']} files, "
        f"{report['seconds'] / 3600.0:.2f} h, {len(report['errors'])} errors"
    )
    for bad in report["errors"][:10]:
        print(f"  error: {bad['id']}: {bad['error']}")

    if args.delete_audio:
        if report["errors"]:
            print("build errors - raw audio kept")
            return 1
        index = load_index(Path(args.out_dir) / "index.json")
        current_rows = rows[:args.limit] if args.limit is not None else rows
        candidates = [row for row in current_rows if row["id"] in index]
        rows_by_id = {row["id"]: row for row in candidates}
        if len(candidates) == len(current_rows) and verify_sample(
            index, args.out_dir, rows_by_id, args.audio_root,
            sample_ids=list(rows_by_id), n=len(candidates),
        ):
            removed = delete_audio(candidates, args.audio_root)
            print(f"verification ok - deleted {removed} raw audio files")
        else:
            print("spot check FAILED — raw audio kept")
            return 1
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
