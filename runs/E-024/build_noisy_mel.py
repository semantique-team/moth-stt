"""E-024 — build the noisy half of Common Voice IT, and the clean-half manifest.

Splits the CV-IT training manifest deterministically (by id hash) into two halves of equal clips.
The **clean** half needs no shards: it is a manifest of ids that already live in
``data/shards/cv_it_train``, so only the noisy half is computed (~4.8 GB instead of 9.7).

The noisy half is mixed on the raw audio and goes straight to mel — no intermediate wav files,
which would cost another 10 GB on a machine with 18 GB free.

    uv run python runs/E-024/build_noisy_mel.py --check      # sizes and disk, build nothing
    uv run python runs/E-024/build_noisy_mel.py              # build (~2 h, 8 workers)

SNR per clip is drawn from a hash of the clip id, so the dataset is reproducible and a resume
rebuilds byte-identically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from stt.data.build_mel import DEFAULT_SHARD_BYTES, DTYPE, TARGET_RATE, load_audio_16k, write_index
from stt.features import log_mel

REPO = Path(__file__).resolve().parents[2]
CV_MANIFEST = REPO / "data/manifests/cv_it_train.jsonl"
#: The manifest's ``audio`` field already starts with ``cv_it_27/``, so the root is ``data/raw``.
AUDIO_ROOT = REPO / "data/raw"
CLEAN_MANIFEST = REPO / "data/manifests/cv_it_train_clean50.jsonl"
NOISY_MANIFEST = REPO / "data/manifests/cv_it_train_noisy50.jsonl"
NOISY_SHARDS = REPO / "data/shards/cv_it_noisy50"
NOISE_DIRS = (REPO / "data/raw/noise/noise_musan", REPO / "data/raw/noise/noise_room_tone")

#: SNR range, uniform in dB. Below 10 dB nothing short of a denoiser helps (see PLAN.md).
SNR_LOW, SNR_HIGH = 10.0, 30.0


def bucket(identifier: str) -> int:
    """Deterministic 0/1 split, stable across machines and Python versions."""
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def snr_for(identifier: str) -> float:
    digest = hashlib.sha256(f"snr:{identifier}".encode()).hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    return round(SNR_LOW + (SNR_HIGH - SNR_LOW) * fraction, 2)


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)) + 1e-12))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Same rule as ``runs/audit-2026-11-09/noise_curve.py``, so train and eval conditions match."""
    if noise.size < speech.size:
        noise = np.tile(noise, int(np.ceil(speech.size / noise.size)))
    noise = noise[: speech.size]
    speech_power, noise_power = rms(speech) ** 2, rms(noise) ** 2
    if speech_power <= 0 or noise_power <= 0:
        return speech
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10.0)))
    mixed = speech + noise * scale
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


_NOISE_CACHE: list[np.ndarray] = []


def _noise_pool() -> list[np.ndarray]:
    """Loaded once per worker process."""
    if not _NOISE_CACHE:
        for directory in NOISE_DIRS:
            for path in sorted(directory.glob("*.wav")):
                _NOISE_CACHE.append(load_audio_16k(path))
    return _NOISE_CACHE


def _job(args: tuple[dict, str, int]) -> dict:
    row, audio_root, noise_index = args
    try:
        speech = load_audio_16k(Path(audio_root) / row["audio"])
        pool = _noise_pool()
        if not pool:
            return {"id": row["id"], "error": "no noise clips found"}
        noise = pool[noise_index % len(pool)]
        snr = snr_for(row["id"])
        mixed = mix_at_snr(speech, noise, snr)
        mel = log_mel(mixed).astype(DTYPE)
        return {
            "id": row["id"],
            "mel": mel,
            "frames": int(mel.shape[1]),
            "duration": round(len(mixed) / TARGET_RATE, 3),
            "snr_db": snr,
        }
    except Exception as exc:  # noqa: BLE001 - reported per clip, never crashes the build
        return {"id": row["id"], "error": str(exc)}


def load_split() -> tuple[list[dict], list[dict]]:
    rows = [
        json.loads(line)
        for line in CV_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clean = [row for row in rows if bucket(str(row["id"])) == 0]
    noisy = [row for row in rows if bucket(str(row["id"])) == 1]
    return clean, noisy


def main() -> int:
    parser = argparse.ArgumentParser(prog="build_noisy_mel.py")
    parser.add_argument("--check", action="store_true", help="report sizes and exit")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--shard-size-mb", type=int, default=DEFAULT_SHARD_BYTES // (1 << 20))
    parser.add_argument("--limit", type=int, default=None, help="debug: first N noisy clips")
    args = parser.parse_args()

    if not CV_MANIFEST.is_file():
        raise SystemExit(f"missing {CV_MANIFEST}")
    if not AUDIO_ROOT.is_dir():
        raise SystemExit(f"missing raw audio {AUDIO_ROOT}")
    noise_clips = sum(len(list(directory.glob("*.wav"))) for directory in NOISE_DIRS)
    if not noise_clips:
        raise SystemExit(f"no noise clips under {NOISE_DIRS}")

    clean, noisy = load_split()
    clean_hours = sum(float(r["duration"]) for r in clean) / 3600
    noisy_hours = sum(float(r["duration"]) for r in noisy) / 3600
    mel_gb = noisy_hours * 57.6 / 1000
    print(
        f"split: clean {len(clean)} clips / {clean_hours:.1f} h | "
        f"noisy {len(noisy)} clips / {noisy_hours:.1f} h | noise pool {noise_clips} clips"
    )
    print(f"noisy mel on disk: ~{mel_gb:.1f} GB | only the noisy half is computed")
    snrs = [snr_for(str(r["id"])) for r in noisy[:2000]]
    print(
        f"SNR draw over a 2k sample: min {min(snrs):.1f} median {sorted(snrs)[len(snrs) // 2]:.1f} "
        f"max {max(snrs):.1f} dB"
    )
    if args.check:
        return 0

    CLEAN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean), encoding="utf-8"
    )
    print(f"wrote {CLEAN_MANIFEST.relative_to(REPO)} ({len(clean)} rows)")

    rows = noisy[: args.limit] if args.limit else noisy
    NOISY_SHARDS.mkdir(parents=True, exist_ok=True)
    shard_bytes = args.shard_size_mb * (1 << 20)
    index: dict[str, dict] = {}
    snr_report: dict[str, float] = {}
    errors: list[dict] = []
    buffer: list[np.ndarray] = []
    buffered = 0
    shard_id = 0
    offset = 0
    seconds = 0.0
    started = time.time()

    def flush() -> None:
        nonlocal buffer, buffered, shard_id, offset
        if not buffer:
            return
        flat = np.concatenate([mel.T.reshape(-1) for mel in buffer])
        np.save(NOISY_SHARDS / f"shard-{shard_id:03d}.npy", flat)
        buffer, buffered, shard_id, offset = [], 0, shard_id + 1, 0

    jobs = [(row, str(AUDIO_ROOT), position) for position, row in enumerate(rows)]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(_job, jobs, chunksize=16):
            done += 1
            if "error" in result:
                errors.append({"id": result["id"], "error": result["error"]})
            else:
                mel = result["mel"]
                clip_bytes = int(mel.size) * 2
                if buffer and buffered + clip_bytes > shard_bytes:
                    flush()
                # New id: the clean half keeps the CV ids, so the two sources cannot collide.
                uid = f"cv-noisy-{result['id']}"
                index[uid] = {
                    "shard": f"shard-{shard_id:03d}.npy",
                    "offset": offset,
                    "frames": result["frames"],
                    "duration": result["duration"],
                }
                snr_report[uid] = result["snr_db"]
                buffer.append(mel)
                buffered += clip_bytes
                offset += result["frames"]
                seconds += result["duration"]
            if done % 5000 == 0:
                elapsed = time.time() - started
                rate = done / elapsed
                print(
                    f"  {done}/{len(rows)} clips, {seconds / 3600:.1f} h, "
                    f"{rate:.0f} clips/s, eta {(len(rows) - done) / rate / 60:.0f} min",
                    flush=True,
                )
    flush()
    write_index(index, NOISY_SHARDS)

    text = "".join(
        json.dumps(
            {
                **{k: v for k, v in row.items() if k != "id"},
                "id": f"cv-noisy-{row['id']}",
                "source": "cv_it_noisy",
                "snr_db": snr_report.get(f"cv-noisy-{row['id']}"),
            },
            ensure_ascii=False,
        )
        + "\n"
        for row in rows
        if f"cv-noisy-{row['id']}" in index
    )
    NOISY_MANIFEST.write_text(text, encoding="utf-8")

    report = {
        "clips": len(index),
        "shards": shard_id,
        "hours": round(seconds / 3600, 2),
        "errors": errors[:20],
        "error_count": len(errors),
        "snr_low": SNR_LOW,
        "snr_high": SNR_HIGH,
        "manifest": str(NOISY_MANIFEST.relative_to(REPO)),
        "shards_dir": str(NOISY_SHARDS.relative_to(REPO)),
        "minutes": round((time.time() - started) / 60, 1),
    }
    (NOISY_SHARDS / "build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=1))
    print(f"wrote {NOISY_MANIFEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())