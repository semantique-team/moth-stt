"""Dataset: mmap mel shards, duration buckets, collate, augmentation (design: ``docs/04`` §3–4).

Batching: clips are ordered by duration (shuffled within equal durations), packed into batches of
at most **320 audio-seconds** and **64 clips**, then the batch order is shuffled. The encoder
receives a key-padding mask built from ``lengths`` (in the model), CTC receives the true lengths.

Augmentation is applied **on the cached mel**, at load time: SpecAugment (2 frequency masks ≤ 15
bins; up to 5 time masks of ≤ 40 frames, ≤ 20% of the frames total) and a mel-domain tempo stretch
of 0.9–1.1 by interpolation (no pitch shift). ``AugmentConfig.light()`` keeps frequency masks only
(H1). The dev/eval loader never augments.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stt.data.build_mel import read_clip

__all__ = [
    "AugmentConfig",
    "MelDataset",
    "augment",
    "collate",
    "mask_freq",
    "mask_time",
    "plan_batches",
    "tempo_stretch",
]


@dataclass(frozen=True)
class AugmentConfig:
    """Starting points from ``docs/04`` §4."""

    freq_masks: int = 2
    freq_width: int = 15
    time_masks: int = 5
    time_width: int = 40
    time_max_fraction: float = 0.2
    tempo: bool = True
    tempo_range: tuple[float, float] = (0.9, 1.1)

    @classmethod
    def light(cls) -> AugmentConfig:
        """H1: light SpecAugment only (frequency masks, no time masks, no tempo stretch)."""
        return cls(freq_masks=2, time_masks=0, tempo=False)


def plan_batches(
    durations: list[float],
    max_seconds: float = 320.0,
    max_clips: int = 64,
    seed: int = 0,
    epoch: int = 0,
) -> list[list[int]]:
    """Duration-bucketed batches (shuffled batch order, deterministic per seed/epoch)."""
    rng = random.Random(seed + epoch * 1_000_003)
    order = list(range(len(durations)))
    rng.shuffle(order)
    order.sort(key=lambda index: durations[index])  # stable: mixes equal durations

    batches: list[list[int]] = []
    current: list[int] = []
    seconds = 0.0
    for index in order:
        duration = durations[index]
        if current and (seconds + duration > max_seconds or len(current) >= max_clips):
            batches.append(current)
            current = []
            seconds = 0.0
        current.append(index)
        seconds += duration
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def collate(items: list[dict]) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Pad ``[80, T]`` mels to the batch maximum (zeros); return tensor, lengths, rows."""
    mels = [item["mel"] for item in items]
    max_frames = max(mel.shape[1] for mel in mels)
    batch = torch.zeros(len(mels), 80, max_frames, dtype=torch.float32)
    lengths = torch.zeros(len(mels), dtype=torch.long)
    for position, mel in enumerate(mels):
        frames = mel.shape[1]
        batch[position, :, :frames] = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32))
        lengths[position] = frames
    return batch, lengths, [item["row"] for item in items]


def tempo_stretch(mel: np.ndarray, factor: float) -> np.ndarray:
    """Time-interpolation tempo stretch (factor > 1 is faster → shorter mel). No pitch shift."""
    if factor == 1.0:
        return mel
    frames = mel.shape[1]
    new_frames = max(2, round(frames / factor))
    positions = np.linspace(0.0, frames - 1.0, new_frames)
    left = np.floor(positions).astype(int)
    right = np.minimum(left + 1, frames - 1)
    fraction = (positions - left).astype(np.float32)
    return (mel[:, left] * (1.0 - fraction) + mel[:, right] * fraction).astype(np.float32)


def mask_freq(
    mel: np.ndarray, fill: float, count: int, max_width: int, rng: np.random.Generator
) -> np.ndarray:
    """``count`` frequency masks of width ≤ ``max_width`` bins."""
    out = mel.copy()
    if count <= 0:
        return out
    bins = out.shape[0]
    for _ in range(count):
        width = int(rng.integers(1, min(max_width, bins) + 1))
        start = int(rng.integers(0, bins - width + 1))
        out[start : start + width, :] = fill
    return out


def mask_time(
    mel: np.ndarray,
    fill: float,
    count: int,
    max_width: int,
    max_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Up to ``count`` time masks of width ≤ ``max_width`` frames, ≤ ``max_fraction`` of frames."""
    out = mel.copy()
    if count <= 0:
        return out
    frames = out.shape[1]
    budget = int(frames * max_fraction)
    masked = 0
    for _ in range(count):
        remaining = budget - masked
        if remaining <= 0:
            break
        width = int(rng.integers(1, min(max_width, remaining) + 1))
        start = int(rng.integers(0, frames - width + 1))
        out[:, start : start + width] = fill
        masked += width
    return out


def augment(mel: np.ndarray, config: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """Tempo stretch first (on the whole mel), then SpecAugment on the resulting length."""
    out = np.asarray(mel, dtype=np.float32)
    if config.tempo:
        out = tempo_stretch(out, float(rng.uniform(*config.tempo_range)))
    fill = float(out.min())
    if config.freq_masks:
        out = mask_freq(out, fill, config.freq_masks, config.freq_width, rng)
    if config.time_masks:
        count = max(0, min(config.time_masks, out.shape[1] // 200 or 1))
        out = mask_time(out, fill, count, config.time_width, config.time_max_fraction, rng)
    return out


class MelDataset:
    """Memory-mapped mel cache over manifest rows; deterministic augmentation per item and epoch.

    ``epoch`` is mutable: the training loop sets it each epoch so augmentation varies across
    epochs while staying reproducible.
    """

    def __init__(
        self,
        index: dict[str, dict],
        shards_dir: Path | str,
        rows: list[dict],
        augment: AugmentConfig | None = None,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        self.index = index
        self.shards_dir = Path(shards_dir)
        self.rows = rows
        self.augment_config = augment
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def raw_mel(self, position: int) -> np.ndarray:
        """The cached mel, without augmentation."""
        return read_clip(self.index, self.shards_dir, self.rows[position]["id"])

    def __getitem__(self, position: int) -> dict:
        row = self.rows[position]
        mel = self.raw_mel(position)
        if self.augment_config is not None:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + position)
            mel = augment(mel, self.augment_config, rng)
        return {"mel": mel, "row": row, "id": row["id"]}
