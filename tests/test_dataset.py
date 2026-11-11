import numpy as np
import pytest
import soundfile as sf
import torch

from stt.data.build_mel import build_shards, load_index, read_clip
from stt.data.dataset import (
    AugmentConfig,
    MelDataset,
    augment,
    collate,
    mask_freq,
    mask_time,
    plan_batches,
    tempo_stretch,
)


def _sine(seconds: float = 2.0, sample_rate: int = 16000, frequency: float = 220.0) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def _mel(shape=(80, 400), seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _tiny_cache(tmp_path, clips: int = 2) -> tuple[dict, dict]:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for index in range(clips):
        name = f"c{index}.wav"
        sf.write(audio_dir / name, _sine(2.0, frequency=220.0 + 55.0 * index), 16000)
        rows.append({"id": f"c{index}", "audio": name, "duration": 2.0})
    build_shards(rows, audio_dir, tmp_path / "shards", workers=1)
    index = load_index(tmp_path / "shards" / "index.json")
    return index, {row["id"]: row for row in rows}


def test_plan_batches_respects_limits_and_covers_everything() -> None:
    durations = [10.0] * 200 + [40.0] * 30
    batches = plan_batches(durations, max_seconds=320.0, max_clips=64, seed=0, epoch=0)
    assert sorted(index for batch in batches for index in batch) == list(range(len(durations)))
    for batch in batches:
        assert len(batch) <= 64
        assert sum(durations[index] for index in batch) <= 320.0 + 1e-9
    assert plan_batches(durations, max_seconds=320.0, max_clips=64, seed=0, epoch=0) == batches
    assert plan_batches(durations, max_seconds=320.0, max_clips=64, seed=0, epoch=1) != batches


def test_collate_pads_with_lengths() -> None:
    first = np.zeros((80, 100), dtype=np.float32)
    second = np.ones((80, 80), dtype=np.float32)
    tensor, lengths, rows = collate(
        [{"mel": first, "row": {"id": "a"}}, {"mel": second, "row": {"id": "b"}}]
    )
    assert tensor.shape == (2, 80, 100)
    assert lengths.tolist() == [100, 80]
    assert float(tensor[1, :, 80:].abs().sum()) == 0.0
    assert rows[0]["id"] == "a"
    assert tensor.dtype == torch.float32


def test_mask_freq_bounds() -> None:
    mel = _mel()
    out = mask_freq(mel, fill=float(mel.min()), count=2, max_width=15, rng=np.random.default_rng(0))
    assert out.shape == mel.shape
    changed = np.where((out != mel).any(axis=1))[0]
    assert 0 < len(changed) <= 30  # two masks of at most 15 bins


def test_mask_time_respects_fraction_budget() -> None:
    mel = _mel(shape=(80, 400))
    out = mask_time(
        mel,
        fill=float(mel.min()),
        count=5,
        max_width=40,
        max_fraction=0.2,
        rng=np.random.default_rng(1),
    )
    changed = np.where((out != mel).any(axis=0))[0]
    assert 0 < len(changed) <= 80  # at most 20% of 400 frames


def test_tempo_stretch_length_bounds() -> None:
    mel = _mel(shape=(80, 500), seed=3)
    assert abs(tempo_stretch(mel, 1.1).shape[1] - 500 / 1.1) <= 1
    assert abs(tempo_stretch(mel, 0.9).shape[1] - 500 / 0.9) <= 1
    assert np.array_equal(tempo_stretch(mel, 1.0), mel)


@pytest.mark.parametrize("frames,factor,expected", [
    (2700, 0.9, 3000), (2701, 0.9, 2701), (3000, 0.9, 3000),
    (3000, 1.0, 3000), (3000, 1.1, 2727),
])
def test_tempo_limit_keeps_both_endpoints(frames, factor, expected) -> None:
    mel = _mel((80, frames))
    out = tempo_stretch(mel, factor)
    assert out.shape[1] == expected
    np.testing.assert_array_equal(out[:, 0], mel[:, 0])
    np.testing.assert_array_equal(out[:, -1], mel[:, -1])
    if expected == frames:
        np.testing.assert_array_equal(out, mel)


@pytest.mark.parametrize("factor", [0.0, -1.0, float("nan"), float("inf")])
def test_tempo_rejects_invalid_factors(factor) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        tempo_stretch(_mel(), factor)


@pytest.mark.parametrize("augmented", [False, True])
def test_dataset_rejects_overlong_source(monkeypatch, augmented) -> None:
    dataset = MelDataset({}, ".", [{"id": "too-long"}],
                         augment=AugmentConfig() if augmented else None)
    monkeypatch.setattr(dataset, "raw_mel", lambda _position: _mel((80, 3001)))
    with pytest.raises(ValueError, match="too-long: input exceeds 3000"):
        dataset[0]
    with pytest.raises(ValueError, match="input exceeds 3000"):
        tempo_stretch(_mel((80, 3001)), 1.1)


def test_overflow_fallback_is_deterministic_and_preserves_transcript(monkeypatch) -> None:
    mel = _mel((80, 2999))
    row = {"id": "last-word", "norm": "ciao fino alla fine"}
    config = AugmentConfig(freq_masks=0, time_masks=0, tempo_range=(0.9, 0.9))
    dataset = MelDataset({}, ".", [row], augment=config, seed=3)
    monkeypatch.setattr(dataset, "raw_mel", lambda _position: mel)
    first, second = dataset[0], dataset[0]
    np.testing.assert_array_equal(first["mel"], mel)
    np.testing.assert_array_equal(first["mel"], second["mel"])
    assert first["row"] is row and second["row"] is row


def test_tempo_augmentation_deterministic_across_resume(monkeypatch) -> None:
    mel = _mel((80, 2800))
    config = AugmentConfig(freq_masks=0, time_masks=0)
    dataset = MelDataset({}, ".", [{"id": "x"}], augment=config, seed=42, epoch=2)
    monkeypatch.setattr(dataset, "raw_mel", lambda _position: mel)
    first = dataset[0]["mel"].copy()
    dataset.epoch = 3
    later = dataset[0]["mel"]
    dataset.epoch = 2
    np.testing.assert_array_equal(first, dataset[0]["mel"])
    assert first.shape != later.shape or not np.array_equal(first, later)


def test_augment_flags_and_determinism() -> None:
    mel = _mel(shape=(80, 400), seed=2)
    untouched = AugmentConfig(freq_masks=0, time_masks=0, tempo=False)
    assert np.array_equal(augment(mel, untouched, np.random.default_rng(0)), mel)

    light_first = augment(mel, AugmentConfig.light(), np.random.default_rng(3))
    light_second = augment(mel, AugmentConfig.light(), np.random.default_rng(3))
    assert light_first.shape == mel.shape
    assert not np.array_equal(light_first, mel)
    assert np.array_equal(light_first, light_second)


def test_read_clip_caches_shard_memmaps(tmp_path, monkeypatch) -> None:
    index, _ = _tiny_cache(tmp_path)
    shards = tmp_path / "shards"
    cache: dict[str, np.ndarray] = {}
    calls = []
    original = np.load

    def counting(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "load", counting)
    first = read_clip(index, shards, "c0", cache)
    second = read_clip(index, shards, "c1", cache)

    assert len(calls) == 1
    assert set(cache) == {index["c0"]["shard"]}
    assert first.shape[0] == 80 and second.shape[0] == 80


def test_dataset_augments_training_and_never_eval(tmp_path) -> None:
    index, rows_by_id = _tiny_cache(tmp_path)
    shards = tmp_path / "shards"
    rows = list(rows_by_id.values())

    train = MelDataset(index, shards, rows, augment=AugmentConfig(), seed=1, epoch=0)
    item = train[0]
    raw = read_clip(index, shards, "c0")
    assert item["id"] == "c0"
    assert not np.array_equal(item["mel"], raw)

    eval_dataset = MelDataset(index, shards, rows, augment=None)
    assert np.array_equal(eval_dataset[0]["mel"], raw)

    again = MelDataset(index, shards, rows, augment=AugmentConfig(), seed=1, epoch=0)
    assert np.array_equal(again[0]["mel"], train[0]["mel"])

    later = MelDataset(index, shards, rows, augment=AugmentConfig(), seed=1, epoch=1)
    assert not np.array_equal(later[0]["mel"], train[0]["mel"])
