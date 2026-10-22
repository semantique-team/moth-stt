import inspect

import numpy as np
import torch

import stt.model
from stt.model import EncoderConfig, MothSTT, sinusoids


def _mel(batch: int = 1, frames: int = 100, seed: int = 0) -> torch.Tensor:
    data = np.random.default_rng(seed).standard_normal((batch, 80, frames)).astype(np.float32)
    return torch.from_numpy(data) * 0.1


def test_parameter_count_matches_design() -> None:
    model = MothSTT(vocab_size=1024)
    total = sum(parameter.numel() for parameter in model.parameters())
    encoder_params = 7_632_384  # docs/02 §4 (conv1 + conv2 + 4 blocks + final LN)
    head_params = 384 * (1024 + 1) + (1024 + 1)  # V_ctc = V_sp + 1 (docs/02 §5)
    assert total == encoder_params + head_params


def test_head_blocks_add_parameters_and_keep_the_forward_contract() -> None:
    model = MothSTT(vocab_size=1024, head_blocks=2)
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_575_169
    model.eval()
    with torch.no_grad():
        log_probs, out_lengths = model(_mel(batch=2, frames=100), torch.tensor([100, 81]))
    assert log_probs.shape == (2, 50, 1025)
    assert out_lengths.tolist() == [50, 41]


def test_head_block_padding_is_masked() -> None:
    model = MothSTT(vocab_size=64, encoder_config=EncoderConfig(n_layers=2), head_blocks=2)
    model.eval()
    short = _mel(frames=60, seed=1)
    long = _mel(frames=100, seed=2)
    batch = torch.cat([torch.nn.functional.pad(short, (0, 40)), long], dim=0)
    with torch.no_grad():
        batched, _ = model(batch, torch.tensor([60, 100]))
        single, _ = model(short, torch.tensor([60]))
    assert torch.allclose(batched[0, :28], single[0, :28], atol=1e-4)


def test_forward_shapes_and_log_probs() -> None:
    model = MothSTT(vocab_size=64, encoder_config=EncoderConfig(n_layers=2))
    model.eval()
    mel = _mel(batch=2, frames=100)
    lengths = torch.tensor([100, 81])
    with torch.no_grad():
        log_probs, out_lengths = model(mel, lengths)
    assert log_probs.shape == (2, 50, 65)
    assert out_lengths.tolist() == [50, 41]
    probabilities = log_probs.exp()
    assert torch.allclose(probabilities.sum(-1), torch.ones(2, 50), atol=1e-4)


def test_positions_are_buffers_not_parameters_and_cover_30_seconds() -> None:
    model = MothSTT(vocab_size=64)
    assert model.encoder_config.max_frames == 1500  # 30 s at 50 Hz
    assert "positions" in dict(model.encoder.named_buffers())
    assert "positions" not in model.state_dict()  # not persisted in checkpoints


def test_too_long_input_is_rejected() -> None:
    model = MothSTT(vocab_size=64, encoder_config=EncoderConfig(n_layers=1, max_frames=40))
    mel = _mel(frames=200)
    try:
        model(mel, torch.tensor([200]))
    except ValueError as error:
        assert "chunk" in str(error)
    else:  # pragma: no cover - the raise is the contract
        raise AssertionError("expected a ValueError for inputs beyond 30 s")


def test_padding_never_reaches_the_attention_of_real_frames() -> None:
    """A short clip must produce the same logits inside a padded batch as alone (conv edges aside)."""
    model = MothSTT(vocab_size=64, encoder_config=EncoderConfig(n_layers=2))
    model.eval()
    short = _mel(frames=60, seed=1)
    long = _mel(frames=100, seed=2)
    batch = torch.cat([torch.nn.functional.pad(short, (0, 40)), long], dim=0)
    lengths = torch.tensor([60, 100])
    with torch.no_grad():
        batched, _ = model(batch, lengths)
        single, _ = model(short, torch.tensor([60]))
    # Frames 0..27 are at least two encoder frames away from the padded boundary.
    assert torch.allclose(batched[0, :28], single[0, :28], atol=1e-4)


def test_batched_masked_equals_single_for_middle_frames() -> None:
    model = MothSTT(vocab_size=64, encoder_config=EncoderConfig(n_layers=2))
    model.eval()
    first = _mel(frames=90, seed=3)
    second = _mel(frames=90, seed=4)
    with torch.no_grad():
        both, _ = model(torch.cat([first, second], dim=0), torch.tensor([90, 90]))
        alone, _ = model(first, torch.tensor([90]))
    assert torch.allclose(both[0], alone[0], atol=1e-4)
    assert torch.allclose(both[1], alone[0], atol=1e-3) is False  # different audio ⇒ different output


def test_sinusoids_shape_and_range() -> None:
    table = sinusoids(1500, 384)
    assert table.shape == (1500, 384)
    assert float(table.min()) >= -1.0
    assert float(table.max()) <= 1.0


def test_model_module_does_not_import_transformers() -> None:
    source = inspect.getsource(stt.model)
    assert "import transformers" not in source
    assert "from transformers" not in source
