import pytest
import torch

from stt.convert_whisper import convert_encoder, map_state_dict
from stt.model import EncoderConfig

transformers = pytest.importorskip("transformers")


def _hf_encoder():
    from transformers import WhisperConfig

    try:
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
    except ImportError:  # older layouts
        from transformers import WhisperEncoder  # type: ignore[attr-defined]

    config = WhisperConfig(
        d_model=384,
        encoder_layers=4,
        encoder_attention_heads=6,
        encoder_ffn_dim=1536,
        num_mel_bins=80,
        max_source_positions=1500,
    )
    torch.manual_seed(0)
    encoder = WhisperEncoder(config)
    encoder.eval()
    return encoder


def test_map_state_dict_reports_skipped_positions_and_no_unknown() -> None:
    hf = _hf_encoder()
    state = {"encoder." + key: value for key, value in hf.state_dict().items()}
    mapped, report = map_state_dict(state)
    assert "conv1.weight" in mapped
    assert "blocks.0.query.weight" in mapped
    assert "blocks.3.mlp2.bias" in mapped
    assert "attn_ln.weight" not in mapped
    assert report["mapped"] == len(mapped) == len(hf.state_dict()) - 1
    assert report["skipped"] == ["embed_positions.weight"]
    assert report["unknown"] == []
    assert report["position_max_diff"] is not None
    assert report["position_max_diff"] < 1e-4


def test_convert_encoder_copies_weights() -> None:
    hf = _hf_encoder()
    encoder, _ = convert_encoder(hf.state_dict())
    reference = hf.state_dict()
    assert torch.equal(encoder.conv1.weight, reference["conv1.weight"])
    assert torch.equal(encoder.conv2.bias, reference["conv2.bias"])
    assert encoder.blocks[2].query.weight is not None
    assert torch.equal(encoder.blocks[2].key.weight, reference["layers.2.self_attn.k_proj.weight"])
    assert torch.equal(encoder.final_ln.weight, reference["layer_norm.weight"])
    assert not hasattr(encoder, "head")  # encoder only: the CTC head lives in MothSTT


def test_offline_parity_with_transformers_whisper_encoder() -> None:
    """Random weights, same mapping: our forward must equal WhisperEncoder within 1e-4 on 30 s."""
    hf = _hf_encoder()
    encoder, _report = convert_encoder(hf.state_dict(), EncoderConfig())
    encoder.eval()

    torch.manual_seed(1)
    mel = torch.randn(1, 80, 3000) * 0.1
    with torch.no_grad():
        expected = hf(mel).last_hidden_state
        got, lengths = encoder(mel, torch.tensor([3000]))
    assert lengths.tolist() == [1500]
    assert got.shape == expected.shape == (1, 1500, 384)
    assert torch.max(torch.abs(got - expected)).item() < 1e-4


def test_unknown_key_rejected() -> None:
    state = {"conv1.weight": torch.zeros(384, 80, 3), "mystery.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="mystery"):
        convert_encoder(state, EncoderConfig())
