"""Whisper-tiny encoder + CTC head (design: ``docs/02-architecture.md`` §3–4).

A ~100-line plain-PyTorch re-implementation of the Whisper-tiny encoder (width 384, 4 pre-LN
blocks, 6 heads × 64, FFN 1536, fixed sinusoidal positions sliced to the true length) with a
linear CTC head over the shared IT+EN BPE vocabulary. No decoder, no language token. Training
uses a **key-padding mask** built from the per-clip lengths; inference runs batch 1 at the true
length. Positional sinusoids are generated, never stored in checkpoints. ``transformers`` is a
dev-only dependency (weight-mapping parity test), never imported here.

Edge note: the convolutional front-end sees the zero-padding of a batch at the clip boundary
(the same zeros it would see from its own conv padding), so padding influences at most the last
encoder frame — attention itself never lets padding leak (the mask is tested).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["EncoderConfig", "MothSTT", "WhisperTinyEncoder", "sinusoids"]


@dataclass(frozen=True)
class EncoderConfig:
    """Whisper-tiny dimensions; ``max_frames`` is the 30 s sinusoid-table limit."""

    n_mels: int = 80
    width: int = 384
    n_layers: int = 4
    n_heads: int = 6
    ffn_dim: int = 1536
    max_frames: int = 1500
    max_timescale: float = 10000.0


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> torch.Tensor:
    """Fixed sinusoidal table, computed like Whisper's (float32 arithmetic, not float64)."""
    if channels % 2 != 0:
        raise ValueError(f"channels must be even, got {channels}")
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class _Block(nn.Module):
    """Pre-LN transformer block: ``x = x + Attn(LN1(x)); x = x + MLP(LN2(x))``."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.heads = config.n_heads
        self.attn_ln = nn.LayerNorm(config.width)
        self.mlp_ln = nn.LayerNorm(config.width)
        self.query = nn.Linear(config.width, config.width)
        self.key = nn.Linear(config.width, config.width, bias=False)  # no bias, like Whisper
        self.value = nn.Linear(config.width, config.width)
        self.out = nn.Linear(config.width, config.width)
        self.mlp1 = nn.Linear(config.width, config.ffn_dim)
        self.mlp2 = nn.Linear(config.ffn_dim, config.width)

    def forward(self, x: torch.Tensor, keep_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self._attention(self.attn_ln(x), keep_mask)
        return x + self.mlp2(F.gelu(self.mlp1(self.mlp_ln(x))))

    def _attention(self, x: torch.Tensor, keep_mask: torch.Tensor | None) -> torch.Tensor:
        batch, frames, width = x.shape
        head_dim = width // self.heads
        scale = head_dim**-0.25  # q and k are each scaled by head_dim^-0.25, like Whisper

        def split(projection: nn.Linear, tensor: torch.Tensor) -> torch.Tensor:
            return projection(tensor).view(batch, frames, self.heads, head_dim).transpose(1, 2)

        q = split(self.query, x) * scale
        k = split(self.key, x) * scale
        v = split(self.value, x)
        scores = q @ k.transpose(-1, -2)
        if keep_mask is not None:
            scores = scores.masked_fill(~keep_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        merged = (weights @ v).transpose(1, 2).reshape(batch, frames, width)
        return self.out(merged)


class WhisperTinyEncoder(nn.Module):
    """Conv stem (50 Hz) + blocks + final LayerNorm; returns hidden states and encoder lengths."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        self.conv1 = nn.Conv1d(self.config.n_mels, self.config.width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            self.config.width, self.config.width, kernel_size=3, stride=2, padding=1
        )
        self.register_buffer(
            "positions",
            sinusoids(self.config.max_frames, self.config.width, self.config.max_timescale),
            persistent=False,  # generated, never stored in checkpoints
        )
        self.blocks = nn.ModuleList(_Block(self.config) for _ in range(self.config.n_layers))
        self.final_ln = nn.LayerNorm(self.config.width)

    def forward(
        self, mel: torch.Tensor, mel_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mel.shape[1] != self.config.n_mels:
            raise ValueError(f"expected {self.config.n_mels} mel bins, got {mel.shape[1]}")
        if mel.shape[2] > self.config.max_frames * 2:
            raise ValueError(
                f"input longer than 30 s ({mel.shape[2]} mel frames) — chunk the audio"
            )
        lengths = torch.as_tensor(mel_lengths, dtype=torch.long, device=mel.device)

        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # [B, T_enc, width], T_enc = ceil(T / 2) at 50 Hz
        enc_lengths = torch.div(lengths + 1, 2, rounding_mode="floor")

        x = x + self.positions[: x.shape[1]].to(x.dtype)
        keep_mask = None
        if bool((enc_lengths < x.shape[1]).any()):
            positions = torch.arange(x.shape[1], device=x.device)[None, :]
            keep_mask = (positions < enc_lengths[:, None])[:, None, None, :]
        for block in self.blocks:
            x = block(x, keep_mask)
        return self.final_ln(x), enc_lengths


class MothSTT(nn.Module):
    """The shippable model: encoder + ``Linear(width, V_ctc)`` head, ``V_ctc = vocab_size + 1``."""

    def __init__(self, vocab_size: int = 1024, encoder_config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.encoder_config = encoder_config or EncoderConfig()
        self.encoder = WhisperTinyEncoder(self.encoder_config)
        self.head = nn.Linear(self.encoder_config.width, vocab_size + 1)

    def forward(
        self, mel: torch.Tensor, mel_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``mel [B, 80, T]`` → ``log_probs [B, ceil(T/2), V_ctc]`` (fp32) + out lengths."""
        hidden, out_lengths = self.encoder(mel, mel_lengths)
        logits = self.head(hidden)
        return F.log_softmax(logits.float(), dim=-1), out_lengths
