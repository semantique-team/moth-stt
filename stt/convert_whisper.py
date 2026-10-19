"""One-off conversion: ``openai/whisper-tiny`` encoder weights → our module (design: ``docs/02`` §3).

The encoder in :mod:`stt.model` is a plain-PyTorch re-implementation (~100 lines) that never
depends on ``transformers`` at train/export time. This script maps the Hugging Face state dict
onto it once, verifies parity on a 30 s padded input (gate: max |Δ| ≤ 1e-4, fp32) and writes
``models/whisper_tiny_encoder.pt`` (gitignored; reproducible from the script + the HF revision).

``map_state_dict`` is pure name/shape mapping, so the parity test can run offline with random
weights: build a ``transformers.WhisperEncoder``, map its state dict, and compare forward outputs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import warnings
from pathlib import Path

import torch

from stt.model import EncoderConfig, WhisperTinyEncoder, sinusoids

__all__ = ["convert_encoder", "main", "map_state_dict"]

_PREFIXES = ("model.encoder.", "encoder.")
_POSITIONAL = "embed_positions.weight"
_PARITY_GATE = 1e-4

#: Module-name renames between the HF encoder and ours.
_RENAMES = {
    "q_proj": "query",
    "k_proj": "key",
    "v_proj": "value",
    "out_proj": "out",
    "self_attn_layer_norm": "attn_ln",
    "final_layer_norm": "mlp_ln",
    "fc1": "mlp1",
    "fc2": "mlp2",
}


def _strip_prefix(name: str) -> str:
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _map_name(name: str) -> str | None:
    """HF key (prefix stripped) → our key; ``None`` when unknown."""
    if name in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"):
        return name
    if name in ("layer_norm.weight", "layer_norm.bias"):
        return f"final_ln.{name.split('.', 1)[1]}"
    parts = name.split(".")
    if parts[0] == "layers" and len(parts) >= 4:
        index = parts[1]
        if parts[2] == "self_attn" and parts[3] in ("q_proj", "k_proj", "v_proj", "out_proj"):
            if len(parts) == 5:
                return f"blocks.{index}.{_RENAMES[parts[3]]}.{parts[4]}"
        elif parts[2] in ("self_attn_layer_norm", "final_layer_norm", "fc1", "fc2") and len(parts) == 4:
            return f"blocks.{index}.{_RENAMES[parts[2]]}.{parts[3]}"
    return None


def map_state_dict(hf_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    """Map an HF encoder state dict onto our names; report skips, unknowns and position drift."""
    mapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    unknown: list[str] = []
    position_max_diff: float | None = None

    for original, tensor in hf_state.items():
        name = _strip_prefix(original)
        if name == _POSITIONAL:
            skipped.append(name)
            expected = sinusoids(tensor.shape[0], tensor.shape[1])
            position_max_diff = float((tensor.to(torch.float32) - expected).abs().max())
            continue
        target = _map_name(name)
        if target is None:
            unknown.append(name)
        else:
            mapped[target] = tensor

    report = {
        "mapped": len(mapped),
        "skipped": skipped,
        "unknown": unknown,
        "position_max_diff": position_max_diff,
    }
    return mapped, report


def convert_encoder(
    hf_state: dict[str, torch.Tensor], config: EncoderConfig | None = None
) -> tuple[WhisperTinyEncoder, dict]:
    """Instantiate our encoder and load the mapped weights (strict)."""
    mapped, report = map_state_dict(hf_state)
    if report["unknown"]:
        raise ValueError(f"unknown state-dict keys: {', '.join(report['unknown'])}")
    encoder = WhisperTinyEncoder(config or EncoderConfig())
    encoder.load_state_dict(mapped, strict=True)
    return encoder, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="convert_whisper.py",
        description="Map openai/whisper-tiny encoder weights onto stt.model and verify parity.",
    )
    parser.add_argument("--model", default="openai/whisper-tiny")
    parser.add_argument("--out", default="models/whisper_tiny_encoder.pt")
    parser.add_argument("--skip-verify", action="store_true", help="write even if parity fails")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import transformers
    from transformers import WhisperModel  # heavy import only when this script runs

    warnings.filterwarnings("ignore", category=UserWarning)
    hf_model = WhisperModel.from_pretrained(args.model)
    hf_encoder = hf_model.encoder.eval()
    encoder, report = convert_encoder(hf_encoder.state_dict())
    encoder.eval()

    torch.manual_seed(args.seed)
    mel = torch.randn(1, 80, 3000) * 0.1  # 30 s padded input, the design's parity case
    with torch.no_grad():
        expected = hf_encoder(mel).last_hidden_state
        got, _lengths = encoder(mel, torch.tensor([3000]))
    max_abs_diff = float((got - expected).abs().max())
    print(f"parity on 30 s padded input: max |delta| = {max_abs_diff:.3e} (gate {_PARITY_GATE})")
    if max_abs_diff > _PARITY_GATE and not args.skip_verify:
        print("parity gate FAILED — nothing written")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "config": dataclasses.asdict(encoder.config),
            "source": args.model,
            "transformers_version": transformers.__version__,
            "max_abs_diff": max_abs_diff,
            "position_max_diff": report["position_max_diff"],
        },
        out,
    )
    sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {
        "file": out.name,
        "sha256": sha256,
        "source": args.model,
        "transformers_version": transformers.__version__,
        "max_abs_diff": max_abs_diff,
        "position_max_diff": report["position_max_diff"],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), sha256 {sha256[:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
