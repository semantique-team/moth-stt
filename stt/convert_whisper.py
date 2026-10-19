"""One-off conversion: ``openai/whisper-tiny`` encoder weights → our module (design: ``docs/02`` §3).

The encoder in :mod:`stt.model` is a plain-PyTorch re-implementation (~100 lines) that never
depends on ``transformers`` at train/export time. This script maps the Hugging Face state dict
onto it once, verifies it on a 30 s padded input and writes ``models/whisper_tiny_encoder.pt``
(gitignored; reproducible from the script + the HF revision).

Verification (measured 2026-10-19 on whisper-tiny): the reference is ``transformers``' **eager**
attention path with the checkpoint's ``embed_positions`` table substituted into a scratch copy
(`_PARITY_GATE` = 1e-3). Two measured facts force this shape of the check: (a) the released
checkpoint stores a position table that differs from the pure sinusoid formula by ≈3e-4 (a
precision artifact of the original training stack; our model regenerates the formula, per
``docs/02`` §3), and (b) switching ``transformers`` to its own sdpa kernel moves the encoder
output by 4e-4–9e-4 — the fp32 noise floor of any independent re-implementation, amplified by
the 4 blocks. The strict offline parity test (random weights, ≤1e-4) lives in
``tests/test_convert.py`` and runs in CI; ``map_state_dict`` is pure name/shape mapping so that
test needs no network.
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
#: Absolute gate on the 30 s parity case, positions matched, eager reference (see module docstring).
_PARITY_GATE = 1e-3

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
    length = torch.tensor([3000])
    with torch.no_grad():
        hf_encoder.config._attn_implementation = "eager"  # the reference implementation
        expected = hf_encoder(mel).last_hidden_state
        hf_encoder.config._attn_implementation = "sdpa"
        sdpa = hf_encoder(mel).last_hidden_state
        got_generated, _lengths = encoder(mel, length)
        # scratch copy of our forward with the checkpoint's positions substituted
        encoder.positions[:1500] = hf_encoder.state_dict()[_POSITIONAL].to(torch.float32)
        got_matched, _lengths = encoder(mel, length)
    # positions are non-persistent buffers: the substitution never reaches state_dict()

    kernel_floor = float((sdpa - expected).abs().max())
    generated_diff = float((got_generated - expected).abs().max())
    matched_diff = float((got_matched - expected).abs().max())
    scale = float(expected.abs().max())
    matched_relative = matched_diff / scale if scale else 0.0
    print(
        f"parity, positions matched vs eager: max |delta| = {matched_diff:.3e} "
        f"(gate {_PARITY_GATE}; relative {matched_relative:.2e})"
    )
    print(
        f"generated-vs-checkpoint positions: table |delta| = {report['position_max_diff']:.3e}; "
        f"end-to-end |delta| = {generated_diff:.3e}"
    )
    print(f"reference noise floor (transformers sdpa vs eager): {kernel_floor:.3e}")
    if matched_diff > _PARITY_GATE and not args.skip_verify:
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
            "max_abs_diff": matched_diff,
            "max_abs_diff_generated_positions": generated_diff,
            "kernel_floor": kernel_floor,
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
        "parity": {
            "reference": "transformers eager, checkpoint positions substituted",
            "gate": _PARITY_GATE,
            "max_abs_diff": matched_diff,
            "relative_max_abs_diff": matched_relative,
            "max_abs_diff_generated_positions": generated_diff,
            "kernel_floor_sdpa_vs_eager": kernel_floor,
            "position_max_diff": report["position_max_diff"],
        },
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), sha256 {sha256[:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
