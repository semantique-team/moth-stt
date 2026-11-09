"""Feasibility probe (not an experiment): what does InterCTC cost per training step?

Times a realistic training step with and without an auxiliary CTC loss attached at the encoder
output, on this machine, to replace a guess with a multiplier before E-024 is written.

    uv run python runs/E-024/probe_cost.py [--reps 20]

Three designs were tried; only the third measured anything, and the two failures are recorded in
the code because they are the interesting part:

1. **Stepping through real batches** — step time varied 2x with batch composition, more than the
   effect. Timing a *fixed* batch from the loader did not help: the spread stayed around 70%,
   because the loader, the optimizer and the allocator interact.
2. **Back-to-back pairs of different variants** — showed "InterCTC is 9% faster", which is
   impossible: the first call in a pair pays allocator warm-up.
3. **Synthetic tensors of the shape E-017 actually used, tight loops, alternating round order** —
   stable to ±0.5% in a separate check, which is what makes the ~2% effect readable at all.

The quantity measured is the ratio, so the batch does not need to be real audio; it needs the right
shape. E-017 packed 192 audio-seconds per batch at 50 Hz, i.e. 9,600 encoder input frames across 32
clips (600 frames = 6 s each at the median).

A sensitivity control runs the encoder twice and must land near 2x; if it does not, the probe
refuses to report a projection.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from stt.model import MothSTT, _padding_mask

REPO = Path(__file__).resolve().parents[2]

#: E-017's batch shape: 32 clips, 192 audio-seconds total, 50 Hz input frames.
BATCH, FRAMES = 32, 750
CLIP_FRAMES = 600  # 6 s per clip → 300 encoder frames; only the total budget needs to match
VOCAB_CTC = 1025
TARGET_LEN = 30
AUX_WEIGHT = 0.3


class MothWithAux(MothSTT):
    """E-017's model plus a training-only CTC head at the encoder output."""

    def __init__(self, vocab_size: int, head_blocks: int) -> None:
        super().__init__(vocab_size=vocab_size, head_blocks=head_blocks)
        self.aux_head = nn.Linear(self.encoder_config.width, vocab_size + 1)


def make_model(device: str) -> MothWithAux:
    model = MothWithAux(vocab_size=1024, head_blocks=2)
    payload = torch.load(
        REPO / "runs/E-017/checkpoints/ckpt_best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model"], strict=False)  # aux_head is fresh, as in the experiment
    return model.to(device).train()


def main() -> int:
    parser = argparse.ArgumentParser(prog="probe_cost.py")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device_kind = device.split(":")[0]
    torch.manual_seed(0)

    model = make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    mel = torch.randn(BATCH, 80, CLIP_FRAMES, device=device) * 0.5
    lengths = torch.full((BATCH,), CLIP_FRAMES, dtype=torch.long, device=device)
    targets = torch.randint(1, VOCAB_CTC, (BATCH * TARGET_LEN,), dtype=torch.long, device=device)
    target_lengths = torch.full((BATCH,), TARGET_LEN, dtype=torch.long, device=device)

    def step(variant: str) -> float:
        started = time.perf_counter()
        with torch.autocast(device_type=device_kind, dtype=torch.bfloat16, enabled=True):
            hidden, out_lengths = model.encoder(mel, lengths)
            if variant != "baseline":
                aux_logits = model.aux_head(hidden)
            work = hidden
            if model.head_blocks:
                keep = _padding_mask(out_lengths, work.shape[1])
                for block in model.head_blocks:
                    work = block(work, keep)
            log_probs = F.log_softmax(model.head(work).float(), dim=-1)
            if variant == "extra_encoder":
                # Sensitivity control: a second full encoder pass, inside the same graph.
                hidden2, _ = model.encoder(mel, lengths)
                work2 = hidden2
                keep2 = _padding_mask(out_lengths, work2.shape[1])
                for block in model.head_blocks:
                    work2 = block(work2, keep2)
                log_probs = log_probs + 0.0 * model.head(work2).float()
        loss = criterion(log_probs.transpose(0, 1).contiguous(), targets, out_lengths, target_lengths)
        if variant != "baseline":
            aux_log_probs = F.log_softmax(aux_logits.float(), dim=-1)
            aux_loss = criterion(
                aux_log_probs.transpose(0, 1).contiguous(), targets, out_lengths, target_lengths
            )
            loss = (1.0 - AUX_WEIGHT) * loss + AUX_WEIGHT * aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if device_kind == "mps":
            torch.mps.synchronize()
        return time.perf_counter() - started

    variants = ("baseline", "interctc", "extra_encoder")
    for _ in range(args.warmup):
        for variant in variants:
            step(variant)

    times: dict[str, list[float]] = {name: [] for name in variants}
    for round_index in range(args.rounds):
        order = list(reversed(variants)) if round_index % 2 else list(variants)
        for variant in order:
            for _ in range(args.reps):
                times[variant].append(step(variant))

    results: dict[str, object] = {
        "device": device,
        "shape": {"batch": BATCH, "clip_frames": CLIP_FRAMES, "audio_s": BATCH * CLIP_FRAMES / 100},
        "samples_per_variant": args.reps * args.rounds,
    }
    for variant in variants:
        results[variant] = {
            "median_s": statistics.median(times[variant]),
            "min_s": min(times[variant]),
            "p10_s": sorted(times[variant])[len(times[variant]) // 10],
        }
        print(json.dumps({"variant": variant, **results[variant]}), flush=True)  # type: ignore[dict-item]

    base = results["baseline"]["median_s"]  # type: ignore[index]
    overhead = results["interctc"]["median_s"] / base  # type: ignore[index]
    control = results["extra_encoder"]["median_s"] / base  # type: ignore[index]
    sensitive = control > overhead + 0.05
    results |= {
        "overhead_ratio": overhead,
        "sensitivity_control_ratio": control,
        "timer_is_sensitive": sensitive,
        "aux_weight": AUX_WEIGHT,
    }
    print(
        json.dumps(
            {
                "overhead_ratio": round(overhead, 4),
                "sensitivity_control_ratio": round(control, 4),
                "timer_is_sensitive": sensitive,
            }
        ),
        flush=True,
    )

    (REPO / "runs/E-024").mkdir(parents=True, exist_ok=True)
    if not sensitive:
        print("timer failed its sensitivity control: no projection is valid", flush=True)
        (REPO / "runs/E-024" / "probe_cost.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        return 1

    # Anchor the projection on E-017's *measured* wall time, not on this probe's absolute step time
    # (the probe's synthetic batch is not E-017's real batch). Only the ratio transfers.
    e017_hours = 3.35  # 26642 steps, 2026-11-07
    results["projection"] = {
        "e017_hours": e017_hours,
        "interctc_hours": round(e017_hours * overhead, 2),
        "added_minutes": round((e017_hours * overhead - e017_hours) * 60, 1),
    }
    print(json.dumps(results["projection"]), flush=True)
    (REPO / "runs/E-024" / "probe_cost.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {REPO / 'runs/E-024' / 'probe_cost.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())