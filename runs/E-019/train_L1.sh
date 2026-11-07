#!/usr/bin/env bash
# E-019 arm L1 — Whistle LoRA on the two gold speakers (D28, exploratory side artifact).
#
# Baseline sanity check first (step 3 of the plan): L0 re-run on T-S3 must reproduce
# runs/E-017/baselines/whistle-it/ exactly. It does — hypotheses bit-identical,
# WER 28.3646 / ETA 52.1127 on 200 clips (see runs/E-019/l0/).
#
# Environment: .venv-whistle (cactus-needle 3.2.10, jax 0.12.0.1, CPU backend).
# CPU timing measured on this host: ~5.7 s/step at batch 16, so 3 epochs x 72 steps
# is ~20 min, plus validation and the CLI's test decode.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=runs/E-019/data
OUT=runs/E-019/lora-L1.safetensors

NEEDLE_TELEMETRY=0 caffeinate -i .venv-whistle/bin/needle whistle finetune \
    "$DATA/train.jsonl" "$DATA/val.jsonl" "$DATA/test.jsonl" \
    --language it \
    --epochs 3 --batch-size 16 --lr 1e-4 \
    --lora-rank 16 --lora-alpha 32 --seed 0 \
    --checkpoint-dir runs/E-019/checkpoints \
    --out "$OUT"