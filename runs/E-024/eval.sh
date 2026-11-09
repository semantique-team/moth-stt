#!/usr/bin/env bash
# E-024 evaluation — noise curve (primary) + clean gate (secondary).
#
# The control is E-017, the same recipe without noise augmentation. Both arms are evaluated by the
# identical script, on the identical clips with the identical noise sequence, so every comparison is
# paired. Decode parameters are frozen from E-022.
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE=runs/E-017/checkpoints/ckpt_best.pt
NOISY=runs/E-024/checkpoints/ckpt_best.pt

# 1. The control: E-017 has not been measured on the noise curve in this directory.
uv run python runs/E-024/noise_eval.py --checkpoint "$BASE" --out runs/E-024/eval/base

# 2. The arm. --skip-clean first is tempting (it is the expensive part), but the clean gate is the
#    reason the arm can ship at all, so it is part of the same run.
uv run python runs/E-024/noise_eval.py --checkpoint "$NOISY" --out runs/E-024/eval/noisy

# 3. Paired comparison of the two curves.
uv run python runs/E-024/compare.py

# 4. The noise set with the VAD guard, to confirm no words appear on non-speech (unchanged from
#    E-017 unless the acoustic model got more/less trigger-happy).
uv run python runs/E-005/eval_sets.py --checkpoint "$NOISY" --config runs/E-024/config.yaml \
    --sets noise --out runs/E-024/eval/noise-set --decoder beam \
    --lm runs/E-008/lm_bpe3_aug.pkl.gz --lm-weight 0.5 --vad-guard
