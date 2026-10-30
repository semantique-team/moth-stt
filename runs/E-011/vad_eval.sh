#!/usr/bin/env bash
# E-011 addendum — re-measure the noise and speech sets with the Silero VAD guard (D25).
# Decode-only: E-016 best checkpoint and the augmented E-008 LM, alpha 0.2, as in release 0.3.1.
# Acceptance: 0 non-music words on the noise set (both decoders), 0 clips gated on T-400 and
# FLEURS-IT.
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=runs/E-016/checkpoints/ckpt_best.pt
CONFIG=runs/E-016/config.yaml
LM=runs/E-008/lm_bpe3_aug.pkl.gz
EVAL=runs/E-005/eval_sets.py

uv run python "$EVAL" --checkpoint "$CKPT" --config "$CONFIG" \
    --sets noise --out runs/E-011/vad-noise-greedy --vad-guard

uv run python "$EVAL" --checkpoint "$CKPT" --config "$CONFIG" \
    --sets noise --out runs/E-011/vad-noise-beam-lm --decoder beam \
    --lm "$LM" --lm-weight 0.2 --vad-guard

uv run python "$EVAL" --checkpoint "$CKPT" --config "$CONFIG" \
    --sets recorded_t,fleurs_it_test --out runs/E-011/vad-speech-greedy --vad-guard

uv run python "$EVAL" --checkpoint "$CKPT" --config "$CONFIG" \
    --sets recorded_t,fleurs_it_test --out runs/E-011/vad-speech-beam-lm --decoder beam \
    --lm "$LM" --lm-weight 0.2 --vad-guard
