#!/usr/bin/env bash
# E-016 held-out evaluation: greedy and beam 5 + LM 0.2 on the four ship-gate sets,
# plus the noise set with the no-speech guard (E-014 parity check).
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=runs/E-016/checkpoints/ckpt_best.pt
CONFIG=runs/E-016/config.yaml
SETS=fleurs_it_test,fleurs_en_test,cv_it_test,recorded_t

uv run python runs/E-005/eval_sets.py --checkpoint "$CKPT" --config "$CONFIG" \
    --sets "$SETS" --out runs/E-016/eval/greedy

uv run python runs/E-005/eval_sets.py --checkpoint "$CKPT" --config "$CONFIG" \
    --sets "$SETS" --out runs/E-016/eval/beam-lm \
    --decoder beam --lm runs/E-008/lm_bpe3.pkl.gz --lm-weight 0.2

uv run python runs/E-005/eval_sets.py --checkpoint "$CKPT" --config "$CONFIG" \
    --sets noise --out runs/E-016/eval/noise-greedy --no-speech-guard

uv run python runs/E-005/eval_sets.py --checkpoint "$CKPT" --config "$CONFIG" \
    --sets noise --out runs/E-016/eval/noise-beam-lm --decoder beam \
    --lm runs/E-008/lm_bpe3.pkl.gz --lm-weight 0.2 --no-speech-guard
