#!/usr/bin/env bash
# E-014 held-out evaluation: greedy and beam 5 + LM 0.2 on the four ship-gate sets.
# FLEURS, CV test and T were never trained on; T is the mixed set that carries the ETA.
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=runs/E-014/checkpoints/ckpt_best.pt
CONFIG=runs/E-014/config.yaml
SETS=fleurs_it_test,fleurs_en_test,cv_it_test,recorded_t

uv run python runs/E-005/eval_sets.py --checkpoint "$CKPT" --config "$CONFIG" \
    --sets "$SETS" --out runs/E-014/eval/greedy

uv run python runs/E-005/eval_sets.py --checkpoint "$CKPT" --config "$CONFIG" \
    --sets "$SETS" --out runs/E-014/eval/beam-lm \
    --decoder beam --lm runs/E-008/lm_bpe3.pkl.gz --lm-weight 0.2
