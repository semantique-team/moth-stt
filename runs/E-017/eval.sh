#!/usr/bin/env bash
# E-017 evaluation — cross-speaker gold (trained on S1 + S2, S3 held out).
# Decode-only. Mirrors runs/E-016/eval.sh plus the S3/S2 slices and the room-tone noise set.
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=runs/E-017/checkpoints/ckpt_best.pt
CONFIG=runs/E-017/config.yaml
LM=runs/E-008/lm_bpe3_aug.pkl.gz
EVAL=runs/E-005/eval_sets.py
OUT=runs/E-017/eval
S3=runs/E-017/manifests/T-S3.jsonl
S2=runs/E-017/manifests/T-S2.jsonl

# 1. Ship-gate sets, greedy and beam+LM.
uv run python "$EVAL" --checkpoint "$CKPT" --config "$CONFIG" \
    --sets fleurs_it_test,fleurs_en_test,cv_it_test,recorded_t --out "$OUT/greedy"

uv run python "$EVAL" --checkpoint "$CKPT" --config "$CONFIG" \
    --sets fleurs_it_test,fleurs_en_test,cv_it_test,recorded_t --out "$OUT/beam-lm" \
    --decoder beam --lm "$LM" --lm-weight 0.2

# 2. Noise set with the VAD guard (room tone included).
uv run python "$EVAL" --checkpoint "$CKPT" --config "$CONFIG" \
    --sets noise --out "$OUT/noise-greedy" --vad-guard
uv run python "$EVAL" --checkpoint "$CKPT" --config "$CONFIG" \
    --sets noise --out "$OUT/noise-beam-lm" --decoder beam \
    --lm "$LM" --lm-weight 0.2 --vad-guard

# 3. Per-speaker slices. S3 is the held-out speaker (the generalisation claim);
#    S2 is a training speaker under E-017 and is reported separately.
for dec in greedy beam-lm; do
    hyp="$OUT/$dec/recorded_t.hyp.jsonl"
    uv run python -m stt.eval --manifest "$S3" --hyp "$hyp" --out "$OUT/s3-$dec.metrics.json"
    uv run python -m stt.eval --manifest "$S2" --hyp "$hyp" --out "$OUT/s2-$dec.metrics.json"
done

# 4. Paired differences on S3: E-017 vs E-016 (same recipe, one more gold speaker)
#    and vs E-014 (before any gold).
uv run python -m stt.eval --manifest "$S3" --hyp "$OUT/greedy/recorded_t.hyp.jsonl" \
    --vs "$OUT/e016-s3-greedy.metrics.json" --out "$OUT/e017-vs-e016-s3-greedy.diff.json"
uv run python -m stt.eval --manifest "$S3" --hyp "$OUT/beam-lm/recorded_t.hyp.jsonl" \
    --vs "$OUT/e016-s3-aug-lm.metrics.json" --out "$OUT/e017-vs-e016-s3-beam-lm.diff.json"
uv run python -m stt.eval --manifest "$S3" --hyp "$OUT/greedy/recorded_t.hyp.jsonl" \
    --vs "$OUT/e014-s3-greedy.metrics.json" --out "$OUT/e017-vs-e014-s3-greedy.diff.json"
uv run python -m stt.eval --manifest "$S3" --hyp "$OUT/beam-lm/recorded_t.hyp.jsonl" \
    --vs "$OUT/e014-s3-aug-lm.metrics.json" --out "$OUT/e017-vs-e014-s3-beam-lm.diff.json"