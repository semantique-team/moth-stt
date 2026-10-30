#!/usr/bin/env bash
# E-017 pre-check — zero-shot cross-speaker read on S3 (2026-10-30).
# Decode-only: E-016 (gold fine-tune on S1) and E-014 (before gold) on the 600-clip T,
# plus the Whisper/Whistle/Parakeet baselines on the S3 slice. No training.
set -euo pipefail
cd "$(dirname "$0")/../.."

LM=runs/E-008/lm_bpe3_aug.pkl.gz
EVAL=runs/E-005/eval_sets.py
M=runs/E-017/manifests
B=runs/E-017/baselines

uv run python runs/E-017/make_speaker_manifest.py --speaker S3 --out "$M/T-S3.jsonl"
uv run python runs/E-017/make_speaker_manifest.py --speaker S2 --out "$M/T-S2.jsonl"

# Moth on the full T-600 (greedy and beam+LM), both checkpoints
run_moth() {
    local name=$1 ckpt=$2 cfg=$3
    uv run python "$EVAL" --checkpoint "$ckpt" --config "$cfg" \
        --sets recorded_t --out "runs/$name/eval/t600-greedy"
    uv run python "$EVAL" --checkpoint "$ckpt" --config "$cfg" \
        --sets recorded_t --out "runs/$name/eval/t600-aug-lm" \
        --decoder beam --lm "$LM" --lm-weight 0.2
}
run_moth E-016 runs/E-016/checkpoints/ckpt_best.pt runs/E-016/config.yaml
run_moth E-014 runs/E-014/checkpoints/ckpt_best.pt runs/E-014/config.yaml

# Baselines on the S3 slice
uv run baselines/whisper_runner.py --model tiny --variant auto --decode beam5 \
    --sets T-S3 --manifests-dir "$M" --audio-root recordings/audio \
    --out "$B/whisper-tiny-auto-beam5"
uv run baselines/whisper_runner.py --model tiny --variant it --decode beam5 \
    --sets T-S3 --manifests-dir "$M" --audio-root recordings/audio \
    --out "$B/whisper-tiny-it-beam5"
PYTHONPATH=. .venv-baselines/bin/python baselines/whistle_needle.py --variant it --hotwords none \
    --sets T-S3 --manifests-dir "$M" --audio-root recordings/audio --out "$B/whistle-it"
PYTHONPATH=. .venv-baselines/bin/python baselines/parakeet_sherpa.py \
    --sets T-S3 --manifests-dir "$M" --audio-root recordings/audio --out "$B/parakeet-v3"

# S3-only metrics and paired differences vs E-014 / vs each baseline
for dec in greedy aug-lm; do
    for who in E-016 E-014; do
        uv run python -m stt.eval --manifest "$M/T-S3.jsonl" \
            --hyp "runs/$who/eval/t600-$dec/recorded_t.hyp.jsonl" \
            --out "runs/E-017/eval/$(echo "$who" | tr '[:upper:]' '[:lower:]' | tr -d '-')-s3-$dec.metrics.json"
    done
    uv run python -m stt.eval --manifest "$M/T-S3.jsonl" \
        --hyp runs/E-016/eval/t600-$dec/recorded_t.hyp.jsonl \
        --vs "runs/E-017/eval/e014-s3-$dec.metrics.json" \
        --out "runs/E-017/eval/e016-vs-e014-s3-$dec.diff.json"
done
for name in whisper-tiny-auto-beam5 whisper-tiny-it-beam5 whistle-it parakeet-v3; do
    uv run python -m stt.eval --manifest "$M/T-S3.jsonl" --hyp "$B/$name/T-S3.jsonl" \
        --out "$B/$name/T-S3.metrics.json"
    uv run python -m stt.eval --manifest "$M/T-S3.jsonl" \
        --hyp runs/E-016/eval/t600-aug-lm/recorded_t.hyp.jsonl \
        --vs "$B/$name/T-S3.metrics.json" --out "$B/$name/e016-vs-$name.diff.json"
done
