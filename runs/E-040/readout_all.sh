#!/bin/sh
# E-040 — export and read out the remaining replicates, one at a time.
#
# Each run's best checkpoint is exported to int8 (the shipped format) and then decoded with the
# A3 LM and the E-030 decoder. Logs go to runs/E-040/readout.log; stops on the first failure.
set -eu

cd "$(dirname "$0")/../.."
LOG="runs/E-040/readout.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >>"$LOG"; }

for entry in \
  "seed1 last2 s1-last2" \
  "seed2 frozen s2-frozen" \
  "seed2 last2 s2-last2" \
  "seed3 frozen s3-frozen" \
  "seed3 last2 s3-last2"; do
  set -- $entry
  seed=$1
  arm=$2
  name=$3
  run="runs/E-040/$seed/$arm"
  if [ -f "runs/E-040/eval/$name/readout.json" ]; then
    say "skip $name (already read out)"
    continue
  fi
  say "export $run"
  if ! caffeinate -i .venv/bin/python -m stt.export \
    --out "$run/release-int8" \
    --checkpoint "$run/checkpoints/ckpt_best.pt" \
    --vocab release/vocab.json --head-blocks 2 --version "e040-$name" \
    >"$run/export.out" 2>&1; then
    say "EXPORT FAILED $run"
    exit 1
  fi
  say "readout $name"
  if ! caffeinate -i .venv/bin/python runs/E-040/readout.py --run "$run" --name "$name" \
    >"runs/E-040/eval-$name.out" 2>&1; then
    say "READOUT FAILED $name"
    exit 1
  fi
  say "done $name"
done

say "readouts complete"
