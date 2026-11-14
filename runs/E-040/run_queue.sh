#!/bin/sh
# E-040 — run the remaining multi-seed trainings one after another.
#
# Waits for any E-040 run already in flight, then trains seed2/frozen, seed2/last2, seed3/frozen
# and seed3/last2 sequentially, preflighting each config and stopping the queue on the first
# failure. One run at a time, same recipe as E-036; only the seed changes.
set -eu

cd "$(dirname "$0")/../.."
LOG="runs/E-040/queue.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >>"$LOG"; }

say "queue starting; waiting for any in-flight E-040 training"
while pgrep -f "stt/train.py --config runs/E-040" >/dev/null 2>&1; do
  sleep 60
done
say "no E-040 training in flight"

for config in \
  runs/E-040/seed2/frozen/config.yaml \
  runs/E-040/seed2/last2/config.yaml \
  runs/E-040/seed3/frozen/config.yaml \
  runs/E-040/seed3/last2/config.yaml; do
  out="${config%/config.yaml}"
  say "preflight $config"
  if ! .venv/bin/python stt/train.py --config "$config" --check-config >"$out/preflight.json" 2>&1; then
    say "PREFLIGHT FAILED $config"
    exit 1
  fi
  say "start $config"
  if ! caffeinate -i .venv/bin/python stt/train.py --config "$config" --device mps \
    >"$out/train.out" 2>&1; then
    say "TRAINING FAILED $config"
    exit 1
  fi
  say "done $config"
done

say "queue complete"
