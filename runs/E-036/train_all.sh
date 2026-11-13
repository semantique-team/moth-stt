#!/bin/sh
# E-036 — run both arms sequentially on the M5, stopping on the first failure.
set -eu
cd "$(dirname "$0")/../.."
caffeinate -i .venv/bin/python -m stt.train --config runs/E-036/frozen/config.yaml \
  > runs/E-036/frozen/train.out 2>&1
caffeinate -i .venv/bin/python -m stt.train --config runs/E-036/last2/config.yaml \
  > runs/E-036/last2/train.out 2>&1
