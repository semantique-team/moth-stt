#!/usr/bin/env bash
# Reproduce from pinned pre-E-032 inputs; never replace the historical receipt.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python runs/E-031/reproduce_corpus.py "$@"
