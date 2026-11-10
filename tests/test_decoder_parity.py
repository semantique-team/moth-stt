"""Identical synthetic scores, no release assets or Node packages required."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from stt.decode import beam_search, greedy
from stt.lm import BOS_ID, EOS_ID

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is required for decoder parity")


def test_python_node_decoder_parity() -> None:
    pieces = ["", "\u2581l", "'", "inglese", "\u2581meeting"]
    fixtures = []
    # Exact ties, blank inside/outside top-k, blank-only, repeats, and empty/limited input.
    rows = [
        [[0, -0.1, -0.2, -10, -10]],
        [[-10, -0.5, -0.5, -0.5, -0.5]],
        [[0, -10, -10, -10, -10]] * 3,
        [[-10, 0, -10, -10, -10]] * 3,
        [[-10, 0, -10, -10, -10], [0, -10, -10, -10, -10], [-10, 0, -10, -10, -10]],
        [[-1, -0.5, -10, -10, -10], [-0.5, -10, -1.2, -10, -10]],
        [
            [-10, 0, -10, -10, -10],
            [-10, -10, 0, -10, -10],
            [-10, -10, -10, 0, -10],
            [-10, -10, -10, -10, 0],
        ],
    ]
    rng = np.random.default_rng(42)
    rows.extend(rng.integers(-8, 1, size=(8, 5)).tolist() for _ in range(12))
    for matrix in rows:
        for top_k in [1, 2, 4, 8]:
            for bonus in [0, 1, 2.4]:
                for use_lm in [False, True]:
                    fixtures.append(
                        {
                            "rows": matrix,
                            "frames": len(matrix),
                            "topK": top_k,
                            "tokenBonus": bonus,
                            "useLM": use_lm,
                        }
                    )
    fixtures.extend(
        {"rows": rows[-1], "frames": n, "topK": 2, "tokenBonus": 0, "useLM": False}
        for n in [0, 1, 3]
    )

    class TinyLM:
        """Asymmetric table-free scorer exposes BOS/EOS and CTC-to-LM ID mistakes."""

        def score(self, prev2: int, prev1: int, token: int) -> float:
            assert prev2 in {BOS_ID, 0, 1, 2, 3}
            assert prev1 in {BOS_ID, 0, 1, 2, 3}
            assert token in {EOS_ID, 0, 1, 2, 3}
            return -((prev2 * 3 + prev1 * 5 + token * 7) % 11) / 4

    script = """
import { readFileSync } from 'node:fs';
import { beamSearch, greedy } from './web/decode.js';
const {fixtures, pieces} = JSON.parse(readFileSync(0, 'utf8'));
const lm = { score: (a, b, c) => -((a * 3 + b * 5 + c * 7) % 11) / 4 };
console.log(JSON.stringify(fixtures.map(f => {
  const scores = Float32Array.from(f.rows.flat());
  return [greedy(scores, f.frames, pieces.length, pieces),
    beamSearch(scores, f.frames, pieces.length, pieces, {
      beamWidth: 8, topK: f.topK, tokenBonus: f.tokenBonus,
      lm: f.useLM ? lm : null, lmWeight: 0.5,
    })];
})));
"""
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        cwd=ROOT,
        input=json.dumps({"fixtures": fixtures, "pieces": pieces}),
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    outputs = json.loads(result.stdout)
    for fixture, actual in zip(fixtures, outputs, strict=True):
        scores = np.asarray(fixture["rows"], dtype=np.float32)
        expected = [
            greedy(scores, fixture["frames"], pieces),
            beam_search(
                scores,
                fixture["frames"],
                pieces,
                beam_width=8,
                top_k=fixture["topK"],
                token_bonus=fixture["tokenBonus"],
                lm=TinyLM() if fixture["useLM"] else None,
                lm_weight=0.5,
            ),
        ]
        assert actual == expected, fixture
