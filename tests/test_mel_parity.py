"""Python/JS mel parity with synthetic PCM and the packaged filterbank only."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from stt.features import HOP_LENGTH, N_MELS, SAMPLE_RATE, load_mel_filters, log_mel

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is required for mel parity")


@pytest.mark.parametrize(
    ("signal", "samples"),
    [("zeros", 400 + 160), ("sine", 1600), ("noise", 1600), ("noise", 1637)],
    ids=["minimum-silence", "sine", "noise", "nonmultiple-hop"],
)
def test_python_node_mel_parity(signal: str, samples: int) -> None:
    if signal == "zeros":
        pcm = np.zeros(samples, dtype=np.float32)
    elif signal == "sine":
        pcm = (0.2 * np.sin(2 * np.pi * 440 * np.arange(samples) / SAMPLE_RATE)).astype(np.float32)
    else:
        pcm = (0.1 * np.random.default_rng(42).standard_normal(samples)).astype(np.float32)
    filters = load_mel_filters()
    expected = log_mel(pcm, filters=filters)

    script = """
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
const { computeMel } = await import(pathToFileURL(resolve('web/mel.js')).href);
const { pcm, filters } = JSON.parse(readFileSync(0, 'utf8'));
const { data, frames, bins } = computeMel(Float32Array.from(pcm), Float32Array.from(filters));
console.log(JSON.stringify({ data: Array.from(data), frames, bins }));
"""
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        cwd=ROOT,
        input=json.dumps({"pcm": pcm.tolist(), "filters": filters.ravel().tolist()}),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    output = json.loads(result.stdout)
    assert output["bins"] == N_MELS
    assert output["frames"] == samples // HOP_LENGTH
    actual = np.asarray(output["data"], dtype=np.float32).reshape(output["bins"], output["frames"])
    assert actual.shape == expected.shape == (N_MELS, samples // HOP_LENGTH)
    # JS uses float32 Hann/DFT tables rather than NumPy's float64 FFT.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
