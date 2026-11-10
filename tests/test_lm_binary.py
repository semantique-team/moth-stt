import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from stt.lm import BOS_ID, EOS_ID, LM_VOCAB_SIZE, BinaryBPELM, BPETrigramLM


@pytest.fixture
def binary(tmp_path):
    lm = BPETrigramLM.from_sequences(
        [[1, 2], [1, 2], [1, 3], [4], []], min_bigram_count=1, min_trigram_count=1
    )
    path = tmp_path / "lm.bin"
    lm.save_browser(path)
    return lm, path, path.read_bytes()


QUERIES = [
    (BOS_ID, BOS_ID, 1),  # BOS trigram
    (BOS_ID, BOS_ID, EOS_ID),  # empty sequence
    (BOS_ID, 1, 2),
    (1, 2, EOS_ID),  # EOS trigram
    (99, BOS_ID, 1),  # BOS bigram
    (99, 1, 2),
    (99, 2, EOS_ID),  # EOS bigram
    (BOS_ID, BOS_ID, 2),  # known context, missing token
    (99, 99, 2),  # unigram
    (99, 99, 999),  # smoothed unseen unigram
    (99, 99, EOS_ID),
    (99, 99, BOS_ID),
]


def test_writer_roundtrip_and_float32_fallback(binary):
    original, path, data = binary
    lm = BinaryBPELM.load(path)
    assert BinaryBPELM(data).score(*QUERIES[0]) == lm.score(*QUERIES[0])
    for query in QUERIES:
        assert lm.score(*query) == pytest.approx(original.score(*query), abs=0.005001)
        prev2, prev1, token = query
        quantized = original._trigram_log.get((prev2, prev1), {}).get(token)
        if quantized is None:
            quantized = original._bigram_log.get(prev1, {}).get(token)
        if quantized is not None:
            assert lm.score(*query) == round(quantized / 0.01) * lm.log_scale
        else:
            expected = np.float32(float(lm.unigram[token]) + 2 * math.log(lm.backoff))
            assert lm.score(*query) == float(expected)
    for value in vars(lm).values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value[0] = 0
    assert not hasattr(lm, "_trigram_counts")
    for token in [-1, LM_VOCAB_SIZE]:
        with pytest.raises(ValueError, match="vocabulary"):
            lm.score(BOS_ID, BOS_ID, token)


def malformed_buffers(data):
    bigram_offsets = 28 + LM_VOCAB_SIZE * 4 + 4
    bigram_entries = struct.unpack_from("<I", data, bigram_offsets + LM_VOCAB_SIZE * 4)[0]
    bigram_tokens = bigram_offsets + (LM_VOCAB_SIZE + 1) * 4
    trigram_count = bigram_tokens + bigram_entries * 4
    contexts = struct.unpack_from("<I", data, trigram_count)[0]
    trigram_keys = trigram_count + 4
    trigram_offsets = trigram_keys + contexts * 4
    trigram_tokens = trigram_offsets + (contexts + 1) * 4

    mutations = [
        ("magic", 0, "B", 0),
        ("version", 8, "I", 2),
        ("vocab", 12, "I", 1024),
        ("unigram length", 20, "I", 0xFFFFFFFF),
        ("bigram contexts", bigram_offsets - 4, "I", 0xFFFFFFFF),
        ("trigram contexts", trigram_count, "I", 0xFFFFFFFF),
        ("trigram key range", trigram_keys, "I", LM_VOCAB_SIZE**2),
        (
            "trigram duplicate key",
            trigram_keys + 4,
            "I",
            struct.unpack_from("<I", data, trigram_keys)[0],
        ),
        ("trigram unsorted keys", trigram_keys, "I", LM_VOCAB_SIZE**2 - 1),
    ]
    for field, offset in [("backoff", 16), ("scale", 24)]:
        for value in [0.0, -0.1, float("nan"), float("inf"), -float("inf")]:
            mutations.append((f"{field} {value}", offset, "f", value))
    for value in [float("nan"), float("inf"), -float("inf")]:
        mutations.append((f"unigram {value}", 28, "f", value))
    for name, offsets, tokens, rows in [
        ("bigram", bigram_offsets, bigram_tokens, LM_VOCAB_SIZE),
        ("trigram", trigram_offsets, trigram_tokens, contexts),
    ]:
        row_offsets = struct.unpack_from(f"<{rows + 1}I", data, offsets)
        row = next(i for i in range(rows) if row_offsets[i + 1] - row_offsets[i] >= 2)
        tokens += row_offsets[row] * 2
        mutations.extend(
            [
                (f"{name} offset origin", offsets, "I", 1),
                (f"{name} descending offsets", offsets + 4, "I", 0xFFFFFFFF),
                (f"{name} oversized entries", offsets + rows * 4, "I", 0xFFFFFFFF),
                (f"{name} token range", tokens, "H", LM_VOCAB_SIZE),
                (
                    f"{name} duplicate token",
                    tokens + 2,
                    "H",
                    struct.unpack_from("<H", data, tokens)[0],
                ),
                (f"{name} unsorted tokens", tokens, "H", LM_VOCAB_SIZE - 1),
            ]
        )
    result = [("trailing bytes", data + b"\x00")]
    for name, offset, fmt, value in mutations:
        changed = bytearray(data)
        struct.pack_into("<" + fmt, changed, offset, value)
        result.append((name, bytes(changed)))
    return result


def test_rejects_malformed_buffers(binary):
    for _, data in malformed_buffers(binary[2]):
        with pytest.raises(ValueError):
            BinaryBPELM(data)


def test_rejects_every_truncation(binary):
    data = binary[2]
    for length in range(len(data)):
        with pytest.raises(ValueError):
            BinaryBPELM(data[:length])


def test_empty_csr_roundtrip(tmp_path):
    path = tmp_path / "empty.bin"
    BPETrigramLM.from_sequences([]).save_browser(path)
    lm = BinaryBPELM.load(path)
    assert len(lm.trigram_keys) == len(lm.bigram_tokens) == 0
    assert math.isfinite(lm.score(BOS_ID, BOS_ID, EOS_ID))


def test_writer_omits_empty_trigram_rows(tmp_path):
    lm = BPETrigramLM.from_sequences([[1, 2], [1, 2], [3, 4]])
    assert any(not row for row in lm._trigram_counts.values())
    path = tmp_path / "pruned.bin"
    lm.save_browser(path)
    expected = path.read_bytes()
    # Enforce the writer contract even if an empty log row reaches serialization.
    lm._trigram_log[(3, 4)] = {}
    lm.save_browser(path)
    assert path.read_bytes() == expected
    binary = BinaryBPELM(expected)
    assert np.all(binary.trigram_offsets[1:] > binary.trigram_offsets[:-1])
    assert len(binary.trigram_keys) == 3


def test_browser_validation_and_score_parity(binary):
    if not shutil.which("node"):
        pytest.skip("Node.js is required for browser LM parity")
    data = binary[2]
    module = (Path(__file__).resolve().parents[1] / "web" / "lm.js").as_uri()
    script = """
import { readFileSync } from 'node:fs';
import { BPELM } from MODULE;
const input = JSON.parse(readFileSync(0, 'utf8'));
const buffer = hex => Uint8Array.from(Buffer.from(hex, 'hex')).buffer;
const lm = new BPELM(buffer(input.valid));
for (const [name, hex] of input.invalid) {
  let rejected = false;
  try { new BPELM(buffer(hex)); } catch (error) { rejected = true; }
  if (!rejected) throw new Error(`accepted malformed LM: ${name}`);
}
const valid = buffer(input.valid);
for (let length = 0; length < valid.byteLength; length += 1) {
  let rejected = false;
  try { new BPELM(valid.slice(0, length)); } catch (error) { rejected = true; }
  if (!rejected) throw new Error(`accepted truncated LM: ${length}`);
}
console.log(JSON.stringify(input.queries.map(query => lm.score(...query))));
""".replace("MODULE", json.dumps(module))
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(
            {
                "valid": data.hex(),
                "invalid": [(name, bad.hex()) for name, bad in malformed_buffers(data)],
                "queries": QUERIES,
            }
        ),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lm = BinaryBPELM(data)
    assert json.loads(result.stdout) == [lm.score(*query) for query in QUERIES]
