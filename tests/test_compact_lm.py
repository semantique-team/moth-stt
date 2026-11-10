import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stt.compact_lm import compact_lm
from stt.lm import LM_VOCAB_SIZE, BinaryBPELM


def tiny_binary(empty=True, all_empty=False):
    """Hand-pack unusual exact scores, not counts or a freshly quantized LM."""
    vocab = LM_VOCAB_SIZE
    data = b"MOTHLMB1" + struct.pack("<IIfIf", 1, vocab, 0.37, vocab, 0.0037)
    data += np.linspace(-19.125, -0.0625, vocab, dtype="<f4").tobytes()
    offsets = np.full(vocab + 1, 2, dtype="<u4")
    offsets[:3] = 0
    data += struct.pack("<I", vocab) + offsets.tobytes()
    data += struct.pack("<2H2h", 3, 5, -32768, 32767)
    keys, offsets = ([1, 2, 3, 4, 5], [0, 0, 2, 2, 3, 3]) if empty else ([2, 4], [0, 2, 3])
    if all_empty:
        offsets = [0] * (len(keys) + 1)
    data += struct.pack("<I", len(keys)) + struct.pack(f"<{len(keys)}I", *keys)
    data += struct.pack(f"<{len(offsets)}I", *offsets)
    if not all_empty:
        data += struct.pack("<3H3h", 1, 7, 9, -32768, -12345, 32767)
    return data


@pytest.mark.parametrize(
    "empty,all_empty,removed", [(True, False, 3), (False, False, 0), (True, True, 5)]
)
def test_lossless_rows_and_all_token_scores(tmp_path, empty, all_empty, removed):
    source, output, report = (tmp_path / name for name in ("old.bin", "new.bin", "report.json"))
    data = tiny_binary(empty, all_empty)
    source.write_bytes(data)
    result = compact_lm(source, output, report)
    compacted = output.read_bytes()
    before, after = BinaryBPELM(data), BinaryBPELM(compacted)
    assert source.read_bytes() == data
    assert result == json.loads(report.read_text())
    assert result["source_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["output_sha256"] == hashlib.sha256(compacted).hexdigest()
    assert result["empty_trigram_rows_removed"] == removed
    assert result["saved_bytes"] == len(data) - len(compacted) == 8 * removed
    assert result["verification"]["scores_exact"]
    assert result["quality_claim"].startswith("None.")
    if not removed:
        assert compacted == data
    for name in (
        "unigram",
        "unigram_floor",
        "bigram_offsets",
        "bigram_tokens",
        "bigram_logq",
        "trigram_tokens",
        "trigram_logq",
    ):
        assert getattr(before, name).tobytes() == getattr(after, name).tobytes()
    assert before.backoff == after.backoff
    assert before.log_scale == after.log_scale
    assert np.all(after.trigram_offsets[1:] > after.trigram_offsets[:-1])
    for row, key in enumerate(before.trigram_keys):
        start, end = before.trigram_offsets[row : row + 2]
        if start == end:
            assert key not in after.trigram_keys
        else:
            index = int(np.searchsorted(after.trigram_keys, key))
            assert after.trigram_keys[index] == key
            assert tuple(after.trigram_offsets[index : index + 2]) == (start, end)
    # Every token for retained, removed and absent contexts, including both fallback levels.
    for prev2, prev1 in [(0, key) for key in range(7)] + [(99, 99)]:
        for token in range(LM_VOCAB_SIZE):
            assert before.score(prev2, prev1, token) == after.score(prev2, prev1, token)


def test_malformed_inputs_leave_no_outputs(tmp_path):
    source, output, report = (tmp_path / name for name in ("old.bin", "new.bin", "report.json"))
    data = tiny_binary()
    lm = BinaryBPELM(data)
    trigram_start = len(data) - 4 - lm.trigram_keys.nbytes - lm.trigram_offsets.nbytes - 12
    invalid = [data[:n] for n in (0, 8, 27, 28, len(data) - 1)] + [data + b"\x00"]
    for offset, fmt, value in [
        (0, "B", 0),
        (8, "I", 2),
        (16, "f", float("nan")),
        (trigram_start, "I", 0xFFFFFFFF),
        (trigram_start + 8, "I", 1),  # duplicate key
        (trigram_start + 4 + lm.trigram_keys.nbytes, "I", 1),
        (len(data) - 12, "H", LM_VOCAB_SIZE),
    ]:
        changed = bytearray(data)
        struct.pack_into("<" + fmt, changed, offset, value)
        invalid.append(bytes(changed))
    for bad in invalid:
        source.write_bytes(bad)
        with pytest.raises(ValueError):
            compact_lm(source, output, report)
        assert source.read_bytes() == bad
        assert not output.exists()
        assert not report.exists()


@pytest.mark.parametrize("existing", ["output", "report", "directory"])
def test_refuses_existing_paths(tmp_path, existing):
    source, output, report = (tmp_path / name for name in ("old.bin", "new.bin", "report.json"))
    source.write_bytes(tiny_binary())
    path = report if existing == "report" else output
    if existing == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        compact_lm(source, output, report)
    if existing != "directory":
        assert path.read_bytes() == b"do not overwrite"
    assert not (output if path == report else report).exists()


def test_refuses_aliases_and_missing_parent(tmp_path):
    source = tmp_path / "old.bin"
    source.write_bytes(tiny_binary())
    for output, report in [
        (source, tmp_path / "report.json"),
        (tmp_path / "new.bin", source),
        (tmp_path / "new.bin", tmp_path / "new.bin"),
        (tmp_path / "missing" / "new.bin", tmp_path / "report.json"),
    ]:
        with pytest.raises(ValueError):
            compact_lm(source, output, report)
    assert source.read_bytes() == tiny_binary()


def test_cli(tmp_path):
    source, output, report = (tmp_path / name for name in ("old.bin", "new.bin", "report.json"))
    source.write_bytes(tiny_binary())
    command = [
        sys.executable,
        "-m",
        "stt.compact_lm",
        "--input",
        str(source),
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(report.read_text())
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert repeated.returncode == 2
    assert "refusing existing path" in repeated.stderr


def test_exact_python_js_parity_before_and_after(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node.js is required for browser LM parity")
    source, output, report = (tmp_path / name for name in ("old.bin", "new.bin", "report.json"))
    source.write_bytes(tiny_binary())
    compact_lm(source, output, report)
    queries = [(0, key, token) for key in range(7) for token in range(LM_VOCAB_SIZE)]
    queries += [(99, 99, token) for token in range(LM_VOCAB_SIZE)]
    module = (Path(__file__).resolve().parents[1] / "web" / "lm.js").as_uri()
    script = """
import { readFileSync } from 'node:fs';
import { BPELM } from MODULE;
const input = JSON.parse(readFileSync(0, 'utf8'));
const load = hex => new BPELM(Uint8Array.from(Buffer.from(hex, 'hex')).buffer);
const before = load(input.before), after = load(input.after);
const scores = input.queries.map(query => {
  const score = before.score(...query);
  if (!Object.is(score, after.score(...query))) throw new Error(`score changed: ${query}`);
  return score;
});
console.log(JSON.stringify(scores));
""".replace("MODULE", json.dumps(module))
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(
            {
                "before": source.read_bytes().hex(),
                "after": output.read_bytes().hex(),
                "queries": queries,
            }
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    before, after = BinaryBPELM.load(source), BinaryBPELM.load(output)
    scores = json.loads(result.stdout)
    assert scores == [before.score(*query) for query in queries]
    assert scores == [after.score(*query) for query in queries]
