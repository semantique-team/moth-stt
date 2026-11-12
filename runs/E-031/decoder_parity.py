"""E-031 — decoder-level parity: the same acoustic logits through JS and Python.

The browser fixture differs from Python by two words with **both** the 0.8.0 and the 0.8.1 LM,
so the difference is in the browser acoustic/ORT/VAD path, not the LM. This check isolates the
other half: it takes the Python ONNX logits for the same fixture and runs the **JS beam search +
LM reader** (`web/decode.js`, `web/lm.js`) and the Python `beam_search` + `BinaryBPELM` on exactly
the same matrix. Exact text equality means the LM binary and decoder carry over.

    uv run python runs/E-031/decoder_parity.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from audit import digest, require_new

from stt.data.build_mel import load_audio_16k
from stt.decode import beam_search, load_pieces
from stt.features import load_mel_filters, log_mel
from stt.lm import BinaryBPELM
from stt.onnx_runner import OnnxMoth

REPO = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).resolve().parent / "release"
AUDIO = REPO / "web/tests/fixtures/speech.wav"
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="new parity output directory")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be one directory name")
    out = Path(__file__).resolve().parent / "decoder-parity" / args.name
    require_new(out)
    out.mkdir(parents=True)
    moth = OnnxMoth(RELEASE / "model.int8.onnx", RELEASE / "vocab.json")
    pieces = load_pieces(RELEASE / "vocab.json")
    lm = BinaryBPELM.load(RELEASE / "lm_bpe3.bin")
    wav = load_audio_16k(AUDIO)
    mel = log_mel(wav, filters=load_mel_filters(RELEASE / "mel_filters.bin"))
    log_probs = np.asarray(moth.log_probs(mel), dtype=np.float32)
    log_probs.tofile(out / "logits.bin")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "frames": int(log_probs.shape[0]),
                "vocab": int(log_probs.shape[1]),
                "params": PARAMS,
                "audio_sha256": hashlib.sha256(AUDIO.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    python_text = beam_search(
        log_probs,
        log_probs.shape[0],
        pieces,
        beam_width=PARAMS["beam"],
        top_k=PARAMS["top_k"],
        lm=lm,
        lm_weight=PARAMS["lm_weight"],
        token_bonus=PARAMS["token_bonus"],
    )
    completed = subprocess.run(
        [
            "node",
            str(Path(__file__).resolve().parent / "js_beam.mjs"),
            str(out / "logits.bin"),
            str(RELEASE / "lm_bpe3.bin"),
            str(RELEASE / "vocab.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    js_text = completed.stdout.strip()
    result = {
        "python_text": python_text,
        "js_text": js_text,
        "exact_match": python_text == js_text,
        "frames": int(log_probs.shape[0]),
        "params": PARAMS,
        "scope": "same-logits decoder only",
        "assets": {"model": digest((RELEASE / "model.int8.onnx").read_bytes()),
                   "lm": digest((RELEASE / "lm_bpe3.bin").read_bytes()),
                   "vocab": digest((RELEASE / "vocab.json").read_bytes())},
        "code": {name: digest((REPO / name).read_bytes()) for name in (
            "stt/decode.py", "stt/lm.py", "web/decode.js", "web/lm.js",
            "runs/E-031/decoder_parity.py", "runs/E-031/js_beam.mjs")},
        "audio_sha256": digest(AUDIO.read_bytes()),
        "logits_sha256": digest((out / "logits.bin").read_bytes()),
    }
    (out / "parity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["exact_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
