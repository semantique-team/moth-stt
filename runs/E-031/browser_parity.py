"""E-031 — browser↔Python parity for the 0.8.1 release candidate.

The acceptance run in `browser-acceptance/` decodes the versioned fixture in real Chromium/WASM
with the E-031 LM and writes the transcript per project. This script decodes the **same fixture**
with the Python ONNX path and the same binary LM, and compares the normalized texts. It also
checks that all browser projects agree. It is a parity check on one public 3 s fixture, not a WER
benchmark.

    uv run python runs/E-031/browser_parity.py
    uv run python runs/E-031/browser_parity.py --release release \
        --reports runs/E-031/browser-acceptance-0.8.0-control \
        --out runs/E-031/browser-parity-0.8.0.json --tag control-0.8.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit import digest, require_new

from stt.data.build_mel import load_audio_16k
from stt.decode import beam_search, load_pieces
from stt.eval import score_rows
from stt.features import load_mel_filters, log_mel
from stt.lm import BinaryBPELM
from stt.onnx_runner import OnnxMoth
from stt.text import normalize

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AUDIO = REPO / "web/tests/fixtures/speech.wav"
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.55, "token_bonus": 2.7}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=OUT / "release")
    parser.add_argument("--reports", type=Path, default=OUT / "browser-acceptance")
    parser.add_argument("--out", type=Path, default=OUT / "browser-parity.json")
    parser.add_argument("--tag", default="e031-0.8.1")
    args = parser.parse_args()
    release = args.release.resolve()
    require_new(args.out)

    reports = {}
    for path in sorted(args.reports.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        reports[path.stem] = {
            "passed": data.get("passed"),
            "release_version": data.get("release_version"),
            "threads": data["results"]["threads"],
            "text": data["results"]["single"]["text"],
            "model_sha256": next(
                asset["sha256"] for asset in data["assets"] if asset["name"] == "model.int8.onnx"
            ),
            "lm_sha256": next(
                asset["sha256"] for asset in data["assets"] if asset["name"] == "lm_bpe3.bin"
            ),
        }
    if not reports:
        raise SystemExit(f"no reports in {args.reports}")
    config = json.loads((release / "config.json").read_text(encoding="utf-8"))
    for report in reports.values():
        if (report["passed"] is not True or report["release_version"] != config["version"]
                or report["model_sha256"] != digest((release / "model.int8.onnx").read_bytes())
                or report["lm_sha256"] != digest((release / "lm_bpe3.bin").read_bytes())):
            raise SystemExit("failed or mismatched browser report; rerun acceptance for this release")
    if any(config["decode"].get(key) != value for key, value in PARAMS.items()):
        raise SystemExit("release decoder differs from parity parameters")
    texts = {normalize(row["text"]) for row in reports.values()}
    if len(texts) != 1:
        raise SystemExit(f"browser projects disagree: {texts}")

    moth = OnnxMoth(release / "model.int8.onnx", release / "vocab.json")
    pieces = load_pieces(release / "vocab.json")
    lm = BinaryBPELM.load(release / "lm_bpe3.bin")
    wav = load_audio_16k(AUDIO)
    mel = log_mel(wav, filters=load_mel_filters(release / "mel_filters.bin"))
    log_probs = moth.log_probs(mel)
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
    browser_text = next(iter(texts))
    scored = score_rows(
        [
            {
                "id": "fixture",
                "text": browser_text,
                "norm": browser_text,
                "duration": round(len(wav) / 16_000, 3),
                "source": "parity_fixture",
                "split": "dev",
                "lang": "it",
            }
        ],
        {"fixture": python_text},
        bootstrap=0,
        seed=0,
    )
    result = {
        "tag": args.tag,
        "release": str(release.relative_to(REPO)) if release.is_relative_to(REPO) else str(release),
        "audio": {
            "path": str(AUDIO.relative_to(REPO)),
            "seconds": round(len(wav) / 16_000, 3),
            "sha256": hashlib.sha256(AUDIO.read_bytes()).hexdigest(),
        },
        "params": PARAMS,
        "browser_text": browser_text,
        "python_text": python_text,
        "exact_normalized_match": normalize(python_text) == browser_text,
        "wer_browser_as_reference": scored["overall"]["wer"],
        "projects": reports,
        "note": "One public 3 s fixture; runtime parity, not a WER or natural-speech benchmark.",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("browser_text", "python_text",
                                            "exact_normalized_match", "wer_browser_as_reference")},
                     indent=2))
    return 0 if result["exact_normalized_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
