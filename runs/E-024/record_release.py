"""Record the 0.7.0 release build: hashes, sizes and browser RTFx, in the E-021 record format.

The publisher (`runs/E-021/upload_hf.py`) verifies `release/` against a record before uploading;
this writes the one for 0.7.0 so the guard keeps working after the decode change.

    uv run python runs/E-024/record_release.py
"""

from __future__ import annotations

import json
from pathlib import Path

from stt.data.sources import sha256_file

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RELEASE = REPO / "release"
PAYLOAD = ("model.int8.onnx", "vocab.json", "mel_filters.bin", "silero_vad.onnx", "lm_bpe3.bin")
ALSO = ("config.json", "NOTICE")


def main() -> int:
    config = json.loads((RELEASE / "config.json").read_text(encoding="utf-8"))
    version = config["version"]
    files = {}
    for name in (*PAYLOAD, *ALSO):
        path = RELEASE / name
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    total = sum(entry["bytes"] for entry in files.values())
    # ``package_bytes`` must mean what the publisher checks: the five runtime files, not the
    # config/NOTICE pair (that is the convention in runs/E-021/release-0.5.1.json).
    payload = sum(files[name]["bytes"] for name in PAYLOAD)

    browser = json.loads((OUT / "browser.json").read_text(encoding="utf-8"))
    record = {
        "release": version,
        "date": "2026-11-09",
        "checkpoint": "runs/E-024/checkpoints/ckpt_best.pt (step 26442, noise-augmented fine-tune from E-014)",
        "lm": "runs/E-008/lm_bpe3_aug.pkl.gz (unchanged)",
        "reason": (
            "New acoustic checkpoint (E-024): the E-017 recipe re-run with half of the Common Voice "
            "IT clips heard through noise at a random 10-30 dB SNR. Architecture, tokenizer, LM, "
            "decode and package size are unchanged from 0.6.0; only the int8 weights differ. Buys "
            "6.3 WER at 15 dB SNR and 11.5 at 10 dB, with clean performance a tie."
        ),
        "build": (
            "uv run python stt/export.py --out release --checkpoint "
            "runs/E-024/checkpoints/ckpt_best.pt --vocab release/vocab.json --head-blocks 2 "
            f"--version {version}"
        ),
        "package": (
            "uv run python runs/E-010/package_browser.py --lm runs/E-008/lm_bpe3_aug.pkl.gz "
            "--beam 12 --top-k 10 --lm-weight 0.5 --token-bonus 2.4"
        ),
        "decode": config["decode"],
        "package_bytes": payload,
        "package_mb": round(payload / 1e6, 2),
        "payload_mb": round(total / 1e6, 2),
        "files": files,
        "vad": config["vad"],
        "verification": {
            "fp32": {
                "gate": "seeded sweep, D29: median <= 1e-3, worst <= 1e-2",
                "median_abs": 1.888e-4,
                "worst_abs": 1.217e-3,
                "draws_over_1e-3": "1/80",
                "source": "stt/export.py stdout for this build",
            },
            "int8": {"max_abs": 3.063, "top1_agreement": 1.0},
        },
        "wer": {
            "t600": {"wer": 0.2520, "eta": 0.5417},
            "t-s3": {"wer": 0.2692, "eta": 0.4845},
            "fleurs_it_test": {"wer": 0.2014},
            "fleurs_en_test": {"wer": 0.4914},
            "cv_it_test": {"wer": 0.2437},
            "source": "runs/E-024/eval/gate/ (PyTorch) and eval/gate/int8/ (this artifact)",
        },
        "browser_rtfx": browser,
        "browser_parity": json.loads((OUT / "parity-wer.json").read_text(encoding="utf-8")),
    }
    target = REPO / "runs" / "E-024" / f"release-{version}.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(REPO)}: {record['package_mb']} MB, {len(files)} files")
    for name, entry in files.items():
        print(f"  {name:<20} {entry['bytes']:>10}  sha256 {entry['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())