"""Package the E-008 LM and decoder settings into the browser release.

Run after ``stt/export.py`` and after the E-008 LM exists. The release stays gitignored; the
binary LM and the patched config are covered by ``runs/E-010/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stt.data.sources import sha256_file
from stt.lm import BPETrigramLM
from stt.release import validate_release

RELEASE = Path("release")
#: The shipped LM is the *augmented* one (runs/E-008/build_lm_augmented.py, adopted in 0.3.1):
#: the base LM has no prior for the recorded gold vocabulary. The default used to point at
#: ``lm_bpe3.pkl.gz``, which every release since 0.3.1 overrode by hand — a footgun found while
#: recording 0.6.0 (E-022).
LM_PICKLE = Path("runs/E-008/lm_bpe3_aug.pkl.gz")
LM_BINARY = Path("release/lm_bpe3.bin")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the LM and decode settings into release/")
    parser.add_argument("--lm", default=str(LM_PICKLE), help="BPETrigramLM pickle to package")
    parser.add_argument("--beam", type=int, default=12, help="beam width in config.json")
    parser.add_argument("--top-k", type=int, default=10, dest="top_k")
    parser.add_argument("--lm-weight", type=float, default=0.5, dest="lm_weight")
    parser.add_argument("--budget-bytes", type=int, default=20_000_000)
    parser.add_argument(
        "--token-bonus",
        type=float,
        default=2.4,
        dest="token_bonus",
        help="per-token score bonus (gamma); tune it *with* --lm-weight, see runs/E-022/",
    )
    args = parser.parse_args()
    lm_pickle = Path(args.lm)
    config_path = RELEASE / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not (RELEASE / "model.int8.onnx").exists():
        raise SystemExit("release/model.int8.onnx missing: run stt/export.py first")

    lm = BPETrigramLM.load(lm_pickle)
    lm.save_browser(LM_BINARY)
    # ``hotword_bonus`` was written here for three releases but no runtime ever loaded a hotword
    # index, so it advertised a feature that did not exist (runs/E-022 audit §5). Dropped.
    config["decode"] = {
        "default": "beam",
        "beam": args.beam,
        "top_k": args.top_k,
        "lm": LM_BINARY.name,
        "lm_weight": args.lm_weight,
        "token_bonus": args.token_bonus,
    }
    config["files"][LM_BINARY.name] = f"sha256:{sha256_file(LM_BINARY)}"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _, report = validate_release(RELEASE, budget_bytes=args.budget_bytes)
    print(f"{LM_BINARY}: {LM_BINARY.stat().st_size / 1e6:.2f} MB")
    print(f"decode: {json.dumps(config['decode'], sort_keys=True)}")
    print(f"required release files: {report['total_bytes']} bytes (not browser transfer size)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
