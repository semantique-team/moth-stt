"""E-022 — browser/Python transcript parity at the new decode parameters.

The decoder is now two implementations (`stt/decode.py`, `web/decode.js`). The unit test in
`web/tests/decoder.spec.ts` proves they agree on a synthetic tie; this proves it on **real audio**
through the **shipped int8 graph**, which is what the browser actually runs.

Method: take 25 real T clips, decode them in Python from the shipped `model.int8.onnx`, and write
the hypotheses to JSON. The Playwright test `tests/parity.spec.ts` re-decodes the same clips in the
browser from the same graph and the same config, and this script compares the two.

    uv run python runs/E-022/parity.py          # writes parity-python.jsonl + the expected table
    cd web && npx playwright test tests/parity.spec.ts
    uv run python runs/E-022/parity.py --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stt.decode import beam_search, load_pieces
from stt.lm import BPETrigramLM

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
SPEECH = REPO / "web/tests/fixtures/speech.wav"


def main() -> int:
    parser = argparse.ArgumentParser(prog="E-022/parity.py")
    parser.add_argument("--compare", action="store_true", help="compare against the browser output")
    parser.add_argument("--clips", type=int, default=25)
    parser.add_argument("--all", action="store_true", help="every T clip, for the WER sample")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (REPO / "recordings/manifests/T.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if not args.all:
        rows = rows[: args.clips]
    raw = json.loads((REPO / "data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    index = {uid: {**e, "shard": f"recorded_t/{e['shard']}"} for uid, e in raw.items()}
    shard = np.load(REPO / "data/shards/recorded_t/shard-000.npy", mmap_mode="r")

    if args.compare:
        expected = {
            json.loads(line)["id"]: json.loads(line)["hyp"]
            for line in (OUT / "parity-python.jsonl").read_text(encoding="utf-8").splitlines()
        }
        actual_path = REPO / "web/test-results/parity-browser.json"
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
        same = 0
        for uid, text in actual.items():
            if expected.get(uid) == text:
                same += 1
            else:
                print(f"MISMATCH {uid}\n  python: {expected.get(uid)!r}\n  browser: {text!r}")
        total = len(actual)
        print(f"identical {same}/{total} = {same / total:.1%}")
        (OUT / "parity.json").write_text(
            json.dumps({"clips": total, "identical": same, "ratio": same / total}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0 if same == total else 1

    from stt.onnx_runner import OnnxMoth

    moth = OnnxMoth(REPO / "release/model.int8.onnx", REPO / "release/vocab.json")
    pieces = load_pieces(REPO / "release/vocab.json")
    lm = BPETrigramLM.load(REPO / "runs/E-008/lm_bpe3_aug.pkl.gz")
    config = json.loads((REPO / "release/config.json").read_text(encoding="utf-8"))["decode"]

    entries = []
    with (OUT / "parity-python.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            entry = index[row["id"]]
            start = entry["offset"] * 80
            mel = np.asarray(
                shard[start : start + entry["frames"] * 80], dtype=np.float32
            ).reshape(entry["frames"], 80).T
            log_probs = moth.log_probs(mel)
            text = beam_search(
                log_probs,
                log_probs.shape[0],
                pieces,
                beam_width=config["beam"],
                top_k=config["top_k"],
                lm=lm,
                lm_weight=config["lm_weight"],
                token_bonus=config["token_bonus"],
            )
            handle.write(json.dumps({"id": row["id"], "hyp": text}, ensure_ascii=False) + "\n")
            entries.append({"id": row["id"], "entry": entry})
    (OUT / "parity-clips.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} python hypotheses; now run web/tests/parity.spec.ts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())