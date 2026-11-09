"""E-022 — evaluate the frozen decode arm through the repository's own decoders.

One script, one arm, no search: the parameters come from the plan (`runs/E-022/PLAN.md`) and are
read from ``ARMS`` below, which is frozen. Sets are the project's ship-gate sets plus the held-out
speaker slice, all on the E-017 checkpoint.

    uv run python runs/E-022/eval.py --arm frozen
    uv run python runs/E-022/eval.py --arm shipped      # the control: reproduces runs/E-017/
    uv run python runs/E-022/eval.py --arm cheap        # the pre-registered fallback
    uv run python runs/E-022/eval.py --arm frozen --int8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from stt.data.build_mel import load_audio_16k
from stt.data.dataset import MelDataset, plan_batches
from stt.decode import beam_search, greedy, load_pieces
from stt.eval import score_rows
from stt.features import log_mel
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import _prepare_batch, load_config, load_sources
from stt.vad import has_speech

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "E-022"
LM_PATH = REPO / "runs/E-008/lm_bpe3_aug.pkl.gz"
CHECKPOINT = REPO / "runs/E-017/checkpoints/ckpt_best.pt"

#: Frozen decode arms. ``shipped`` is what runs/E-017/ was produced with; ``frozen`` is the E-022
#: candidate chosen on dev only; ``cheap`` is the pre-registered fallback if the browser bar fails.
ARMS = {
    "shipped": {"beam": 5, "top_k": 8, "lm_weight": 0.2, "token_bonus": 0.0},
    "cheap": {"beam": 8, "top_k": 8, "lm_weight": 0.5, "token_bonus": 2.4},
    "frozen": {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4},
}


def sets() -> dict[str, tuple[dict, list[dict], Path]]:
    """Every evaluated set, as the ``(index, rows, shards_root)`` triples the loader wants."""
    t_rows = [
        json.loads(line)
        for line in (REPO / "recordings/manifests/T.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    raw = json.loads((REPO / "data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    t_index = {uid: {**e, "shard": f"recorded_t/{e['shard']}"} for uid, e in raw.items()}
    root = REPO / "data/shards"

    def load(name: str, limit: int = 0) -> tuple[dict, list[dict], Path]:
        index, rows, shards = load_sources(
            [{"manifest": f"data/manifests/{name}.jsonl", "shards": f"data/shards/{name}"}]
        )
        if limit:
            rows = rows[:limit]
            keep = {r["id"] for r in rows}
            index = {uid: e for uid, e in index.items() if uid in keep}
        return index, rows, shards

    result = {
        "t600": (t_index, t_rows, root),
        "t-s3": (t_index, [r for r in t_rows if r.get("speaker") == "S3"], root),
        "t-s1s2": (t_index, [r for r in t_rows if r.get("speaker") in ("S1", "S2")], root),
        "fleurs_it_test": load("fleurs_it_test"),
        "fleurs_en_test": load("fleurs_en_test"),
        "cv_it_dev": load("cv_it_dev", limit=600),
        "noise": load("noise"),
    }
    s4 = [r for r in t_rows if r.get("speaker") == "S4"]
    if s4:
        result["t-s4"] = (t_index, s4, root)
    return result


def decode_greedy(model, pieces, index, rows, root) -> dict[str, str]:
    """Greedy is VAD-free and length-neutral, so it is the control for the whole change."""
    dataset = MelDataset(index, root, rows)
    batches = plan_batches(
        [float(r["duration"]) for r in rows], max_seconds=192, max_clips=32, seed=0, epoch=0
    )
    hypotheses: dict[str, str] = {}
    with torch.inference_mode():
        for positions in batches:
            items = [dataset[p] for p in positions]
            tensor, lengths, batch_rows = _prepare_batch(items, "truncated")
            log_probs, out_lengths = model(tensor, lengths)
            for offset, row in enumerate(batch_rows):
                hypotheses[row["id"]] = greedy(
                    log_probs[offset], int(out_lengths[offset]), pieces
                )
    return hypotheses


def decode_torch(model, pieces, lm, index, rows, root, params) -> dict[str, str]:
    dataset = MelDataset(index, root, rows)
    batches = plan_batches(
        [float(r["duration"]) for r in rows], max_seconds=192, max_clips=32, seed=0, epoch=0
    )
    hypotheses: dict[str, str] = {}
    with torch.inference_mode():
        for positions in batches:
            items = [dataset[p] for p in positions]
            tensor, lengths, batch_rows = _prepare_batch(items, "truncated")
            log_probs, out_lengths = model(tensor, lengths)
            for offset, row in enumerate(batch_rows):
                hypotheses[row["id"]] = beam_search(
                    log_probs[offset],
                    int(out_lengths[offset]),
                    pieces,
                    beam_width=params["beam"],
                    top_k=params["top_k"],
                    lm=lm,
                    lm_weight=params["lm_weight"],
                    token_bonus=params["token_bonus"],
                )
    return hypotheses


def decode_int8(moth_onnx, pieces, lm, index, rows, root, params) -> dict[str, str]:
    """Run the shipped int8 graph on the cached mel (no torch, no re-encoding)."""
    shards: dict[str, np.ndarray] = {}
    hypotheses: dict[str, str] = {}
    for row in rows:
        entry = index[row["id"]]
        name = Path(entry["shard"]).name
        array = shards.get(name)
        if array is None:
            array = np.load(root / entry["shard"], mmap_mode="r")
            shards[name] = array
        start = entry["offset"] * 80
        mel = np.asarray(array[start : start + entry["frames"] * 80], dtype=np.float32)
        log_probs = moth_onnx.log_probs(mel.reshape(entry["frames"], 80).T)
        hypotheses[row["id"]] = beam_search(
            log_probs,
            log_probs.shape[0],
            pieces,
            beam_width=params["beam"],
            top_k=params["top_k"],
            lm=lm,
            lm_weight=params["lm_weight"],
            token_bonus=params["token_bonus"],
        )
    return hypotheses


def write_jsonl(path: Path, hypotheses: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
            for key, value in hypotheses.items()
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="E-022/eval.py")
    parser.add_argument("--arm", choices=sorted(ARMS), default="frozen")
    parser.add_argument("--int8", action="store_true", help="also decode the shipped int8 artifact")
    parser.add_argument("--sets", default=None, help="comma list of set names")
    parser.add_argument("--greedy", action="store_true", help="also run greedy (VAD-free control)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    params = ARMS[args.arm]
    out_dir = Path(args.out) if args.out else OUT / "eval" / args.arm
    config = load_config(REPO / "runs/E-017/config.yaml")
    processor = load_tokenizer(REPO / config["tokenizer"])
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    lm = BPETrigramLM.load(LM_PATH)

    model = MothSTT(vocab_size=1024, head_blocks=2)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"arm {args.arm}: {json.dumps(params)}", flush=True)

    available = sets()
    wanted = (
        {name.strip() for name in args.sets.split(",") if name.strip()} if args.sets else set(available)
    )
    unknown = wanted - set(available)
    if unknown:
        parser.error(f"unknown sets: {sorted(unknown)}")

    summary: dict[str, dict] = {}
    for name in sorted(wanted):
        index, rows, root = available[name]
        if name == "noise":
            continue
        started = time.time()
        hypotheses = decode_torch(model, pieces, lm, index, rows, root, params)
        elapsed = time.time() - started
        audio_seconds = sum(float(r["duration"]) for r in rows)
        write_jsonl(out_dir / f"{name}.hyp.jsonl", hypotheses)
        scored = score_rows(rows, hypotheses, bootstrap=1000, seed=0)
        (out_dir / f"{name}.metrics.json").write_text(
            json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        overall = scored["overall"]
        summary[name] = {
            "wer": overall["wer"],
            "eta": overall["eta"],
            "wer_ci": overall["wer_ci"],
            "deletions": overall["del"] / overall["ref_words"],
            "insertions": overall["ins"] / overall["ref_words"],
            "rtfx": audio_seconds / elapsed,
        }
        print(
            json.dumps(
                {
                    "set": name,
                    "wer": round(overall["wer"], 4),
                    "eta": None if overall["eta"] is None else round(overall["eta"], 4),
                    "del": round(overall["del"] / overall["ref_words"], 4),
                    "ins": round(overall["ins"] / overall["ref_words"], 4),
                    "rtfx": round(audio_seconds / elapsed, 1),
                }
            ),
            flush=True,
        )

        if args.greedy:
            started = time.time()
            greedy_hypotheses = decode_greedy(model, pieces, index, rows, root)
            write_jsonl(out_dir / f"{name}.greedy.hyp.jsonl", greedy_hypotheses)
            scored_greedy = score_rows(rows, greedy_hypotheses, bootstrap=1000, seed=0)
            (out_dir / f"{name}.greedy.metrics.json").write_text(
                json.dumps(scored_greedy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "set": name,
                        "decoder": "greedy",
                        "wer": round(scored_greedy["overall"]["wer"], 4),
                        "rtfx": round(audio_seconds / (time.time() - started), 1),
                    }
                ),
                flush=True,
            )

    # noise: the VAD decides, the decoder never sees a gated clip. The audio root is not the shards
    # root — `sets()` returns the mel cache path, and the noise clips live under data/raw/noise.
    if "noise" in wanted:
        index, rows, _shards_root = available["noise"]
        audio_root = REPO / "data/raw/noise"
        if not audio_root.is_dir():
            raise SystemExit(f"noise audio missing at {audio_root}; cannot run the VAD guard")
        hypotheses = {row["id"]: "" for row in rows}
        decoded = 0
        for row in rows:
            audio = load_audio_16k(audio_root / row["audio"])
            if not has_speech(audio):
                continue
            mel = log_mel(audio)
            with torch.inference_mode():
                log_probs, out_lengths = model(
                    torch.from_numpy(mel[None]), torch.tensor([mel.shape[1]])
                )
            hypotheses[row["id"]] = beam_search(
                log_probs[0],
                int(out_lengths[0]),
                pieces,
                beam_width=params["beam"],
                top_k=params["top_k"],
                lm=lm,
                lm_weight=params["lm_weight"],
                token_bonus=params["token_bonus"],
            )
            decoded += 1
        write_jsonl(out_dir / "noise.hyp.jsonl", hypotheses)
        scored = score_rows(rows, hypotheses, bootstrap=0, seed=0)
        (out_dir / "noise.metrics.json").write_text(
            json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summary["noise"] = {"decoded_after_vad": decoded, **scored["noise"]}
        print(json.dumps({"set": "noise", **scored["noise"]}), flush=True)

    if args.int8:
        from stt.onnx_runner import OnnxMoth

        moth = OnnxMoth(REPO / "release/model.int8.onnx", REPO / "release/vocab.json")
        int8_pieces = load_pieces(REPO / "release/vocab.json")
        for name in ("t600", "t-s3"):
            if name not in available:
                continue
            index, rows, root = available[name]
            hypotheses = decode_int8(moth, int8_pieces, lm, index, rows, root, params)
            target = out_dir / "int8" / name
            write_jsonl(target.with_name(f"{name}.hyp.jsonl"), hypotheses)
            scored = score_rows(rows, hypotheses, bootstrap=1000, seed=0)
            (target.with_name(f"{name}.metrics.json")).write_text(
                json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            summary[f"int8/{name}"] = {
                "wer": scored["overall"]["wer"],
                "eta": scored["overall"]["eta"],
                "wer_ci": scored["overall"]["wer_ci"],
            }
            print(
                json.dumps(
                    {
                        "set": f"int8/{name}",
                        "wer": round(scored["overall"]["wer"], 4),
                        "eta": round(scored["overall"]["eta"], 4),
                    }
                ),
                flush=True,
            )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())