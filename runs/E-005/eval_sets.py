"""E-005 post-run evaluation: checkpoint averaging + hypotheses/metrics for each held-out set.

Inference only (no training). Run from the repository root:

    uv run python runs/E-005/eval_sets.py --checkpoint runs/E-005/checkpoints/ckpt_best.pt \
        --sets mls_it_dev,ls_en_dev --out runs/E-005/eval/best
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from stt.data.build_mel import load_audio_16k
from stt.data.dataset import MelDataset, plan_batches
from stt.decode import beam_search, greedy
from stt.eval import score_rows
from stt.hotwords import HotwordIndex, load_hotwords
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.no_speech import is_no_speech
from stt.tokenizer import encode, load_tokenizer
from stt.train import _prepare_batch, load_config, load_sources
from stt.vad import GATE_THRESHOLD
from stt.vad import has_speech as vad_has_speech

#: ``name -> (manifest, shards, audio root)``; ``None`` where the raw audio was deleted after the
#: mel cache. The VAD guard needs the audio root, the mel-based energy guard does not.
SETS = {
    "mls_it_dev": ("data/manifests/mls_it_dev.jsonl", "data/shards/mls_it_dev", None),
    "ls_en_dev": ("data/manifests/ls_en_dev.jsonl", "data/shards/ls_en_dev", None),
    "fleurs_it_test": (
        "data/manifests/fleurs_it_test.jsonl",
        "data/shards/fleurs_it_test",
        "data/raw",
    ),
    "fleurs_en_test": (
        "data/manifests/fleurs_en_test.jsonl",
        "data/shards/fleurs_en_test",
        "data/raw",
    ),
    "cv_it_dev": ("data/manifests/cv_it_dev.jsonl", "data/shards/cv_it_dev", "data/raw"),
    "cv_it_test": ("data/manifests/cv_it_test.jsonl", "data/shards/cv_it_test", "data/raw"),
    "recorded_t": (
        "recordings/manifests/T.jsonl",
        "data/shards/recorded_t",
        "recordings/audio",
    ),
    "noise": ("data/manifests/noise.jsonl", "data/shards/noise", "data/raw/noise"),
}


def _drop_silent(items: list[dict], hyps: dict[str, str]) -> list[dict]:
    """Remove clips with no speech-like energy, recording an empty hypothesis for them."""
    keep = []
    for item in items:
        if is_no_speech(item["mel"]):
            hyps[item["id"]] = ""
        else:
            keep.append(item)
    return keep


def decode(
    ckpt: str,
    names: list[str],
    out_dir: Path,
    config_path: str,
    *,
    decoder: str = "greedy",
    beam_width: int = 5,
    top_k: int = 8,
    hotwords_path: str | None = None,
    hotword_bonus: float = 1.0,
    hotword_prefix_bonus: float = 0.0,
    lm_path: str | None = None,
    lm_weight: float = 0.0,
    token_bonus: float = 0.0,
    limit: int = 0,
    no_speech_guard: bool = False,
    vad_guard: bool = False,
    vad_threshold: float = GATE_THRESHOLD,
) -> dict:
    config = load_config(config_path)
    batch = config["batch"]
    input_mode = config["train"]["input"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    processor = load_tokenizer(config["tokenizer"])
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    hotword_index = None
    hotword_terms = 0
    if hotwords_path:
        terms = load_hotwords(hotwords_path)
        hotword_index = HotwordIndex([encode(processor, term) for term in terms])
        hotword_terms = len(terms)
        print(
            f"hotwords: {hotword_terms} terms, max {hotword_index.max_len} pieces",
            flush=True,
        )
    lm = BPETrigramLM.load(lm_path) if lm_path else None
    if lm is not None:
        print(f"lm: {lm_path} (weight {lm_weight})", flush=True)
    print(
        f"decoder: {decoder}"
        + (f" (beam {beam_width}, top-k {top_k})" if decoder == "beam" else ""),
        flush=True,
    )

    model = MothSTT(
        vocab_size=int(config["model"]["vocab_size"]),
        head_blocks=int(config["model"].get("head_blocks", 0)),
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    print(f"checkpoint {ckpt} (step {payload.get('step')}) on {device}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        manifest, shards, audio_root = SETS[name]
        index, rows, root = load_sources([{"manifest": manifest, "shards": shards}])
        if limit and limit < len(rows):
            rows = random.Random(0).sample(rows, limit)
        all_rows = rows
        hyps: dict[str, str] = {}
        guarded = 0
        if vad_guard:
            # D25: Silero VAD on the 16 kHz PCM, before the mel cache. Sets whose raw audio was
            # deleted after the cache fall back to no gate, with a warning (never silently).
            if audio_root is None or not Path(audio_root).is_dir():
                print(
                    f"warning: {name}: no audio under {audio_root}; VAD guard skipped",
                    flush=True,
                )
            else:
                kept = []
                for row in rows:
                    if vad_has_speech(
                        load_audio_16k(Path(audio_root) / row["audio"]), vad_threshold
                    ):
                        kept.append(row)
                    else:
                        hyps[row["id"]] = ""
                guarded = len(rows) - len(kept)
                rows = kept
                print(
                    f"{name}: VAD gated {guarded}/{len(all_rows)} clips "
                    f"(threshold {vad_threshold})",
                    flush=True,
                )
        dataset = MelDataset(index, root, rows)
        batches = plan_batches(
            [float(row["duration"]) for row in rows],
            max_seconds=float(batch["max_seconds"]),
            max_clips=int(batch["max_clips"]),
            seed=0,
            epoch=0,
        )
        with torch.inference_mode():
            for positions in batches:
                items = [dataset[position] for position in positions]
                if no_speech_guard:
                    items = _drop_silent(items, hyps)
                    guarded += len(positions) - len(items)
                    if not items:
                        continue
                tensor, lengths, brows = _prepare_batch(items, input_mode)
                log_probs, out_lengths = model(tensor.to(device), lengths.to(device))
                for offset, row in enumerate(brows):
                    if decoder == "beam":
                        hyps[row["id"]] = beam_search(
                            log_probs[offset],
                            int(out_lengths[offset]),
                            pieces,
                            beam_width=beam_width,
                            top_k=top_k,
                            hotwords=hotword_index,
                            hotword_bonus=hotword_bonus,
                            hotword_prefix_bonus=hotword_prefix_bonus,
                            lm=lm,
                            lm_weight=lm_weight,
                            token_bonus=token_bonus,
                        )
                    else:
                        hyps[row["id"]] = greedy(
                            log_probs[offset], int(out_lengths[offset]), pieces
                        )
        (out_dir / f"{name}.hyp.jsonl").write_text(
            "".join(json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
                    for key, value in hyps.items()),
            encoding="utf-8",
        )
        scored = score_rows(all_rows, hyps, bootstrap=1000, seed=0)
        metrics = {
            "checkpoint": ckpt,
            "set": name,
            "n_utterances": scored["n_utterances"],
            "decode": {
                "method": decoder,
                "beam_width": beam_width if decoder == "beam" else None,
                "top_k": top_k if decoder == "beam" else None,
                "hotwords": hotwords_path,
                "hotword_terms": hotword_terms,
                "hotword_bonus": hotword_bonus if decoder == "beam" else None,
                "hotword_prefix_bonus": hotword_prefix_bonus if decoder == "beam" else None,
                "lm": lm_path,
                "lm_weight": lm_weight if decoder == "beam" else None,
                "token_bonus": token_bonus if decoder == "beam" else None,
                "limit": limit,
                "no_speech_guard": no_speech_guard,
                "vad_guard": vad_guard,
                "vad_threshold": vad_threshold if vad_guard else None,
                "guarded": guarded,
            },
            "overall": scored["overall"],
            "breakdown": scored["breakdown"],
            "noise": scored["noise"],
            "per_utterance": scored["per_utterance"],
        }
        (out_dir / f"{name}.metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        results[name] = scored["overall"]
        print(f"{name}: WER {scored['overall']['wer']:.4f} "
              f"CI {scored['overall'].get('wer_ci')}"
              + (f" (guarded {guarded})" if no_speech_guard or vad_guard else ""), flush=True)
        if device == "mps":
            torch.mps.empty_cache()
    return results


def average_checkpoints(paths: list[str], out: str) -> None:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    averaged = {}
    for key, first in payloads[0]["model"].items():
        tensors = [payload["model"][key] for payload in payloads]
        averaged[key] = (
            torch.stack([tensor.float() for tensor in tensors]).mean(0).to(first.dtype)
            if first.is_floating_point()
            else first
        )
    base = dict(payloads[-1])
    base["model"] = averaged
    base["averaged_from"] = paths
    torch.save(base, out)
    print(f"averaged {len(paths)} checkpoints -> {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="runs/E-005/config.yaml")
    parser.add_argument("--average", default=None, help="comma list of checkpoints to average first")
    parser.add_argument("--decoder", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hotwords", default=None, help="one normalized term per line")
    parser.add_argument("--hotword-bonus", type=float, default=1.0)
    parser.add_argument("--hotword-prefix-bonus", type=float, default=0.0)
    parser.add_argument("--lm", default=None, help="BPE n-gram LM pickle (.pkl or .pkl.gz)")
    parser.add_argument("--lm-weight", type=float, default=0.0)
    #: Per-token bonus (E-022). Must be set together with --lm-weight; the pair is what the shipped
    #: decode uses (0.5 / 2.4). Default 0 reproduces every pre-E-022 number in the repository.
    parser.add_argument("--token-bonus", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="random subset per set (tuning only)")
    parser.add_argument(
        "--no-speech-guard",
        action="store_true",
        help="legacy E-011 energy guard: skip silent clips on the mel (stt.no_speech)",
    )
    parser.add_argument(
        "--vad-guard",
        action="store_true",
        help="skip clips with no speech (Silero VAD, stt.vad, D25); needs the raw audio",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=GATE_THRESHOLD,
        help=f"speech-window fraction above which a clip is decoded (default {GATE_THRESHOLD})",
    )
    args = parser.parse_args()

    if args.vad_guard and args.no_speech_guard:
        parser.error("--vad-guard and --no-speech-guard are mutually exclusive")

    if args.average:
        average_checkpoints([path.strip() for path in args.average.split(",")], args.checkpoint)
    decode(
        args.checkpoint,
        [name.strip() for name in args.sets.split(",")],
        Path(args.out),
        args.config,
        decoder=args.decoder,
        beam_width=args.beam_width,
        top_k=args.top_k,
        hotwords_path=args.hotwords,
        hotword_bonus=args.hotword_bonus,
        hotword_prefix_bonus=args.hotword_prefix_bonus,
        lm_path=args.lm,
        lm_weight=args.lm_weight,
        token_bonus=args.token_bonus,
        limit=args.limit,
        no_speech_guard=args.no_speech_guard,
        vad_guard=args.vad_guard,
        vad_threshold=args.vad_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
