"""Can the shipped VAD recover the multi-sentence penalty? (probe, 2026-11-09)

`long_clip_probe.py` showed that 5 sentences in one 12.5 s clip cost 9.4 WER points against
decoding them separately — with the *true* boundaries, so that is an upper bound.

This probe uses the fix path the product would actually take: raw audio, Silero VAD (already in the
19.44 MB package) to find the silences, cut there, decode each segment, join. It reports what
fraction of the 9.4 points a real segmenter recovers, and how well it handles the failure modes
(over-segmentation losing a sentence's first word, under-segmentation leaving two sentences joined).

    uv run python runs/audit-2026-11-09/vad_chunk_probe.py [--groups 40] [--per-group 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stt.data.build_mel import load_audio_16k
from stt.decode import beam_search
from stt.features import log_mel
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import load_config
from stt.vad import WINDOW, WINDOW_THRESHOLD, speech_probabilities

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4}

#: Silence needed to cut. The VAD window is 32 ms; 250 ms is a normal inter-sentence pause and well
#: under the ~1 s between sentences in read speech.
MIN_SILENCE_S = 0.25
#: Do not cut inside a segment shorter than this (it would chop a word).
MIN_SEGMENT_S = 0.40
#: A segment longer than this gets force-split at its quietest internal window.
MAX_SEGMENT_S = 12.0


def segment_bounds(audio: np.ndarray, margin_s: float = 0.0) -> list[tuple[int, int]]:
    """Sample offsets of speech segments, from the shipped Silero VAD.

    ``margin_s`` keeps a little audio on each side of the detected speech. Without it the cut lands
    on the first voiced window, which removes the ~0.3 s of leading silence the model is trained
    on — the same effect measured on single clips, where trimming silence was neutral-to-worse.
    """
    probabilities = speech_probabilities(audio)
    if probabilities.size == 0:
        return []
    voiced = probabilities > WINDOW_THRESHOLD
    segments: list[tuple[int, int]] = []
    index = 0
    total = len(voiced)
    while index < total:
        if not voiced[index]:
            index += 1
            continue
        start = index
        while index < total and voiced[index]:
            index += 1
        segments.append((start, index))
    if not segments:
        return []

    # Merge across pauses shorter than MIN_SILENCE_S.
    merged = [segments[0]]
    for start, end in segments[1:]:
        gap = start - merged[-1][1]
        if gap * (WINDOW / 16000) < MIN_SILENCE_S:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    bounds: list[tuple[int, int]] = []
    for start, end in merged:
        if (end - start) * (WINDOW / 16000) < MIN_SEGMENT_S:
            continue
        margin = int(margin_s * 16000)
        bounds.append((max(0, start * WINDOW - margin), min(len(audio), end * WINDOW + margin)))
    return bounds


def decode_mel(model, pieces, lm, mel: np.ndarray) -> str:
    if mel.shape[1] < 30:
        return ""
    tensor = torch.from_numpy(mel[None])
    with torch.inference_mode():
        log_probs, out_lengths = model(tensor, torch.tensor([mel.shape[1]]))
    return beam_search(
        log_probs[0],
        int(out_lengths[0]),
        pieces,
        beam_width=PARAMS["beam"],
        top_k=PARAMS["top_k"],
        lm=lm,
        lm_weight=PARAMS["lm_weight"],
        token_bonus=PARAMS["token_bonus"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="vad_chunk_probe.py")
    parser.add_argument("--groups", type=int, default=40)
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument(
        "--margin", type=float, default=0.0, help="seconds of context kept on each side of a cut"
    )
    parser.add_argument(
        "--rule",
        choices=("always", "multi"),
        default="always",
        help="'multi' segments only when the VAD finds 2+ speech runs; single runs decode whole",
    )
    args = parser.parse_args()

    from stt.eval import _compare, score_rows

    cfg = load_config(REPO / "runs/E-017/config.yaml")
    processor = load_tokenizer(REPO / cfg["tokenizer"])
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    model = MothSTT(vocab_size=1024, head_blocks=2)
    payload = torch.load(
        REPO / "runs/E-017/checkpoints/ckpt_best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model"])
    model.eval()
    lm = BPETrigramLM.load(REPO / "runs/E-008/lm_bpe3_aug.pkl.gz")

    rows = [
        json.loads(line)
        for line in (REPO / "recordings/manifests/T.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    s3 = [r for r in rows if r.get("speaker") == "S3"][: args.groups * args.per_group]
    groups = [
        s3[i : i + args.per_group] for i in range(0, len(s3) - args.per_group + 1, args.per_group)
    ]
    print(f"{len(groups)} groups of {args.per_group} clips", flush=True)

    whole_hyps: dict[str, str] = {}
    vad_hyps: dict[str, str] = {}
    scored_rows: list[dict] = []
    segment_counts: list[int] = []

    for group_index, group in enumerate(groups):
        audios = [load_audio_16k(REPO / "recordings/audio" / r["audio"]) for r in group]
        silence = np.zeros(int(0.3 * 16000), dtype=np.float32)
        concatenated = np.concatenate(
            [np.concatenate([audio, silence]) for audio in audios]
        ).astype(np.float32)

        uid = f"group-{group_index:03d}"
        norm = " ".join(r["norm"] for r in group)
        en_spans: list[list[int]] = []
        cursor = 0
        for r in group:
            for start, end in r.get("en_spans") or []:
                en_spans.append([start + cursor, end + cursor])
            cursor += len(r["norm"].split())
        scored_rows.append(
            {
                "id": uid,
                "norm": norm,
                "duration": len(concatenated) / 16000,
                "en_spans": en_spans,
            }
        )

        whole_hyps[uid] = decode_mel(model, pieces, lm, log_mel(concatenated))
        bounds = segment_bounds(concatenated, margin_s=args.margin)
        segment_counts.append(len(bounds))
        if args.rule == "multi" and len(bounds) < 2:
            # One speech run: segmenting it can only lose the leading silence the model expects,
            # which measured +1.6 to +2.9 WER on single clips.
            vad_hyps[uid] = whole_hyps[uid]
            continue
        pieces_text = [
            decode_mel(model, pieces, lm, log_mel(concatenated[start:end]))
            for start, end in bounds
        ]
        vad_hyps[uid] = " ".join(text for text in pieces_text if text)

    whole = score_rows(scored_rows, whole_hyps, bootstrap=1000, seed=0)
    vad = score_rows(scored_rows, vad_hyps, bootstrap=1000, seed=0)
    diff = _compare(vad, whole, bootstrap=1000, seed=0)
    truth_bound = 0.2657580919931857  # from long_clip_probe.py: true boundaries
    summary = {
        "groups": len(groups),
        "clips_per_group": args.per_group,
        "margin_s": args.margin,
        "rule": args.rule,
        "mean_seconds": sum(r["duration"] for r in scored_rows) / len(scored_rows),
        "whole_wer": whole["overall"]["wer"],
        "vad_split_wer": vad["overall"]["wer"],
        "whole_eta": whole["overall"]["eta"],
        "vad_split_eta": vad["overall"]["eta"],
        "vad_minus_whole": diff["wer_diff"],
        "ci": diff["wer_diff_ci"],
        "tie": diff["wer_tie"],
        "true_boundary_wer": truth_bound,
        "recovered_fraction": (whole["overall"]["wer"] - vad["overall"]["wer"])
        / (whole["overall"]["wer"] - truth_bound),
        "segments_per_group_median": float(np.median(segment_counts)),
    }
    print(json.dumps(summary, indent=1), flush=True)
    (OUT / "vad_chunk_probe.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "vad_chunk_hyps.json").write_text(
        json.dumps({"whole": whole_hyps, "vad": vad_hyps}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())