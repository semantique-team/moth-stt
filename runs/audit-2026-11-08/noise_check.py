"""Audit probe 12 — does the token bonus resurrect words on non-speech?

A positive per-token bonus literally rewards emitting tokens, so the noise gate has to be re-checked
in the arm that ships it. Two readings:

1. **with the VAD** (the shipping path): words on the 337 non-music clips and on the 120 music clips;
2. **VAD off** (the failure mode the guard exists to cover): the honest worst case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import beam_with_bonus

from stt.data.build_mel import load_audio_16k
from stt.features import log_mel
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.text import normalize
from stt.tokenizer import load_tokenizer
from stt.train import load_config
from stt.vad import has_speech


def main() -> int:
    cfg = load_config("runs/E-017/config.yaml")
    proc = load_tokenizer(cfg["tokenizer"])
    pieces = {i + 1: proc.id_to_piece(i) for i in range(proc.vocab_size())}
    model = MothSTT(vocab_size=1024, head_blocks=2)
    payload = torch.load(
        "runs/E-017/checkpoints/ckpt_best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model"])
    model.eval()
    lm = BPETrigramLM.load("runs/E-008/lm_bpe3_aug.pkl.gz")

    rows = [json.loads(line) for line in (Path("data/manifests/noise.jsonl")).read_text(encoding="utf-8").splitlines()]
    audio_root = Path("data/raw/noise")

    arms = {"shipped-0.2-g0": (0.2, 0.0), "recal-0.5-g2.4": (0.5, 2.4)}
    summary = {}
    for arm, (lm_weight, gamma) in arms.items():
        stats = {"vad_on": {"words": 0, "clips": 0}, "vad_off": {"words": 0, "clips": 0}}
        per_kind = {
            "noise_musan": 0, "noise_musan_clips": 0,
            "noise_music": 0, "noise_music_clips": 0,
            "noise_room_tone": 0, "noise_room_tone_clips": 0,
            "noise_silence": 0, "noise_silence_clips": 0,
        }
        with torch.inference_mode():
            for row in rows:
                audio = load_audio_16k(audio_root / row["audio"])
                gated = not has_speech(audio)
                mel = log_mel(audio)
                log_probs, out_lengths = model(
                    torch.from_numpy(mel[None]), torch.tensor([mel.shape[1]])
                )
                text = beam_with_bonus(
                    log_probs[0], int(out_lengths[0]), pieces,
                    beam_width=8, top_k=8, lm=lm, lm_weight=lm_weight, gamma=gamma, beta=0.0,
                )
                words = len(normalize(text, "it").split())
                source = row.get("source", "?")
                if gated:
                    stats["vad_on"]["clips"] += 1
                else:
                    stats["vad_on"]["words"] += words
                    stats["vad_on"]["clips"] += 1
                stats["vad_off"]["words"] += words
                stats["vad_off"]["clips"] += 1
                if source in per_kind:
                    per_kind[source] += words
                    per_kind[f"{source}_clips"] += int(words > 0)
        summary[arm] = {"with_vad": stats["vad_on"], "vad_off": stats["vad_off"], "vad_off_words_by_source": per_kind}
        print(json.dumps({"arm": arm, **summary[arm]}), flush=True)

    Path("runs/audit-2026-11-08/noise-token-bonus.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())