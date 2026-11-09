"""How much does environmental noise cost? (probe, 2026-11-09)

Motivated by testing the demo with a real laptop microphone: "with less noise, a bit better".
The project has known since `docs/11` §7.4 that noise augmentation is the robustness gap it never
closed, but the *size* of the gap on our own test set was never measured. This measures it.

Method: take the 120 MUSAN noise clips already used to build the noise set, mix each into a T clip
at a target SNR (scaled on RMS relative to the speech), recompute the mel, decode with the shipped
parameters. Same clips, paired across conditions, so the only variable is the noise.

    uv run python runs/audit-2026-11-09/noise_curve.py [--clips 100]

SNR reference points, for scale: a quiet room is 30-40 dB, a laptop fan in the same room ~25-30,
an office ~15-20, a café ~5-10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stt.data.build_mel import load_audio_16k
from stt.decode import beam_search
from stt.eval import _compare, score_rows
from stt.features import log_mel
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import load_config
from stt.vad import has_speech

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4}
SNRS = [None, 30.0, 20.0, 15.0, 10.0, 5.0]  # None = clean


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)) + 1e-12))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``speech`` at ``snr_db`` (10 log10(P_speech / P_noise)), tiled to length."""
    if noise.size < speech.size:
        repeats = int(np.ceil(speech.size / noise.size))
        noise = np.tile(noise, repeats)
    noise = noise[: speech.size]
    speech_power = rms(speech) ** 2
    noise_power = rms(noise) ** 2
    if noise_power <= 0 or speech_power <= 0:
        return speech
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10.0)))
    mixed = speech + noise * scale
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(prog="noise_curve.py")
    parser.add_argument("--clips", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
    s3 = [r for r in rows if r.get("speaker") == "S3"][: args.clips]

    noise_dir = REPO / "data/raw/noise/noise_musan"
    noise_paths = sorted(noise_dir.glob("*.wav"))
    if not noise_paths:
        raise SystemExit(f"no MUSAN clips under {noise_dir}")
    rng = np.random.default_rng(args.seed)
    noise_cache = [load_audio_16k(path) for path in noise_paths]
    print(f"{len(s3)} T clips (S3) x {len(noise_paths)} noise sources at {SNRS}", flush=True)

    def decode(audio: np.ndarray) -> str:
        mel = log_mel(audio)
        with torch.inference_mode():
            log_probs, out_lengths = model(
                torch.from_numpy(mel[None]), torch.tensor([mel.shape[1]])
            )
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

    results: dict[str, dict] = {}
    per_condition: dict[str, dict] = {}
    gated = {str(snr): 0 for snr in SNRS}
    for snr in SNRS:
        hypotheses: dict[str, str] = {}
        for index, row in enumerate(s3):
            speech = load_audio_16k(REPO / "recordings/audio" / row["audio"])
            if snr is None:
                audio = speech
            else:
                noise = noise_cache[int(rng.integers(len(noise_cache)))]
                audio = mix_at_snr(speech, noise, snr)
                if not has_speech(audio):
                    gated[str(snr)] += 1
            hypotheses[row["id"]] = decode(audio)
        scored = score_rows(s3, hypotheses, bootstrap=1000, seed=0)
        per_condition[str(snr)] = scored
        results[str(snr)] = {
            "wer": scored["overall"]["wer"],
            "eta": scored["overall"]["eta"],
            "del": scored["overall"]["del"] / scored["overall"]["ref_words"],
            "ins": scored["overall"]["ins"] / scored["overall"]["ref_words"],
            "gated_by_vad": gated[str(snr)],
        }
        print(
            json.dumps(
                {
                    "snr_db": "clean" if snr is None else snr,
                    "wer": round(scored["overall"]["wer"], 4),
                    "eta": round(scored["overall"]["eta"], 4),
                    "deletions": round(results[str(snr)]["del"], 4),
                    "insertions": round(results[str(snr)]["ins"], 4),
                    "vad_gated": gated[str(snr)],
                }
            ),
            flush=True,
        )

    clean = per_condition[str(None)]
    deltas = {}
    for snr in SNRS[1:]:
        diff = _compare(per_condition[str(snr)], clean, bootstrap=1000, seed=0)
        deltas[str(snr)] = {
            "wer_diff": diff["wer_diff"],
            "wer_ci": diff["wer_diff_ci"],
            "tie": diff["wer_tie"],
        }
        print(
            json.dumps(
                {
                    "snr_db": snr,
                    "delta_vs_clean": round(diff["wer_diff"], 4),
                    "ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                    "tie": diff["wer_tie"],
                }
            ),
            flush=True,
        )

    summary = {
        "clips": len(s3),
        "conditions": results,
        "deltas_vs_clean": deltas,
        "params": PARAMS,
    }
    (OUT / "noise_curve.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'noise_curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())