"""E-025 — the E-024 noise curve, re-measured on a room the model has never heard.

Both checkpoints decode the **same mixed audio** at every condition, so the comparison is paired.
The clean condition is a harness check: the audio is identical to E-024's, so the numbers must
reproduce the ones in ``runs/E-024/eval/{base,noisy}/clean.metrics.json``; if they do not, the
harness is wrong and the curve below is not read.

    uv run python runs/E-025/holdout_eval.py \
        --noise data/raw/noise/room_holdout/room2_2026-11-09_trimmed_16k.wav \
        --out runs/E-025/eval
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

REPO = Path(__file__).resolve().parents[2]
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4}
SNRS = [None, 30.0, 20.0, 15.0, 10.0, 5.0]
ARMS = {
    "base": REPO / "runs/E-017/checkpoints/ckpt_best.pt",
    "new": REPO / "runs/E-024/checkpoints/ckpt_best.pt",
}
#: E-024's clean numbers on these same 100 clips, for the harness check.
E024_CLEAN = {"base": 0.2224, "new": 0.2394}


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)) + 1e-12))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if noise.size < speech.size:
        noise = np.tile(noise, int(np.ceil(speech.size / noise.size)))
    noise = noise[: speech.size]
    speech_power, noise_power = rms(speech) ** 2, rms(noise) ** 2
    if speech_power <= 0 or noise_power <= 0:
        return speech
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10.0)))
    mixed = speech + noise * scale
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = np.asarray(mixed) / peak
    return mixed.astype(np.float32)


def mix_with_offset(
    speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Use a random part of the room recording, so a 38 s take is not one 3 s loop."""
    if noise.size > speech.size:
        start = int(rng.integers(0, noise.size - speech.size + 1))
        segment = noise[start : start + speech.size]
    else:
        segment = noise
    return mix_at_snr(speech, segment, snr_db)


def build_model(checkpoint: Path, head_blocks: int) -> MothSTT:
    model = MothSTT(vocab_size=1024, head_blocks=head_blocks)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return model.eval()


def main() -> int:
    parser = argparse.ArgumentParser(prog="holdout_eval.py")
    parser.add_argument("--noise", required=True, help="held-out room tone (16 kHz mono wav)")
    parser.add_argument("--config", default="runs/E-024/config.yaml")
    parser.add_argument("--out", default="runs/E-025/eval")
    parser.add_argument("--clips", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / args.config)
    head_blocks = int(cfg["model"].get("head_blocks", 0))
    processor = load_tokenizer(REPO / cfg["tokenizer"])
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    lm = BPETrigramLM.load(REPO / "runs/E-008/lm_bpe3_aug.pkl.gz")
    models = {name: build_model(path, head_blocks) for name, path in ARMS.items()}

    rows = [
        json.loads(line)
        for line in (REPO / "recordings/manifests/T.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    s3 = [row for row in rows if row.get("speaker") == "S3"][: args.clips]
    noise_path = Path(args.noise)
    noise_path = noise_path if noise_path.is_absolute() else REPO / noise_path
    noise = load_audio_16k(noise_path)
    if noise.size < 16000:
        raise SystemExit(f"{noise_path}: room tone is too short ({noise.size / 16000:.2f} s)")

    result: dict[str, object] = {
        "noise": str(noise_path.relative_to(REPO)),
        "noise_seconds": round(noise.size / 16000, 2),
        "clips": len(s3),
        "arms": {name: str(path.relative_to(REPO)) for name, path in ARMS.items()},
        "conditions": {},
    }

    for condition_index, snr in enumerate(SNRS):
        # Independent offset stream per condition, identical for both arms: the pairing is exact.
        rng = np.random.default_rng(args.seed + condition_index)
        hypotheses: dict[str, dict[str, str]] = {name: {} for name in ARMS}
        for row in s3:
            speech = load_audio_16k(REPO / "recordings/audio" / row["audio"])
            audio = speech if snr is None else mix_with_offset(speech, noise, snr, rng)
            mel = log_mel(audio)
            with torch.inference_mode():
                for name, model in models.items():
                    log_probs, out_lengths = model(
                        torch.from_numpy(mel[None]), torch.tensor([mel.shape[1]])
                    )
                    hypotheses[name][row["id"]] = beam_search(
                        log_probs[0],
                        int(out_lengths[0]),
                        pieces,
                        beam_width=PARAMS["beam"],
                        top_k=PARAMS["top_k"],
                        lm=lm,
                        lm_weight=PARAMS["lm_weight"],
                        token_bonus=PARAMS["token_bonus"],
                    )
        tag = "clean" if snr is None else f"{snr:g}"
        scored = {
            name: score_rows(s3, hypotheses[name], bootstrap=1000, seed=0) for name in ARMS
        }
        for name in ARMS:
            (out / f"{tag}.{name}.metrics.json").write_text(
                json.dumps(scored[name], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (out / f"{tag}.{name}.hyp.jsonl").write_text(
                "".join(
                    json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
                    for key, value in hypotheses[name].items()
                ),
                encoding="utf-8",
            )
        delta = _compare(scored["new"], scored["base"], bootstrap=1000, seed=0)
        result["conditions"][tag] = {
            "snr_db": "clean" if snr is None else snr,
            "base": {
                "wer": scored["base"]["overall"]["wer"],
                "eta": scored["base"]["overall"]["eta"],
            },
            "new": {
                "wer": scored["new"]["overall"]["wer"],
                "eta": scored["new"]["overall"]["eta"],
            },
            "delta": {
                "wer": delta["wer_diff"],
                "wer_ci": delta["wer_diff_ci"],
                "wer_tie": delta["wer_tie"],
                "eta": delta["eta_diff"],
                "eta_ci": delta["eta_diff_ci"],
                "eta_tie": delta["eta_tie"],
            },
        }
        print(
            json.dumps(
                {
                    "condition": tag,
                    "base": round(scored["base"]["overall"]["wer"], 4),
                    "new": round(scored["new"]["overall"]["wer"], 4),
                    "delta": round(delta["wer_diff"], 4),
                    "ci": [round(value, 4) for value in delta["wer_diff_ci"]],
                    "tie": delta["wer_tie"],
                }
            ),
            flush=True,
        )

        if snr is None:
            for name, expected in E024_CLEAN.items():
                actual = scored[name]["overall"]["wer"]
                if abs(actual - expected) > 0.005:
                    print(
                        f"HARNESS MISMATCH: {name} clean {actual:.4f} != E-024 {expected:.4f}; "
                        "the curve is not comparable",
                        flush=True,
                    )
                    result["harness_check"] = {"ok": False, name: actual, "expected": expected}
                else:
                    result.setdefault("harness_check", {"ok": True, "deltas": {}})
                    result["harness_check"]["deltas"][name] = round(actual - expected, 5)

    gains = [
        result["conditions"][tag]["delta"]["wer"] for tag in ("30", "20", "15", "10")
    ]
    result["mean_gain_10_30"] = float(np.mean(gains))
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
