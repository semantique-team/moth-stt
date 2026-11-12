"""E-024 — noise curve and clean gate for a checkpoint.

The primary readout of E-024 is the **noise curve**, not clean WER: a model that trades 0.3 clean
points for 8 points at 15 dB is a good trade, and clean WER alone would hide it. Secondary is the
clean gate, because a model that wins on noise and loses on clean is not shippable.

    uv run python runs/E-024/noise_eval.py --checkpoint runs/E-017/checkpoints/ckpt_best.pt \
        --out runs/E-024/eval/base
    uv run python runs/E-024/noise_eval.py --checkpoint runs/E-024/checkpoints/ckpt_best.pt \
        --out runs/E-024/eval/noisy

The noise is mixed with the same rule the training set used
(`runs/E-024/build_noisy_mel.py::mix_at_snr`), which is also the same pool — so the gain measured
here is an **upper bound** for a room the mixer has never seen. Stated in the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stt.data.build_mel import load_audio_16k
from stt.data.dataset import MelDataset, plan_batches
from stt.decode import beam_search
from stt.eval import _compare, score_rows
from stt.features import log_mel
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import _prepare_batch, load_config, load_sources

REPO = Path(__file__).resolve().parents[2]
PARAMS = {"beam": 12, "top_k": 10, "lm_weight": 0.5, "token_bonus": 2.4}
SNRS = [None, 30.0, 20.0, 15.0, 10.0, 5.0]
NOISE_DIRS = (REPO / "data/raw/noise/noise_musan", REPO / "data/raw/noise/noise_room_tone")
CLEAN_SETS = {
    "t600": ("recordings/manifests/T.jsonl", "data/shards/recorded_t"),
    "fleurs_it_test": ("data/manifests/fleurs_it_test.jsonl", "data/shards/fleurs_it_test"),
    "cv_it_test": ("data/manifests/cv_it_test.jsonl", "data/shards/cv_it_test"),
}


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


def build_model(checkpoint: Path, head_blocks: int) -> MothSTT:
    model = MothSTT(vocab_size=1024, head_blocks=head_blocks)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return model.eval()


def main() -> int:
    parser = argparse.ArgumentParser(prog="noise_eval.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="runs/E-024/config.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--clips", type=int, default=100)
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--lm-weight", type=float, default=PARAMS["lm_weight"])
    parser.add_argument("--token-bonus", type=float, default=PARAMS["token_bonus"])
    parser.add_argument(
        "--lm",
        default="runs/E-008/lm_bpe3_aug.pkl.gz",
        help="pickle trigram LM used by the PyTorch decode (default: shipped augmented LM)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = {
        "beam": PARAMS["beam"],
        "top_k": PARAMS["top_k"],
        "lm_weight": args.lm_weight,
        "token_bonus": args.token_bonus,
    }
    cfg = load_config(REPO / args.config)
    processor = load_tokenizer(REPO / cfg["tokenizer"])
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    lm = BPETrigramLM.load(REPO / args.lm)
    model = build_model(REPO / args.checkpoint, int(cfg["model"].get("head_blocks", 0)))

    rows = [
        json.loads(line)
        for line in (REPO / "recordings/manifests/T.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    s3 = [r for r in rows if r.get("speaker") == "S3"][: args.clips]

    noise_paths = [p for directory in NOISE_DIRS for p in sorted(directory.glob("*.wav"))]
    if not noise_paths:
        raise SystemExit(f"no noise clips under {NOISE_DIRS}")
    noise_pool = [load_audio_16k(path) for path in noise_paths]
    rng = np.random.default_rng(args.seed)

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
            beam_width=params["beam"],
            top_k=params["top_k"],
            lm=lm,
            lm_weight=params["lm_weight"],
            token_bonus=params["token_bonus"],
        )

    curve: dict[str, dict] = {}
    scored_by_snr: dict[str, dict] = {}
    for snr in SNRS:
        hypotheses: dict[str, str] = {}
        for row in s3:
            speech = load_audio_16k(REPO / "recordings/audio" / row["audio"])
            if snr is None:
                audio = speech
            else:
                # Same noise index sequence per SNR as the audit probe, so conditions are paired.
                audio = mix_at_snr(speech, noise_pool[int(rng.integers(len(noise_pool)))], snr)
            hypotheses[row["id"]] = decode(audio)
        scored = score_rows(s3, hypotheses, bootstrap=1000, seed=0)
        scored_by_snr[str(snr)] = scored
        # Persist per-utterance metrics so the arms can be compared **paired** later.
        tag = "clean" if snr is None else f"{snr:g}"
        (out / f"{tag}.metrics.json").write_text(
            json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (out / f"{tag}.hyp.jsonl").write_text(
            "".join(
                json.dumps({"id": k, "hyp": v}, ensure_ascii=False) + "\n"
                for k, v in hypotheses.items()
            ),
            encoding="utf-8",
        )
        curve[str(snr)] = {
            "wer": scored["overall"]["wer"],
            "eta": scored["overall"]["eta"],
            "deletions": scored["overall"]["del"] / scored["overall"]["ref_words"],
        }
        print(
            json.dumps(
                {
                    "snr_db": "clean" if snr is None else snr,
                    "wer": round(scored["overall"]["wer"], 4),
                    "eta": round(scored["overall"]["eta"], 4),
                }
            ),
            flush=True,
        )

    clean = scored_by_snr[str(None)]
    for snr in SNRS[1:]:
        diff = _compare(scored_by_snr[str(snr)], clean, bootstrap=1000, seed=0)
        curve[str(snr)]["delta_vs_clean"] = diff["wer_diff"]
        curve[str(snr)]["ci"] = diff["wer_diff_ci"]

    result: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "decode": params,
        "lm": args.lm,
        "clips": len(s3),
        "noise_curve": curve,
        "mean_noise_wer_10_30": float(
            np.mean([curve[str(s)]["wer"] for s in (10.0, 15.0, 20.0, 30.0)])
        ),
    }

    if not args.skip_clean:
        clean_sets: dict[str, dict] = {}
        for name, (manifest, shards) in CLEAN_SETS.items():
            index, set_rows, root = load_sources(
                [{"manifest": manifest, "shards": shards}]
            )
            dataset = MelDataset(index, root, set_rows)
            batches = plan_batches(
                [float(r["duration"]) for r in set_rows],
                max_seconds=192,
                max_clips=32,
                seed=0,
                epoch=0,
            )
            hypotheses = {}
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
            scored = score_rows(set_rows, hypotheses, bootstrap=1000, seed=0)
            clean_sets[name] = {
                "wer": scored["overall"]["wer"],
                "eta": scored["overall"]["eta"],
                "n": scored["n_utterances"],
            }
            (out / f"clean-{name}.metrics.json").write_text(
                json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(json.dumps({"clean_set": name, **clean_sets[name]}), flush=True)
        result["clean_sets"] = clean_sets

    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())