"""Audit probe 4 — paired bootstrap for the token bonus, and the noise gate check.

Decodes T-600 and FLEURS-IT at ``gamma`` in {0, 1.2, 1.5} (and ``gamma=1.5`` on the noise set with
the VAD guard), writes hypothesis files, and compares each arm against ``gamma = 0`` with the
project's own paired bootstrap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import decode_set
from sweep_token_bonus_ext import collect

from stt.data.build_mel import load_audio_16k
from stt.eval import _compare, score_rows
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import load_config
from stt.vad import has_speech

OUT = Path("runs/audit-2026-11-08/paired")


def write_hyps(path: Path, hyps: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": key, "hyp": value}, ensure_ascii=False) + "\n"
            for key, value in hyps.items()
        ),
        encoding="utf-8",
    )


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

    t_rows = [
        json.loads(line) for line in (Path("recordings/manifests/T.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    raw = json.loads(Path("data/shards/recorded_t/index.json").read_text(encoding="utf-8"))
    t_index = {uid: {**e, "shard": f"recorded_t/{e['shard']}"} for uid, e in raw.items()}
    t_root = Path("data/shards")

    fleurs_idx, fleurs_rows, fleurs_root = collect([("fleurs_it_test", 0)])

    def run(idx, rows, root, gamma):
        return decode_set(
            model, pieces, rows, idx, root, cfg, lm=lm, lm_weight=0.2, gamma=gamma, beta=0.0
        )

    def scored(rows, hyps, tag):
        result = score_rows(rows, hyps, bootstrap=1000, seed=0)
        write_hyps(OUT / f"{tag}.hyp.jsonl", hyps)
        return result

    arms = {}
    for gamma in (0.0, 1.2, 1.5):
        hyps = run(t_index, t_rows, t_root, gamma)
        arms[f"t600-g{gamma}"] = scored(t_rows, hyps, f"t600-g{gamma}")
    for gamma in (0.0, 1.2, 1.5):
        hyps = run(fleurs_idx, fleurs_rows, fleurs_root, gamma)
        arms[f"fleurs-it-g{gamma}"] = scored(fleurs_rows, hyps, f"fleurs-it-g{gamma}")

    for gamma in (1.2, 1.5):
        for set_name in ("t600", "fleurs-it"):
            ours = arms[f"{set_name}-g{gamma}"]
            base = arms[f"{set_name}-g0.0"]
            diff = _compare(ours, base, bootstrap=1000, seed=0)
            print(
                json.dumps(
                    {
                        "set": set_name,
                        "gamma": gamma,
                        "wer": round(ours["overall"]["wer"], 4),
                        "base": round(base["overall"]["wer"], 4),
                        "wer_diff": round(diff["wer_diff"], 4),
                        "wer_ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                        "wer_tie": diff["wer_tie"],
                        "eta": ours["overall"]["eta"] and round(ours["overall"]["eta"], 4),
                        "eta_diff": diff["eta_diff"] and round(diff["eta_diff"], 4),
                        "eta_ci": diff["eta_diff_ci"] and [round(x, 4) for x in diff["eta_diff_ci"]],
                        "eta_tie": diff["eta_tie"],
                    }
                ),
                flush=True,
            )

    # Noise: VAD gates everything, so re-check that the bonus cannot resurrect words.
    _, noise_rows, _ = collect([("noise", 0)])
    noise_hyps = {}
    for row in noise_rows:
        audio = Path("data/raw/noise") / row["audio"]
        noise_hyps[row["id"]] = "" if not has_speech(load_audio_16k(audio)) else None
    undecided = [row for row in noise_rows if noise_hyps[row["id"]] is None]
    print(f"noise: {len(noise_rows)} clips, {len(undecided)} not gated by the VAD", flush=True)
    noise = score_rows(noise_rows, {k: (v or "") for k, v in noise_hyps.items()}, bootstrap=0)
    print(json.dumps(noise["noise"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())