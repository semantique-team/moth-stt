"""Audit probe 11 — does trimming the leading/trailing silence change the transcript?

Motivation: on T-600 the model deletes the **first** reference word in 13.3% of clips but the last
in 0.5%. The recordings carry a median 0.35 s of leading room tone. If the onset is the problem, a
VAD-trimmed front-end fixes it for free (the VAD already ships in the package).

Uses ``stt.vad.speech_probabilities`` to find the first and last speech window, trims with a
300 ms margin, recomputes the mel, and re-decodes. Same decode parameters on both arms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_token_bonus import beam_with_bonus

from stt.data.build_mel import load_audio_16k
from stt.eval import _compare, score_rows
from stt.features import log_mel
from stt.lm import BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import load_config
from stt.vad import WINDOW, WINDOW_THRESHOLD, speech_probabilities

MARGIN_S = 0.30


def trim(pcm: np.ndarray, margin_s: float = MARGIN_S) -> np.ndarray:
    probabilities = list(speech_probabilities(pcm))
    if not probabilities:
        return pcm
    speech = [i for i, p in enumerate(probabilities) if p > WINDOW_THRESHOLD]
    if not speech:
        return pcm
    rate = 16000
    start = max(0, speech[0] * WINDOW - int(margin_s * rate))
    end = min(len(pcm), (speech[-1] + 1) * WINDOW + int(margin_s * rate))
    if end - start < int(0.5 * rate):
        return pcm
    return pcm[start:end]


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

    rows = [
        json.loads(line) for line in (Path("recordings/manifests/T.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    arms = {"raw": {}, "trimmed": {}}
    trimmed_saved = []
    with torch.inference_mode():
        for row in rows:
            audio = load_audio_16k(Path("recordings/audio") / row["audio"])
            for arm, pcm in (("raw", audio), ("trimmed", trim(audio))):
                if arm == "trimmed":
                    trimmed_saved.append(len(audio) / 16000 - len(pcm) / 16000)
                mel = log_mel(pcm)
                tensor = torch.from_numpy(mel[None])
                lengths = torch.tensor([mel.shape[1]])
                log_probs, out_lengths = model(tensor, lengths)
                arms[arm][row["id"]] = beam_with_bonus(
                    log_probs[0], int(out_lengths[0]), pieces,
                    beam_width=8, top_k=8, lm=lm, lm_weight=0.5, gamma=2.4, beta=0.0,
                )

    results = {}
    for arm, hyps in arms.items():
        scored = score_rows(rows, hyps, bootstrap=1000, seed=0)
        results[arm] = scored
        o = scored["overall"]
        print(
            json.dumps(
                {
                    "arm": arm,
                    "wer": round(o["wer"], 4),
                    "eta": round(o["eta"], 4),
                    "del": round(o["del"] / o["ref_words"], 4),
                    "ins": round(o["ins"] / o["ref_words"], 4),
                }
            ),
            flush=True,
        )
        Path(f"runs/audit-2026-11-08/eval/trim-{arm}").mkdir(parents=True, exist_ok=True)
        Path(f"runs/audit-2026-11-08/eval/trim-{arm}/t600.hyp.jsonl").write_text(
            "".join(
                json.dumps({"id": k, "hyp": v}, ensure_ascii=False) + "\n"
                for k, v in hyps.items()
            ),
            encoding="utf-8",
        )

    diff = _compare(results["trimmed"], results["raw"], bootstrap=1000, seed=0)
    print(
        json.dumps(
            {
                "trimmed_minus_raw": round(diff["wer_diff"], 4),
                "ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                "tie": diff["wer_tie"],
                "eta_diff": round(diff["eta_diff"], 4),
                "eta_tie": diff["eta_tie"],
                "mean_trimmed_s": round(float(np.mean(trimmed_saved)), 3),
            }
        ),
        flush=True,
    )

    # first-word deletion rate, both arms
    from stt.eval import align
    from stt.text import normalize as norm

    for arm, hyps in arms.items():
        first = 0
        for row in rows:
            ref = norm(row["norm"], "it").split()
            hyp = norm(hyps[row["id"]], "it").split()
            dels = {i for kind, i, _ in align(ref, hyp) if kind == "del"}
            first += int(0 in dels)
        print(json.dumps({"arm": arm, "first_word_deleted": first, "n": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())