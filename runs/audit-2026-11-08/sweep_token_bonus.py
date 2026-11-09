"""Audit probe (2026-11-08) — is the CTC/LM combination under-emitting?

Diagnostic only, not an experiment entry. Beam search as in ``stt.decode`` plus a per-token
bonus ``gamma * n_tokens`` on the emitted prefix (the usual CTC "insertion/length" fix), and the
option of dropping the LM entirely to separate the two causes of the deletion bias.

Tuning set: 200 clips of ``mls_it_dev`` + 200 of ``ls_en_dev`` (dev, never T).
Report set: ``recordings/manifests/T.jsonl`` (600 clips) — reported with the explicit caveat that
it is a diagnostic read, not an adopted number.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from stt.data.dataset import MelDataset, plan_batches
from stt.decode import BLANK_ID, _ids_to_text, _lm_bonus, _logaddexp
from stt.eval import score_rows
from stt.lm import BOS_ID, EOS_ID, BPETrigramLM
from stt.model import MothSTT
from stt.tokenizer import load_tokenizer
from stt.train import _prepare_batch, load_config, load_sources

NEG_INF = float("-inf")


def beam_with_bonus(log_probs, length, pieces, *, beam_width, top_k, lm, lm_weight, gamma, beta):
    """Prefix beam search + ``gamma`` per emitted token + ``beta`` length normalisation."""
    import numpy as np

    array = log_probs.detach().cpu().numpy() if hasattr(log_probs, "detach") else log_probs
    frames = int(length)
    if frames <= 0:
        return ""
    array = array[:frames]
    beams: dict[tuple[int, ...], list[float]] = {(): [0.0, NEG_INF]}
    for frame in array:
        candidates = np.argpartition(frame, -top_k)[-top_k:].astype(int).tolist()
        if BLANK_ID not in candidates:
            candidates.append(BLANK_ID)
        candidates.sort(key=lambda token: (-float(frame[token]), token))
        nxt: dict[tuple[int, ...], list[float]] = {}
        for prefix, (p_blank, p_nonblank) in beams.items():
            total = _logaddexp(p_blank, p_nonblank)
            entry = nxt.setdefault(prefix, [NEG_INF, NEG_INF])
            entry[0] = _logaddexp(entry[0], total + float(frame[BLANK_ID]))
            last = prefix[-1] if prefix else -1
            for token in candidates:
                if token == BLANK_ID:
                    continue
                logp = float(frame[token])
                bonus = gamma + _lm_bonus(lm, prefix, token, lm_weight)
                if token == last:
                    entry[1] = _logaddexp(entry[1], p_nonblank + logp)
                    new_prefix = prefix + (token,)
                    np_entry = nxt.setdefault(new_prefix, [NEG_INF, NEG_INF])
                    np_entry[1] = _logaddexp(np_entry[1], p_blank + logp + bonus)
                else:
                    new_prefix = prefix + (token,)
                    np_entry = nxt.setdefault(new_prefix, [NEG_INF, NEG_INF])
                    np_entry[1] = _logaddexp(np_entry[1], total + logp + bonus)
        ranked = sorted(nxt.items(), key=lambda item: -_logaddexp(item[1][0], item[1][1]))
        beams = dict(ranked[:beam_width])

    def final(item):
        prefix, (p_blank, p_nonblank) = item
        score = _logaddexp(p_blank, p_nonblank)
        if lm is not None and lm_weight:
            prev1 = prefix[-1] - 1 if prefix else BOS_ID
            prev2 = prefix[-2] - 1 if len(prefix) >= 2 else BOS_ID
            score += lm_weight * lm.score(prev2, prev1, EOS_ID)
        return score + beta * len(prefix) * math.log(1.0 + len(prefix))

    best = max(beams.items(), key=final)[0]
    return _ids_to_text(best, pieces)


def decode_set(model, pieces, rows, index, root, cfg, *, lm, lm_weight, gamma, beta):
    batch = cfg["batch"]
    ds = MelDataset(index, root, rows)
    batches = plan_batches(
        [float(r["duration"]) for r in rows],
        max_seconds=float(batch["max_seconds"]),
        max_clips=int(batch["max_clips"]),
        seed=0,
        epoch=0,
    )
    hyps = {}
    with torch.inference_mode():
        for positions in batches:
            items = [ds[p] for p in positions]
            tensor, lengths, brows = _prepare_batch(items, "truncated")
            log_probs, out_lengths = model(tensor, lengths)
            for offset, row in enumerate(brows):
                hyps[row["id"]] = beam_with_bonus(
                    log_probs[offset],
                    int(out_lengths[offset]),
                    pieces,
                    beam_width=5,
                    top_k=8,
                    lm=lm,
                    lm_weight=lm_weight,
                    gamma=gamma,
                    beta=beta,
                )
    return hyps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/E-017/checkpoints/ckpt_best.pt")
    parser.add_argument("--config", default="runs/E-017/config.yaml")
    parser.add_argument("--lm", default="runs/E-008/lm_bpe3_aug.pkl.gz")
    parser.add_argument("--out", default="runs/audit-2026-11-08/token-bonus.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    proc = load_tokenizer(cfg["tokenizer"])
    pieces = {i + 1: proc.id_to_piece(i) for i in range(proc.vocab_size())}
    model = MothSTT(vocab_size=1024, head_blocks=int(cfg["model"].get("head_blocks", 0)))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    lm = BPETrigramLM.load(args.lm)

    dev_rows = []
    dev_index, dev_root = {}, None
    for name, limit in (("mls_it_dev", 200), ("ls_en_dev", 200)):
        index, rows, root = load_sources(
            [{"manifest": f"data/manifests/{name}.jsonl", "shards": f"data/shards/{name}"}]
        )
        subset = rows[:limit]
        for uid, entry in index.items():
            if any(r["id"] == uid for r in subset):
                dev_index[uid] = entry
        dev_rows.extend(subset)
        dev_root = root
    print(f"dev subset: {len(dev_rows)} clips", flush=True)

    t_index, t_rows, t_root = load_sources(
        [{"manifest": "recordings/manifests/T.jsonl", "shards": "data/shards/recorded_t"}]
    )

    grid = []
    for gamma, beta in [
        (0.0, 0.0),
        (0.2, 0.0),
        (0.4, 0.0),
        (0.6, 0.0),
        (0.8, 0.0),
        (1.0, 0.0),
        (0.1, 0.05),
        (0.1, 0.1),
        (0.2, 0.1),
    ]:
        hyps = decode_set(
            model, pieces, dev_rows, dev_index, dev_root, cfg,
            lm=lm, lm_weight=0.2, gamma=gamma, beta=beta,
        )
        scored = score_rows(dev_rows, hyps, bootstrap=0, seed=0)
        o = scored["overall"]
        row = {
            "gamma": gamma,
            "beta": beta,
            "dev_wer": o["wer"],
            "dev_del": o["del"] / o["ref_words"],
            "dev_sub": o["sub"] / o["ref_words"],
            "dev_ins": o["ins"] / o["ref_words"],
        }
        hyps_t = decode_set(
            model, pieces, t_rows, t_index, t_root, cfg,
            lm=lm, lm_weight=0.2, gamma=gamma, beta=beta,
        )
        scored_t = score_rows(t_rows, hyps_t, bootstrap=0, seed=0)
        ot = scored_t["overall"]
        row |= {
            "t600_wer": ot["wer"],
            "t600_eta": ot["eta"],
            "t600_del": ot["del"] / ot["ref_words"],
            "t600_sub": ot["sub"] / ot["ref_words"],
            "t600_ins": ot["ins"] / ot["ref_words"],
        }
        grid.append(row)
        print(json.dumps(row), flush=True)

    # Same sweep without the LM, to separate the LM from the acoustic model.
    for gamma in (0.0, 0.4, 0.8):
        hyps_t = decode_set(
            model, pieces, t_rows, t_index, t_root, cfg,
            lm=None, lm_weight=0.0, gamma=gamma, beta=0.0,
        )
        ot = score_rows(t_rows, hyps_t, bootstrap=0, seed=0)["overall"]
        print(json.dumps({"lm": "off", "gamma": gamma, "t600_wer": ot["wer"],
                          "t600_del": ot["del"] / ot["ref_words"],
                          "t600_sub": ot["sub"] / ot["ref_words"]}), flush=True)

    Path(args.out).write_text(json.dumps(grid, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())