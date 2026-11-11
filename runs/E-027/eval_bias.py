"""E-027 — contextual-bias pilot on the published 0.7.0 release.

No training, no weight changes, no release changes. The glossary is pre-registered in
``PLAN.md``; this script only decodes the same acoustic logits with and without a bounded
``ContextualBias``. T is reused test evidence, not an independent dev set.

    uv run python runs/E-027/eval_bias.py --out runs/E-027
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from stt.data.build_mel import load_audio_16k
from stt.decode import beam_search
from stt.eval import _compare, score_rows
from stt.features import load_mel_filters, log_mel
from stt.hotwords import ContextualBias
from stt.onnx_runner import ReleaseMoth
from stt.text import normalize
from stt.tokenizer import encode, load_tokenizer

REPO = Path(__file__).resolve().parents[2]
T_MANIFEST = REPO / "recordings/manifests/T.jsonl"
NATURAL = (
    "data/manifests/cv_it_train.jsonl",
    "data/manifests/mls_it_train.jsonl",
    "data/manifests/spliced_it_en.jsonl",
)
GOLD = ("recordings/manifests/R-S1.jsonl", "recordings/manifests/R-S2.jsonl")
PRODUCT_TERMS = [
    "meeting", "call", "team", "sprint", "ticket", "deploy", "deployment", "release",
    "deadline", "backlog", "roadmap", "standup", "feedback", "update", "report", "budget",
    "forecast", "target", "task", "project", "workflow", "pipeline", "feature", "bug",
]
IRRELEVANT_TERMS = [
    "banana", "guitar", "mountain", "elephant", "garden", "bicycle", "kitchen", "thunder",
    "marble", "violin", "rabbit", "meadow", "lantern", "cactus", "harbor", "tunnel", "pillow",
    "rocket", "forest", "camel", "mirror", "pumpkin", "dolphin", "volcano",
]
MAX_SAMPLES = 30 * 16_000


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def _vocabulary(paths: tuple[str, ...]) -> Counter:
    counts: Counter = Counter()
    for path in paths:
        for line in (REPO / path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # R-S1-0642 onward is the E-015 addendum itself.
            if path.endswith("R-S1.jsonl") and int(row["id"].rsplit("-", 1)[1]) >= 642:
                continue
            counts.update(_tokens(row.get("text") or row.get("norm") or ""))
    return counts


def _term_occurrences(row: dict) -> list[str]:
    words = _tokens(row["text"])
    return [word for start, end in (row.get("en_spans") or []) for word in words[start:end]]


def _read_rows(path: Path, limit: int | None = None, category: str | None = None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if category is not None:
        rows = [row for row in rows if row.get("category") == category]
    return rows[:limit] if limit is not None else rows


def _audio_rows(rows: list[dict], root: Path) -> tuple[list[dict], list[str]]:
    kept, skipped = [], []
    for row in rows:
        if float(row.get("duration", 0)) > 30:
            skipped.append(row["id"])
            continue
        audio = load_audio_16k(root / row["audio"])
        if not 560 <= len(audio) <= MAX_SAMPLES:
            skipped.append(row["id"])
            continue
        kept.append((row, audio))
    return kept, skipped


def _presence(
    rows: list[dict], hyps: dict[str, str], natural: Counter, gold: Counter
) -> dict:
    correct = total = 0
    for row in rows:
        terms = [
            term
            for term in _term_occurrences(row)
            if term not in natural and term not in gold
        ]
        if not terms:
            continue
        total += 1
        correct += int(any(term in set(_tokens(hyps[row["id"]])) for term in terms))
    return {"correct": correct, "total": total, "accuracy": correct / total if total else None}


def _glossary_hits(hyps: dict[str, str], terms: set[str]) -> int:
    return sum(1 for hyp in hyps.values() for word in _tokens(hyp) if word in terms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100, help="control-set clips per set")
    args = parser.parse_args()

    out = args.out
    hyp_dir = out / "hyp"
    if (out / "metrics.json").exists() or hyp_dir.exists():
        raise SystemExit(f"refusing to overwrite {out / 'metrics.json'} or {hyp_dir}")
    out.mkdir(parents=True, exist_ok=True)
    hyp_dir.mkdir()

    runtime = ReleaseMoth(REPO / "release")
    cfg = runtime.config["decode"]
    filters = load_mel_filters(REPO / "release" / runtime.config["features"]["mel_filters"])
    tokenizer = load_tokenizer(REPO / "data/tokenizer/moth_bpe.model")
    pieces = runtime.model.pieces

    def ctc(word: str) -> list[int]:
        return encode(tokenizer, word)

    def bias(terms: list[str], bonus: float, prefix: float) -> ContextualBias:
        return ContextualBias.from_terms(
            terms, pieces, ctc, bonus=bonus, prefix_bonus=prefix
        )

    arms = {
        "baseline": None,
        "bias_1.0": bias(PRODUCT_TERMS, 1.0, 0.25),
        "bias_0.5": bias(PRODUCT_TERMS, 0.5, 0.0),
        "irrelevant_1.0": bias(IRRELEVANT_TERMS, 1.0, 0.25),
        "zero_bonus": bias(PRODUCT_TERMS, 0.0, 0.0),
    }

    t_rows = _read_rows(T_MANIFEST)
    natural, gold = _vocabulary(NATURAL), _vocabulary(GOLD)
    rare_ids = {
        row["id"]
        for row in t_rows
        if any(term not in natural and term not in gold for term in _term_occurrences(row))
    }
    sets = {
        "t600": (t_rows, REPO / "recordings/audio"),
        "t_control_it": (
            [row for row in t_rows if row.get("category") == "control_it"],
            REPO / "recordings/audio",
        ),
        "cv_it_dev": (
            _read_rows(REPO / "data/manifests/cv_it_dev.jsonl", limit=args.limit),
            REPO / "data/raw",
        ),
        "fleurs_en_test": (
            _read_rows(REPO / "data/manifests/fleurs_en_test.jsonl", limit=args.limit),
            REPO / "data/raw",
        ),
    }

    # Decode each clip once per arm, reusing the acoustic logits across arms.
    hyps: dict[str, dict[str, dict[str, str]]] = {
        arm: {name: {} for name in sets} for arm in arms
    }
    skipped: dict[str, list[str]] = {}
    for name, (rows, root) in sets.items():
        kept, missing = _audio_rows(rows, root)
        skipped[name] = missing
        print(f"{name}: {len(kept)} clips ({len(missing)} skipped)", flush=True)
        for row, audio in kept:
            mel = log_mel(audio, 16_000, filters=filters)
            log_probs = runtime.model.log_probs(mel)
            common = {
                "beam_width": int(cfg["beam"]),
                "top_k": int(cfg["top_k"]),
                "lm": runtime.lm,
                "lm_weight": float(cfg["lm_weight"]),
                "token_bonus": float(cfg["token_bonus"]),
            }
            for arm, contextual in arms.items():
                hyps[arm][name][row["id"]] = beam_search(
                    log_probs,
                    log_probs.shape[0],
                    pieces,
                    **common,
                    **({"contextual": contextual} if contextual is not None else {}),
                )

    metrics: dict[str, dict] = {}
    for name, (rows, _) in sets.items():
        kept_ids = set(hyps["baseline"][name])
        kept_rows = [row for row in rows if row["id"] in kept_ids]
        metrics[name] = {
            arm: score_rows(kept_rows, hyps[arm][name], bootstrap=0) for arm in arms
        }
        for arm in arms:
            (hyp_dir / f"{name}.{arm}.jsonl").write_text(
                "".join(
                    json.dumps({"id": uid, "hyp": hyp}, ensure_ascii=False) + "\n"
                    for uid, hyp in hyps[arm][name].items()
                ),
                encoding="utf-8",
            )

    rare_rows = [row for row in t_rows if row["id"] in rare_ids and row["id"] in hyps["baseline"]["t600"]]
    rare_metrics = {
        arm: score_rows(rare_rows, hyps[arm]["t600"], bootstrap=0) for arm in arms
    }
    presence = {
        arm: _presence(rare_rows, hyps[arm]["t600"], natural, gold) for arm in arms
    }

    product_set = set(PRODUCT_TERMS)
    irrelevant_set = set(IRRELEVANT_TERMS)
    hits = {
        name: {
            arm: {
                "product": _glossary_hits(hyps[arm][name], product_set),
                "irrelevant": _glossary_hits(hyps[arm][name], irrelevant_set),
            }
            for arm in arms
        }
        for name in sets
    }

    comparisons = {}
    for arm in arms:
        if arm == "baseline":
            continue
        comparisons[arm] = {
            "t600": _compare(metrics["t600"][arm], metrics["t600"]["baseline"], 1000, 0),
            "rare": _compare(rare_metrics[arm], rare_metrics["baseline"], 1000, 0),
        }

    zero_equal = hyps["zero_bonus"] == hyps["baseline"]
    report = {
        "release": "0.7.0",
        "decode": {
            "beam": cfg["beam"],
            "top_k": cfg["top_k"],
            "lm_weight": cfg["lm_weight"],
            "token_bonus": cfg["token_bonus"],
        },
        "glossary": {
            "product": PRODUCT_TERMS,
            "irrelevant": IRRELEVANT_TERMS,
            "source": "data/en_terms.txt, first 24 single-word work/meeting entries",
        },
        "sets": {
            name: {
                "n": len(metrics[name]["baseline"]["per_utterance"]),
                "skipped": skipped[name],
                "arms": {
                    arm: {
                        "overall": metrics[name][arm]["overall"],
                        "noise": metrics[name][arm]["noise"],
                    }
                    for arm in arms
                },
            }
            for name in sets
        },
        "rare": {
            "n": len(rare_rows),
            "arms": {
                arm: {
                    "overall": rare_metrics[arm]["overall"],
                    "presence": presence[arm],
                }
                for arm in arms
            },
        },
        "glossary_hits": hits,
        "comparisons": comparisons,
        "zero_bonus_equals_baseline": zero_equal,
    }
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for arm in ("bias_1.0", "bias_0.5", "irrelevant_1.0"):
        row = comparisons[arm]["t600"]
        rare = comparisons[arm]["rare"]
        print(
            f"{arm:14s} T600 WER {row['wer_diff'] * 100:+.2f} "
            f"[{row['wer_diff_ci'][0] * 100:+.2f},{row['wer_diff_ci'][1] * 100:+.2f}] "
            f"ETA {row['eta_diff'] * 100:+.2f} "
            f"[{row['eta_diff_ci'][0] * 100:+.2f},{row['eta_diff_ci'][1] * 100:+.2f}] "
            f"| rare ETA {rare['eta_diff'] * 100:+.2f}"
        )
    print(f"zero_bonus == baseline: {zero_equal}")
    print(f"wrote {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
