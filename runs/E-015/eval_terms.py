"""E-015 — the rare-term readout, by provenance, with a paired bootstrap.

The primary metric is the 47 T utterances that contain one of the 44 terms never seen in any
training text. The 16 terms the gold already covers are a regression check; the natural/gold-only
groups are context. The baseline is 0.7.0's `t600.hyp.jsonl`; the new arm is E-015's.

The historical E-015 readout used term *presence* (a term anywhere in the hypothesis). Since
2026-11-11 this script also reports **aligned ETA** on the same rows, which is the canonical
metric: a term only counts when it is aligned to the reference occurrence. Presence remains a
diagnostic. The historical result is not rewritten.

    uv run python runs/E-015/eval_terms.py \
        --hyp runs/E-015/eval/gate/t600.hyp.jsonl \
        --baseline runs/E-024/eval/gate/t600.hyp.jsonl \
        --out runs/E-015/eval/terms-eta.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from stt.eval import score_rows
from stt.text import normalize

REPO = Path(__file__).resolve().parents[2]
T_MANIFEST = REPO / "recordings/manifests/T.jsonl"
NATURAL = (
    "data/manifests/cv_it_train.jsonl",
    "data/manifests/mls_it_train.jsonl",
    "data/manifests/spliced_it_en.jsonl",
)
GOLD = ("recordings/manifests/R-S1.jsonl", "recordings/manifests/R-S2.jsonl")
#: The terms added to the gold on 2026-10-24; the historical constant was named TERMS18 but
#: contains 16 strings. The regression check is reported as ``gold_terms`` (16).
GOLD_TERMS = [
    "backlog", "cfo", "cpu", "crm", "deadline", "deploy", "erp", "kpi", "merge", "milestone",
    "okr", "onboarding", "paypal", "phishing", "url", "warranty",
]


def toks(text: str) -> list[str]:
    return normalize(text).split()


def vocabulary(paths: tuple[str, ...]) -> Counter:
    """Token counts of the training text. R-S1-0642 and R-S2-0624 onward are the E-015/E-028
    term addenda, so the baseline's gold is everything before them: the groups must describe what
    0.7.0 had seen."""
    counts: Counter = Counter()
    for path in paths:
        for line in (REPO / path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if path.endswith("R-S1.jsonl") and int(row["id"].rsplit("-", 1)[1]) >= 642:
                continue
            if path.endswith("R-S2.jsonl") and int(row["id"].rsplit("-", 1)[1]) >= 624:
                continue
            counts.update(toks(row.get("text") or row.get("norm") or ""))
    return counts


def load_hyp(path: Path) -> dict[str, str]:
    return {
        json.loads(line)["id"]: json.loads(line)["hyp"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def term_occurrences(row: dict) -> list[str]:
    words = toks(row["text"])
    out: list[str] = []
    for span in row.get("en_spans") or []:
        out.extend(words[span[0] : span[1]])
    return out


def score(rows: dict[str, dict], hyps: dict[str, str], group: str, natural: Counter, gold: Counter):
    occurrences = correct = 0
    for rid, row in rows.items():
        if rid not in hyps:
            continue
        hyp = set(toks(hyps[rid]))
        for term in term_occurrences(row):
            bucket = "natural" if term in natural else "gold" if term in gold else "never"
            if bucket != group:
                continue
            occurrences += 1
            correct += int(term in hyp)
    return correct, occurrences


def rare_rows(rows: dict[str, dict], hyps: dict[str, str], natural: Counter, gold: Counter):
    """The T utterances with at least one never-seen term, and their per-row correctness."""
    out = []
    for rid, row in rows.items():
        if rid not in hyps:
            continue
        terms = [t for t in term_occurrences(row) if t not in natural and t not in gold]
        if not terms:
            continue
        hyp = set(toks(hyps[rid]))
        out.append((rid, terms, any(term in hyp for term in terms)))
    return out


def bootstrap(
    base_rows: list[tuple[str, list[str], bool]],
    new_rows: list[tuple[str, list[str], bool]],
    rng: np.random.Generator,
    resamples: int = 1000,
) -> dict[str, float]:
    base = {rid: ok for rid, _, ok in base_rows}
    new = {rid: ok for rid, _, ok in new_rows}
    ids = sorted(set(base) & set(new))
    base_values = np.array([base[rid] for rid in ids], dtype=float)
    new_values = np.array([new[rid] for rid in ids], dtype=float)
    diffs = []
    for _ in range(resamples):
        sample = rng.integers(0, len(ids), len(ids))
        diffs.append(new_values[sample].mean() - base_values[sample].mean())
    diffs = np.asarray(diffs)
    return {
        "n": len(ids),
        "base": float(base_values.mean()),
        "new": float(new_values.mean()),
        "delta": float(new_values.mean() - base_values.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "bootstrap": resamples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="eval_terms.py")
    parser.add_argument("--hyp", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in T_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    new = load_hyp(Path(args.hyp))
    base = load_hyp(Path(args.baseline))
    natural, gold = vocabulary(NATURAL), vocabulary(GOLD)
    rng = np.random.default_rng(args.seed)

    result: dict[str, object] = {"hyp": args.hyp, "baseline": args.baseline, "groups": {}}
    for group in ("natural", "gold", "never"):
        for arm, hyps in (("base", base), ("new", new)):
            correct, total = score(rows, hyps, group, natural, gold)
            result["groups"].setdefault(group, {})[arm] = {
                "correct": correct,
                "occurrences": total,
                "accuracy": correct / total if total else None,
            }
        print(
            f"{group:8s} base {result['groups'][group]['base']['correct']:3d}/"
            f"{result['groups'][group]['base']['occurrences']:4d} "
            f"new {result['groups'][group]['new']['correct']:3d}/"
            f"{result['groups'][group]['new']['occurrences']:4d}"
        )

    base_rare = rare_rows(rows, base, natural, gold)
    new_rare = rare_rows(rows, new, natural, gold)
    result["rare"] = bootstrap(base_rare, new_rare, rng)
    print(
        f"\n47 rare utterances: base {result['rare']['base'] * 100:.1f}% "
        f"new {result['rare']['new'] * 100:.1f}% "
        f"delta {result['rare']['delta'] * 100:+.1f} "
        f"[{result['rare']['ci_low'] * 100:+.1f}, {result['rare']['ci_high'] * 100:+.1f}] "
        f"(n={result['rare']['n']})"
    )

    # Canonical metric: aligned ETA on the same 47 rows, and on all 600 T rows.
    def aligned_eta(rows_subset: list[dict], hyps: dict[str, str]) -> dict:
        metrics = score_rows(rows_subset, hyps, bootstrap=0)
        overall = metrics["overall"]
        per = metrics["per_utterance"].values()
        return {
            "eta": overall["eta"],
            "wer": overall["wer"],
            "en_hit": sum(item["en_hit"] for item in per),
            "en_ref": sum(item["en_ref"] for item in per),
        }

    rare_ids = sorted(rid for rid, _, _ in base_rare)
    rare_subset = [rows[rid] for rid in rare_ids]
    all_subset = list(rows.values())
    result["aligned_eta"] = {
        "rare": {"base": aligned_eta(rare_subset, base), "new": aligned_eta(rare_subset, new)},
        "all": {"base": aligned_eta(all_subset, base), "new": aligned_eta(all_subset, new)},
    }
    for name, subset in (("47 rare", rare_subset), ("T-600", all_subset)):
        base_eta = result["aligned_eta"]["rare" if name == "47 rare" else "all"]["base"]
        new_eta = result["aligned_eta"]["rare" if name == "47 rare" else "all"]["new"]
        print(
            f"aligned ETA {name}: base {base_eta['eta'] * 100:.1f}% "
            f"({base_eta['en_hit']}/{base_eta['en_ref']}) "
            f"new {new_eta['eta'] * 100:.1f}% ({new_eta['en_hit']}/{new_eta['en_ref']})"
        )

    # Regression check: the 16 terms already covered by the gold.
    def gold_terms_accuracy(hyps: dict[str, str]) -> tuple[int, int]:
        correct = total = 0
        for rid, row in rows.items():
            if rid not in hyps:
                continue
            hyp = set(toks(hyps[rid]))
            present = [t for t in term_occurrences(row) if t in GOLD_TERMS]
            if not present:
                continue
            total += 1
            correct += int(any(t in hyp for t in present))
        return correct, total

    base_gold, new_gold = gold_terms_accuracy(base), gold_terms_accuracy(new)
    result["gold_terms"] = {"base": base_gold, "new": new_gold}
    print(
        f"16 gold terms: base {base_gold[0]}/{base_gold[1]} = "
        f"{base_gold[0] / base_gold[1] * 100:.1f}% "
        f"new {new_gold[0]}/{new_gold[1]} = {new_gold[0] / new_gold[1] * 100:.1f}%"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
