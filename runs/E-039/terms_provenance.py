"""E-039 — corrected term provenance after the PR #12 review.

The E-015 boundary counted "natural" exposure from CV-IT, MLS-IT and spliced IT+EN only, and
omitted `ls_en_train` even though E-014/E-024/E-028 train on it. This recomputes the taxonomy over
the acoustic source text actually used by the accepted lineage, with the addendum boundary kept so
the historical 47-utterance readout stays interpretable.

Scope: presence in source text, not epoch-weighted exposure, not Whisper pretraining, not the
tokenizer or LM corpus. `spliced_it_en` repeats CV/MLS/LS text and therefore adds no new term.

    uv run python runs/E-039/terms_provenance.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from stt.text import normalize

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = REPO / "recordings/manifests/T.jsonl"
NATURAL = (
    "data/manifests/cv_it_train.jsonl",
    "data/manifests/mls_it_train.jsonl",
    "data/manifests/ls_en_train.jsonl",
    "data/manifests/spliced_it_en.jsonl",
)
GOLD = (
    ("recordings/manifests/R-S1.jsonl", 642),
    ("recordings/manifests/R-S2.jsonl", 624),
)


def vocabulary(paths: tuple[str, ...]) -> Counter:
    counts: Counter = Counter()
    for path in paths:
        for line in (REPO / path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                counts.update(normalize(row.get("norm") or row.get("text") or "").split())
    return counts


def main() -> int:
    natural_other = vocabulary((
        "data/manifests/cv_it_train.jsonl",
        "data/manifests/mls_it_train.jsonl",
        "data/manifests/spliced_it_en.jsonl",
    ))
    natural_ls = vocabulary(("data/manifests/ls_en_train.jsonl",))
    natural = natural_other + natural_ls
    gold_pre: Counter = Counter()
    addendum: Counter = Counter()
    terms3: Counter = Counter()
    for path, boundary in GOLD:
        for line in (REPO / path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            number = int(row["id"].rsplit("-", 1)[1])
            words = normalize(row.get("text") or row.get("norm") or "").split()
            if number < boundary:
                gold_pre.update(words)
            elif number < boundary + 50:
                addendum.update(words)
            else:
                terms3.update(words)

    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terms = sorted({
        term
        for row in rows
        for start, end in row.get("en_spans") or []
        for term in row["norm"].split()[start:end]
    })
    counts = {
        term: {
            "natural": natural[term],
            "gold_pre": gold_pre[term],
            "addendum": addendum[term],
            "terms3": terms3[term],
        }
        for term in terms
    }
    pre_groups = {"natural": [], "gold": [], "never": []}
    shipped_groups = {"natural": [], "gold": [], "addendum": [], "never": []}
    for term, seen in counts.items():
        pre_groups["natural" if seen["natural"] else "gold" if seen["gold_pre"] else "never"].append(term)
        if seen["natural"]:
            shipped_groups["natural"].append(term)
        elif seen["gold_pre"]:
            shipped_groups["gold"].append(term)
        elif seen["addendum"]:
            shipped_groups["addendum"].append(term)
        else:
            shipped_groups["never"].append(term)

    never_pre = set(pre_groups["never"])
    never_rows = sorted(
        row["id"]
        for row in rows
        if any(term in never_pre for start, end in row.get("en_spans") or []
               for term in row["norm"].split()[start:end])
    )
    # The E-015 boundary counted natural exposure without ls_en_train, so reproduce that set and
    # show which of its terms the omitted corpus actually covers.
    e015_never = {
        term for term in terms if not natural_other[term] and not gold_pre[term]
    }
    moved_by_ls = sorted(term for term in e015_never if natural_ls[term])
    ls_only = sorted(
        term for term in terms
        if natural_ls[term] and not natural_other[term]
        and not gold_pre[term] and not addendum[term]
    )
    result = {
        "method": {
            "natural_sources": list(NATURAL),
            "gold_boundaries": {path: boundary for path, boundary in GOLD},
            "unit": "presence in acoustic source text; not epoch-weighted exposure",
            "excludes": ["Whisper pretraining", "tokenizer text", "LM corpus"],
        },
        "terms": len(terms),
        "pre_addendum": {
            "groups": {key: {"count": len(value), "terms": value} for key, value in pre_groups.items()},
            "never_term_utterances": len(never_rows),
            "never_term_ids": never_rows,
        },
        "shipped_e028": {
            "groups": {key: {"count": len(value), "terms": value[:80]}
                       for key, value in shipped_groups.items()},
        },
        "e015_boundary": {
            "never_terms": len(e015_never),
            "never_utterances": len(sorted(
                row["id"] for row in rows
                if any(term in e015_never for start, end in row.get("en_spans") or []
                       for term in row["norm"].split()[start:end])
            )),
        },
        "e015_never_but_in_ls_en": {"count": len(moved_by_ls), "terms": moved_by_ls},
        "ls_en_only_exposure": {"count": len(ls_only), "terms": ls_only},
        "counts": counts,
        "note": ("supersedes the E-015 boundary, which omitted ls_en_train; the historical "
                 "47-utterance readout keeps its own fixed definition"),
    }
    (OUT / "terms-provenance.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"terms {len(terms)}")
    print("pre-addendum: " + ", ".join(
        f"{key}={value['count']}" for key, value in result["pre_addendum"]["groups"].items()
    ) + f" | never utterances {len(never_rows)}")
    print("shipped: " + ", ".join(
        f"{key}={value['count']}" for key, value in result["shipped_e028"]["groups"].items()
    ))
    print(f"E-015 'never' but present in ls_en_train: {len(moved_by_ls)} {moved_by_ls}")
    print(f"terms whose only exposure is ls_en_train: {len(ls_only)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
