"""E-039 — leakage audit, first pass (read-only).

Checks exact/near text duplicates across every text source, hashes our own recorded audio per set,
checks corpus id disjointness, and recomputes the natural/gold/never term classification against
the training text actually used (including the E-015/E-028 addenda boundary).

    uv run python runs/E-039/audit.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from stt.data.sources import sha256_file
from stt.text import normalize

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NEAR_THRESHOLD = 0.8
NEAR_LIMIT = 200

TEXT_SOURCES: list[tuple[str, str, str]] = [
    ("script/T", "recordings/scripts/T.jsonl", "script"),
    ("script/A", "recordings/scripts/A.jsonl", "script"),
    ("script/R-S1", "recordings/scripts/R-S1.jsonl", "script"),
    ("script/R-S2", "recordings/scripts/R-S2.jsonl", "script"),
    ("script/pool", "recordings/scripts/pool.jsonl", "script"),
    ("rec/T", "recordings/manifests/T.jsonl", "manifest"),
    ("rec/T-S2", "recordings/manifests/T-S2.jsonl", "manifest"),
    ("rec/A-S1", "recordings/manifests/A-S1.jsonl", "manifest"),
    ("rec/A-S2", "recordings/manifests/A-S2.jsonl", "manifest"),
    ("rec/A-S3", "recordings/manifests/A-S3.jsonl", "manifest"),
    ("rec/R-S1", "recordings/manifests/R-S1.jsonl", "manifest"),
    ("rec/R-S2", "recordings/manifests/R-S2.jsonl", "manifest"),
    ("rec/R-S1-terms", "recordings/manifests/R-S1-terms.jsonl", "manifest"),
    ("rec/R-S1-terms3", "recordings/manifests/R-S1-terms3.jsonl", "manifest"),
    ("rec/R-S2-terms", "recordings/manifests/R-S2-terms.jsonl", "manifest"),
    ("data/cv_it_train", "data/manifests/cv_it_train.jsonl", "manifest"),
    ("data/cv_it_dev", "data/manifests/cv_it_dev.jsonl", "manifest"),
    ("data/cv_it_test", "data/manifests/cv_it_test.jsonl", "manifest"),
    ("data/mls_it_train", "data/manifests/mls_it_train.jsonl", "manifest"),
    ("data/mls_it_dev", "data/manifests/mls_it_dev.jsonl", "manifest"),
    ("data/mls_it_test", "data/manifests/mls_it_test.jsonl", "manifest"),
    ("data/ls_en_train", "data/manifests/ls_en_train.jsonl", "manifest"),
    ("data/ls_en_dev", "data/manifests/ls_en_dev.jsonl", "manifest"),
    ("data/spliced_it_en", "data/manifests/spliced_it_en.jsonl", "manifest"),
    ("data/fleurs_it_dev", "data/manifests/fleurs_it_dev.jsonl", "manifest"),
    ("data/fleurs_it_test", "data/manifests/fleurs_it_test.jsonl", "manifest"),
    ("data/fleurs_en_dev", "data/manifests/fleurs_en_dev.jsonl", "manifest"),
    ("data/fleurs_en_test", "data/manifests/fleurs_en_test.jsonl", "manifest"),
    ("data/noise", "data/manifests/noise.jsonl", "manifest"),
]

OWN_SOURCES = {name for name, _, _ in TEXT_SOURCES if name.startswith(("script/", "rec/"))}
#: Structural overlaps that follow from how the data was recorded, not leakage. Each pair carries
#: the reason, so an unexpected pair is visibly different from a known one.
EXPECTED_PAIRS = {
    frozenset({"script/A", "script/T"}): "anchor A is drawn from 60 reviewed T sentences",
    frozenset({"script/A", "rec/T"}): "anchor A is drawn from 60 reviewed T sentences",
    frozenset({"rec/A-S1", "rec/T"}): "S1 anchor reads T sentences",
    frozenset({"rec/A-S2", "rec/T"}): "S2 anchor reads T sentences",
    frozenset({"rec/A-S3", "rec/T"}): "S3 anchor reads T sentences",
    frozenset({"rec/A-S1", "script/A"}): "anchor manifest mirrors the A script",
    frozenset({"rec/A-S2", "script/A"}): "anchor manifest mirrors the A script",
    frozenset({"rec/A-S3", "script/A"}): "anchor manifest mirrors the A script",
    frozenset({"rec/A-S1", "script/T"}): "S1 anchor reads T sentences",
    frozenset({"rec/A-S2", "script/T"}): "S2 anchor reads T sentences",
    frozenset({"rec/A-S3", "script/T"}): "S3 anchor reads T sentences",
    frozenset({"rec/A-S1", "rec/A-S2"}): "all anchors read the same 60 T sentences",
    frozenset({"rec/A-S1", "rec/A-S3"}): "all anchors read the same 60 T sentences",
    frozenset({"rec/A-S2", "rec/A-S3"}): "all anchors read the same 60 T sentences",
    frozenset({"rec/A-S1", "rec/T-S2"}): "anchor subset overlaps the S2 T rows it repeats",
    frozenset({"rec/A-S2", "rec/T-S2"}): "anchor subset overlaps the S2 T rows it repeats",
    frozenset({"rec/A-S3", "rec/T-S2"}): "anchor subset overlaps the S2 T rows it repeats",
    frozenset({"rec/T", "script/T"}): "T manifest is the recorded T script",
    frozenset({"rec/T-S2", "script/T"}): "T-S2 is the S2 subset of T",
    frozenset({"rec/T-S2", "rec/T"}): "T-S2 is the S2 subset of T",
    frozenset({"rec/T-S2", "script/A"}): "T-S2 contains the S2 anchor sentences read as T",
    frozenset({"rec/R-S1", "script/R-S1"}): "R-S1 manifest is the recorded R-S1 script",
    frozenset({"rec/R-S2", "script/R-S2"}): "R-S2 manifest is the recorded R-S2 script",
    frozenset({"rec/R-S1-terms", "rec/R-S1"}): "term addendum takes are part of the R-S1 script",
    frozenset({"rec/R-S1-terms3", "rec/R-S1"}): "term addendum takes are part of the R-S1 script",
    frozenset({"rec/R-S1-terms", "script/R-S1"}): "term addendum takes read R-S1 cues",
    frozenset({"rec/R-S1-terms3", "script/R-S1"}): "term addendum takes read R-S1 cues",
    frozenset({"rec/R-S1-terms", "rec/R-S1-terms3"}): "two term addenda over the same script",
    frozenset({"rec/R-S2-terms", "rec/R-S2"}): "term addendum takes are part of the R-S2 script",
    frozenset({"rec/R-S2-terms", "script/R-S2"}): "term addendum takes read R-S2 cues",
}
#: Corpus text repeats by design (crowd-sourced prompts read by different speakers), so identical
#: normalized text across a corpus's own splits is not audio leakage. Still reported.
CORPUS_TEXT_REPEAT_PAIRS = {
    frozenset({"data/cv_it_dev", "data/cv_it_test"}),
    frozenset({"data/cv_it_train", "data/cv_it_dev"}),
    frozenset({"data/cv_it_train", "data/cv_it_test"}),
    frozenset({"data/cv_it_train", "data/spliced_it_en"}),
}
CORPUS_SPLITS = {
    "cv_it": ("data/cv_it_train", "data/cv_it_dev", "data/cv_it_test"),
    "mls_it": ("data/mls_it_train", "data/mls_it_dev", "data/mls_it_test"),
    "ls_en": ("data/ls_en_train", "data/ls_en_dev"),
}

# E-014 -> E-024 -> E-028: CV's clean/noisy halves retain the parent transcripts.
# Count the parent once, plus all other natural acoustic sources (including clean LS-EN).
NATURAL_TRAIN_MANIFESTS = (
    "data/manifests/cv_it_train.jsonl",
    "data/manifests/mls_it_train.jsonl",
    "data/manifests/ls_en_train.jsonl",
    "data/manifests/spliced_it_en.jsonl",
)


def text_of(row: dict, kind: str) -> str:
    raw = row.get("norm") or row.get("text") or ""
    return raw if kind == "manifest" else normalize(raw)


def load_source(name: str, path: str, kind: str) -> list[dict]:
    rows = []
    for line in (REPO / path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append({"source": name, "id": str(row["id"]), "text": text_of(row, kind)})
    return rows


def trigrams(text: str) -> frozenset[tuple[str, str, str]]:
    words = text.split()
    return frozenset(zip(words, words[1:], words[2:]))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def exact_duplicates(rows_by_source: dict[str, list[dict]]) -> dict:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, rows in rows_by_source.items():
        for row in rows:
            if row["text"]:
                index[row["text"]].append((name, row["id"]))
    cross: dict[tuple[str, str], dict] = defaultdict(lambda: {"pairs": 0, "examples": []})
    within: dict[str, int] = defaultdict(int)
    multi_source = 0
    for text, hits in index.items():
        sources = sorted({name for name, _ in hits})
        if len(sources) > 1:
            multi_source += 1
            for left in range(len(sources)):
                for right in range(left + 1, len(sources)):
                    key = (sources[left], sources[right])
                    entry = cross[key]
                    entry["pairs"] += 1
                    if len(entry["examples"]) < 5:
                        entry["examples"].append(
                            {"text": text, "ids": [f"{s}:{i}" for s, i in hits[:4]]}
                        )
        elif len(hits) > 1:
            within[sources[0]] += 1
    return {
        "multi_source_texts": multi_source,
        "cross_source_pairs": {
            f"{a}|{b}": {"pairs": v["pairs"], "examples": v["examples"],
                         "class": (
                             "structural" if frozenset({a, b}) in EXPECTED_PAIRS
                             else "corpus_text_repeat"
                             if frozenset({a, b}) in CORPUS_TEXT_REPEAT_PAIRS else "REVIEW"
                         ),
                         "reason": EXPECTED_PAIRS.get(frozenset({a, b}))}
            for (a, b), v in sorted(cross.items())
        },
        "within_source_duplicate_texts": dict(sorted(within.items())),
    }


def near_duplicates(rows_by_source: dict[str, list[dict]]) -> dict:
    rows = [row for name in sorted(OWN_SOURCES) for row in rows_by_source[name] if row["text"]]
    grams = [trigrams(row["text"]) for row in rows]
    inverted: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, gram in enumerate(grams):
        for token in gram:
            inverted[token].append(idx)
    seen: set[tuple[int, int]] = set()
    evaluated = 0
    highest = {"jaccard": 0.0, "a": None, "b": None}
    pairs: dict[str, dict] = defaultdict(lambda: {"count": 0, "examples": []})
    for idx, row in enumerate(rows):
        candidates: set[int] = set()
        for token in grams[idx]:
            candidates.update(inverted[token])
        for other in candidates:
            if other <= idx:
                continue
            key = (idx, other)
            if key in seen:
                continue
            seen.add(key)
            left, right = row, rows[other]
            if left["text"] == right["text"]:
                continue
            score = jaccard(grams[idx], grams[other])
            evaluated += 1
            if score > highest["jaccard"]:
                highest = {"jaccard": round(score, 4),
                           "a": f"{row['source']}:{row['id']}",
                           "b": f"{rows[other]['source']}:{rows[other]['id']}"}
            if score <= NEAR_THRESHOLD:
                continue
            pair = "|".join(sorted((left["source"], right["source"])))
            entry = pairs[pair]
            entry["count"] += 1
            if len(entry["examples"]) < 10:
                entry["examples"].append(
                    {"a": f"{left['source']}:{left['id']}", "b": f"{right['source']}:{right['id']}",
                     "jaccard": round(score, 4), "text_a": left["text"], "text_b": right["text"]}
                )
    total = sum(entry["count"] for entry in pairs.values())
    ordered = dict(sorted(pairs.items(), key=lambda item: -item[1]["count"]))
    trimmed = {key: value for key, value in list(ordered.items())[:NEAR_LIMIT]}
    return {"threshold": NEAR_THRESHOLD, "total_pairs": total, "by_pair": trimmed,
            "truncated_pairs_groups": max(0, len(ordered) - len(trimmed)),
            "candidate_pairs_evaluated": evaluated, "highest_below_check": highest}


def manifest_audio_root(name: str) -> Path:
    """Where a recording manifest's ``audio`` paths are relative to.

    T lives at the legacy root, the anchor under ``anchor/`` and each R set under its own tree
    (``R-S1/``, ``R-S2/``); the addendum manifests inherit their R set.
    """
    if name.startswith("R-"):
        return REPO / "recordings/audio" / name.split("-terms")[0][:4].rstrip("-")
    if name.startswith("A-"):
        return REPO / "recordings/audio/anchor"
    return REPO / "recordings/audio"


def audio_registry() -> dict:
    root = REPO / "recordings/audio"
    by_set: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(root.rglob("*.wav")):
        rel = path.relative_to(root)
        parts = rel.parts
        label = parts[0] if len(parts) > 1 else "root"
        if label == "anchor" and len(parts) > 2:
            label = f"anchor/{parts[1]}"
        elif len(parts) > 2:
            label = f"{parts[0]}/{parts[1]}"
        by_set[label].append({"path": str(rel), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path)})
    hashes: dict[str, list[str]] = defaultdict(list)
    for label, entries in by_set.items():
        for entry in entries:
            hashes[entry["sha256"]].append(f"{label}:{entry['path']}")
    collisions = {
        digest: paths for digest, paths in hashes.items()
        if len(paths) > 1 and len({path.split(":", 1)[0] for path in paths}) > 1
    }
    repeated = {digest: paths for digest, paths in hashes.items() if len(paths) > 1}
    missing = []
    for path in sorted((REPO / "recordings/manifests").glob("*.jsonl")):
        base = manifest_audio_root(path.name)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            audio = row.get("audio")
            if audio and not (base / audio).is_file():
                missing.append(f"{path.name}:{row['id']} -> {audio}")
    return {
        "files": sum(len(entries) for entries in by_set.values()),
        "sets": {label: {"files": len(entries), "bytes": sum(e["bytes"] for e in entries)}
                 for label, entries in sorted(by_set.items())},
        "repeated_hashes": len(repeated),
        "cross_set_collisions": {digest: paths for digest, paths in collisions.items()},
        "missing_audio": missing[:50],
        "missing_audio_count": len(missing),
    }


def id_disjointness(ids_by_source: dict[str, set[str]]) -> dict:
    result = {}
    for corpus, sources in CORPUS_SPLITS.items():
        pairs = {}
        for position, left in enumerate(sources):
            for right in sources[position + 1:]:
                if left in ids_by_source and right in ids_by_source:
                    shared = ids_by_source[left] & ids_by_source[right]
                    pairs[f"{left}|{right}"] = {"shared": len(shared),
                                                "examples": sorted(shared)[:5]}
        result[corpus] = pairs
    return result


def term_provenance() -> dict:
    """Classify every English term of T against the training text each model actually saw.

    The historical 44-term taxonomy omitted clean LS-EN and must be recomputed, not assumed.
    0.8.1 (E-028) trained on the S1 and S2 addenda. R-S1-0692+ (``terms3``) was used in
    rejected E-032/E-036 experiments, not in the shipped model.
    """
    t_rows = [json.loads(line) for line in
              (REPO / "recordings/manifests/T.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    natural = Counter()
    for path in NATURAL_TRAIN_MANIFESTS:
        for line in (REPO / path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                natural.update(normalize(row.get("norm") or row.get("text") or "").split())

    gold_pre = Counter()
    addendum_028 = Counter()
    terms3 = Counter()
    for path, pre_end in (("recordings/manifests/R-S1.jsonl", 642),
                          ("recordings/manifests/R-S2.jsonl", 624)):
        for line in (REPO / path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            number = int(re.search(r"(\d+)$", row["id"]).group(1))
            words = normalize(row.get("text") or row.get("norm") or "").split()
            if number < pre_end:
                gold_pre.update(words)
            elif number < pre_end + 50:
                addendum_028.update(words)
            elif number >= pre_end + 50:
                terms3.update(words)

    terms = sorted({t for row in t_rows for span in row.get("en_spans") or []
                    for t in row["norm"].split()[span[0]:span[1]]})
    counts = {term: {"natural": natural[term], "gold_pre": gold_pre[term],
                     "addendum_028": addendum_028[term], "terms3": terms3[term]}
              for term in terms}
    pre_groups = {"natural": [], "gold_pre": [], "never": []}
    shipped_groups = {"natural": [], "gold_pre": [], "addendum": [], "never": []}
    for term, seen in counts.items():
        if seen["natural"]:
            key = "natural"
        elif seen["gold_pre"]:
            key = "gold_pre"
        else:
            key = "never"
        pre_groups[key].append(term)
        if seen["natural"]:
            shipped_groups["natural"].append(term)
        elif seen["gold_pre"]:
            shipped_groups["gold_pre"].append(term)
        elif seen["addendum_028"]:
            shipped_groups["addendum"].append(term)
        else:
            shipped_groups["never"].append(term)

    pre_never = set(pre_groups["never"])
    rare_rows = sorted(
        row["id"] for row in t_rows
        if any(term in pre_never for span in row.get("en_spans") or []
               for term in row["norm"].split()[span[0]:span[1]])
    )
    return {
        "terms": len(terms),
        "natural_manifests": list(NATURAL_TRAIN_MANIFESTS),
        "natural_definition": "token occurrences in current natural acoustic source manifests; "
                              "CV parent counted once instead of clean/noisy halves, "
                              "spliced reuse counted separately; not epoch-weighted exposure "
                              "or upstream Whisper/tokenizer/LM provenance",
        "pre_addendum": {
            "groups": {key: {"count": len(value), "terms": value[:60]}
                       for key, value in pre_groups.items()},
            "never_term_utterances": len(rare_rows),
            "never_term_ids": rare_rows,
            "definition": "0.7.0 acoustic source-text coverage before E-015/E-028 addenda; "
                          "recomputed with clean LS-EN, not the historical E-015 never44 boundary",
        },
        "shipped": {
            "groups": {key: {"count": len(value), "terms": value[:60]}
                       for key, value in shipped_groups.items()},
            "definition": "E-028 (0.8.1) acoustic source-text coverage, "
                          "S1 and S2 addenda included; not tokenizer/LM/pretraining exposure",
        },
        "terms3_only": sorted(term for term, seen in counts.items()
                              if seen["terms3"] and not seen["natural"]
                              and not seen["gold_pre"] and not seen["addendum_028"]),
        "counts": counts,
    }


def main() -> int:
    rows_by_source = {}
    for name, path, kind in TEXT_SOURCES:
        if (REPO / path).is_file():
            rows_by_source[name] = load_source(name, path, kind)
    ids_by_source = {name: {row["id"] for row in rows} for name, rows in rows_by_source.items()}

    result = {
        "sources": {name: {"path": path, "rows": len(rows_by_source.get(name, []))}
                    for name, path, _ in TEXT_SOURCES},
        "exact": exact_duplicates(rows_by_source),
        "near": near_duplicates(rows_by_source),
        "audio": audio_registry(),
        "id_disjointness": id_disjointness(ids_by_source),
        "terms": term_provenance(),
        "note": "read-only audit; findings require human review, nothing is fixed here",
    }
    result["source_sha256"] = {
        name: sha256_file(REPO / path) for name, path, _ in TEXT_SOURCES
        if (REPO / path).is_file()
    }
    (OUT / "audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    exact = result["exact"]
    print(f"exact: {exact['multi_source_texts']} texts shared across sources")
    for key, entry in exact["cross_source_pairs"].items():
        print(f"  {key:35s} {entry['pairs']:6d}  {entry['class']}")
    near = result["near"]
    print(f"near: {near['total_pairs']} pairs > {near['threshold']} within our data "
          f"({near['candidate_pairs_evaluated']} candidate pairs evaluated, "
          f"max {near['highest_below_check']['jaccard']})")
    for key, entry in list(near["by_pair"].items())[:10]:
        print(f"  {key:45s} {entry['count']:5d}")
    audio = result["audio"]
    print(f"audio: {audio['files']} files, {audio['repeated_hashes']} repeated hashes, "
          f"{len(audio['cross_set_collisions'])} cross-set collisions, "
          f"{audio['missing_audio_count']} missing")
    for corpus, pairs in result["id_disjointness"].items():
        for key, entry in pairs.items():
            print(f"ids: {corpus} {key} shared={entry['shared']}")
    terms = result["terms"]
    print(f"terms: {terms['terms']} English terms")
    print("  pre-addendum (0.7.0): "
          + ", ".join(f"{k}={v['count']}" for k, v in terms["pre_addendum"]["groups"].items())
          + f", never-term utterances={terms['pre_addendum']['never_term_utterances']}")
    print("  shipped (0.8.1):      "
          + ", ".join(f"{k}={v['count']}" for k, v in terms["shipped"]["groups"].items())
          + f", terms3-only={len(terms['terms3_only'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
