"""Recompute E-031 development readouts from pinned retained counts/hypotheses, no weights."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from audit import E031_REVISION, REPO, digest, git_bytes, read_pinned_hypotheses, require_new
from gate import target_rare_rows

from stt.eval import _compare, macro_wer_bootstrap, score_rows

OUT = Path(__file__).resolve().parent
ARMS = ("A0_shipped", "A1_frozen_2_2", "A2_frozen_1_2", "A3_frozen_1_2_terms")
CODE_HASH_METHOD = "SHA-256 of source bytes with CRLF normalized to LF"


def code_digest(content):
    return digest(content.replace(b"\r\n", b"\n"))


def macro_readout(rows, metrics, bootstrap=1000, seed=0):
    ids = {str(row["id"]) for row in rows}
    if len(ids) != len(rows):
        raise ValueError("duplicate dev reference ids")
    for arm, scored in metrics.items():
        if set(scored["per_utterance"]) != ids:
            raise ValueError(f"{arm}: retained dev counts differ from reference ids")
        for row in rows:
            if scored["per_utterance"][str(row["id"])]["ref_words"] != len(row["norm"].split()):
                raise ValueError(f"{arm}: retained reference word count mismatch")
    base = metrics["A0_shipped"]
    result = {}
    for arm, scored in metrics.items():
        entry = {"macro_wer": macro_wer_bootstrap(rows, scored, bootstrap=bootstrap, seed=seed)}
        if arm != "A0_shipped":
            entry["vs_A0"] = macro_wer_bootstrap(rows, scored, base, bootstrap=bootstrap, seed=seed)
        result[arm] = entry
    return result


def rare_readout(rows, candidate, baseline, bootstrap=1000, seed=0):
    targets = target_rare_rows(rows)
    new = score_rows(targets, candidate, bootstrap=0)
    base = score_rows(targets, baseline, bootstrap=0)

    def counts(scored):
        per = scored["per_utterance"].values()
        return {"en_ref": sum(item["en_ref"] for item in per),
                "en_hit": sum(item["en_hit"] for item in per),
                "aligned_eta": scored["overall"]["eta"]}

    return {"n_rows": len(rows), "candidate": counts(new), "baseline_0.8.0": counts(base),
            "vs_0.8.0": _compare(new, base, bootstrap=bootstrap, seed=seed),
            "historical_all_span_tokens": sum(end - start for row in rows
                                               for start, end in row.get("en_spans", [])),
            "definition": "exactly aligned annotated occurrences of the 44 frozen target terms",
            "gate": False}


def readout(bootstrap=1000, seed=0):
    inputs = {}

    def content(path):
        data = git_bytes(path)
        inputs[path] = digest(data)
        return data

    rows = []
    for name in ("mls_it_dev", "ls_en_dev"):
        rows.extend(json.loads(line) for line in content(f"data/manifests/{name}.jsonl").splitlines()
                    if line.strip())
    metrics = {arm: json.loads(content(f"runs/E-031/dev/{arm}/dev.metrics.json")) for arm in ARMS}
    t_rows = {row["id"]: row for row in (
        json.loads(line) for line in content("recordings/manifests/T.jsonl").splitlines()
        if line.strip())}
    rare = [json.loads(line) for line in content("runs/E-026/rare47-16k.jsonl").splitlines()
            if line.strip()]
    for row in rare:
        if any(row.get(key) != t_rows[row["id"]].get(key) for key in ("norm", "en_spans", "lang")):
            raise ValueError(f"{row['id']}: pinned rare/T annotations disagree")
    hyps = []
    for path in ("runs/E-031/eval/frozen_1_2_terms/t600.hyp.jsonl",
                 "runs/E-030/eval/e028-int8/t600.hyp.jsonl"):
        content(path)
        hyps.append(read_pinned_hypotheses(path))
    return {
        "evidence": "Retained development evidence, not an independent test or new decode. "
                    "T/S3 repeatedly informed development; dev selection CIs are post-hoc recomputation.",
        "revision": E031_REVISION, "inputs_sha256": inputs,
        "code_hash_method": CODE_HASH_METHOD,
        "code_sha256": {name: code_digest((REPO / name).read_bytes()) for name in (
            "runs/E-031/audit_readout.py", "runs/E-031/audit.py", "runs/E-031/gate.py",
            "runs/E-031/build_lm.py", "stt/eval.py", "stt/text.py", "stt/data/sources.py")},
        "versions": {name: version(name) for name in ("num2words", "numpy")},
        "bootstrap": {"resamples": bootstrap, "seed": seed, "confidence": 0.95,
                      "macro": "paired utterance resampling within language; equal-language macro",
                      "eta": "paired utterance ratio bootstrap on target occurrences"},
        "dev": macro_readout(rows, metrics, bootstrap, seed),
        "rare_target_only": rare_readout(rare, hyps[0], hyps[1], bootstrap, seed),
        "historical_gate": {"rare_threshold": 0.284, "unchanged": True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "audit-readout.json")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    require_new(args.out)
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    result = readout(args.bootstrap, args.seed)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"dev": result["dev"], "rare_target_only": result["rare_target_only"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
