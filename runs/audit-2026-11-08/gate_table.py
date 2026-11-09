"""Audit probe 15 — the ship gate, read at the final decode parameters.

The two arms are "shipped decode" and "candidate decode" on the *same* E-017 checkpoint. Against the
external baselines, only the candidate row is new; the shipped row reproduces the numbers in
``runs/E-017/`` and is reported as a control that the harness is unchanged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from stt.eval import _compare, score_rows

T = Path("recordings/manifests/T.jsonl")
rows = [json.loads(line) for line in T.open(encoding="utf-8")]
ids = {r["id"]: r for r in rows}
s3_ids = {r["id"] for r in rows if r.get("speaker") == "S3"}


def load(path: Path) -> dict[str, str]:
    return {json.loads(line)["id"]: json.loads(line)["hyp"] for line in path.open(encoding="utf-8")}


def subsets(name: str) -> list[dict]:
    if name == "t600":
        return rows
    if name == "t-s3":
        return [ids[u] for u in s3_ids]
    return [ids[u] for u in s3_ids]


def compare(name: str, hyp_path: Path, baseline_path: Path, rows_subset=None, tag="") -> dict:
    subset = rows_subset if rows_subset is not None else rows
    ours = score_rows(subset, load(hyp_path), bootstrap=1000, seed=0)
    base = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    if subset is not rows:
        keep = {r["id"] for r in subset}
        base = copy.deepcopy(base)
        base["per_utterance"] = {k: v for k, v in base["per_utterance"].items() if k in keep}
    diff = _compare(ours, base, bootstrap=1000, seed=0)
    return {
        "vs": tag or Path(baseline_path).parent.name,
        "set": name,
        "moth_wer": round(ours["overall"]["wer"], 4),
        "moth_eta": None if ours["overall"]["eta"] is None else round(ours["overall"]["eta"], 4),
        "wer_diff": round(diff["wer_diff"], 4),
        "wer_ci": [round(x, 4) for x in diff["wer_diff_ci"]],
        "wer_won": (not diff["wer_tie"]) and diff["wer_diff"] < 0,
        "eta_diff": None if diff["eta_diff"] is None else round(diff["eta_diff"], 4),
        "eta_ci": None
        if diff["eta_diff_ci"] is None
        else [round(x, 4) for x in diff["eta_diff_ci"]],
        "eta_won": (not diff["eta_tie"]) and (diff["eta_diff"] or 0) > 0,
        "eta_tie": diff["eta_tie"],
    }


def main() -> int:
    final = Path("runs/audit-2026-11-08/final/final")
    shipped = Path("runs/audit-2026-11-08/final/shipped")
    baseline_dir = Path("runs/E-017/baselines-t600")

    out = []
    for arm, root in (("candidate", final), ("shipped-control", shipped)):
        for name, hyp in (("t600", "t600"), ("t-s3", "t-s3")):
            for baseline in ("whisper-tiny-it-beam5", "whistle-it", "parakeet-v3"):
                rec = compare(
                    name,
                    root / f"{hyp}.hyp.jsonl",
                    baseline_dir / baseline / "T.metrics.json",
                    subsets(name),
                    tag=baseline,
                )
                rec["arm"] = arm
                out.append(rec)
                print(json.dumps(rec), flush=True)

    # FLEURS-IT: the set the ship gate has failed on since E-005.
    for arm, root in (("candidate", final), ("shipped-control", shipped)):
        for baseline in ("whisper-tiny-auto-beam5", "whistle-it"):
            hyp = load(root / "fleurs_it.hyp.jsonl")
            fleurs_rows = [
                json.loads(line)
                for line in Path("data/manifests/fleurs_it_test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            ours = score_rows(fleurs_rows, hyp, bootstrap=1000, seed=0)
            base = json.loads(
                (Path("baselines/runs") / baseline / "fleurs_it_test.metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            diff = _compare(ours, base, bootstrap=1000, seed=0)
            rec = {
                "arm": arm,
                "vs": baseline,
                "set": "fleurs_it",
                "moth_wer": round(ours["overall"]["wer"], 4),
                "wer_diff": round(diff["wer_diff"], 4),
                "wer_ci": [round(x, 4) for x in diff["wer_diff_ci"]],
                "wer_won": (not diff["wer_tie"]) and diff["wer_diff"] < 0,
            }
            out.append(rec)
            print(json.dumps(rec), flush=True)

    Path("runs/audit-2026-11-08/gate-table.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())