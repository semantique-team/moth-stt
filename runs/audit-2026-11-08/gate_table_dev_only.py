"""Audit probe 17 — the ship gate at the **dev-only** candidate.

Rewrites ``gate_table.py`` against ``runs/audit-2026-11-08/dev-only/dev-only/`` so that §1.3 of
``AUDIT.md`` quotes numbers from the arm that dev selected, not from the cell that saw S3.
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


def main() -> int:
    out = []
    root = Path("runs/audit-2026-11-08/dev-only/dev-only")
    baseline_dir = Path("runs/E-017/baselines-t600")

    for set_name, subset in (("t600", rows), ("t-s3", [ids[u] for u in s3_ids])):
        hyp = load(root / f"{set_name}.hyp.jsonl")
        ours = score_rows(subset, hyp, bootstrap=1000, seed=0)
        for baseline in ("whisper-tiny-it-beam5", "whisper-tiny-auto-beam5", "whistle-it", "parakeet-v3"):
            base = json.loads((baseline_dir / baseline / "T.metrics.json").read_text(encoding="utf-8"))
            if set_name == "t-s3":
                keep = {r["id"] for r in subset}
                base = copy.deepcopy(base)
                base["per_utterance"] = {k: v for k, v in base["per_utterance"].items() if k in keep}
            diff = _compare(ours, base, bootstrap=1000, seed=0)
            rec = {
                "set": set_name,
                "vs": baseline,
                "moth_wer": round(ours["overall"]["wer"], 4),
                "moth_eta": round(ours["overall"]["eta"], 4),
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
            out.append(rec)
            print(json.dumps(rec), flush=True)

    fleurs_rows = [
        json.loads(line)
        for line in Path("data/manifests/fleurs_it_test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    ours = score_rows(fleurs_rows, load(root / "fleurs_it.hyp.jsonl"), bootstrap=1000, seed=0)
    for baseline in ("whisper-tiny-auto-beam5", "whistle-it"):
        base = json.loads(
            (Path("baselines/runs") / baseline / "fleurs_it_test.metrics.json").read_text(
                encoding="utf-8"
            )
        )
        diff = _compare(ours, base, bootstrap=1000, seed=0)
        rec = {
            "set": "fleurs_it",
            "vs": baseline,
            "moth_wer": round(ours["overall"]["wer"], 4),
            "wer_diff": round(diff["wer_diff"], 4),
            "wer_ci": [round(x, 4) for x in diff["wer_diff_ci"]],
            "wer_won": (not diff["wer_tie"]) and diff["wer_diff"] < 0,
        }
        out.append(rec)
        print(json.dumps(rec), flush=True)

    Path("runs/audit-2026-11-08/gate-table-dev-only.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())