"""The one scorer (design: ``docs/05-evaluation.md``).

Baselines and our model are different *producers of hypothesis files*; they never touch this
code. ``eval.py`` reads a reference manifest (JSONL, see ``docs/03-data.md`` §4) and a
hypothesis file (``{"id", "hyp", ...}``), scores both under the one text rule
(:func:`stt.text.normalize`) and writes a single ``metrics.json``.

Metrics: corpus-level WER (word-level Levenshtein: substitutions + deletions + insertions over
reference words), per-category and per-speaker(-group) breakdowns, insertion rate per 100
reference words, ETA (English-term accuracy over ``en_spans``: the share of reference English
words aligned as an exact match, with the per-term errors listed), noise-set counting (words
emitted on empty-transcript clips, music reported separately) and a seeded paired bootstrap over
utterances (1,000 resamples by default) for the 95% CI of each WER and of the WER/ETA difference
against another system's ``metrics.json`` (``--vs``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

from stt import text as text_module
from stt.text import normalize

__all__ = ["align", "main", "op_counts", "score_rows"]

_BOOTSTRAP_LO = 0.025
_BOOTSTRAP_HI = 0.975

#: One alignment operation: (kind, ref_index, hyp_index); -1 when a side is missing.
Op = tuple[str, int, int]


def align(ref: list[str], hyp: list[str]) -> list[Op]:
    """Minimum-cost word alignment between ``ref`` and ``hyp``.

    Tie-breaking order implements the project convention: match, then substitution, then
    deletion, then insertion (substitutions are preferred over delete+insert pairs).
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        prev, row = d[i - 1], d[i]
        ref_word = ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(
                prev[j - 1] + (0 if ref_word == hyp[j - 1] else 1),
                prev[j] + 1,
                row[j - 1] + 1,
            )

    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append(("match", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))
            i -= 1
        else:
            ops.append(("ins", -1, j - 1))
            j -= 1
    ops.reverse()
    return ops


def op_counts(ops: list[Op]) -> tuple[int, int, int]:
    """Count ``(substitutions, deletions, insertions)`` in an alignment."""
    sub = sum(1 for kind, _, _ in ops if kind == "sub")
    dele = sum(1 for kind, _, _ in ops if kind == "del")
    ins = sum(1 for kind, _, _ in ops if kind == "ins")
    return sub, dele, ins


def score_rows(
    rows: list[dict],
    hyps: dict[str, str],
    bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    """Score hypothesis texts against manifest rows under the one text rule.

    Rows whose ``norm`` is empty are treated as noise/silence clips: they contribute nothing to
    WER and are summarized in the ``noise`` block instead. Returns the core ``metrics`` dict
    (``n_utterances``, ``overall``, ``breakdown``, ``noise``, ``per_utterance``).
    """
    missing = [str(row["id"]) for row in rows if str(row["id"]) not in hyps]
    if missing:
        raise KeyError(f"hypotheses missing for {len(missing)} ids: {missing[:5]}")

    per_utterance: dict[str, dict[str, int]] = {}
    category_groups: dict[str, list[str]] = {}
    speaker_groups: dict[str, list[str]] = {}
    speaker_group_groups: dict[str, list[str]] = {}
    eta_errors: list[dict] = []
    noise = {
        "clips": 0,
        "total_words": 0,
        "non_empty_clips": 0,
        "non_music_clips": 0,
        "non_music_words": 0,
        "non_music_non_empty_clips": 0,
        "music_clips": 0,
        "music_words": 0,
        "music_non_empty_clips": 0,
    }

    for row in rows:
        uid = str(row["id"])
        lang = row.get("lang")
        hint = lang if lang in ("it", "en") else "it"
        hyp_norm = normalize(hyps[uid], hint)
        norm = row.get("norm") or ""

        if not norm:
            words = len(hyp_norm.split())
            non_empty = int(words > 0)
            is_music = row.get("category") == "music"
            noise["clips"] += 1
            noise["total_words"] += words
            noise["non_empty_clips"] += non_empty
            if is_music:
                noise["music_clips"] += 1
                noise["music_words"] += words
                noise["music_non_empty_clips"] += non_empty
            else:
                noise["non_music_clips"] += 1
                noise["non_music_words"] += words
                noise["non_music_non_empty_clips"] += non_empty
            continue

        ref_words = norm.split()
        hyp_words = hyp_norm.split()
        ops = align(ref_words, hyp_words)
        sub, dele, ins = op_counts(ops)

        ref_to_op = {ref_idx: (kind, hyp_idx) for kind, ref_idx, hyp_idx in ops if ref_idx >= 0}
        en_ref = en_hit = 0
        for start, end in row.get("en_spans") or []:
            for ref_idx in range(start, end):
                en_ref += 1
                kind, hyp_idx = ref_to_op[ref_idx]
                if kind == "match":
                    en_hit += 1
                else:
                    got = hyp_words[hyp_idx] if hyp_idx >= 0 else None
                    eta_errors.append({"id": uid, "term": ref_words[ref_idx], "got": got})

        per_utterance[uid] = {
            "ref_words": len(ref_words),
            "sub": sub,
            "del": dele,
            "ins": ins,
            "en_ref": en_ref,
            "en_hit": en_hit,
        }

        category = row.get("category")
        if category is not None:
            category_groups.setdefault(str(category), []).append(uid)
        speaker = row.get("speaker")
        if speaker is not None:
            speaker_groups.setdefault(str(speaker), []).append(uid)
            speaker_group_groups.setdefault(_speaker_group(str(speaker)), []).append(uid)

    overall = _stats(list(per_utterance), per_utterance)
    overall["eta_errors"] = eta_errors
    if bootstrap > 0:
        errors = [_errors(per_utterance[uid]) for uid in per_utterance]
        refs = [per_utterance[uid]["ref_words"] for uid in per_utterance]
        overall["wer_ci"] = _bootstrap_ratio(errors, refs, bootstrap, seed)
    else:
        overall["wer_ci"] = None

    breakdown: dict[str, dict] = {}
    if category_groups:
        breakdown["category"] = {
            key: _stats(uids, per_utterance) for key, uids in category_groups.items()
        }
    if speaker_groups:
        breakdown["speaker"] = {
            key: _stats(uids, per_utterance) for key, uids in speaker_groups.items()
        }
        breakdown["speaker_group"] = {
            key: _stats(uids, per_utterance) for key, uids in speaker_group_groups.items()
        }

    return {
        "n_utterances": len(per_utterance),
        "overall": overall,
        "breakdown": breakdown,
        "noise": noise if noise["clips"] else None,
        "per_utterance": per_utterance,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.py",
        description="Score a hypothesis file against a reference manifest.",
    )
    parser.add_argument("--manifest", required=True, help="reference manifest JSONL")
    parser.add_argument("--hyp", required=True, help="hypothesis JSONL (id, hyp)")
    parser.add_argument("--out", default=None, help="output metrics.json (default: <hyp>.metrics.json)")
    parser.add_argument("--breakdown", default=None, help="comma list: category,speaker")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples (0 disables)")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--vs", default=None, help="another metrics.json for a paired difference")
    parser.add_argument("--config", default=None, help="run config path to record in metrics.json")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    hyp_path = Path(args.hyp)
    rows = _read_jsonl(manifest_path)
    hyps = {str(row["id"]): row.get("hyp", "") for row in _read_jsonl(hyp_path)}

    result = score_rows(rows, hyps, bootstrap=args.bootstrap, seed=args.seed)
    result |= {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "hyp": str(hyp_path),
        "hyp_sha256": _sha256(hyp_path),
        "normalizer": "stt.text.normalize",
        "normalizer_sha256": _sha256(Path(text_module.__file__)),
        "git_sha": _git_sha(),
        "config_path": args.config,
        "extra_hypotheses": len(set(hyps) - {str(row["id"]) for row in rows}),
    }

    if args.breakdown:
        wanted = {part.strip() for part in args.breakdown.split(",") if part.strip()}
        unknown = wanted - {"category", "speaker"}
        if unknown:
            parser.error(f"unknown breakdown sections: {', '.join(sorted(unknown))}")
        sections = result["breakdown"]
        keep: dict[str, dict] = {}
        if "category" in wanted and "category" in sections:
            keep["category"] = sections["category"]
        if "speaker" in wanted:
            if "speaker" in sections:
                keep["speaker"] = sections["speaker"]
            if "speaker_group" in sections:
                keep["speaker_group"] = sections["speaker_group"]
        result["breakdown"] = keep or None
    else:
        result["breakdown"] = None

    if args.vs:
        other = json.loads(Path(args.vs).read_text(encoding="utf-8"))
        if other.get("manifest_sha256") != result["manifest_sha256"]:
            print(f"error: {args.vs} was scored on a different manifest")
            return 2
        result["vs"] = _compare(result, other, bootstrap=args.bootstrap, seed=args.seed)

    out_path = Path(args.out) if args.out else hyp_path.with_name(hyp_path.name + ".metrics.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(result, out_path)
    return 0


# --------------------------------------------------------------------------------------------
# internals


def _speaker_group(speaker: str) -> str:
    return "S1+S2" if speaker in ("S1", "S2") else "S3+S4"


def _errors(stats: dict) -> int:
    return stats["sub"] + stats["del"] + stats["ins"]


def _stats(uids: list[str], per_utterance: dict[str, dict[str, int]]) -> dict:
    ref_words = sum(per_utterance[uid]["ref_words"] for uid in uids)
    sub = sum(per_utterance[uid]["sub"] for uid in uids)
    dele = sum(per_utterance[uid]["del"] for uid in uids)
    ins = sum(per_utterance[uid]["ins"] for uid in uids)
    en_ref = sum(per_utterance[uid]["en_ref"] for uid in uids)
    en_hit = sum(per_utterance[uid]["en_hit"] for uid in uids)
    errors = sub + dele + ins
    return {
        "n": len(uids),
        "ref_words": ref_words,
        "sub": sub,
        "del": dele,
        "ins": ins,
        "wer": errors / ref_words if ref_words else 0.0,
        "insertion_rate": 100.0 * ins / ref_words if ref_words else 0.0,
        "eta": en_hit / en_ref if en_ref else None,
    }


def _bootstrap_ratio(
    numerators: list[int], denominators: list[int], resamples: int, seed: int
) -> list[float] | None:
    """95% CI of the corpus ratio (Σnum / Σden) by resampling utterances."""
    size = len(numerators)
    if size == 0:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        num = den = 0
        for _ in range(size):
            idx = rng.randrange(size)
            num += numerators[idx]
            den += denominators[idx]
        values.append(num / den if den else 0.0)
    values.sort()
    lo = values[min(resamples - 1, int(_BOOTSTRAP_LO * resamples))]
    hi = values[min(resamples - 1, int(_BOOTSTRAP_HI * resamples))]
    return [lo, hi]


def _bootstrap_diff(
    num_a: list[int],
    den_a: list[int],
    num_b: list[int],
    den_b: list[int],
    resamples: int,
    seed: int,
) -> tuple[float, list[float] | None, bool]:
    """Paired bootstrap: point difference (ratio_a − ratio_b), 95% CI, and the tie flag."""
    ratio_a = sum(num_a) / sum(den_a) if sum(den_a) else 0.0
    ratio_b = sum(num_b) / sum(den_b) if sum(den_b) else 0.0
    point = ratio_a - ratio_b
    if resamples <= 0 or not num_a:
        return point, None, False
    rng = random.Random(seed)
    size = len(num_a)
    values: list[float] = []
    for _ in range(resamples):
        na = da = nb = db = 0
        for _ in range(size):
            idx = rng.randrange(size)
            na += num_a[idx]
            da += den_a[idx]
            nb += num_b[idx]
            db += den_b[idx]
        a = na / da if da else 0.0
        b = nb / db if db else 0.0
        values.append(a - b)
    values.sort()
    lo = values[min(resamples - 1, int(_BOOTSTRAP_LO * resamples))]
    hi = values[min(resamples - 1, int(_BOOTSTRAP_HI * resamples))]
    return point, [lo, hi], lo <= 0.0 <= hi


def _compare(result: dict, other: dict, bootstrap: int, seed: int) -> dict:
    ours, theirs = result["per_utterance"], other["per_utterance"]
    if set(ours) != set(theirs):
        raise ValueError("the two metrics.json files cover different utterance ids")
    ids = list(ours)
    wer_diff, wer_ci, wer_tie = _bootstrap_diff(
        [_errors(ours[uid]) for uid in ids],
        [ours[uid]["ref_words"] for uid in ids],
        [_errors(theirs[uid]) for uid in ids],
        [theirs[uid]["ref_words"] for uid in ids],
        bootstrap,
        seed,
    )
    eta_diff = eta_ci = None
    eta_tie = False
    en_ref_ours = sum(ours[uid]["en_ref"] for uid in ids)
    en_ref_theirs = sum(theirs[uid]["en_ref"] for uid in ids)
    if en_ref_ours and en_ref_theirs:
        eta_diff, eta_ci, eta_tie = _bootstrap_diff(
            [ours[uid]["en_hit"] for uid in ids],
            [ours[uid]["en_ref"] for uid in ids],
            [theirs[uid]["en_hit"] for uid in ids],
            [theirs[uid]["en_ref"] for uid in ids],
            bootstrap,
            seed,
        )
    return {
        "other": other.get("hyp"),
        "other_manifest_sha256": other.get("manifest_sha256"),
        "wer_diff": wer_diff,
        "wer_diff_ci": wer_ci,
        "wer_tie": wer_tie,
        "eta_diff": eta_diff,
        "eta_diff_ci": eta_ci,
        "eta_tie": eta_tie,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _print_summary(result: dict, out_path: Path) -> None:
    overall = result["overall"]
    wer = overall["wer"]
    ci = overall["wer_ci"]
    eta = overall["eta"]
    line = f"WER {wer:.4f}"
    if ci:
        line += f" [{ci[0]:.4f}, {ci[1]:.4f}]"
    if eta is not None:
        line += f"  ETA {eta:.4f}"
    line += f"  ins/100w {overall['insertion_rate']:.2f}  ({overall['ref_words']} ref words)"
    print(line)
    if overall["eta_errors"]:
        print(f"ETA errors: {len(overall['eta_errors'])}")
    if result.get("vs"):
        vs = result["vs"]
        print(f"vs {vs['other']}: WER diff {vs['wer_diff']:+.4f} (tie: {vs['wer_tie']})")
    if result.get("noise"):
        noise = result["noise"]
        print(
            f"noise: non-music words {noise['non_music_words']} "
            f"of {noise['non_music_clips']} clips; music words {noise['music_words']}"
        )
    print(f"metrics written to {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
