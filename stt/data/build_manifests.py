"""Build the real manifests from the raw datasets (design: ``docs/03-data.md``).

The M2 pipeline, concretely: the **dev/test sets first** (they define the overlap hashes), then the
**train sets**, sampled to their target hours and filtered against every dev/test text
(``norm_hash``). Audio-health checks (corrupt/silent/clipping) run later, during the mel shard
build, which decodes every file anyway (``stt/data/build_mel.py``).

Outputs (committed, they are the reproducible input of training/evaluation):
``data/manifests/<set>.jsonl`` + ``data/manifests/build_report.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from stt.data.filters import FilterConfig, apply_filters, build_report
from stt.data.sources import plan_records, sample_by_duration, write_records
from stt.manifest import norm_hash

__all__ = ["SetSpec", "build_manifests", "default_sets", "main"]


@dataclass(frozen=True)
class SetSpec:
    name: str
    source: str
    relative_root: str
    split: str
    target_hours: float | None = None


def default_sets(mls_hours: float = 170.0, ls_hours: float = 100.0) -> list[SetSpec]:
    """The M2 build list (``docs/03`` §2 targets); CV IT slots in here once downloaded."""
    return [
        SetSpec("fleurs_it_dev", "fleurs_it", "fleurs/it_it", "dev"),
        SetSpec("fleurs_it_test", "fleurs_it", "fleurs/it_it", "test"),
        SetSpec("fleurs_en_dev", "fleurs_en", "fleurs/en_us", "dev"),
        SetSpec("fleurs_en_test", "fleurs_en", "fleurs/en_us", "test"),
        SetSpec("mls_it_dev", "mls_it", "mls_it/mls_italian/dev", "dev"),
        SetSpec("mls_it_test", "mls_it", "mls_it/mls_italian/test", "test"),
        SetSpec("ls_en_dev", "librispeech", "librispeech/LibriSpeech/dev-clean", "dev"),
        SetSpec("mls_it_train", "mls_it", "mls_it/mls_italian/train", "train", mls_hours),
        SetSpec("ls_en_train", "librispeech", "librispeech/LibriSpeech/train-clean-100", "train", ls_hours),
    ]


def build_manifests(
    raw_root: Path | str,
    out_dir: Path | str,
    sets: list[SetSpec],
    config: FilterConfig | None = None,
    seed: int = 0,
    progress: object = None,
) -> dict:
    """Run the pipeline; returns the build report (also written as JSON)."""
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = config or FilterConfig()
    report: dict = {"sets": {}, "seed": seed}

    # Pass 1 — dev/test: filtered, then their hashes guard every train set.
    eval_hashes: set[str] = set()
    for spec in [s for s in sets if s.split != "train"]:
        rows = plan_records(spec.source, raw_root / spec.relative_root, spec.split)
        _prefix_audio(rows, spec.relative_root)
        kept, stats = apply_filters(rows, config)
        eval_hashes |= {norm_hash(row["text"]) for row in kept}
        _write_set(spec, rows, kept, stats, out_dir, report)

    # Pass 2 — train: sampled to target hours, then filtered (overlap included).
    for spec in [s for s in sets if s.split == "train"]:
        rows = plan_records(spec.source, raw_root / spec.relative_root, spec.split)
        _prefix_audio(rows, spec.relative_root)
        sampling = None
        if spec.target_hours:
            sample = sample_by_duration(rows, spec.target_hours * 3600.0, seed=seed)
            rows = sample["selected"]
            sampling = {
                "target_hours": spec.target_hours,
                "available_hours": round(sample["total_seconds"] / 3600.0, 1),
                "short": sample["short"],
            }
        kept, stats = apply_filters(rows, config, eval_hashes=eval_hashes)
        _write_set(spec, rows, kept, stats, out_dir, report, sampling)

    (out_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _prefix_audio(rows: list[dict], relative_root: str) -> None:
    """Make audio paths relative to the raw root: one ``--audio-root`` works for every set."""
    for row in rows:
        row["audio"] = f"{relative_root}/{row['audio']}"


def _write_set(
    spec: SetSpec,
    rows: list[dict],
    kept: list[dict],
    stats: dict,
    out_dir: Path,
    report: dict,
    sampling: dict | None = None,
) -> None:
    write_records(kept, out_dir / f"{spec.name}.jsonl")
    hours = stats["seconds"] / 3600.0
    entry = {
        "source": spec.source,
        "split": spec.split,
        "rows_after_sampling": len(rows),
        "kept": stats["kept"],
        "dropped": stats["dropped"],
        "hours": round(hours, 2),
    }
    if sampling:
        entry["sampling"] = sampling
    report["sets"][spec.name] = entry
    print(build_report(spec.name, {**stats, "total": len(rows)}, kept))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_manifests.py", description="Build the M2 manifests from data/raw."
    )
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-dir", default="data/manifests")
    parser.add_argument("--mls-hours", type=float, default=170.0)
    parser.add_argument("--ls-hours", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    specs = default_sets(args.mls_hours, args.ls_hours)
    build_manifests(args.raw_root, args.out_dir, specs, seed=args.seed)
    print(f"\nmanifests written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
