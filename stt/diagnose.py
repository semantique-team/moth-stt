"""No-training, reference-assisted audit (docs/12), not a production accuracy claim.

Run ``python -m stt.diagnose RELEASE --manifest refs.jsonl --out NEW_DIRECTORY``.
Relative audio paths use --audio-root (default: manifest directory). Supply mono 16 kHz
single utterances, 560 samples to 30 seconds; no resampling, truncation, VAD or segmentation.
--logits instead accepts an NPZ keyed by exact manifest IDs, each [T, 1025] raw logits
or log probabilities. These supplied arrays are normalized, but their model provenance
cannot be verified. No SentencePiece encoding is guessed from vocab.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from stt.decode import beam_search, greedy, load_pieces
from stt.eval import _git_sha, score_rows
from stt.lm import BinaryBPELM
from stt.manifest import load_manifest
from stt.release import validate_release
from stt.text import normalize


def log_softmax(values: np.ndarray) -> np.ndarray:
    """Normalize real [T, V] scores; -inf is allowed, NaN/+inf/all-impossible rows are not."""
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] < 1 or array.dtype.kind not in "fiu":
        raise ValueError("scores must be a real numeric [T, V] array")
    array = array.astype(np.float64)
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError("scores contain NaN or positive infinity")
    if np.isneginf(array).all(axis=1).any():
        raise ValueError("each frame must have a finite score")
    with np.errstate(over="ignore"):
        shifted = array - array.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _check_log_probs(log_probs: np.ndarray) -> np.ndarray:
    array = np.asarray(log_probs)
    if array.ndim != 2 or array.shape[1] < 1 or array.dtype.kind not in "fiu":
        raise ValueError("log_probs must be a real numeric [T, V] array")
    array = array.astype(np.float64)
    if (
        np.isnan(array).any()
        or np.isposinf(array).any()
        or not np.allclose(np.logaddexp.reduce(array, axis=1), 0.0, atol=1e-5, rtol=0)
    ):
        raise ValueError("log_probs must be normalized, without NaN/+inf or impossible frames")
    return array


def top_k_mask(log_probs: np.ndarray, top_k: int) -> np.ndarray:
    """Decoder candidate support: blank plus top-k nonblanks, lower ID wins ties."""
    array = _check_log_probs(log_probs)
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    mask = np.zeros(array.shape, dtype=bool)
    mask[:, 0] = True
    # Stable sorting preserves ascending IDs on equal scores, like stt.decode.
    ids = np.argsort(-array[:, 1:], axis=1, kind="stable")[:, :top_k] + 1
    np.put_along_axis(mask, ids, True, axis=1)
    return mask


def ctc_forward(log_probs: np.ndarray, labels: list[int], mask: np.ndarray | None = None) -> float:
    """Exact log sum of CTC paths collapsing to known nonblank IDs (blank=0).

    A mask removes frame/label paths without renormalizing probabilities. This tests
    candidate pruning only, NOT prefix-beam survival. Empty references/zero frames
    have the usual CTC semantics; impossible references return -inf, never NaN.
    """
    array = _check_log_probs(log_probs)
    if any(
        isinstance(label, (bool, np.bool_))
        or not isinstance(label, (int, np.integer))
        or not 1 <= label < array.shape[1]
        for label in labels
    ):
        raise ValueError("reference labels must be integer nonblank IDs inside the vocabulary")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != array.shape or mask.dtype != np.bool_:
            raise ValueError("mask must be boolean with the same shape as log_probs")
        array = np.where(mask, array, -np.inf)
    states = np.zeros(2 * len(labels) + 1, dtype=np.int64)
    states[1::2] = labels
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != 0) & (states[2:] != states[:-2])
    alpha = np.full(len(states), -np.inf)
    alpha[0] = 0.0
    for frame in array:
        step = np.full(len(states), -np.inf)
        step[1:] = alpha[:-1]
        total = np.logaddexp(alpha, step)
        step[:] = -np.inf
        step[2:] = np.where(skip[2:], alpha[:-2], -np.inf)
        alpha = np.logaddexp(total, step) + frame[states]
    return float(np.logaddexp(alpha[-1], alpha[-2])) if labels else float(alpha[0])


def reference_support(log_probs: np.ndarray, labels: list[int], top_k: int) -> dict:
    """Oracle reference likelihood and support under frame candidate pruning."""
    full = ctc_forward(log_probs, labels)
    restricted = ctc_forward(log_probs, labels, top_k_mask(log_probs, top_k))
    full_possible, restricted_possible = math.isfinite(full), math.isfinite(restricted)
    return {
        "ctc_ids": labels,
        "full_log_probability": full if full_possible else None,
        "top_k_log_probability": restricted if restricted_possible else None,
        "full_reachable": full_possible,
        "top_k_reachable": restricted_possible,
        "retained_probability_fraction": (
            min(1.0, math.exp(restricted - full)) if full_possible else None
        ),
        "log_probability_loss": full - restricted if restricted_possible else None,
        "all_reference_mass_pruned": full_possible and not restricted_possible,
    }


def verify_tokenizer(processor: object, pieces: dict[int, str]) -> None:
    """Check every SP ID, including unk, against all 1024 release pieces."""
    if (
        processor.get_piece_size() != 1024
        or set(pieces) != set(range(1, 1025))
        or any(processor.id_to_piece(i) != pieces[i + 1] for i in range(1024))
    ):
        raise ValueError("SentencePiece must match all 1024 release piece IDs exactly")


def diagnose_utterance(
    log_probs: np.ndarray,
    row: dict,
    pieces: dict[int, str],
    settings: dict,
    lm: object,
    processor: object | None = None,
) -> tuple[dict[str, str], dict]:
    """Decode the same frames in three frozen configurations and optionally score refs."""
    array = _check_log_probs(log_probs)
    if not 1 <= len(array) <= 1500 or array.shape[1] != len(pieces) + 1:
        raise ValueError("expected 1..1500 frames and one output per piece plus blank")
    common = {"beam_width": settings["beam"], "top_k": settings["top_k"]}
    hint = row["lang"] if row["lang"] in ("it", "en") else "it"
    hypotheses = {
        "greedy": greedy(array, len(array), pieces),
        "beam_acoustic0": beam_search(array, len(array), pieces, **common),
        "beam_lm": beam_search(
            array,
            len(array),
            pieces,
            **common,
            lm=lm,
            lm_weight=settings["lm_weight"],
            token_bonus=settings["token_bonus"],
        ),
    }
    hypotheses = {name: normalize(hyp, hint) for name, hyp in hypotheses.items()}
    words = row["norm"].split()
    english_indices = [i for start, end in row["en_spans"] for i in range(start, end)]
    detail = {
        "id": row["id"],
        "reference": row["norm"],
        "frames": len(array),
        "hypotheses": hypotheses,
        "english_word_presence_not_eta": {
            name: {
                "reference_word_occurrences": len(english_indices),
                "present_anywhere": sum(words[i] in hyp.split() for i in english_indices),
            }
            for name, hyp in hypotheses.items()
        },
        "forced_reference": {"available": False, "reason": "no verified SentencePiece model"},
    }
    if processor is not None:
        verify_tokenizer(processor, pieces)
        sp_ids = processor.encode(row["norm"], out_type=int)
        if processor.unk_id() in sp_ids:
            detail["forced_reference"]["reason"] = "reference contains SentencePiece unk"
        elif processor.decode(sp_ids) != row["norm"]:
            detail["forced_reference"]["reason"] = "reference does not round-trip exactly"
        else:
            detail["forced_reference"] = {
                "available": True,
                **reference_support(array, [i + 1 for i in sp_ids], settings["top_k"]),
            }
    return hypotheses, detail


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--out", required=True, type=Path, help="new output directory; never overwrite"
    )
    parser.add_argument("--audio-root", type=Path, help="relative audio path root")
    parser.add_argument(
        "--tokenizer", type=Path, help="matching SentencePiece .model, forced refs only"
    )
    parser.add_argument(
        "--logits", type=Path, help="NPZ keyed by manifest ID; no audio/model execution"
    )
    parser.add_argument("--budget-bytes", type=int, default=20_000_000)
    args = parser.parse_args(argv)
    try:
        config, release_report = validate_release(args.release, args.budget_bytes)
        rows = load_manifest(args.manifest)
        if not rows:
            raise ValueError("manifest must contain existing references")
        for row in rows:
            if not isinstance(row.get("en_spans"), list):
                raise ValueError(  # noqa: TRY004 - invalid manifest data
                    f"{row['id']}: explicit en_spans list required (may be empty)"
                )
            if not math.isfinite(row["duration"]) or not 0 < row["duration"] <= 30:
                raise ValueError(f"{row['id']}: duration must be finite, positive and <=30 seconds")
            indices = [i for start, end in row["en_spans"] for i in range(start, end)]
            if len(set(indices)) != len(indices):
                raise ValueError(f"{row['id']}: overlapping en_spans")
        if args.out.exists():
            raise ValueError("output directory already exists; refusing to overwrite")
        root = Path(release_report["release"])
        pieces = load_pieces(root / config["tokenizer"]["vocab"])
        if set(pieces) != set(range(1, 1025)) or any(
            not isinstance(piece, str) for piece in pieces.values()
        ):
            raise ValueError("release must contain 1024 string pieces at CTC IDs 1..1024")
        processor = None
        if args.tokenizer is not None:
            import sentencepiece as spm

            processor = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
            verify_tokenizer(processor, pieces)
        settings = config["decode"]
        lm = BinaryBPELM.load(root / settings["lm"])
        archive = None
        audio_paths = {}
        if args.logits is not None:
            archive = np.load(args.logits, allow_pickle=False)
            if not isinstance(archive, np.lib.npyio.NpzFile):
                raise ValueError("--logits must be an NPZ archive")
            if len(archive.files) != len(rows) or set(archive.files) != {r["id"] for r in rows}:
                archive.close()
                raise ValueError("NPZ keys must match manifest IDs exactly")
        else:
            from stt.features import load_mel_filters, log_mel
            from stt.onnx_runner import OnnxMoth

            audio_root = (args.audio_root or args.manifest.parent).resolve()
            for row in rows:
                if not row.get("audio"):
                    raise ValueError(f"{row['id']}: audio path required")
                path = (audio_root / row["audio"]).resolve()
                if not path.is_file():
                    raise ValueError(f"{row['id']}: audio file missing: {path}")
                audio_paths[row["id"]] = path
            filters = load_mel_filters(root / config["features"]["mel_filters"])
            if not np.isfinite(filters).all():
                raise ValueError("release mel filters must be finite")
            model = OnnxMoth(root / config["model"]["file"], root / config["tokenizer"]["vocab"])
            if (model.input_name, model.output_name) != (
                config["model"]["input"],
                config["model"]["output"],
            ):
                raise ValueError("ONNX tensor names differ from release config")
        hypotheses = {name: {} for name in ("greedy", "beam_acoustic0", "beam_lm")}
        details = []
        try:
            for row in rows:
                if archive is not None:
                    array = log_softmax(archive[row["id"]])
                else:
                    import soundfile as sf

                    audio, rate = sf.read(audio_paths[row["id"]], dtype="float32")
                    if (
                        rate != 16000
                        or audio.ndim != 1
                        or not 560 <= len(audio) <= 480000
                        or not np.isfinite(audio).all()
                        or np.any(np.abs(audio) > 1)
                    ):
                        raise ValueError(
                            f"{row['id']}: expected mono finite PCM [-1,1], 16 kHz, <=30s"
                        )
                    array = model.log_probs(log_mel(audio, filters=filters))
                hyps, detail = diagnose_utterance(array, row, pieces, settings, lm, processor)
                if archive is None:
                    detail["audio"] = str(audio_paths[row["id"]])
                    detail["audio_sha256"] = _sha256(audio_paths[row["id"]])
                    detail["audio_seconds"] = len(audio) / rate
                for name, hyp in hyps.items():
                    hypotheses[name][row["id"]] = hyp
                details.append(detail)
        finally:
            if archive is not None:
                archive.close()
        report = {
            "release": release_report,
            "version": config["version"],
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": _sha256(args.manifest),
            "normalizer": "stt.text.normalize",
            "code_sha256": {
                name: _sha256(Path(__file__).with_name(name))
                for name in ("diagnose.py", "decode.py", "eval.py", "features.py", "text.py")
            },
            "git_sha": _git_sha(),
            "tokenizer": (
                {"path": str(args.tokenizer.resolve()), "sha256": _sha256(args.tokenizer)}
                if args.tokenizer
                else None
            ),
            "input": (
                {
                    "kind": "supplied_logits_unverified",
                    "path": str(args.logits.resolve()),
                    "sha256": _sha256(args.logits),
                }
                if args.logits
                else {"kind": "release_onnx_audio"}
            ),
            "decode": {
                "greedy": {"method": "greedy"},
                "beam_acoustic0": {
                    "method": "beam",
                    "beam": settings["beam"],
                    "top_k": settings["top_k"],
                    "lm_weight": 0,
                    "token_bonus": 0,
                },
                "beam_lm": {
                    "method": "beam",
                    "beam": settings["beam"],
                    "top_k": settings["top_k"],
                    "lm_weight": settings["lm_weight"],
                    "token_bonus": settings["token_bonus"],
                    "lm": settings["lm"],
                },
            },
            "limitations": [
                "No training, tuning, invented labels or independent quality claim.",
                "References and en_spans are supplied annotations, never inferred or repaired.",
                "No VAD gating or segmentation; each row is assumed to be one utterance <=30s.",
                "beam_acoustic0 vs beam_lm changes BOTH LM weight and token bonus; not causal LM isolation.",
                "Reference CTC likelihood/top-k support uses oracle refs, not deployable accuracy.",
                "Top-k support is not prefix-beam survival; prefix survival and N-best are NOT implemented.",
                "No Viterbi alignment/timestamps; exact forward sums all paths for one reference encoding.",
                "Reachability or low reference likelihood is not proof of an acoustic/model-capacity cause.",
                "Term presence anywhere is not aligned ETA; only stt.eval scores are canonical metrics.",
                "Supplied logits are not evidence that the validated release produced those arrays.",
            ],
            "metrics": {
                name: score_rows(rows, hyps, bootstrap=0) for name, hyps in hypotheses.items()
            },
            "utterances": details,
        }
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.out.mkdir(parents=True, exist_ok=False)
        for name, hyps in hypotheses.items():
            (args.out / f"{name}.hyp.jsonl").write_text(
                "".join(json.dumps({"id": uid, "hyp": hyp}) + "\n" for uid, hyp in hyps.items()),
                encoding="utf-8",
            )
        (args.out / "report.json").write_text(payload, encoding="utf-8")
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps({"ok": True, "report": str((args.out / "report.json").resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
