"""Opt-in, inference-only finalist evaluation (docs/12, A1/A3/A4).

Integration API: load_evaluation_config(run_config), build_dev_sets(run_config),
evaluate_model(model, dev_sets, pieces, settings, lm=...), or evaluate_log_probs
with {set_name: {utterance_id: [T, V] ndarray}} (already trimmed to true lengths).
Both return train-compatible wer/wer_ci/n_utterances/sets plus canonical metrics.
No training loop, checkpoint writes, auxiliary heads, or export are involved.

Example additions to an existing run YAML (paths relative to the working directory)::

    data:
      train: [{manifest: snapshots/train.jsonl, shards: data/shards/train}]
      dev: [{manifest: snapshots/it-dev.jsonl, shards: data/shards/it-dev}]
      dev_en: [{manifest: snapshots/en-dev.jsonl, shards: data/shards/en-dev}]
      dev_mixed: [{manifest: snapshots/mixed-dev.jsonl, shards: data/shards/mixed-dev}]
    evaluation:
      selection_metric: mixed_wer
      decoder:
        mode: beam
        lm_path: release/lm_bpe3.bin
        lm_sha256: <64 lowercase hex characters>
        beam_width: 12
        top_k: 10
        lm_weight: 0.5
        token_bonus: 2.4
      wer_margin: 0.005
      min_eta: 0.0
      insertion_margin: 0.0

Without evaluation, settings are greedy/macro_wer; opt-in defaults to release beam.
Macro averages IT/EN only, never mixed. Mixed rows require English annotations except
control_it. Baseline gates require IT+EN, allow +0.005 absolute WER by default,
and require mixed ETA >= max(min_eta, baseline ETA) and mixed insertion_rate <=
baseline + insertion_margin (words per 100 reference words). WER/ETA are ratios.
Strictly lower primary WER wins; float ties (absolute tolerance 1e-12) keep baseline,
then lexicographically smallest candidate name. These are point gates, not CI claims.

Run only on a few final contenders, not every training step::

    python -m stt.training_eval --config run.yaml --checkpoint candidate.pt --out eval.json
    python -m stt.training_eval --config run.yaml --check-config --out preflight.json
    python -m stt.training_eval --config run.yaml --checkpoint a.pt --checkpoint b.pt \
        --baseline-metrics baseline.json --out finalists.json

Checkpoint inputs are trusted local torch training payloads (including RNG pickle).
Only payload['model'] is loaded, strictly, into the shippable MothSTT. All candidates
must match the configured architecture; their declared train manifests are also checked.
The explicit train inventory must include historical acoustic exposure: these guards
cannot discover omitted data, near duplicates, consent, or renamed unlabelled sources.
This is FP32 checkpoint evaluation, not a reproduction of the release INT8/VAD path.
Batch padding is preserved and fingerprinted, including actual plans and mel lengths.
Cached evaluations accept acoustic_provenance; its explicit default is an external
cache with unspecified batching, never implicitly the model evaluator's batch plan.
Model provenance hashes each unpadded consumed mel as C-order little-endian float32,
with shape/dtype recorded. No shard rereads or candidate-dependent logit hashes are used.
Cached provenance is caller-trusted: include acoustic content identity and batch convention,
and explicitly attest acoustic_provenance_verified=true inside acoustic_provenance to rank.
Absent that attestation, scoring works but comparative ranking is refused.
The config must declare the original SentencePiece tokenizer. Optional vocab.json
must match its ID-to-piece mapping exactly. Checkpoints must pin tokenizer_sha256
or retain a readable declared tokenizer file; matching vocabulary sizes is insufficient.
The binary LM format only encodes vocabulary size, not tokenizer identity: the pinned
LM's training-tokenizer compatibility remains an operator attestation.
--check-config validates config, tokenizer, LM, split guards and file inventory without
loading checkpoints, instantiating a model, or running forward. No training is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from stt import decode as decode_module
from stt import eval as eval_module
from stt import text as text_module
from stt.decode import beam_search, greedy, load_pieces
from stt.eval import score_rows
from stt.lm import BinaryBPELM
from stt.manifest import load_manifest
from stt.text import normalize


@dataclass(frozen=True)
class DecoderConfig:
    mode: str = "greedy"
    beam_width: int = 12
    top_k: int = 10
    lm_weight: float = 0.5
    token_bonus: float = 2.4
    lm_path: str | None = None
    lm_sha256: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    selection_metric: str = "macro_wer"
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    wer_margin: float = 0.005
    min_eta: float = 0.0
    insertion_margin: float = 0.0


def _number(name, value, minimum=0.0, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    return value


def load_evaluation_config(config: Mapping) -> EvaluationConfig:
    """Validate settings without loading assets; reject typos and coercion of strings/bools."""
    raw = config.get("evaluation", {})
    if not isinstance(raw, Mapping):
        raise TypeError("evaluation must be a mapping")
    unknown = set(raw) - set(EvaluationConfig.__dataclass_fields__)
    decoder = raw.get("decoder", {})
    if not isinstance(decoder, Mapping):
        raise TypeError("evaluation.decoder must be a mapping")
    unknown |= set(decoder) - set(DecoderConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown evaluation settings: {sorted(unknown)}")
    dec = DecoderConfig(**({"mode": "beam"} | dict(decoder) if "evaluation" in config else {}))
    if dec.mode not in ("greedy", "beam"):
        raise ValueError("decoder mode must be greedy or beam")
    for name in ("beam_width", "top_k"):
        value = getattr(dec, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("lm_weight", "token_bonus"):
        _number(name, getattr(dec, name))
    if dec.mode == "beam":
        if not isinstance(dec.lm_path, str) or not dec.lm_path:
            raise ValueError("beam requires lm_path to a binary LM")
        if not isinstance(dec.lm_sha256, str) or not re.fullmatch("[0-9a-f]{64}", dec.lm_sha256):
            raise ValueError("beam requires pinned lm_sha256 (64 lowercase hex characters)")
    settings = EvaluationConfig(**(dict(raw) | {"decoder": dec}))
    if settings.selection_metric not in ("macro_wer", "mixed_wer"):
        raise ValueError("selection_metric must be macro_wer or mixed_wer")
    _number("wer_margin", settings.wer_margin)
    _number("min_eta", settings.min_eta, maximum=1.0)
    _number("insertion_margin", settings.insertion_margin)
    return settings


def load_lm(settings: EvaluationConfig) -> BinaryBPELM | None:
    """Verify the exact bytes before parsing MOTHLMB1; never load a pickle LM."""
    if settings.decoder.mode == "greedy":
        return None
    data = Path(settings.decoder.lm_path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != settings.decoder.lm_sha256:
        raise ValueError("LM sha256 mismatch")
    lm = BinaryBPELM(data)
    lm.sha256 = digest
    return lm


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_evaluation_tokenizer(config, lm=None):
    """Use tokenizer bytes as identity and its actual mapping as decoder vocabulary."""
    from stt.tokenizer import load_tokenizer

    if not config.get("tokenizer"):
        raise ValueError("evaluation requires a readable tokenizer source file")
    digest = _sha256(config["tokenizer"])
    processor = load_tokenizer(config["tokenizer"])
    pieces = {i + 1: processor.id_to_piece(i) for i in range(processor.vocab_size())}
    if len(pieces) != config["model"]["vocab_size"]:
        raise ValueError("tokenizer/model vocabulary mismatch")
    if config.get("vocab") and load_pieces(config["vocab"]) != pieces:
        raise ValueError("vocab ID-to-piece mapping differs from tokenizer")
    if lm is not None and len(pieces) + 2 != len(lm.unigram):
        raise ValueError("binary LM/tokenizer vocabulary mismatch")
    return pieces, digest


def check_config(config):
    """Preflight data and file inventory only; never instantiate an acoustic model."""
    settings = load_evaluation_config(config)
    dev_sets = build_dev_sets(config)
    lm = load_lm(settings)
    pieces, digest = load_evaluation_tokenizer(config, lm)
    _number("max_seconds", config.get("batch", {}).get("max_seconds", 192), minimum=1e-12)
    clips = config.get("batch", {}).get("max_clips", 32)
    if type(clips) is not int or clips <= 0:
        raise ValueError("max_clips must be a positive integer")
    if config.get("train", {}).get("input", "truncated") not in ("truncated", "padded30"):
        raise ValueError("unknown input_mode")
    paths = {Path(config["tokenizer"])}
    if config.get("vocab"):
        paths.add(Path(config["vocab"]))
    if lm is not None:
        paths.add(Path(settings.decoder.lm_path))
    for key in ("train", "dev", "dev_en", "dev_mixed"):
        for src in config["data"].get(key, []):
            paths.add(Path(src["manifest"]))
            if key != "train":
                paths.add(Path(src["shards"]) / "index.json")
    for _, dataset in dev_sets:
        paths.update(Path(entry["shard"]) for entry in dataset.index.values())
    inventory = {str(path): _sha256(path) for path in sorted(paths)}
    return {
        "checked": True,
        "tokenizer_sha256": digest,
        "vocab_size": len(pieces),
        "lm_sha256": lm.sha256 if lm is not None else None,
        "sets": {name: len(dataset.rows) for name, dataset in dev_sets},
        "files_sha256": inventory,
    }


def validate_dev_rows(rows: list[dict], train_rows: list[dict], *, mixed=False) -> None:
    """Pure split/role, exact ID/text overlap and mixed ETA annotation guard."""
    if not rows:
        raise ValueError("empty dev set")
    train_ids = {row["id"] for row in train_rows}
    train_norms = {normalize(row["norm"]) for row in train_rows if row["norm"]}
    seen = set()
    for row in rows:
        uid = row["id"]
        source = str(row.get("source", "")).lower()
        role = str(row.get("role", "dev")).lower()
        if (
            row.get("split") != "dev"
            or role != "dev"
            or source.startswith(("recorded_t", "recorded_a", "recorded_r"))
            or re.match(r"^[tar](?:[-_]|$)", str(uid), re.IGNORECASE)
            or str(row.get("speaker", "")).upper() == "S3"
        ):
            raise ValueError(f"{uid}: forbidden dev split/role or historical T/A/R/S3 reuse")
        if uid in seen:
            raise ValueError(f"duplicate dev id: {uid}")
        seen.add(uid)
        norm = normalize(row["norm"])
        if not norm or norm != row["norm"]:
            raise ValueError(f"{uid}: dev norm must be nonempty and canonical")
        if uid in train_ids or norm in train_norms:
            raise ValueError(f"{uid}: exact train ID/norm overlap")
        if mixed:
            if row.get("lang") != "mix":
                raise ValueError(f"{uid}: mixed dev requires lang=mix")
            spans = row.get("en_spans")
            if not isinstance(spans, list) or (not spans and row.get("category") != "control_it"):
                raise ValueError(f"{uid}: missing ETA en_spans annotations")
            previous = 0
            for span in spans:
                if (
                    not isinstance(span, list)
                    or len(span) != 2
                    or any(type(x) is not int for x in span)
                    or not previous <= span[0] < span[1] <= len(norm.split())
                ):
                    raise ValueError(f"{uid}: ETA spans must be sorted, nonoverlapping, nonempty")
                previous = span[1]
    if mixed and not any(row.get("en_spans") for row in rows):
        raise ValueError("mixed dev has no annotated English terms for ETA")


def build_dev_sets(config: Mapping) -> list[tuple[str, object]]:
    """Load existing manifest/shards sources without importing or invoking stt.train.

    data.train is a required explicit acoustic exposure inventory; only its manifests
    are read. Sources need not share a shard root. Returned datasets never augment.
    """
    from stt.data.build_mel import load_index
    from stt.data.dataset import MelDataset

    settings = load_evaluation_config(config)
    data = config["data"]
    if not data.get("train"):
        raise ValueError("data.train must explicitly inventory training sources")
    train_rows = [row for src in data["train"] for row in load_manifest(src["manifest"])]
    sets = []
    all_rows = []
    for name, key in (("it", "dev"), ("en", "dev_en"), ("mixed", "dev_mixed")):
        if not data.get(key):
            continue
        rows, index = [], {}
        for src in data[key]:
            source_rows = load_manifest(src["manifest"])
            root = Path(src["shards"]).resolve()
            source_index = load_index(root / "index.json")
            for row in source_rows:
                uid = row["id"]
                if uid in index:
                    raise ValueError(f"duplicate dev id: {uid}")
                entry = source_index[uid]
                index[uid] = dict(entry, shard=str(root / entry["shard"]))
            rows.extend(source_rows)
        validate_dev_rows(rows, train_rows, mixed=name == "mixed")
        if name != "mixed" and any(row["lang"] != name for row in rows):
            raise ValueError(f"{name} dev contains wrong-language rows")
        all_rows.extend(rows)
        sets.append((name, MelDataset(index, Path("."), rows, augment=None)))
    validate_dev_rows(all_rows, train_rows)
    if settings.selection_metric == "mixed_wer" and not any(name == "mixed" for name, _ in sets):
        raise ValueError("mixed_wer requires data.dev_mixed")
    return sets


def _protocol(rows_by_set, pieces, settings, input_mode, acoustic_provenance, lm_sha256):
    # Include metadata used for scoring, normalizer, decoder and acoustic input convention.
    payload = {
        "rows": rows_by_set,
        "pieces": sorted(pieces.items()),
        "settings": asdict(settings),
        "input_mode": input_mode,
        "acoustic_provenance": acoustic_provenance,
        "lm_sha256": lm_sha256,
        "code_sha256": {
            name: _sha256(path)
            for name, path in {
                "training_eval": __file__,
                "decode": decode_module.__file__,
                "eval": eval_module.__file__,
                "model": Path(__file__).with_name("model.py"),
                "dataset": Path(__file__).parent / "data" / "dataset.py",
                "lm": Path(__file__).with_name("lm.py"),
                "hotwords": Path(__file__).with_name("hotwords.py"),
                "tokenizer": Path(__file__).with_name("tokenizer.py"),
            }.items()
        },
        "normalizer_sha256": hashlib.sha256(Path(text_module.__file__).read_bytes()).hexdigest(),
    }
    return payload


def evaluate_log_probs(
    rows_by_set: Mapping[str, list[dict]],
    log_probs: Mapping,
    pieces: Mapping[int, str],
    settings: EvaluationConfig,
    *,
    lm=None,
    bootstrap=0,
    input_mode="truncated",
    acoustic_provenance=None,
) -> dict:
    """Decode complete cached outputs with official decoder and canonical scorer.

    Call validate_dev_rows with the training inventory before using cached outputs.
    Arrays must contain exactly the valid acoustic frames, not batch padding.
    Supplied acoustic provenance is an external attestation, not verified from logits.
    """
    if input_mode not in ("truncated", "padded30"):
        raise ValueError("input_mode must be truncated or padded30")
    if set(rows_by_set) != set(log_probs) or not rows_by_set:
        raise ValueError("cached output sets must exactly match nonempty dev sets")
    if settings.decoder.mode == "beam" and (
        not isinstance(lm, BinaryBPELM)
        or getattr(lm, "sha256", None) != settings.decoder.lm_sha256
        or not isinstance(settings.decoder.lm_sha256, str)
    ):
        raise ValueError("beam evaluation requires verified binary LM from load_lm")
    if acoustic_provenance is None:
        acoustic_provenance = {"producer": "external_cache", "batch_plan": "unspecified"}
    provenance_verified = acoustic_provenance.get("acoustic_provenance_verified") is True
    per_set = {}
    for name, rows in rows_by_set.items():
        if name not in ("it", "en", "mixed"):
            raise ValueError(f"unknown dev set: {name}")
        validate_dev_rows(rows, [], mixed=name == "mixed")
        cache = log_probs[name]
        if set(cache) != {row["id"] for row in rows}:
            raise ValueError(f"{name}: missing or extra cached utterance IDs")
        hyps = {}
        for uid, value in cache.items():
            array = np.asarray(value)
            if (
                array.ndim != 2
                or array.shape[0] == 0
                or set(pieces) != set(range(1, array.shape[1]))
                or np.isnan(array).any()
                or np.isposinf(array).any()
                or not np.isfinite(array).any(axis=1).all()
            ):
                raise ValueError(f"{uid}: invalid log_probs shape/vocab/scores")
            dec = settings.decoder
            hyps[uid] = (
                greedy(array, len(array), pieces)
                if dec.mode == "greedy"
                else beam_search(
                    array,
                    len(array),
                    pieces,
                    beam_width=dec.beam_width,
                    top_k=dec.top_k,
                    lm=lm,
                    lm_weight=dec.lm_weight,
                    token_bonus=dec.token_bonus,
                )
            )
        scored = score_rows(rows, hyps, bootstrap=bootstrap, seed=0)
        per_set[name] = dict(
            scored,
            wer=scored["overall"]["wer"],
            wer_ci=scored["overall"]["wer_ci"],
            hypotheses=hyps,
        )
    macro = [per_set[name]["wer"] for name in ("it", "en") if name in per_set]
    metric = settings.selection_metric
    if metric == "mixed_wer" and "mixed" not in per_set:
        raise ValueError("mixed_wer requires mixed dev metrics")
    if metric == "macro_wer" and not macro:
        raise ValueError("macro_wer requires IT or EN dev")
    selected = per_set["mixed"]["wer"] if metric == "mixed_wer" else sum(macro) / len(macro)
    primary = (
        per_set["mixed"]
        if metric == "mixed_wer"
        else (
            per_set[next(name for name in ("it", "en") if name in per_set)]
            if len(macro) == 1
            else None
        )
    )
    protocol = _protocol(
        rows_by_set,
        pieces,
        settings,
        input_mode,
        acoustic_provenance,
        lm.sha256 if settings.decoder.mode == "beam" else None,
    )
    return {
        "wer": selected,
        "wer_ci": primary["wer_ci"] if primary else None,
        "selection_metric": metric,
        "macro_wer": sum(macro) / len(macro) if macro else None,
        "mixed_wer": per_set.get("mixed", {}).get("wer"),
        "n_utterances": sum(m["n_utterances"] for m in per_set.values()),
        "sets": per_set,
        "protocol": protocol,
        "acoustic_provenance_verified": provenance_verified,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }


def evaluate_model(
    model,
    dev_sets,
    pieces,
    settings: EvaluationConfig,
    *,
    lm=None,
    device="cpu",
    max_seconds=192.0,
    max_clips=32,
    input_mode="truncated",
    bootstrap=0,
) -> dict:
    """Inference-only forward; restore mode even on failure. Never crop reference audio."""
    import torch

    from stt.data.dataset import collate, plan_batches

    _number("max_seconds", max_seconds, minimum=1e-12)
    if type(max_clips) is not int or max_clips < 1:
        raise ValueError("max_clips must be a positive integer")
    if input_mode not in ("truncated", "padded30"):
        raise ValueError("unknown input_mode")
    rows_by_set, cache = {}, {}
    provenance = {
        "producer": "evaluate_model",
        "max_seconds": max_seconds,
        "max_clips": max_clips,
        "padding": "zero-right-batch-max",
        "plans": {},
        "mel_inputs": {},
        "acoustic_provenance_verified": True,
    }
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for name, dataset in dev_sets:
                if name in rows_by_set or dataset.augment_config is not None:
                    raise ValueError("duplicate dev set or augmented evaluation dataset")
                rows_by_set[name] = dataset.rows
                cache[name] = {}
                batches = plan_batches(
                    [float(r["duration"]) for r in dataset.rows],
                    max_seconds=max_seconds,
                    max_clips=max_clips,
                )
                provenance["plans"][name] = []
                provenance["mel_inputs"][name] = {}
                for positions in batches:
                    tensor, lengths, rows = collate([dataset[i] for i in positions])
                    if tensor.shape[-1] > 3000:
                        raise ValueError("dev audio exceeds 3000 mel frames; refusing to crop")
                    for i, row in enumerate(rows):
                        mel = np.ascontiguousarray(
                            tensor[i, :, : int(lengths[i])].numpy(), dtype="<f4"
                        )
                        provenance["mel_inputs"][name][row["id"]] = {
                            "shape": list(mel.shape),
                            "dtype": "<f4",
                            "sha256": hashlib.sha256(mel.tobytes(order="C")).hexdigest(),
                        }
                    if input_mode == "padded30":
                        tensor = torch.nn.functional.pad(tensor, (0, 3000 - tensor.shape[-1]))
                    provenance["plans"][name].append(
                        {
                            "ids": [row["id"] for row in rows],
                            "mel_lengths": lengths.tolist(),
                            "padded_frames": tensor.shape[-1],
                        }
                    )
                    probs, out_lengths = model(tensor.to(device), lengths.to(device))
                    for i, row in enumerate(rows):
                        length = int(out_lengths[i])
                        if not 0 < length <= probs.shape[1]:
                            raise ValueError("invalid model output length")
                        cache[name][row["id"]] = probs[i, :length].float().cpu().numpy()
    finally:
        model.train(was_training)
    return evaluate_log_probs(
        rows_by_set,
        cache,
        pieces,
        settings,
        lm=lm,
        bootstrap=bootstrap,
        input_mode=input_mode,
        acoustic_provenance=provenance,
    )


def rank_candidates(
    candidates: Mapping[str, dict],
    settings: EvaluationConfig,
    *,
    baseline: dict | None = None,
    baseline_name="baseline",
    baseline_source=None,
) -> dict:
    """Pure deterministic ranking/gating; requires matching evaluation fingerprints."""
    if not candidates or baseline_name in candidates:
        raise ValueError("need candidates with names distinct from baseline")
    inputs = [*candidates.values(), *([baseline] if baseline is not None else [])]
    if any(metrics.get("acoustic_provenance_verified") is not True for metrics in inputs):
        raise ValueError("ranking requires verified acoustic provenance for every input")
    reference = baseline if baseline is not None else next(iter(candidates.values()))
    fingerprint = reference.get("protocol_sha256")
    if not fingerprint:
        raise ValueError("metrics require protocol_sha256")
    for metrics in inputs:
        if (
            metrics.get("protocol_sha256") != fingerprint
            or metrics.get("selection_metric") != settings.selection_metric
        ):
            raise ValueError("metrics use different dev/decoder/selection protocols")
        _number("primary WER", metrics["wer"])
        for name, result in metrics["sets"].items():
            _number(f"{name} WER", result["wer"])
            if name == "mixed":
                _number("mixed ETA", result["overall"]["eta"], maximum=1)
                _number("mixed insertions", result["overall"]["insertion_rate"])
        if settings.selection_metric == "mixed_wer" and "mixed" not in metrics["sets"]:
            raise ValueError("missing mixed ETA/WER metrics")
        if baseline is not None and not {"it", "en"} <= set(metrics["sets"]):
            raise ValueError("baseline gates require both IT and EN dev metrics")
    entries = []
    for name, metrics in candidates.items():
        reasons = []
        mixed = metrics["sets"].get("mixed")
        if mixed and mixed["overall"]["eta"] < settings.min_eta:
            reasons.append("mixed ETA below minimum")
        if baseline is not None:
            if set(metrics["sets"]) != set(baseline["sets"]):
                raise ValueError("baseline and candidate sets differ")
            for lang in ("it", "en"):
                if (
                    metrics["sets"][lang]["wer"]
                    > baseline["sets"][lang]["wer"] + settings.wer_margin
                ):
                    reasons.append(f"{lang} WER regression")
            if mixed:
                base = baseline["sets"]["mixed"]["overall"]
                if mixed["overall"]["eta"] < base["eta"]:
                    reasons.append("mixed ETA regression")
                if (
                    mixed["overall"]["insertion_rate"]
                    > base["insertion_rate"] + settings.insertion_margin
                ):
                    reasons.append("mixed insertion regression")
        entries.append(
            {"name": name, "wer": metrics["wer"], "eligible": not reasons, "reasons": reasons}
        )
    entries.sort(key=lambda e: (not e["eligible"], e["wer"], e["name"]))
    winner = baseline_name if baseline is not None else None
    best = baseline["wer"] if baseline is not None else math.inf
    # Name order resolves near-equal candidates without depending on caller order.
    for entry in sorted(entries, key=lambda e: e["name"]):
        value = entry["wer"]
        if (
            entry["eligible"]
            and value < best
            and not math.isclose(value, best, rel_tol=0, abs_tol=1e-12)
        ):
            winner, best = entry["name"], value
    return {
        "selected": winner,
        "ranking": entries,
        "baseline_metrics": baseline,
        "baseline_source": baseline_source,
        "gates": {
            "baseline": baseline_name if baseline is not None else None,
            "wer_margin": settings.wer_margin,
            "min_eta": settings.min_eta,
            "insertion_margin": settings.insertion_margin,
            "tie_tolerance": 1e-12,
        },
    }


def load_checkpoint(path, config, device="cpu"):
    """Load only strict shippable weights from a trusted local training checkpoint."""
    import torch

    from stt.model import MothSTT

    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture = config["model"]
    declared = payload["config"]["model"]
    for key, default in (("vocab_size", 1024), ("head_blocks", 0)):
        if architecture.get(key, default) != declared.get(key, default):
            raise ValueError(f"checkpoint architecture mismatch: {key}")
    if not config.get("tokenizer"):
        raise ValueError("evaluation requires a readable tokenizer source file")
    expected = _sha256(config["tokenizer"])
    saved = payload.get("tokenizer_sha256")
    if saved is None:
        source = payload["config"].get("tokenizer")
        if not source:
            raise ValueError("checkpoint is missing tokenizer identity/source")
        saved = _sha256(source)
    if not isinstance(saved, str) or not re.fullmatch("[0-9a-f]{64}", saved) or saved != expected:
        raise ValueError("checkpoint tokenizer sha256 mismatch")
    model = MothSTT(
        vocab_size=architecture["vocab_size"], head_blocks=architecture.get("head_blocks", 0)
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload["config"]


def main(argv=None) -> int:
    import yaml

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument(
        "--check-config", action="store_true", help="preflight only; no model/forward"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-metrics")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    if not args.check_config and not args.checkpoint:
        parser.error("--checkpoint is required unless --check-config is supplied")
    if args.check_config and (args.checkpoint or args.baseline_metrics):
        parser.error("--check-config does not load checkpoints or baseline metrics")
    output = Path(args.out).resolve()
    inputs = [args.config, *args.checkpoint]
    if args.baseline_metrics:
        inputs.append(args.baseline_metrics)
    if output.exists() or output in {Path(p).resolve() for p in inputs}:
        raise ValueError("output must be a new path, distinct from all inputs")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.check_config:
        result = check_config(config)
        result["files_sha256"][str(Path(args.config))] = _sha256(args.config)
        with Path(args.out).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return 0
    settings = load_evaluation_config(config)
    dev_sets = build_dev_sets(config)
    lm = load_lm(settings)
    pieces, tokenizer_sha256 = load_evaluation_tokenizer(config, lm)
    baseline_bytes = Path(args.baseline_metrics).read_bytes() if args.baseline_metrics else None
    baseline = json.loads(baseline_bytes) if baseline_bytes is not None else None
    baseline_source = (
        {"path": args.baseline_metrics, "sha256": hashlib.sha256(baseline_bytes).hexdigest()}
        if baseline_bytes is not None
        else None
    )
    candidates = {}
    for path in args.checkpoint:
        if path in candidates:
            raise ValueError("duplicate checkpoint")
        model, checkpoint_config = load_checkpoint(path, config, args.device)
        if checkpoint_config.get("train", {}).get("input", "truncated") != config.get(
            "train", {}
        ).get("input", "truncated"):
            raise ValueError("checkpoint acoustic input convention differs from evaluation config")
        sources = checkpoint_config["data"]["train"]
        if not sources:
            raise ValueError("checkpoint must declare training source inventory")
        trained = [row for src in sources for row in load_manifest(src["manifest"])]
        for name, dataset in dev_sets:
            validate_dev_rows(dataset.rows, trained, mixed=name == "mixed")
        candidates[path] = evaluate_model(
            model,
            dev_sets,
            pieces,
            settings,
            lm=lm,
            device=args.device,
            input_mode=config.get("train", {}).get("input", "truncated"),
            max_seconds=config.get("batch", {}).get("max_seconds", 192),
            max_clips=config.get("batch", {}).get("max_clips", 32),
            bootstrap=args.bootstrap,
        )
        with Path(path).open("rb") as handle:
            candidates[path]["checkpoint_sha256"] = hashlib.file_digest(
                handle, "sha256"
            ).hexdigest()
        candidates[path]["tokenizer_sha256"] = tokenizer_sha256
        del model
    result = (
        next(iter(candidates.values()))
        if len(candidates) == 1 and baseline is None
        else {
            **rank_candidates(
                candidates, settings, baseline=baseline, baseline_source=baseline_source
            ),
            "candidates": candidates,
        }
    )
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with Path(args.out).open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
