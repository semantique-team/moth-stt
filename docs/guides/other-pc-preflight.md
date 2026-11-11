# Other-PC Preflight Without Training

This local readiness check implements the preparation step of the
[audit follow-up](audit-improvements-no-training.md), not its deferred experiments.
Read [AGENTS.md](../../AGENTS.md), [STATUS.md](../../STATUS.md), the
[architecture audit](../14-architecture-audit-2026-11-10.md) and the
[mixed-dev protocol](mixed-dev-and-quality-experiments.md) first.
No command below trains, constructs an acoustic model/optimizer, loads checkpoint
payloads, runs forward, builds a corpus/LM, exports weights or uploads data.

## Environment

Use Python 3.12+ and the repository's existing dependency split. From the repository
root, these commands work in PowerShell and Linux shells without activation:

```text
uv sync --locked --no-dev
uv run --locked --no-dev python -m stt.preflight --out host-runtime-new.json
```

Runtime-only setup supports environment/release inspection without torch, YAML or
SentencePiece.

Every report, including environment-only and release-only, requires installed
`numpy`, `onnxruntime`, `soundfile` and `num2words` distributions. Supplying `--config`
also requires `torch`, `sentencepiece` and `pyyaml`, even if config validation would
otherwise pass. Export-only `onnx` is not required. The report's `dependencies.required`
and `dependencies.missing` list package names only; any missing requirement makes
`ok` false and the CLI returns **2** while still writing JSON.

For a training config check, install the existing training extra:

```text
uv sync --locked --no-dev --extra training
uv run --locked --no-dev --extra training python -m stt.preflight --config runs/NEW-ID/config.yaml --out host-config-new.json
```

If the environment is already installed, Windows and Linux use the same module API:

```text
python -m stt.preflight --config run.yaml --release release --out new.json
```

`--config` and `--release` are independently optional; `--out` is required. Its parent
directory must already exist. Existing outputs, including input/output collisions,
are refused. Exit status is **0** for completed checks without issues, **2** for missing
assets, unavailable validation dependencies, rejected configs/releases or output errors.
Config/file failures still produce the new JSON report when output creation is possible.
No inputs or directories are created, copied, archived or modified by preflight.

All config asset paths follow the existing training **working-directory semantics**,
not YAML-directory semantics. Run from the repository root and use relative paths.
Do not silently rebase a historical config when moving it. JSON paths are relative to
cwd or, for a config outside cwd, its directory. Other external locations are opaque
numbered references, not private absolute paths. Move/edit asset references under the
intended root if a portable, actionable inventory is needed. The report does not dump
config values, transcript rows, environment variables, hostnames or raw exceptions.
Review even relative filenames before sharing a report; they can be sensitive too.

Configs and shard indices must be trusted inputs. Their paths can reference files
outside cwd or the config directory, including through absolute paths, `..` or symlinks.
Preflight may stat those external files, read indices and hash eligible files there.
Relative/opaque JSON path reporting is privacy formatting, **not a filesystem sandbox**.

## Checks And Limits

The Python API is `stt.preflight.build_report(config=None, release=None, *,
evaluation=False, hash_checkpoints=False, out=None) -> dict`.
`out` selects a disk probe location only; the API does not write. The CLI writes JSON.

- Environment: allowlisted installed package versions, Python version, OS/release and
  architecture. Torch is imported lazily only if installed; CUDA/MPS availability is
  informational, not a forward test, GPU memory benchmark or device recommendation.
  Dependency presence is checked using distribution version metadata, without importing
  native runtime modules just to check installation. Version strings are informational,
  not compatibility certification. `dependencies.native_runtime_load` remains
  `not_verified`: installed metadata does not prove that native libraries load or run.
- Config: calls `stt.train.check_training_config(config, check_files=False)`. Importing
  this validator requires training dependencies, but never calls the training entry
  point. Its exceptions become issues even if validation fails before returning files.
  Raw exception messages are deliberately omitted because they can contain secrets,
  transcripts or absolute paths; use the existing checker locally for detailed diagnosis.
- Inventory: separately lists configured tokenizer/vocab, encoder/init checkpoint,
  train/dev/dev_en/dev_mixed manifests, indices, shards named by readable indices and
  an explicitly configured evaluation LM. Missing assets are failures, not silently
  substituted fixtures. No recursive raw-audio scan or checkpoint deserialization.
- Hashes: SHA-256 for files up to 16 MiB, except mel shards. Larger checkpoints are
  stat-only unless `--hash-checkpoints` explicitly requests streaming SHA-256. Missing
  indices leave unknown shard names/sizes; file identity does not establish provenance.
- Release: delegates to `stt.release.validate_release`, checking declared hashes,
  required files and the default **20,000,000-byte** raw asset budget, including config
  and NOTICE. This validator streams all declared release files, including large ones.
  Hash consistency is not publisher authentication; compare against an authorized
  historical receipt such as `runs/E-024/release-0.7.0.json` separately.
- Disk: reports raw free/total bytes at relevant existing parent directories, including
  the output location. Multiple parents can refer to the same volume; do not sum them.
  It does not assert enough space for a future run or reserve any space.
- Transfer estimate: unique declared on-disk input bytes are a lower bound, not a ZIP
  size or additional-space requirement. Missing assets, omitted raw audio, wheels,
  caches, retained checkpoints, optimizer state and workspace remain unknown. Unknown
  additional disk/dependency sizes are JSON `null`, never zero or an invented estimate.
  Asset budget is distinct from Python dependencies, ORT/WASM distribution, RAM and
  compressed network download size; the optional 25 MB tier is not silently selected.

Strict evaluation validation is **opt-in**. A training config can contain mixed-dev
references without having a complete finalist-evaluation protocol:

```text
python -m stt.preflight --config finalist.yaml --evaluation --release release --out evaluation-preflight-new.json
python -m stt.preflight --config run.yaml --hash-checkpoints --out checkpoint-preflight-new.json
```

`--evaluation` calls the actual `stt.training_eval.check_config`: reads manifests,
validates splits/annotations/tokenizer/LM, constructs data-only dev datasets and hashes
their shards. It can be expensive, but does not instantiate an acoustic model or run
forward. Without the flag none of these strict evaluation checks run. Ordinary
inventory still reports missing declared mixed-dev files; it does not certify their roles.

## Other-PC Checklist

- [ ] Keep published **0.7.0 INT8** as the unchanged product baseline; E-024 checkpoint
  FP32 evaluation is a different path, not reproduction of INT8/VAD/browser results.
- [ ] Obtain authorized locations and consent/license records for tokenizer, checkpoint,
  historical manifests/indices/shards, LM/corpus receipts and necessary raw recordings.
  Preflight inventories requirements only: it does not package, transfer or upload them.
  Never place audio, checkpoints, shards, signatures, identity maps or credentials in git.
- [ ] Match recordings to their exact manifest IDs, actual audio files, speaker/session,
  reviewed human transcript, duration and processing history on the other PC. Recover
  original takes and checksums; do not recreate or copy prompts as replacement recordings.
  Missing files or hashes cannot be repaired by assigning synthetic identity/provenance.
- [ ] Check that written consent covers the intended recordings, device/host handling,
  private/public use and evaluation; a passing report does not establish consent.
- [ ] Inventory acoustic, tokenizer and LM exposure separately, including historical
  sources. Current mutable scripts or matching vocabulary size do not prove identity.
- [ ] Collect natural mixed-dev and a distinct future holdout using the linked protocol,
  independent elicitation, consent, human transcripts and reviewed `en_spans`. T/A remain
  evaluation-only; R already used in training is not independent dev. T/S3 has been
  repeatedly inspected and is not an untouched holdout. Dev becomes selection evidence
  when inspected; never present it as independent final confirmation.
- [ ] Resolve report issues and compare small-file hashes between authorized hosts.
  Check shard byte totals and disk headroom separately; opt into checkpoint hashing
  when identity verification justifies reading the entire checkpoint.
- [ ] Defer tempo-overflow measurement, matched control/InterCTC training, finalist
  evaluation, bias/LM experiments and CUDA/MPS forward/resume validation until separately
  approved. This check grants no training authorization and demonstrates no quality gain.

For focused developer verification only (no full suite or `test_train`):

```text
uv run --locked pytest -q tests/test_preflight.py
uv run --locked ruff check stt/preflight.py tests/test_preflight.py
```
