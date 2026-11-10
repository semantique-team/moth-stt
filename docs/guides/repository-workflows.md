# Repository Workflows And Reproducibility

> Documentation-audit verification on Windows, 2026-11-10: Ruff passed and JS runtime unit
> tests passed 7/7. Full pytest stops in collection on the Unix-only `resource` import.
> Explicitly excluding `tests/test_train.py` yields 450 passed, 3 skipped and 5 failed
> (baseline command/path handling and LibriSpeech path separators). This is not a green
> full-suite certification; source documentation changes do not fix those portability issues.

This is an operational guide to **main at `cd1ce1b`**, not a list of proposed
features. Commands and configuration keys below were checked against that revision's
source. Lightweight CLI help was also checked in an existing Windows Python 3.12
environment; training help fails there because `resource` is Unix-only. No downloads,
recording, training, export, publishing, or quality experiments were run for this guide.

Start with [AGENTS.md](../../AGENTS.md), [STATUS.md](../../STATUS.md), and
[EXPERIMENTS.md](../../EXPERIMENTS.md). The numbered design documents are explicitly
historical; use [the executed evidence index](../11-evidence-and-decisions.md) for
claims and corrections, and executable source for current arguments.

## Choose A Workflow

| Goal | Start here | Heavy assets needed? |
|---|---|---|
| Transcribe with the published model | [Runtime setup](#runtime-setup) | Complete release bundle and your audio |
| Recalculate a committed result | [Scoring and bootstrap](#scoring-and-bootstrap) | No checkpoint/audio for existing hypotheses |
| Add recordings | [Recording workflow](#recording-workflow) | Consented audio, reviewed scripts |
| Prepare training inputs | [Data workflow](#data-workflow) | Authorized corpora and substantial disk space |
| Run a new acoustic experiment | [New experiments](#new-experiments) | Tokenizer, initialization weights, manifests, mel shards |
| Rebuild a release | [Export and packaging](#export-and-packaging) | Checkpoint, matching vocabulary, trusted training-side LM |
| Check a proposed upload locally | [Publishing checks](#publishing-checks) | Verified bundle; Node dependencies for the Space |
| Change code | [Tests and development](#tests-and-development) | Full Python environment and Node 22 for CI-equivalent checks |

## Command Conventions

Run commands from the **repository root**, except blocks explicitly marked `web/`.
Examples use one physical line per command so they work in Bash and PowerShell without
translating Bash `\` continuations into PowerShell backticks. Quoted paths containing
spaces are required. Select the workflow you need; this is not a script to execute top
to bottom. Setup commands can access package registries; data preparation and training
commands are opt-in, potentially expensive operations.

`E-NEW` below is a placeholder for an approved, unused experiment ID, not an allocated
run. Create the required directories and files before invoking commands that consume
them. Never overwrite an old run's artifacts to make a new experiment fit its paths.
There is no general `stt` console command in `pyproject.toml`; invoke the shown modules
with `python -m`, or the existing experiment scripts by their actual paths.

## Runtime Setup

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then use
a checkout at the intended revision. [.python-version](../../.python-version) selects
Python 3.12; [pyproject.toml](../../pyproject.toml) permits Python >=3.12. Preserve
[uv.lock](../../uv.lock); do not regenerate it as an installation workaround.

```text
git rev-parse HEAD
git status --short --branch
uv sync --locked --no-dev
uv run --locked --no-dev python -m stt.release --help
```

This installs the runtime dependencies: `numpy`, `onnxruntime`, `soundfile`, and
`num2words`. It does **not** request Torch, SentencePiece, or ONNX graph-authoring
dependencies. `uv run` without `--no-dev` can synchronize development dependencies
again, so retain the explicit flags for runtime-only work. The project is installable
as a Python package, but its wheel is not a model-weight distribution: release assets
are separate.

Obtain the complete authorized published bundle from
[the model repository](https://huggingface.co/semantique-team/moth-stt), preserving all
seven files below under `release/`. Pin the remote revision and verify against the
trusted [0.7.0 release record](../../runs/E-024/release-0.7.0.json), not merely a mutable
remote `main`. Downloading is a separate, explicit action, not performed by the CLI.

```text
release/
  config.json
  NOTICE
  model.int8.onnx
  vocab.json
  mel_filters.bin
  silero_vad.onnx
  lm_bpe3.bin
```

```text
uv run --locked --no-dev python -m stt.release release
uv run --locked --no-dev python -m stt.onnx_runner release speech.wav
```

[stt.release](../../stt/release.py) validates configuration semantics, safe file paths,
required files, declared SHA-256 values, the LM signature, and actual asset size.
Config-local hashes prove internal consistency, not publisher authenticity. Compare
all seven external hashes, including config and NOTICE, with the trusted record too.

[ReleaseMoth](../../stt/onnx_runner.py) is the reusable CPU interface:

```python
import soundfile as sf
from stt.onnx_runner import ReleaseMoth

runtime = ReleaseMoth("release")
audio, rate = sf.read("speech.wav", dtype="float32")
text, elapsed_s = runtime.transcribe(audio, sample_rate=rate)
```

Input must be finite **mono 16 kHz PCM in [-1, 1]**, at least 560 samples. There is no
automatic resampling or gain adjustment in this runtime. VAD runs first; non-speech
can return empty text. Segmentation defaults on and is used only when there are at
least two speech runs. Each selected speech span must be at most 30 seconds; a single
overlong span raises rather than silently truncating. Returned time includes VAD,
features, and decoding, but not initial asset/session loading.

The default asset budget is 20,000,000 decimal bytes. An explicitly approved extended
tier can be checked with `--budget-bytes 25000000` and loaded with
`ReleaseMoth("release", budget_bytes=25000000)`. The single-file runner CLI has **no**
budget option. No supported budget exceeds 25,000,000 bytes.

### Platform Boundaries

| Environment | Current behavior and prerequisites |
|---|---|
| Windows native | CPU release runtime and non-training utilities can run with compatible wheels. Use PowerShell commands above. Full training/test collection is not portable unchanged: `stt.train` imports Unix `resource` at module import time. |
| Linux / WSL2 | Use a separate Linux environment, not Windows `.venv`. CPU is the training default when MPS is absent. CUDA must be requested explicitly and needs a compatible Torch build/driver. CI validates Linux CPU, not CUDA quality or speed. |
| macOS Apple Silicon | Historical training platform is M5/MPS. Training auto-selects MPS when available and checks fresh-run CTC loss parity against CPU. `caffeinate` is macOS-only. |
| macOS Intel | No MPS assumption; use CPU and measure feasibility rather than quoting M5 throughput. |
| Recording hardware | The guided capture code uses AVFoundation and `afplay`, so actual capture/playback is macOS-only. Dry-run planning and cue generation do not require a microphone. |

For native Windows training use WSL2/Linux rather than inventing a nonexistent CLI
switch for `resource`. Compatible package availability, audio codecs, microphone
permissions, and acceleration must be checked on the actual machine.

## Training Environment

```text
uv sync --locked --all-extras
```

This matches [.github/workflows/ci.yml](../../.github/workflows/ci.yml): both optional
extras **and** the default dev group. The `training` extra adds `torch==2.14.*`,
`sentencepiece`, `pyyaml`; `export` adds `onnx`, `torch==2.14.*`, `sentencepiece`.
The dev group contains `pytest`, `ruff`, `transformers`, and `openai-whisper`.
Whisper conversion needs `transformers`; the export extra alone is not the complete
developer or historical-training environment. Training-only installation is possible
with `uv sync --locked --no-dev --extra training`, but is not the full test setup.

CI uses `UV_TORCH_BACKEND=cpu`, `UV_DEFAULT_INDEX=https://pypi.org/simple`, and
`UV_KEYRING_PROVIDER=disabled`. Set these as environment variables when reproducing
CI installation, not as arguments to `stt.train`. Linux/macOS shell assignment syntax
differs from PowerShell's `$env:NAME = 'value'`. Do not equate the CI CPU environment
with the historical MPS environment or silently swap the pinned Torch version.

Install `ffmpeg` separately when needed for external audio preparation or macOS
recording (`brew install ffmpeg` is macOS-specific). Python audio loading uses
`soundfile`/libsndfile, not an automatic ffmpeg fallback; codec support, especially
M4A, must be verified before trusting an extension as readable.

## Data Workflow

### Inventory And Fetch

[stt/data/sources.py](../../stt/data/sources.py) is the registry and Python API, **not
a fetch CLI**. It records Common Voice IT 27.0 (CC0), MLS Italian and LibriSpeech
(CC-BY 4.0), FLEURS IT/EN (evaluation-only), and MUSAN noise/music. Accept applicable
dataset terms and retain provenance before obtaining archives. The checkout has no
tracked `data/raw/fetch.sh` to run.

For an authorized URL and known digest, call `download(url, dest, sha256=...)` from
that module. It creates the parent, verifies a supplied checksum after downloading,
and records SHA256SUMS. Existing files are reused without validation if no expected
digest was supplied. This is file reuse, not HTTP partial-download resume. Always
supply an independently obtained digest for reproducibility. Extraction is manual;
inspect archives before extracting them. Do not invent URLs/checksums or bypass
Common Voice access terms.

The default manifest builder expects these extracted roots:

| Source | Root relative to `data/raw/` |
|---|---|
| FLEURS | `fleurs/it_it`, `fleurs/en_us` with `<split>.tsv` and `audio/<split>/` |
| MLS | `mls_it/mls_italian/{train,dev,test}` with transcripts and audio |
| LibriSpeech | `librispeech/LibriSpeech/{train-clean-100,dev-clean}` |
| Optional larger English arm | `librispeech/LibriSpeech/train-clean-360` |
| Common Voice | `cv_it_27/it` with official TSVs and `clips/` |

`plan_records` rejects FLEURS training and unknown splits. It does not itself enforce
all corpus filters or leakage safeguards. Preserve official speaker-disjoint splits;
audit speaker intersections and normalized-text overlap yourself when adding sources.

### Build Manifests Without Replacing Evidence

Use a new output directory for a new snapshot:

```text
uv run --locked --all-extras python -m stt.data.build_manifests --raw-root data/raw --out-dir data/manifests/E-NEW --mls-hours 170 --ls-hours 100 --cv-hours 170 --seed 0
```

[build_manifests](../../stt/data/build_manifests.py) builds dev/test first, then samples
training by duration and filters it against those eval texts. Targets are **before**
filtering and are not guarantees of retained hours. Output is `<set>.jsonl` and
`build_report.json`, with counts, drop reasons, hours, and seed. Defaults produce
12 sets; `--ls360-hours 200` additionally enables `ls_en_train360`.

For an incremental rebuild use `--only cv_it_train` (comma-separated names accepted).
It loads existing exclusion hashes **from that output directory** and merges its
report. A new empty directory containing only a train rebuild does not magically
inherit the original dev/test exclusions. The current helper skips filenames ending
`_train`, not every file whose rows say `split=train`; keep the output inventory
controlled, especially with derived names such as `ls_en_train360`.

Neither default build discovers all T/A scripts, recording manifests, natural mixed-dev,
or future holdout text. Audit/add exclusions before any new training snapshot; use the
Python `extra_eval_hashes` interface when extending this builder, or an explicitly
reviewed experiment-local preparation step. Do not claim automatic comprehensive leakage
prevention. [Filter rules](../../stt/data/filters.py) include charset, digits, 1-30 s,
speaking rate, rough CTC feasibility, and overlap. The default CLI does **not** pass an
audio root to `apply_filters`, so its silence/clipping checks do not run there.

Validate the resulting schema separately:

```text
uv run --locked --all-extras python -c "from stt.manifest import load_manifest; print(len(load_manifest('data/manifests/E-NEW/mls_it_train.jsonl')))"
```

[The schema](../../stt/manifest.py) requires unique IDs and `text`, `norm`, `duration`,
`source`, `split`, `lang`; real loaders also need correct audio/shard paths. Schema
validation is not an audio existence, consent, split-role, or comprehensive annotation
audit. Use `stt.text.normalize`, with `en` for English number expansion and `it` for
Italian/mixed, rather than a second normalizer.

### Tokenizer And Mel Cache

Recover the **original** tokenizer for exact historical model reproduction. For an
approved new tokenizer experiment, explicitly select training-only inputs:

```text
uv run --locked --all-extras python -m stt.data.build_tokenizer --it data/manifests/E-NEW/mls_it_train.jsonl --en data/manifests/E-NEW/ls_en_train.jsonl --pool recordings/scripts/pool.jsonl --terms data/en_terms.txt --out data/tokenizer/E-NEW/moth_bpe --vocab-size 1024 --it-share 0.7 --terms-share 0.05 --seed 0 --report data/manifests/E-NEW/build_report.json
```

[The builder](../../stt/data/build_tokenizer.py) writes `.model`, `.vocab`, and
`.vocab.json`, and merges a `tokenizer` report with corpus statistics and hashes.
Italian share is by words, not audio hours. Seeded corpus construction and stable
piece IDs do not promise byte-identical serialized SentencePiece models. Hash and
archive the actual model used. Do not train a replacement tokenizer and assume its
IDs match an old checkpoint/LM. CTC blank is 0; SentencePiece IDs map to CTC IDs +1.

```text
uv run --locked --all-extras python -m stt.data.build_mel --manifest data/manifests/E-NEW/mls_it_train.jsonl --audio-root data/raw --out-dir data/shards/E-NEW-mls_it_train --workers 2 --shard-size-mb 1500
uv run --locked --all-extras python -m stt.data.build_mel --manifest recordings/manifests/R-S1.jsonl --audio-root recordings/audio/R-S1 --out-dir data/shards/R-S1 --workers 2
```

Run the analogous command for every train/dev/evaluation source you actually need,
using its correct audio root. [build_mel](../../stt/data/build_mel.py) downmixes and
FFT-resamples to 16 kHz explicitly, then writes flat frame-major float16
`shard-NNN.npy`, `index.json` (frame offsets), and `build_report.json`. A complete
cache costs about **57.6 MB per audio hour**, before derived sets and duplicates.
Build into fresh directories; this is not a general incremental cache-resume tool.
`--limit N` makes a partial smoke cache, not a valid cache for the full manifest.
Inspect errors and require every manifest ID in the index before training.

Mel construction catches decode/feature errors, but does **not** implement the
silence/clipping health checks suggested by its upstream manifest builder's docstring.
Run explicit audio health checks before accepting a new corpus. Avoid `--delete-audio`:
it spot-checks only a sample, then can delete all manifest audio, including files absent
from a partial/error cache. Retain evaluation PCM for VAD, browser parity, and later
audits even when training mel exists.

## Recording Workflow

Read the [recording protocol](../../recordings/README.md),
[consent template](../../recordings/consent-template.md), and
[data card](../../recordings/data-card.md) before collection. Signed identities and
consent records stay private, outside git. Permission for private research is not
permission to publish audio or transcripts.

### Freeze Roles Before Collection

T is evaluation-only; A rereads T text and is also evaluation-only. R is training.
Current [script code](../../recordings/generate_scripts.py) allows R for **S1, S2,
S4** and refuses S3. Older prose saying S4 is permanently held out is stale. Declare
each experiment's speaker roles explicitly; if S4 trains, its T score is not
unseen-speaker evidence. T/S3 has been inspected repeatedly, so it is not a pristine
future holdout even though S3 has not trained acoustically.

For natural mixed-dev or new speaker IDs, follow
[the dedicated protocol](mixed-dev-and-quality-experiments.md). The existing guided
recorder/builder handles only T/A/R and the recorder only S1-S4. Do not disguise a
new development collection as R or T to get past these interfaces.

### Import And Review New Script Text

Prepare candidate JSONL externally. Text must already be canonical, and `en_spans`
are **zero-based, half-open word indices**, not character offsets or timestamps:

```json
{"text":"ho spostato il meeting a domani","category":"single_word","en_spans":[[3,4]],"generator":"human","date":"2026-11-10"}
```

```text
uv run --locked --no-dev python -m recordings.generate_scripts import --kind R --speaker S4 --in recordings/candidates/new-r-s4.jsonl
uv run --locked --no-dev python -m recordings.generate_scripts stats
uv run --locked --no-dev python -m recordings.generate_scripts review --kind R --speaker S4 --limit 50
```

The importer appends validated, deduplicated candidates, assigns IDs, and handles T
stratification. It is not an LLM client. Dedupe covers the scripts directory, not every
external training/eval corpus. Import can accept good rows even when other rows are
invalid and its exit code is nonzero: inspect the report and diff. Human-review
wording, English-origin spans (including Italianized terms), categories, naturalness,
and near overlap. Only then mark the exact reviewed IDs:

```text
uv run --locked --no-dev python -m recordings.generate_scripts mark-reviewed --kind R --speaker S4 --reviewer reviewer-id --ids R-S4-0001
```

Replace the example ID with actual imported IDs. Omitting `--ids` marks **all** rows;
it is not a review operation. Review/mark accept T, R, pool, not A. A must be maintained
as a reviewed derivative of T through an explicit human-reviewed change, not an
invented `--kind A` generator invocation. Review all T/A cues before recording:
the recorder itself does not enforce `reviewed`; the builder does for T/A, not R.

### Capture Or Use Cue Sheets

macOS, after installing ffmpeg and granting the terminal microphone access:

```text
uv run --locked --no-dev python -m recordings.record --speaker S4 --kind T --dry-run
uv run --locked --no-dev python -m recordings.record --speaker S4 --kind T --max-seconds 30
uv run --locked --no-dev python -m recordings.record --speaker S4 --kind A --max-seconds 30
```

Enter starts/stops, `p` plays back, `r` retakes, `s` skips, `q` quits. Existing files
are skipped on resume. `--start` is the **1-based cue position**, not the numeric
suffix of a T script ID. `--overwrite` permits replacing existing takes: use it only
deliberately. The default recorder cap is 40 s, while the builder accepts at most
30 s. The recorder's minimum 0.5 s is also weaker than the builder's 1 s minimum.

Windows/Linux/phone capture uses an external recorder and the same frozen cue order:

```text
uv run --locked --no-dev python -m recordings.build_manifest cue --kind T --speaker S4 --out recordings/cues/T-S4.txt
```

Use mono PCM WAV for interchange and validate it; extension support does not imply
codec availability. Files join to text **by position**, with no ASR alignment:

| Kind | Default file path example |
|---|---|
| T | `recordings/audio/S4/S4_0001.wav` |
| A | `recordings/audio/anchor/S4/S4_0001.wav` |
| R | `recordings/audio/R-S4/S4/S4_0001.wav` |

Do not delete a bad cue from the script, reorder cues, renumber later files, or edit
already-recorded wording. Log exclusions/retakes and preserve the original mapping.
`--audio-dir` is a root **above** the speaker directory. Recordings at 48 kHz are
accepted by preprocessing; they are not direct 16 kHz runtime input.

### Build And Inspect A Candidate Manifest

```text
uv run --locked --no-dev python -m recordings.build_manifest build --kind T --out recordings/manifests/T-candidate.jsonl
uv run --locked --no-dev python -m recordings.build_manifest build --kind A --speaker S4 --out recordings/manifests/A-S4-candidate.jsonl
uv run --locked --no-dev python -m recordings.build_manifest build --kind R --speaker S4 --out recordings/manifests/R-S4-candidate.jsonl
```

[The builder](../../recordings/build_manifest.py) rejects missing, unreadable, silent,
over-1%-clipped, or outside-1-30-s takes and reports extra files. It does not verify
the spoken transcript, consent, or mono channel count. Listen/review those separately.
**It writes partial output even when it exits 1.** T iterates all speakers in the
script; `--speaker S4` is not a reliable T-only build filter. With only S4 audio
available, do not replace the committed 600-row T manifest with that partial output.
Review counts, IDs, transcripts, exclusions, and the diff before consolidating an
approved new snapshot. Never use `--allow-unreviewed` as routine acceptance.

Default outputs are `T.jsonl`, `A-<speaker>.jsonl`, `R-<speaker>.jsonl`. R rows have
`split=train`; T/A rows have `split=test`. Build mel only from the accepted snapshot,
then update the experiment config explicitly. Do not train on T/A or move their text
into tokenizer, LM, glossary selection, or synthetic training inputs.

## New Experiments

### Before Running

1. Read the relevant previous PLAN/RESULT and negative results. Allocate an approved
   unused ID and create `runs/E-NEW/PLAN.md` plus an `EXPERIMENTS.md` entry using
   [the repository template](../07-repo-layout-and-roadmap.md#4-experimentsmd-template).
2. Freeze the question, control, intervention, seeds, finite selection grid, budget,
   success/kill criteria, data/speaker roles, and intended inference path. Change one
   variable unless a multi-variable arm is explicitly justified and disclosed.
3. Inventory and hash manifests, raw audio, tokenizer, initialization checkpoint,
   LM, glossaries, code, lockfiles, config, and preparation scripts. Retain retrievable
   authorized snapshots, not just hashes of now-missing files.
4. Audit train/dev/test exact and near overlap. Select on development only, then freeze
   one candidate for the predeclared final gate. Do not select on T/S3 errors and call
   the resulting test comparison independent.
5. Estimate storage and compute. The research ledger is $10 paid compute with review
   at $8; obtain authorization for expensive runs and any changed budget.

For training-side LM text use `python -m stt.data.freeze_lm_corpus`: repeatable
`--train-manifest`, `--train-script`, and `--exclude`, plus required `--out` and
`--report`. At least one training input and one nonempty exclusion are required;
exact overlap fails the freeze. Create output parents first and use fresh paths.
The [LM freeze guide](mixed-dev-and-quality-experiments.md#freeze-training-side-lm-text)
specifies the complete exclusion inventory and deterministic receipt. This utility
does **not** build an LM or plug into historical E-008 builders automatically.

### Create An Actual Config

[E-024's config](../../runs/E-024/config.yaml) is the historical 0.7.0 recipe, not a
fresh-checkout runnable baseline. It needs E-014 initialization, noisy CV shards,
spliced shards, and oversampled gold in addition to the ordinary corpora. For a new
bounded experiment create a complete config with paths to **your frozen inputs**.
This smaller example is illustrative, not the E-024 recipe or a promised quality result:

```yaml
run: E-NEW
seed: 0
model:
  vocab_size: 1024
  head_blocks: 2
  init_checkpoint: runs/E-024/checkpoints/ckpt_best.pt
  freeze_encoder: false
tokenizer: data/tokenizer/moth_bpe.model
data:
  train:
    - manifest: data/manifests/E-NEW/mls_it_train.jsonl
      shards: data/shards/E-NEW-mls_it_train
  dev:
    - manifest: data/manifests/E-NEW/mls_it_dev.jsonl
      shards: data/shards/E-NEW-mls_it_dev
  dev_en:
    - manifest: data/manifests/E-NEW/ls_en_dev.jsonl
      shards: data/shards/E-NEW-ls_en_dev
batch:
  max_seconds: 192
  max_clips: 32
optimizer:
  lr_head: 3.0e-4
  lr_encoder: 5.0e-5
  betas: [0.9, 0.98]
  eps: 1.0e-8
  weight_decay: 0.01
schedule:
  warmup_steps: 500
  min_factor: 0.05
train:
  epochs: 2
  input: truncated
  augment: full
  autocast: bf16
  grad_clip: 1.0
  log_every: 50
  checkpoint_minutes: 10
  keep_last: 3
  max_nonfinite_steps: 20
  empty_cache_every: 10
  eval_every_steps: 2000
```

Choose learning rates and schedule in the plan, not by blindly copying this example.
At least one dev group is needed for meaningful best-checkpoint selection. All source
shard directories within a merged group must share one parent; manifest IDs must be
unique across sources and present in the indexes. The loader does **not** enforce
training split roles itself. Tokenizer vocabulary and head size must agree.

`model.init` loads a converted encoder; `model.init_checkpoint` loads a full Moth
checkpoint, after `init` if both are set. For encoder-only initialization use:

```text
uv run --locked --all-extras python -m stt.convert_whisper --model openai/whisper-tiny --out models/whisper_tiny_encoder.pt --seed 0
```

This can download upstream weights and performs a parity check; it is not an offline
setup probe. Use trusted checkpoints only: training/export and historical LM loading
use pickle-capable serialization. A tokenizer/head shape change is a new experiment,
not a drop-in continuation of the old decoder or LM.

### Train, Smoke, And Resume

Linux/macOS, only after preparing all assets and approving the run:

```text
uv run --locked --all-extras python -m stt.train --config runs/E-NEW/config.yaml --device cpu --max-steps 20
uv run --locked --all-extras python -m stt.train --config runs/E-NEW/config.yaml --device cpu
```

`--max-steps` is an **absolute global-step cap** and does not shorten the cosine
schedule. This smoke command really trains and writes artifacts. Repeating the
second command in the same directory resumes the highest numbered checkpoint,
including loader cursor, optimizer, scheduler, and saved Python/NumPy/Torch CPU RNG.
There is no `--resume` flag. For an independent smoke use a separate config directory
and run ID, not the final run directory. Do not change data/config under an existing
run and mistake automatic resume for a fresh experiment; GPU cross-device bitwise
reproducibility is not guaranteed by the saved CPU RNG state.

On Apple Silicon request `--device mps`; on compatible Linux CUDA request `--device
cuda`. Omitting the flag chooses MPS if available, otherwise CPU, **not CUDA**.
`caffeinate -i uv run --locked --all-extras python -m stt.train --config runs/E-NEW/config.yaml --device mps`
is a macOS-only long-run wrapper. Log hardware, precision, memory, wall time, thermal
conditions, and any interruptions rather than borrowing published M5 timings.

[stt.train](../../stt/train.py) writes beside the **config file**:

| Artifact | Meaning |
|---|---|
| `train.log.jsonl` | Appended start/step/dev/end events, code SHA, Torch version, seed, device, throughput and resume details |
| `checkpoints/ckpt_0000020.pt` etc. | Numbered resumable state; last `keep_last` retained |
| `checkpoints/ckpt_best.pt` | Best observed dev macro WER, independent of last-step retention |
| stdout JSON | Final summary; not an automatically written `summary.json` |

Periodic and final evaluation here uses **greedy** WER. With both dev groups,
checkpoint selection is the macro-average IT/EN WER, not beam+LM release WER.
Training does not automatically produce final test hypotheses or publishable metrics.

### Generate Hypotheses On Current Main

`stt.eval` **scores existing text; it does not run a model**. `stt.onnx_runner` CLI
transcribes one file, not a manifest. For release-path evaluation on arbitrary approved
manifests, an experiment-local producer can use existing APIs as follows. This is a
Python recipe to put in the new run's reviewed evaluation script, not a nonexistent
batch CLI:

```python
import json
from pathlib import Path
from stt.data.build_mel import load_audio_16k
from stt.manifest import load_manifest
from stt.onnx_runner import ReleaseMoth

runtime = ReleaseMoth("release")
rows = load_manifest("recordings/manifests/T.jsonl")
audio_root = Path("recordings/audio")
out = Path("runs/E-NEW/eval/release.hyp.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    for row in rows:
        # Explicit preprocessing outside the strict release runtime.
        pcm = load_audio_16k(audio_root / row["audio"])
        hyp, elapsed = runtime.transcribe(pcm, sample_rate=16000)
        handle.write(json.dumps({"id": row["id"], "hyp": hyp,
                                 "latency_s": elapsed}, ensure_ascii=False) + "\n")
```

Choose the manifest/root/output explicitly for each set, preserve every ID, and
record resampling, segmentation, and timing methodology. Do not include initial model
load in a number labeled steady-state decoding. This recipe does not authorize
repeated test-set selection.

For **checkpoint/mel** evaluation on the repository's fixed sets, the reusable legacy
script is [runs/E-005/eval_sets.py](../../runs/E-005/eval_sets.py), not a new module:

```text
uv run --locked --all-extras python runs/E-005/eval_sets.py --checkpoint runs/E-NEW/checkpoints/ckpt_best.pt --config runs/E-NEW/config.yaml --sets mls_it_dev,ls_en_dev --out runs/E-NEW/eval/dev --decoder beam --beam-width 12 --top-k 10 --lm runs/E-008/lm_bpe3_aug.pkl.gz --lm-weight 0.5 --token-bonus 2.4
```

Its set names map to hardcoded **original** manifest/cache paths, not the config's
`data` section. It supports `mls_it_dev`, `ls_en_dev`, `fleurs_it_test`,
`fleurs_en_test`, `cv_it_dev`, `cv_it_test`, `recorded_t`, `noise`; no arbitrary
`--manifest`, no CUDA option, and no new-snapshot path remapping. The example is
appropriate only if those original sets and the chosen tokenizer/LM are compatible.
For new snapshots, write a small reviewed experiment-local producer using the
existing loaders rather than editing historical set definitions.

It writes `<set>.hyp.jsonl` and `<set>.metrics.json`. `--limit` is a seeded random
subset for tuning, not a full-set result. `--vad-guard` requires raw audio; missing
audio can **warn and skip VAD** rather than fail. Treat that warning as a blocked
release-path evaluation. The old `--no-speech-guard` is the superseded energy guard,
not Silero. This script uses cached mel and no release sentence segmentation; do not
label it end-to-end shipped INT8 evaluation.

## Scoring And Bootstrap

Hypothesis JSONL rows have `id` and `hyp`; extra latency fields are permitted.
Use the same frozen reference bytes for candidate and control. Create output parents
first: the scorer does not create them. A lightweight **offline re-scoring example**
uses only committed manifests and hypotheses:

```text
uv run --locked --no-dev python -m stt.eval --manifest recordings/manifests/T.jsonl --hyp runs/E-024/eval/gate/t600.hyp.jsonl --out runs/E-NEW/t600-rescored.json --breakdown category,speaker --bootstrap 1000 --seed 0
```

For new comparisons, score both producers through the CLI, even when their historical
runner already emitted a metrics file:

```text
uv run --locked --no-dev python -m stt.eval --manifest recordings/manifests/T.jsonl --hyp runs/E-NEW/eval/control.hyp.jsonl --out runs/E-NEW/eval/control.metrics.json --breakdown category,speaker --bootstrap 1000 --seed 0 --config runs/E-NEW/config.yaml
uv run --locked --no-dev python -m stt.eval --manifest recordings/manifests/T.jsonl --hyp runs/E-NEW/eval/candidate.hyp.jsonl --out runs/E-NEW/eval/candidate.metrics.json --breakdown category,speaker --bootstrap 1000 --seed 0 --config runs/E-NEW/config.yaml --vs runs/E-NEW/eval/control.metrics.json
```

[The scorer](../../stt/eval.py) records manifest/hypothesis/normalizer hashes, code
SHA, config **path**, per-utterance counts, requested breakdowns, noise counts, CIs,
and optional paired differences. It does not hash the config, infer checkpoint/LM
provenance, or write hypotheses. Historical direct `score_rows` outputs can lack the
CLI's provenance fields: `--vs` requires the matching `manifest_sha256` and identical
ID coverage, so re-score both first. Missing hypotheses become empty strings;
duplicate hypothesis IDs collapse in a dictionary and extras are counted, not
automatically rejected. Audit complete unique ID coverage before interpreting results.

WER and ETA in JSON are ratios: 0.25 means 25%. Insertion rate is insertions per
100 reference words. ETA counts **aligned exact matches** inside reference English
word spans, not term occurrence anywhere in output. Empty-reference noise is reported
separately, with music separate from non-music. `--bootstrap 0` disables CIs.

Bootstrap uses 1,000 seeded **utterance** resamples by default, not speaker clusters.
`--vs` differences are candidate minus control: better WER is negative, better ETA
positive. A CI containing zero is a tie/inconclusive comparison, not equivalence.
The scorer's `S3+S4` group actually includes every non-S1/S2 speaker; use explicit
roles and per-speaker results for new IDs rather than labeling that aggregate held-out.

### Decision Gates

Freeze experiment-specific thresholds before execution. Historical shipping criteria
in [docs/05](../05-evaluation.md#5-bars-and-gates-all-relative-to-h0-measurements)
include mixed WER CI below zero **and** ETA CI above zero versus the best applicable
Whisper-tiny H0 variant, zero words on non-music noise, browser RTFx >=10 on the named
M5, asset budget, and quantization quality checks. Do not compare against an easier
greedy baseline when the best H0 is beam 5. Report aggregate T and unseen-training-
speaker evidence separately; T-200, T-400, and T-600 are different test snapshots.

The [current quality protocol](mixed-dev-and-quality-experiments.md#selection-and-one-final-gate)
adds full IT/EN dev plus natural mixed-dev selection, then one independent holdout
gate. Its suggested +0.5 WER-point non-regression margin is a **proposal requiring
approval**, not an automatic universal acceptance rule. No missing holdout, reused
room, one-speaker bootstrap, or unavailable benchmark should be presented as a passed
generalization gate. Archive negative results and retain the control if inconclusive.

## Historical Reproduction

These are real scripts, but their frozen assumptions differ from general interfaces:

| Source | What must be recovered or preserved |
|---|---|
| [E-024 config](../../runs/E-024/config.yaml) | E-014 init, original tokenizer, original train/dev cache snapshots, spliced mel, two-speaker gold x50, clean/noisy CV split |
| [E-017 gold builder](../../runs/E-017/prepare_gold.py) | `recordings/manifests/R-S1,R-S2` and matching `data/shards/R-S1,R-S2`; `--sets`, `--factor`, `--out-dir` are supported, but input roots are baked in |
| [E-024 noisy mel builder](../../runs/E-024/build_noisy_mel.py) | Original CV raw audio and exact noise/room-tone pool; deterministic ID hashing does not recover deleted source audio |
| [E-008 augmented LM builder](../../runs/E-008/build_lm_augmented.py) | Hardcoded CV/MLS/LS manifests and R-S1/R-S2/pool text snapshots; output/tokenizer/pruning/backoff flags do not replace its corpus inputs |
| [E-022 evaluator](../../runs/E-022/eval.py) | E-017 config, tokenizer, LM, all fixed cache sets; default E-017 checkpoint, even if a newer checkpoint is passed separately |
| [E-024 shell evaluation](../../runs/E-024/eval.sh) | Bash/Unix environment, E-017/E-024 checkpoints, exact noise conditions and fixed comparison paths; writes historical E-024 outputs |
| [E-024 release recorder](../../runs/E-024/record_release.py) | Version-specific browser output and baked-in quality/export numbers; not a generic measurement/record-generation utility |

For a recovered E-024 checkpoint, the README's historical evaluator can be invoked
without overwriting its original output:

```text
uv run --locked --all-extras python runs/E-022/eval.py --arm frozen --checkpoint runs/E-024/checkpoints/ckpt_best.pt --int8 --out runs/E-NEW/eval/historical-e022
```

This still loads the checkpoint and training-side assets even with `--int8`. Its
INT8 branch reads **`release/model.int8.onnx`**, not an export of `--checkpoint`;
confirm the two actually correspond. It decodes only T-600/T-S3 in that branch,
regardless of `--sets`, and does not implement full `ReleaseMoth` VAD/segmentation
evaluation. `--sets` does not avoid initial loading of every set. `cv_it_dev` in
this harness is only the first 600 clips. Its internal metrics lack CLI manifest
hashes. These facts prevent treating it as a generic complete ship-gate command.

The training-side `.pkl.gz` LMs **are tracked** under `runs/E-008/`. The exact
historical LM corpus snapshot/receipt has not been recovered: rebuilding from today's
mutable R scripts does not prove a byte-identical or leakage-audited historical LM.
Use the committed trusted LM artifact for the old package, or explicitly label a
new corpus-controlled build as a new experiment.

For a paper, distinguish four levels: re-score committed hypotheses; rerun inference
from verified release + licensed audio; retrain from frozen inputs; reproduce the
historical selection process. Availability at one level does not imply the next.
Link every table row to manifest version/count, hypotheses, metrics/diff, config,
weights/tokenizer/LM hashes, code revision, environment, seed, hardware, and decision.
Report single-seed training, repeated test inspection, limited speakers/channels,
public benchmark exposure, and the distinction between controlled-pool and unseen-room
noise evidence. The [runtime acceptance record](../../runs/runtime-2026-11-10/RESULT.md)
is infrastructure evidence, not new acoustic weights or a new quality gain.

## Export And Packaging

### Build The Model Assets

Back up any existing local `release/` first. The following reconstructs the historical
0.7.0 candidate only after obtaining the genuine checkpoint and matching tokenizer:

```text
uv run --locked --all-extras python -m stt.export --out release --checkpoint runs/E-024/checkpoints/ckpt_best.pt --head-blocks 2 --tokenizer data/tokenizer/moth_bpe.model --version 0.7.0 --seed 0
uv run --locked --all-extras python runs/E-010/package_browser.py --lm runs/E-008/lm_bpe3_aug.pkl.gz --beam 12 --top-k 10 --lm-weight 0.5 --token-bonus 2.4 --budget-bytes 20000000
uv run --locked --no-dev python -m stt.release release
```

Use a new version/record/card for a new model; do not overwrite 0.7.0 provenance.
`--vocab PATH` can replace `--tokenizer PATH` when a trusted matching `vocab.json`
exists; they are mutually exclusive. Recovering vocabulary alone permits export,
but is not a substitute for the SentencePiece model needed by training/new LM work.
Head-block count is inferred from checkpoint keys when omitted.

[Export](../../stt/export.py) uses ONNX opset 17, dynamic time, int8 per-channel
weights, and transient fp32 ONNX (removed in `finally`). It writes model, vocabulary,
filters, VAD, config, and generated NOTICE. **Export alone is not a complete release**:
there is no binary LM yet and its decode defaults are not the published recipe.
[The packager](../../runs/E-010/package_browser.py) adds `lm_bpe3.bin`, replaces
decode settings, updates the LM hash, and validates the package. It always operates
on `release/`; there is no `--release` or `--out` packager option. It mutates files
before validation, so a failed budget gate can leave an invalid directory behind.

The fp32 export gate is the current D29 seeded sweep: 20 draws at each 1/5/10/29 s,
median max-absolute gap <=1e-3 and worst <=1e-2. Do not use the older single-draw
description from docs/05 or routinely bypass verification with `--no-verify`.
`verify_int8` **measures** logit gap/top-1 agreement; it does not implement an
end-to-end WER acceptance gate. Real INT8 WER/ETA and browser parity remain separate
required measurements. `--random` is only an infrastructure smoke export, never
quality evidence or publishable weights.

### Validate What Actually Ships

```text
uv run --locked --no-dev python -m stt.release release --budget-bytes 20000000
```

The trusted 0.7.0 record totals **19,440,760 bytes** for all seven required files.
Its legacy `package_bytes=19,436,178` means only the five runtime payload files;
do not use that smaller field for the current all-required-assets budget. Report
ORT/WASM, browser page/JS, dependencies, and compressed cold HTTP transfer separately.
`stt.release --browser-root PATH` counts all raw files in a browser staging directory,
not compressed network traffic. It does not check model authenticity, consent, quality,
or browser compatibility on its own.

## Publishing Checks

The [general Hugging Face guide](publishing-to-huggingface.md) explains accounts and
tokens. For this repo prefer the guarded publishers, **without `--push`** first:

```text
uv run --locked --no-dev python runs/E-021/upload_hf.py --repo semantique-team/moth-stt --release release --card runs/E-024/MODEL_CARD.md --record runs/E-024/release-0.7.0.json
uv run --locked --no-dev python runs/E-022/publish_space.py --check
```

[The model publisher](../../runs/E-021/upload_hf.py) compares record hashes, card
version heading, expected files, and legacy five-file payload count. It does not
replace `stt.release` semantic/budget validation; run both. Defaults target the 0.7.0
card/record. For a new release, supply its own `--release`, `--card`, `--record`,
and approved repo; create a reviewed record from actual new measurements, not by
running the historical E-024 recorder with hardcoded results.

[The Space publisher](../../runs/E-022/publish_space.py) is more version-bound:
source page/card is `runs/E-024/space`, record is E-024's 0.7.0 record, and assets
come from `release/` plus `web/node_modules/onnxruntime-web/dist`. It has no flags
to swap these release/card/record paths. A different-version Space needs an explicit
reviewed code change or new experiment-local publisher, not just `--repo`.
Install the locked Node packages first (see below). `--check` still stages files
locally and is not a no-write operation. Default staging is a temporary directory;
**`--dest PATH` recursively deletes an existing PATH before staging**. Use only a
disposable empty/new staging location. `--serve` serves the staged tree on 8767.

Neither dry run requires a token or contacts the Hub through its publishing code.
`uv` may still synchronize packages unless the environment is already prepared.
Only after explicit publication approval, both scripts use `--push` to upload.
Credentials are taken from `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or cached HF login;
keep tokens out of commands committed to git, logs, chat, and cards. Hub write access
and `huggingface_hub` must exist in the executing Python environment; it is not a
base runtime dependency (the full dev environment includes it transitively). Model
`--private`/Space `--private` control creation, not an automatic privacy migration of
an existing repo. Model upload is several remote commits, not an atomic publication.

After any authorized upload, independently retrieve a pinned remote revision,
compare all seven record hashes, run `stt.release`, and load/transcribe as a user.
For the Space, verify the deployed URL, actual asset requests, no audio upload, and
thread fallback. Static HF Spaces lack COOP/COEP and normally run one WASM thread;
their timing is not the local four-thread result.

## Tests And Development

### Python And Lightweight JavaScript

Linux/macOS with the full environment; Node **22** matches CI:

```text
uv lock --check
uv run --locked --all-extras ruff check .
uv run --locked --all-extras pytest -q
node --experimental-vm-modules --test web/tests/unit-runtime.test.mjs
uv build --wheel
```

[CI](../../.github/workflows/ci.yml) runs lint, Python tests, wheel build, a fresh
runtime-only wheel smoke, and all lightweight `*.test.js/mjs/cjs` Node tests. The
single `.mjs` command above is the current lightweight test file. These Node tests
do not require npm installation, Playwright, a browser, or model downloads.
Python decoder/mel parity tests call Node and **skip if it is missing**; install
Node to avoid mistaking skipped parity for coverage. Conversion tests use synthetic
Transformers configs, not an automatic upstream weight download. Full pytest on
native Windows can fail collection through the Unix `resource` import.

The wheel smoke is important: run from an isolated environment, install only locked
runtime dependencies plus the built wheel, and verify Torch/SentencePiece/ONNX are
absent while runtime imports and vendored mel/VAD assets work. The exact Linux
commands and assertions are in CI; its `/tmp/.../bin/python` paths are not Windows
commands. Building a wheel alone does not establish this isolation property.

### Browser Tests And Local Demo

Commands below run **inside `web/`**; `npm ci` and browser installation can download
packages/binaries and are separate from lightweight CI:

```text
npm ci
npx playwright install chromium
npx playwright test tests/decoder.spec.ts
npx playwright test tests/smoke.spec.ts tests/mic.spec.ts tests/vad.spec.ts
```

The full suite is `npm test`, but do not assume a fresh checkout can pass it.
[Playwright config](../../web/playwright.config.ts) starts `python3 server.py --root
.. --port 8765`, uses one worker/Chromium, and permits fake microphone capture.
Ensure `python3` resolves on PATH. On Windows, start the same server explicitly with
the available Python in a separate terminal, then use its `reuseExistingServer`
behavior rather than assuming a `python3` alias exists:

```text
uv run --locked --no-dev python web/server.py --root . --port 8765
```

That server command runs from the **repository root**; open
`http://127.0.0.1:8765/web/index.html`. Serve only a trusted local checkout: the root
contains manifests and may contain private recordings. Do not expose it publicly.
The custom server sends the isolation headers needed for multithreaded WASM.

| Test group | Additional prerequisites / meaning |
|---|---|
| `unit-runtime.test.mjs` | Node built-ins and synthetic/mocked inputs; no npm/browser/release |
| `decoder.spec.ts` | Playwright and local server; synthetic beam cases plus a parser-score check that requires the original `release/lm_bpe3.bin` matching committed `lm_scores.json` |
| `smoke`, `mic`, `vad` | Complete packaged release, ORT Web files from npm, committed WAV fixtures; measures real browser execution, not benchmark-corpus accuracy |
| `parity.spec.ts` | Default `runs/E-022/parity-clips.json` and corresponding private T PCM; `PARITY_CLIPS`/`PARITY_OUT` override clip list/output, not the hardcoded T audio root |
| `segment.spec.ts` | Five hardcoded S3 recording WAVs plus release assets; absent in a clean checkout |

Playwright writes `web/test-results/` (including timing/parity JSON). Archive selected
results under the new run with hardware/browser versions and method; do not present
a positive-RTFx smoke assertion as the >=10 product gate. Browser/Python real-audio
transcripts can differ on near ties, so report both match counts and paired WER/ETA,
not just synthetic parity. Re-run selected release gates on named platforms/devices;
Linux CI is not evidence for Safari, macOS capture, CUDA, or all x86_64 browsers.

### Adding Something New

Keep the smallest change that answers the approved question. Reuse the canonical
normalizer, manifest schema, decoder, and scorer. A new data source needs licensed
registry/provenance, a reader, split/exclusion audits, filtering, and tests; merely
adding a URL is not an integrated corpus. New acoustic/head shapes need tokenizer
and checkpoint compatibility checks, export tests, and size/latency/quality gates.
New decoding rules need Python/JS parity and real release-path evaluation, not only
an experiment-local Python score. A new public asset must be included in config,
hashing, size accounting, NOTICE/card, and runtime validation as applicable.

Use [tests/](../../tests/) for behavior tests, [web/tests/](../../web/tests/) for
Node/browser checks, and new `runs/<id>/` artifacts for experimental evidence. Do not
silently convert old historical scripts into generic tools or alter old artifacts.
No design-plan proposal or branch-only module is an available main command.

## What Git Does Not Provide

[.gitignore](../../.gitignore) is the authority, not the fact that a working machine
once had a file:

| Item | Status at the audited checkout |
|---|---|
| Code, configs, scripts, manifests, hypotheses, curated terms, many metrics/logs | Tracked evidence; availability does not imply runnable audio/weights |
| `data/raw`, `data/shards`, `data/tokenizer` | Ignored and absent here; no automatic recovery |
| `models/`, `release/`, numbered/best run checkpoints | Ignored and absent here; recover from authorized storage or rebuild with inputs |
| `recordings/audio`, `incoming`, `candidates`, `cues` | Ignored; private raw takes, transfers, and scratch content must not enter git |
| E-008 LM `.pkl.gz` files | Tracked, trusted training-side artifacts; not runtime-required pickle inputs |
| `stt/assets/mel_filters.bin`, vendored `silero_vad.onnx` | Tracked and included in wheels; the VAD is an explicit exception to `*.onnx` ignore |
| `web/node_modules`, test reports, Python environments/caches | Ignored local dependency/output state, not reproduction evidence |
| Top-level `NOTICE` | Not tracked/present; generated `release/NOTICE` and its release-record hash carry the shipped notice |

The checkout has no raw training/evaluation audio, mel shards, original SentencePiece
model, E-024 checkpoint, natural mixed-dev collection, independent future holdout, or
verified local release directory. Two browser fixture WAVs are not a benchmark corpus.
Tracked manifests cannot reconstruct private audio. Some historical source audio was
deleted even on the training machine; noise-source provenance and exact LM text
snapshots remain material reproducibility limits.

Before sharing changes, inspect `git status --short --branch` and the complete diff.
Check that only intended small evidence/code/docs are included and that no personal
data, token, checkpoint, or cache slipped through an unignored path. Do not rely on
ignore rules to recognize every large file. Per AGENTS.md, commits, pushes, tags,
and history changes require explicit approval; this guide does not authorize them.
