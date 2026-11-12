# Natural Mixed-Dev And Quality Experiments

Protocol implementing the infrastructure portion of P1/P2 in the
[2026-11-10 plan](../plans/2026-11-10-lightweight-it-en-runtime-plan.md).
No recordings, corpus freeze of real data, LM builds, training or quality experiments were
performed for this guide. Nothing here establishes a quality gain or authorizes training.
Keep release 0.7.0 INT8 as the control; do not modify the experiment registry/status until
an approved experiment is actually prepared/run under the repository policy.

## Available And Missing Assets

The inspected checkout has scripts, transcript manifests, scorer, hotword matcher and LM
builder code. It does **not** have `data/tokenizer/moth_bpe.model`, the raw evaluation/training
audio under `data/raw/` or `recordings/audio/`, or the E-024 checkpoint. The two small browser
test WAV fixtures are not a benchmark corpus. No natural mixed-dev or independent future
holdout has been collected here. The exact historical LM text snapshot has not been recovered;
current R scripts are not proof of the text used to build the published LM.

Before experiments, obtain authorized asset locations, consent/license records and SHA-256
checksums. Recover/verify the published 0.7.0 release bundle against
`runs/E-024/release-0.7.0.json`. A checkpoint is needed for checkpoint-based reproduction,
not for release-only INT8 inference; a tokenizer is needed for new LM/hotword tokenization.
Do not substitute random weights, synthetic recordings or browser fixtures. The missing
audio/tokenizer and unverified release assets prevent the proposed experiments in this checkout.

## P1: Natural Mixed-Dev

Choose collection targets and speaker roles **before** collecting or inspecting model errors.
Record the intended number of speakers, clips/minutes per speaker, domains, devices, conditions
and stop rule in the collection protocol. These are targets, not claims about existing data.
Prefer several voices and independent sessions rather than many clips from one voice.

1. Obtain written consent before recording; use the existing
   [consent template](../../recordings/consent-template.md) and
   [data card](../../recordings/data-card.md) as starting points. Check that consent actually
   covers spontaneous speech, evaluation and the intended private/public use, not only reading
   prompts. Keep signatures/identity mapping outside git. Record pseudonymous speaker IDs,
   permitted uses, provenance and release license. Private research permission is not CC-BY;
   do not publish audio/transcripts without the appropriate consent and rights.
2. Elicit ordinary Italian conversation/dictation with naturally occurring English words:
   broad work or daily-life tasks, not a prescribed glossary or rereading T/A/R. Include pure
   Italian controls, English controls, brands/acronyms, Italianized forms and longer switches
   when they occur. Preserve the observed distribution; report sparse categories rather than
   manufacturing coverage. Exclude sensitive personal/third-party content.
3. Keep original audio privately, record device/session/sample rate and processing provenance.
   Follow existing audio checks in `recordings/build_manifest.py`: mono PCM, 1-30 s clips,
   non-silent, no material clipping (existing threshold 1%). Segment manually at natural
   boundaries without transcript-aware selection for model success. Keep disfluencies and
   spontaneous wording; log exclusions/retakes and their reasons. The guided recorder and
   builder currently support T/A/R only: do not relabel this collection as R or T to use them.
   Long spontaneous takes are cut into 1-30 s clips with
   [`recordings/segment_notes.py`](../../recordings/segment_notes.py) and imported as
   `natural_mixed_dev`; see the [recordings README](../../recordings/README.md).
4. Human-transcribe and review independently of ASR hypotheses. Normalize through
   `stt.text.normalize`, using `en` for English and `it` for Italian/mixed number expansion.
   Preserve `text` and store canonical `norm`. Annotate English-origin words, including
   Italianized forms, after normalization. Spell numbers unambiguously; do not invent a
   second scoring normalizer. Resolve annotation uncertainty before freezing the manifest.
5. Use the existing manifest schema: unique `id`, real `audio` path, `text`, `norm`, measured
   `duration`, `source="natural_mixed_dev"`, `split="dev"`, pseudonymous `speaker`, `lang`,
   `category`, `en_spans`. Use `lang="mix"` for the categorized mixed protocol, including its
   `control_it`/`control_en` rows; the schema reserves category/spans for mixed rows. Check
   with `stt.manifest.load_manifest`. Never create rows for audio that does not exist.
6. Audit exact normalized duplicates against training corpora/scripts, T/A and other evaluation
   sets; also human-review near duplicates and shared prompts (the script importer already
   has word-trigram Jaccard checks). Keep this collection out of acoustic/tokenizer/LM training.
   Freeze the reviewed manifest, audio hashes, consent references and role inventory before
   candidate selection; record any subsequent version as a new snapshot.

`en_spans` uses existing **zero-based, half-open word indices into `norm.split()`**, not character
offsets, timestamps, or a new format. For annotation illustration only, not a collected sample:

```json
{"norm":"ho spostato il meeting a domani","en_spans":[[3,4]]}
```

Require sorted, nonoverlapping, nonempty spans within bounds on human review (the generic
manifest validator does not enforce all of these). `control_it` has no spans; English controls
cover all reference English words. Use the existing eight categories from
`recordings/generate_scripts.py`; report counts and denominators alongside metrics.

### Importer

`recordings/import_manifest.py` is the separate path for this collection (the guided recorder and
`build_manifest.py` remain T/A/R only). It joins an audio directory with the human transcripts,
enforces mono/1–30 s/non-silent/clipping, the normalized text rule, category and `en_spans`
bounds, and the `source="natural_mixed_dev"` / `split="dev"` fields. Exact normalized duplicates
against the stored scripts and manifests are refused; near duplicates (Jaccard > 0.8) are
reported for human review. Nothing is written unless every row validates.

```bash
uv run python recordings/import_manifest.py \
  --transcripts data/natural_mixed_dev/transcripts.jsonl \
  --audio-dir data/natural_mixed_dev/audio \
  --out data/manifests/natural_mixed_dev.jsonl
```

Transcript rows carry `id`, `text`, `speaker`, `category`, `en_spans` and optionally `audio`;
`lang` is `mix` for this protocol. The tool does not grant consent, freeze roles or verify that
the audio was elicited independently of model errors.

### Development Is Not Holdout

Mixed-dev is for selection/debugging and becomes development evidence as soon as inspected.
Collect a **separate future holdout**, preferably new speakers and sessions, with independent
elicitation and the same consent/annotation discipline. Freeze its roles and access policy
before use; keep its transcripts out of glossaries, LM inputs and pruning/parameter selection.
Do not split one repeatedly inspected speaker's clips and call the remainder unseen-speaker
confirmation. Do not recycle already-consumed R as independent dev.

T and A remain evaluation-only (A repeats T text). T/S3 has been inspected repeatedly and is
not an untouched final test. S3 has not trained acoustically; S4 is allowed as a training speaker
by current script code for E-018, despite older recording prose saying otherwise. If S4 is used
for training, its T results are not unseen-speaker evidence. Freeze per-experiment roles rather
than infer them from an ID. `stt.eval._speaker_group` still labels every non-S1/S2 voice as
`S3+S4`, even new IDs. Use per-speaker results and an explicit role table; do not interpret this
aggregate as held-out. Scorer role changes are outside this task.

## Freeze Training-Side LM Text

`python -m stt.data.freeze_lm_corpus` takes repeatable `--train-manifest`, `--train-script` and
`--exclude`, plus explicit `--out` and `--report`. At least one training input and one exclusion
are required; exclusions containing no nonempty normalized text are refused. There is no
directory scan, default corpus or default exclusion list. Create the
output parent first; existing files and input/output collisions are refused.

Inventory **every** T/A script (including unrecorded prompts), T/A recording manifest, full
IT/EN dev/test manifest, natural mixed-dev and future holdout exclusion snapshot. Enumerate
all of them explicitly in the actual command, not just the abbreviated example below. An
authorized custodian can supply an exclusion snapshot for a sealed holdout without exposing
its references to the candidate designer. Missing exclusions block the protocol; the utility
cannot discover omissions or verify consent. Record why any set is absent/not yet collected.

```bash
# Illustrative paths only. Add every inventory exclusion as another --exclude argument.
python -m stt.data.freeze_lm_corpus \
  --train-manifest snapshots/mls_it_train.jsonl \
  --train-manifest snapshots/ls_en_train.jsonl \
  --train-manifest snapshots/cv_it_train.jsonl \
  --train-script snapshots/R-S1.jsonl \
  --train-script snapshots/R-S2.jsonl \
  --train-script snapshots/pool.jsonl \
  --exclude snapshots/T.jsonl --exclude snapshots/A.jsonl \
  --exclude snapshots/all-dev-test-exclusions.jsonl \
  --exclude snapshots/natural-mixed-dev.jsonl \
  --out frozen/lm-corpus.jsonl --report frozen/lm-corpus.receipt.json
```

Inputs are JSONL objects. Training manifests require `split="train"` on every row; scripts
without split metadata are an explicit training-side attestation by the operator. A declared
non-training split is rejected even through `--train-script`; known T/A names, row IDs and
recorded T/A source tags are also refused. Renaming an unlabelled evaluation script cannot
establish that it is training text: the inventory and exclusion checks remain essential.

Manifest/exclusion text prefers nonempty `norm`, falling back to `text`; scripts use `text`.
All pass through the existing canonical normalizer. `lang` accepts `it`, `en`, `mix` (script
default `mix`); `lang_hint` may explicitly select `it`/`en` number expansion, otherwise English
uses `en` and mixed/Italian use `it`. Overlap and dedupe reuse `stt.manifest.norm_hash` across
languages. Any exact normalized training/exclusion overlap **fails the whole freeze**, rather
than silently dropping the sentence. Review/correct the source selection and rerun to new
paths; do not remove an exclusion to make it pass. Near overlap is not detected by this utility.

The corpus is plain UTF-8 JSONL containing only `text`/`lang`, sorted by canonical text/lang,
one sentence per normalized-text key. Conflicting language labels choose the lexicographically
smallest label, independent of input order; audit label conflicts upstream if label semantics
matter. Empty normalized training rows are skipped and counted. This is a deduplicated corpus,
not a reconstruction of historical sentence weighting/oversampling.

The deterministic JSON receipt records SHA-256 of every input byte snapshot (including
exclusions), input role/path/count, canonical output bytes/hash, duplicate/empty counts,
normalizer source hash and installed `num2words` version. Byte-only source changes alter the
receipt even when the corpus stays identical. Archive the actual snapshots and output with
their receipt, environment lock and code revision; hashes alone do not preserve missing data.
Re-run to fresh paths and compare hashes; do not edit the frozen corpus in place.

### Builder Integration Is Explicit

The historical `runs/E-008/build_lm{,_augmented}.py` scripts read hardcoded mutable inputs;
they do **not** accept this frozen corpus via a CLI flag. Do not invoke them assuming this
utility replaced their input. In a separately approved experiment-local builder, verify the
receipt's corpus SHA-256, load the pinned SentencePiece tokenizer, and feed only
`processor.encode(row["text"], out_type=int)` from this JSONL to
`BPETrigramLM.from_sequences`. Pass explicit `min_bigram_count`, `min_trigram_count` and
`backoff`. Hash the tokenizer, builder/config and output too. No integration or rebuild is
automatic here, and recovering the historical corpus requires its original sources/receipt.

## P2: Frozen Experiments

Before execution, obtain approval and allocate a free experiment ID with the normal registry,
`runs/<id>/PLAN.md`, config and frozen decision criteria. Do not reserve IDs or claim results
for this infrastructure-only delivery. Run one bounded question at a time, keeping acoustic
weights and the release inference path fixed. Use the same audio, VAD and canonical scorer.

### Small Contextual Glossary

Use the existing `stt.hotwords.load_hotwords`, `HotwordIndex` and `beam_search` completion/
prefix bonus support. E-007 already tried a broad 580-term list without a demonstrated
held-out benefit; this is a contextual experiment, not a newly invented decoder feature.

- Freeze a small glossary from product requirements or training-side text before decoding;
  never mine T/S3 errors or future holdout transcripts. Record term provenance and hashes.
- Predeclare a small finite grid of bounded nonnegative completion/prefix bonuses and a
  maximum glossary size; include zero/no glossary. Keep beam 12, top-k 10, LM weight 0.5,
  token bonus 2.4 and all other release settings fixed. Select only on full IT/EN dev plus
  natural mixed-dev, with a predeclared tie-break favoring no change.
- Include an irrelevant glossary matched for size/token length, pure Italian controls and
  empty/non-speech clips. Measure insertion rate, glossary-induced false insertions, WER,
  aligned ETA, bytes and latency, not merely the presence of desired terms in hypotheses.
- Audit SentencePiece word boundaries, shared prefixes and repeated completions. Existing
  suffix matching is not a new whole-word bounded policy. If useful, separately implement
  and verify the same bounded, word-aware rule in JS before browser adoption. No acoustic
  recovery of unknown words is promised.

### Intermediate Trigram LM

Freeze one deduplicated training corpus and tokenizer for all arms. Compare bigram/trigram
minimum counts `(2,2)` control, `(1,2)` and `(2,1)`, backoff 0.4, with fixed release decoder
settings. A rebuilt `(2,2)` is a corpus-controlled comparator, **not** automatically the shipped
byte-identical LM: also retain the original 0.7.0 bundle as the product baseline.
Do not repeat a broad weight/bonus sweep or add a 4-gram format.

Export the actual browser LM binary using the existing export path; gate **binary bytes**
before full benchmarking, never gzip-pickle size. All required model assets must total
<=20,000,000 bytes preferred, <=25,000,000 only for an explicitly approved extended tier.
The release record totals 19,440,760 bytes, including LM 5,116,464 bytes: with other assets
unchanged, theoretical LM maxima are 5,675,704 / 10,675,704 bytes; engineering targets are
<=5.5 / <=10.5 MB. Count new mandatory glossary/config assets too. Report full browser app,
ORT/WASM and cold HTTP transfer separately. Reject over-budget arms before quality gates.
Asymmetric sizes and quality are unknown; more bytes need not improve mixed speech.

## Selection And One Final Gate

Use `python -m stt.eval --manifest ... --hyp ... --out ... --breakdown category,speaker`
and `--vs baseline.metrics.json` on identical frozen manifests. Record WER, aligned canonical
ETA, category/speaker denominators, insertion rate and non-speech emitted words. Term-presence
is diagnostic only, not ETA. Freeze a seed/resample count and declare that the existing paired
bootstrap is by utterance, not speaker cluster. New speaker-cluster analysis would be separate
work; one-speaker clip CIs do not establish population generalization.

Select on full IT/EN dev and natural mixed-dev, then freeze one candidate and run one final
independent holdout gate. Proposed criteria to approve in advance: WER difference CI below
zero and ETA difference CI above zero on mixed holdout; exploratory IT/EN non-regression
margin at most +0.5 WER percentage points, explicitly justified (inconclusive CIs do not
prove equivalence); zero emitted words on the reference non-music set, music/unseen sources
reported separately. Measure asset bytes, load time, peak RAM and RTFx on named hardware,
including x86_64/browser; do not compare different devices as if controlled. Historical RTFx
>=10 is a proposed device-specific gate, not a measurement here.

Archive hypotheses, metrics, manifest/audio/weights/LM/glossary/config hashes, code revision,
environment and timing method. Keep the original when evidence is inconclusive, sample size
is inadequate or any gate fails. No quality, speed or size result follows from this protocol
alone; no experiment, training or new recording has been run for this change.
