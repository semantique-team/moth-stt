# 13 - Current Model and Paper Readiness

Audit date: **2026-11-10**, checkout **main `cd1ce1b`**. This dossier distinguishes executable
architecture, committed measurements, historical publication records and evidence still needed.
It complements [the corrected decision log](11-evidence-and-decisions.md); it does not rewrite
the experiment reports or model cards. Only repository evidence was audited, not live HF hosting.

> **Update 2026-11-12 — 0.8.0 is now the published model.** The audit below was written on
> 2026-11-10 against 0.7.0 and is kept as the 0.7.0 dossier. The new release is **E-028 + the
> E-030 decoder** (same 11,575,169-parameter architecture; int8 graph hash `7bbfad7e…`; package
> 19,440,761 bytes). Published metrics: T-600 **24.71 / 54.17**, T-S3 **25.21 / 50.14**,
> FLEURS-IT **20.25**, FLEURS-EN **49.81**, natural **436/817**, rare presence **11/47**, rare
> aligned ETA **28.4%**. The rare gate target of 30% is missed by two aligned occurrences; this is
> a documented publication exception. Live model and Space were verified on 2026-11-12:
> [release record](../runs/E-028/release-0.8.0.json), [E-030 result](../runs/E-030/RESULT.md),
> [model card](../runs/E-028/MODEL_CARD.md). The 0.7.0 tables below are historical, not the
> current release.
>
> **Update 2026-11-12 (later) — E-031/E-032.** **0.8.1 is published** (2026-11-12): a decode-only
> update, same int8 graph, new LM (`(1,2)` + term sentences ×3), package **19,706,289 B**.
> Published-model comparison: T-600 **24.08 / 55.23** (0.8.0 24.71 / 54.17; WER −0.63
> [−1.11, −0.20], ETA +1.06 [+0.09, +2.04]), T-S3 24.96 / 51.27 (ties), FLEURS-IT 20.16 / EN
> 49.81 (ties), rare aligned ETA **46.9% (38/81)** vs 28.4%, natural **432/817** (gate ≥431).
> Browser acceptance 4/4; decoder-level JS/Python parity exact on identical logits; the full
> noise curve is re-measured and all conditions tie. **E-032**, the S1 acoustic addendum, is
> **rejected**: rare 32.1% (0.8.0 LM) / 51.9% (E-031 LM) but T-S3, FLEURS and natural regress.
> Independent natural mixed-dev and a new-voice rare readout remain missing; the release is not a
> substitute for them. See [17](17-session-handoff-2026-11-12-e031.md).

> **Documentary audit update (2026-11-12):** current release is **0.8.1**, with **19,706,289 B
> of required assets**, not a whole-runtime footprint. The historical dossier below remains
> the 0.7.0 snapshot, including its then-local missing files and test failures.
> "Rare aligned ETA" for E-031/E-032 means all **81 span tokens in 47 rare-bearing utterances**
> (47 target + 34 co-occurring tokens); 24/47 presence is separate, not target-only ETA.
> E-031's original PLAN threshold is **0.284**, not the later historical E-028/E-030 **0.30**
> target; 38/81 passes both. Retained dev CIs are **pooled**, not macro; corrected macro deltas
> are in [E-031 RESULT](../runs/E-031/RESULT.md). The separately retained
> [audit readout](../runs/E-031/audit-readout.json) reports post-hoc macro CIs and target-only
> aligned ETA without changing the historical gate or providing independent-test confirmation.

> x3 adds **579 retained trigrams** (801,858 → 802,437); the shipped change combines corpus,
> pruning and weighting, not an isolated causal effect. An unchanged INT8 graph cannot establish
> unchanged quantization WER cost without a matched fp32 comparison using the same E-031 LM.
> Noise-condition ties and the descriptive 29.47/30.05 mean (no mean CI) establish neither
> equivalence nor an upper bound. E-032 S3 pairs are **26.75/46.76** (0.8.0 LM) and **26.41/47.61**
> (E-031 LM); its original table has six FAIL rows. No historical PLAN/JSON is rewritten.

> [Audit 18](18-audit-fixes-2026-11-12.md) records other agents' no-weight code fixes with
> verification in [audit 18](18-audit-fixes-2026-11-12.md); independent natural mixed-dev/holdout with new speakers and broader
> browser acoustic WER remain missing. No new trained model or executed gates are claimed.
> The Windows `resource` import blocker was fixed before this branch; CPU pytest can now collect
> on Windows, without certifying production training. The 2026-11-10 quoted failures below remain
> historical. No full LM build/staging, gate rerun or model release was performed by this task.

> **Update 2026-11-13 — E-033/E-034 rejected; E-035 advances on dev, no release change.** E-033
> pairs a control and an InterCTC arm (`weight 0.3`, training-only head) from the E-024 checkpoint
> with the E-028 recipe: dev macro **31.0350% vs 31.0101%** (**+0.0249 pp**, 95% CI
> [−0.0998, +0.1557]), a tie; the auxiliary head learns (66.9 → 0.72 over 394,625 training-only
> parameters) but adds no gain, so nothing is exported. E-034 decodes LM variants on dev only:
> keeping single-count trigrams is worth **−0.598 pp** `(2,1)` and **−0.644 pp** `(1,1)` macro
> dev, but 12.03/12.31 MB binaries give **26.36/26.64 MB** packages, over the 20 MB ceiling and
> the 25 MB tier; term weight ×5 is a tie, so the shipped ×3 is saturated. E-035 then adds
> **73,427 single-count trigrams** ranked by context frequency inside the headroom: dev macro
> **22.7849% vs 22.8494%** (**−0.0645 pp**, 95% CI [−0.1144, −0.0134]), binary **5,675,700 B**,
> package **19,999,997 B**; the control reproduces the shipped binary and E-034's hypotheses. The
> product readout then gives a **tie on all seven primary metrics** (T-600 +0.0287 pp, T-S3
> −0.0852, FLEURS-IT +0.0062, FLEURS-EN −0.0437, rare 38/81 in both, T-600 ETA identical), with
> only `noise_music` failing (+2 words) and no significant gain. The candidate is **not adopted**:
> the published weights, LM and package are unchanged and **0.8.1 remains the release**. See
> [E-033](../runs/E-033/RESULT.md), [E-034](../runs/E-034/RESULT.md),
> [E-035 dev](../runs/E-035/RESULT.md) and [E-035 readout](../runs/E-035/READOUT.md).

## 1. Identity and Architecture

The recorded published model is **Moth 0.7.0**, acoustic checkpoint **E-024**. The release labels
the checkpoint step **26,442**, but the training log selects **26,000**; this identity conflict
is unresolved without the actual checkpoint/hash. See E-024 in the [experiment ledger](12-experiment-ledger.md).
The [release record](../runs/E-024/release-0.7.0.json) and
[model card](../runs/E-024/MODEL_CARD.md) report publication on 2026-11-09. Current main adds
runtime infrastructure; it is **not a new trained model or proof of a new deployed demo**.
The Python project version `0.0.1` in [pyproject.toml](../pyproject.toml) is not the model version.

[`stt/model.py`](../stt/model.py) implements:

- Input: 16 kHz mono audio converted to 80-bin log-mel, 10 ms hop; model input `[B,80,T]`.
- Stem: Conv1d `80 -> 384`, kernel 3, stride 1, padding 1, GELU; Conv1d `384 -> 384`,
  kernel 3, stride 2, padding 1, GELU. Output rate 50 Hz, length `ceil(T/2)`.
- Whisper-tiny encoder: width 384, **four** pre-LayerNorm transformer blocks, six attention
  heads of dimension 64, FFN 1,536 with GELU, final encoder LayerNorm. Key projection has no bias.
  Attention is bidirectional, with a real-frame key-padding mask, not causal streaming attention.
- Fixed sinusoidal positions, maximum 1,500 encoder frames / 30 seconds; generated as a
  non-persistent buffer, not learned parameters or stored checkpoint weights.
- Release configuration adds **two full transformer blocks** after the encoder's final LayerNorm,
  then `Linear(384,1025)` and float32 log-softmax. These are not merely a linear head on four
  blocks. `MothSTT()` defaults to zero extra blocks; release construction requires `head_blocks=2`.
- CTC has no autoregressive decoder or language token. Blank is 0; SentencePiece ID `i` maps to
  CTC ID `i+1`, including the unknown piece. 1,024 BPE pieces give 1,025 output classes.

The module docstring and early design descriptions emphasize the original four-block
linear-head model. Read the implementation plus release config, not the introductory shorthand.
No punctuation/case restoration, word timestamps, diarization or overlapping-speech support is
demonstrated. This is record-then-transcribe, not streaming.

### Exact Parameter Count

Verified by instantiating the actual class, without loading weights:

```python
from stt.model import MothSTT
m = MothSTT(vocab_size=1024, head_blocks=2)
print({name: sum(p.numel() for p in module.parameters())
       for name, module in m.named_children()})
print(sum(p.numel() for p in m.parameters()))
```

| Component | Parameters |
|---|---:|
| Four-block encoder, including stem and final LayerNorm | 7,632,384 |
| Two post-encoder blocks, 1,774,080 each | 3,548,160 |
| CTC linear weight and bias, `384*1025+1025` | 394,625 |
| **Total acoustic network** | **11,575,169** |

All 11,575,169 are trainable in a fresh instance; E-024 sets `freeze_encoder: false`.
The old zero-extra-block model is **8,027,009**, not the released architecture. Counting includes
weights and biases but excludes buffers, optimizer state, LM counts and the **separate Silero VAD
network**. The VAD's parameter count was not established here. This verifies the source model's
configured count, not the contents of an absent checkpoint or ONNX graph. Parameter count, fp32
checkpoint bytes, quantized graph bytes and full runtime dependencies are different quantities.

## 2. Provenance and Lineage

### Tokenizer and Text

[`stt/text.py`](../stt/text.py) is the text rule. It applies NFKC/lowercase, folds apostrophes,
joins whitespace after apostrophes, separates digit colons, spells digit sequences using
`num2words` in the language hint, separates hyphens/dashes/slashes, keeps Latin letters/Italian
accented vowels/apostrophe, folds other Latin diacritics, drops other symbols and collapses spaces.
Mixed/unknown language uses Italian fallback in scoring. This is not language-aware mixed-span
numeric normalization. Training data filters digit transcripts; this affects benchmark coverage.

[`build_tokenizer.py`](../stt/data/build_tokenizer.py) defaults to normalized MLS-IT training
transcripts, LibriSpeech-EN training transcripts, the mixed training pool and
[`en_terms.txt`](../data/en_terms.txt). Italian is oversampled toward 70% **by transcript words**,
not audio hours; curated terms target about 5% of final corpus lines. SentencePiece BPE uses
identity normalization, no BOS/EOS/PAD, unknown ID 0, `hard_vocab_limit=False`.
Historical [STATUS](../STATUS.md) reports 100,429 base corpus lines, 1.945 tokens/word,
4.70 tokens/s and 48.8% of terms within three pieces. Those are **historically reported**, not
recomputed here: `data/tokenizer/` is absent and the current
[`build_report.json`](../data/manifests/build_report.json) contains **no `tokenizer` section**.
Do not cite that report as presently containing tokenizer hashes/statistics. Corpus assembly is
seeded; the builder explicitly does not promise byte-identical serialized SentencePiece models.
The release record pins exported `vocab.json`, not the original `.model` or training corpus.

### Acoustic Training

The following is lineage, **not** a sequence in which every numbered experiment initializes from
its predecessor. Sources are [EXPERIMENTS](../EXPERIMENTS.md) and each linked config.

| Stage | Initialization and intervention |
|---|---|
| Whisper conversion | [`convert_whisper.py`](../stt/convert_whisper.py): map pretrained `openai/whisper-tiny` encoder into local implementation; pretrained source is MIT |
| E-004 | Frozen converted encoder, train CTC head; E-005 uses the padded-arm head |
| [E-005](../runs/E-005/config.yaml) | Four-block full fine-tune, natural IT/EN mix, best at 62k; historical augmentation epoch-update bug means repeated augmentation views |
| [E-006](../runs/E-006/config.yaml) | Initialize E-005 best, add two random blocks, full fine-tune; augmentation bug fixed, so delta is not capacity alone |
| [E-014](../runs/E-014/config.yaml) | Initialize E-006 best, add 110 h utterance-level IT/EN splicing; does not initialize from E-012 or E-013 |
| [E-017](../runs/E-017/config.yaml) | Initialize **E-014**, add S1+S2 gold x50; not continued from E-016; historical releases 0.5.0/0.5.1 |
| [E-022](../runs/E-022/release-0.6.0.json) | Same E-017 acoustic weights and augmented LM; change decoder settings; release 0.6.0 |
| [E-024](../runs/E-024/config.yaml) | Initialize **E-014**, rerun E-017 recipe replacing about half of CV-IT by noisy versions; **not continued from E-017/0.6.0**; release 0.7.0 |
| [E-025](../runs/E-025/eval/result.json) | Evaluate E-017/E-024 on new noise recording; card update, **no new checkpoint** |
| [E-015 addendum](../runs/E-015/RESULT.md) | Later rare-term fine-tune rejected; does not replace 0.7.0 |

Natural training sources are Common Voice IT **27.0** (167.79 manifest hours), MLS Italian
(169.89 h), LibriSpeech train-clean-100 English (100 h). Their total is about 437.68 h;
with 110 h spliced exposures and 41.87 h oversampled gold the mixture is about **589.55 h**.
Manifest time and decoded/shard time differ slightly (e.g. CV 167.79 vs historical 167.31 h).
These are exposure hours, **not 589.55 unique hours**: splicing reuses natural clips and gold is
only 50.2 unique minutes, 641 S1 plus 622 valid S2 takes, repeated x50. One invalid S2 take stays
excluded without renumbering. Gold is roughly 7.1% of exposure time.

E-024: seed 0, two epochs, batch cap 192 audio-seconds / 32 clips, AdamW encoder LR 5e-5,
head LR 3e-4, betas `(0.9,0.98)`, eps 1e-8, weight decay .01, warmup 500, cosine floor .05,
gradient clip 1, full augmentation, bf16 training autocast with fp32 CTC, dev selection on
MLS-IT dev plus LibriSpeech-EN dev. The release's best-step label conflicts with the logged
minimum at 26,000; 26,442 is the completed training step count. Historical training
host is Apple M5/MPS, 16 GB; this audit did not reproduce training or verify checkpoint contents.

[`build_noisy_mel.py`](../runs/E-024/build_noisy_mel.py) hashes IDs for the clean/noisy split and
for a **fixed per-clip** 10-30 dB SNR, then caches noisy mels. Noise is not redrawn randomly each
epoch. Workers cycle the sorted pool of 217 MUSAN-noise/room-tone clips. This replaces roughly
83.6 h of CV, not half of all Italian read speech, and adds no training duration. The model card's
"half of its Italian read speech" is too broad. Noise augmentation is speech-plus-noise, not
empty-transcript training.

### LM and Release Assets

The [augmented trigram builder](../runs/E-008/build_lm_augmented.py) reads CV/MLS/LS **train**
text plus R-S1, R-S2 and pool scripts, not T/A. Backoff .4, bigram/trigram count pruning >=2.
The browser format is int16-quantized sparse CSR log-probabilities, not acoustic int8 weights.
E-023's larger LM was evaluated but not adopted. Tokenizer and LM text exposure are distinct
from acoustic exposure; later gold sentences targeted vocabulary discovered missing in T, so
"T was never directly trained on" does not establish a test-independent development process.

Release decoder: **beam 12, top-k 10, LM weight .5, per-token bonus 2.4**, no shipped hotword
bonus. E-022 changed several settings jointly; it is not an isolated causal estimate for the
bonus alone. Greedy is a separate acoustic-only mode, not the release quality default.

The [0.7.0 record](../runs/E-024/release-0.7.0.json) declares dynamic int8 ONNX, float32
log-probability interface, batch one/dynamic time. Its seven file entries are:

| File | Recorded Bytes | Identity / Role |
|---|---:|---|
| `model.int8.onnx` | 11,910,677 | SHA256 `7a075da40eb37de6988947efc717824db8b144769628db078e48dcc35b5ab0af` |
| `lm_bpe3.bin` | 5,116,464 | SHA256 `dcf1d59f075858c7e5f210804b5aabde847fb5d33e8d716e0d534f7c7886003e`; unchanged from 0.6.0 |
| `silero_vad.onnx` | 2,327,524 | Separate Silero VAD v5, MIT |
| `mel_filters.bin` | 64,320 | Float32 `80*201` filterbank |
| `vocab.json` | 17,193 | Exported CTC piece map |
| `config.json` | 1,322 | Version, preprocessing, decoder, asset hashes |
| `NOTICE` | 3,260 | Model/data attribution |
| **Sum** | **19,440,760** | **19.440760 decimal MB**, about **18.54 MiB** |

The record's `package_bytes=19,436,178` is **4,582 bytes smaller than its own file sum**; this is
exactly `config.json + NOTICE`. Payload and complete seven-file package must not be conflated.
The later [runtime report](../runs/runtime-2026-11-10/report.json) records the full 19,440,760 B.
Headroom under 20,000,000 B is 559,240 B. Neither count includes ORT JS/WASM, browser/app code,
Python environment, tokenizer training files, fp32 graph/checkpoint or download compression.
Hashes pin content consistency, not publisher authenticity; an immutable external release pin is
still needed. The release validator supports an explicit <=25 MB tier, not evidence that 0.7.0
used that tier.

## 3. Benchmark Ledger

All WER/ETA values below are **percent**. An absent ETA means no annotated spans, not zero.
FLEURS rows are the project's **filtered test subsets**, not official full-split leaderboard
WER: IT 697 utterances, EN 525, after text/duration filtering. T is scripted read mixed speech,
not a validated spontaneous conversational code-switch corpus. `n` counts utterances, not speakers.

Moth quality rows use decoder B = beam12/top10/LM .5/bonus2.4, Python pickle LM unless explicitly
marked browser binary. PyTorch rows are fp32 evaluation, **not bf16 training dtype** or browser
quality. Their provenance is [E-022 eval code](../runs/E-022/eval.py),
[E-024 gate summary](../runs/E-024/eval/gate/summary.json), and
[E-024 clean/noise code](../runs/E-024/noise_eval.py). These are not all full production VAD and
segmentation paths.

| Model / Decoder / Dtype | Split and n | WER | ETA | Committed Source |
|---|---|---:|---:|---|
| E-017 at old beam5/top8/LM .2/bonus0, fp32 | T-600, 600 | 32.77 | 42.86 | [old arm](../runs/E-022/eval/shipped/t600.metrics.json) |
| E-017 / B, fp32 = 0.6.0 reference | T-600, 600 | 25.92 | 52.64 | [frozen arm](../runs/E-022/eval/frozen/t600.metrics.json) |
| E-017 / B, fp32 | T-S3, 200 | 26.66 | 47.32 | [S3](../runs/E-022/eval/frozen/t-s3.metrics.json) |
| E-017 / B, fp32 | FLEURS-IT test, 697 | 20.46 | - | [IT](../runs/E-022/eval/frozen/fleurs_it_test.metrics.json) |
| E-017 / B, fp32 | FLEURS-EN test, 525 | 49.63 | - | [EN](../runs/E-022/eval/frozen/fleurs_en_test.metrics.json) |
| E-024 / B, fp32 = 0.7.0 reference | T-600, 600 | **25.20** | **54.17** | [T](../runs/E-024/eval/gate/t600.metrics.json) |
| E-024 / B, fp32 | T-S3, 200 | **26.92** | **48.45** | [S3](../runs/E-024/eval/gate/t-s3.metrics.json) |
| E-024 / B, int8 ONNX + Python LM | T-600, 600 | **25.14** | **54.65** | [int8 T](../runs/E-024/eval/gate/int8/t600.metrics.json) |
| E-024 / B, int8 ONNX + Python LM | T-S3, 200 | **27.00** | **48.45** | [int8 S3](../runs/E-024/eval/gate/int8/t-s3.metrics.json) |
| E-024 / B, fp32 | FLEURS-IT test, 697 | **20.14** | - | [IT](../runs/E-024/eval/gate/fleurs_it_test.metrics.json) |
| E-024 / B, fp32 | FLEURS-EN test, 525 | **49.14** | - | [EN](../runs/E-024/eval/gate/fleurs_en_test.metrics.json) |
| E-024 / B, fp32 | CV-IT test, 14,643 | **24.37** | - | [full CV test](../runs/E-024/eval/noisy/clean-cv_it_test.metrics.json) |
| E-024 / B, fp32 | CV-IT dev **first 600**, not 14,697 | **19.85** | - | [dev slice](../runs/E-024/eval/gate/cv_it_dev.metrics.json) |

T-600 fp32 has 3,488 reference words and 1,043 annotated English positions; S3 has 1,174 and
355 respectively. WER nominal 95% CIs: T-600 **[23.14,27.14]%**, S3 **[23.40,30.30]%**.
S1/S2 are training speakers (400/600 clips); S3 contributes no acoustic training audio but shares
S1's microphone and has been repeatedly inspected. The card's "S3 behind both trained voices"
is also numerically imprecise: S2 is 26.93 vs S3 26.92. Neither this ordering nor aggregate wins
establish broad unseen-speaker generalization.

### Baseline Matrix

Each directory below contains metrics plus `meta.json`; audio-inference reruns were not done.
Whisper uses `fp16=False`, padded 30 s, no hotwords. Whistle uses native cactus-needle with no
hotwords and its published compressed `.cact`; exact kernel dtype/beam is **not recorded in the
baseline meta**, so do not label it fp32 or claim a matched beam budget. E-019 describes a W4A8
archive, but that is not a full precision audit of the earlier engine. Parakeet uses the int8
sherpa-onnx transducer export, greedy search, four threads.

| System / Decoder / Dtype | Split and n | WER | ETA | Source |
|---|---|---:|---:|---|
| Whisper-tiny, auto beam5, fp32 | FLEURS-IT 697 / EN 525 | 27.42 / 11.86 | - | [auto baseline](../baselines/runs/whisper-tiny-auto-beam5/) |
| Whisper-tiny, forced IT beam5, fp32 | T-600, 600 | 50.72 | 36.05 | [T baseline](../runs/E-017/baselines-t600/whisper-tiny-it-beam5/) |
| Whisper-tiny, auto beam5, fp32 | T-600, 600 | 68.86 | 39.50 | [auto T](../runs/E-017/baselines-t600/whisper-tiny-auto-beam5/) |
| Whisper-tiny, forced IT beam5, fp32 | T-S3, 200 | 46.51 | 38.03 | [S3 baseline](../runs/E-017/baselines/whisper-tiny-it-beam5/) |
| Whistle, forced IT, native engine precision unspecified | T-600, 600 | 29.42 | 49.28 | [T](../runs/E-017/baselines-t600/whistle-it/) |
| Whistle, forced IT, native engine precision unspecified | T-S3, 200 | 28.36 | 52.11 | [S3](../runs/E-017/baselines/whistle-it/) |
| Whistle, forced IT, native engine precision unspecified | FLEURS-IT 697 / EN 525 | 11.85 / 13.35 | - | [read speech](../baselines/runs/whistle-it/) |
| Whistle, auto, native engine precision unspecified | FLEURS-EN, 525 | 11.70 | - | [auto EN](../baselines/runs/whistle-it-auto/) |
| Whistle, forced IT, native engine precision unspecified | CV-IT test, 14,643 | 20.63 | - | [full CV](../baselines/runs/whistle-it-cv/) |
| Parakeet TDT 0.6B v3, int8, greedy transducer | T-600, 600 | 10.98 | 83.60 | [T](../runs/E-017/baselines-t600/parakeet-v3/) |
| Parakeet TDT 0.6B v3, int8, greedy transducer | FLEURS-IT 697 / EN 525 | 3.65 / 8.17 | - | [read speech](../baselines/runs/parakeet-v3/) |

Whisper CV-test and Parakeet CV-test measurements are absent; no invented cells. E-019's
Whistle CV result (17.59 -> 20.25) is the **first 1,000 clips**, not this full-test baseline.
Do not substitute old T-200/T-400 results for T-600.

Size corrections for fair comparisons: Whisper baseline meta records **148,738,560 parameter
bytes**, not the card's **39 MB** (roughly 39M is a parameter-count description). Whistle base
`.cact` is **16,919,407 B**, not its **220,618,620 B** training safetensors checkpoint.
Parakeet meta records **670,478,772 B = 670.48 decimal MB = 639.42 MiB**, while prose often calls
it "639 MB". Comparing these to Moth's entire seven-file package is not like-for-like: declare
whether VAD, tokenizer, runtime libraries and compression are included for every system.

### Cross-System Gate Conflict

The 0.7.0 card quotes S3 versus forced-IT Whisper as WER -19.51 pp and ETA +8.73 pp (tie), and
versus Whistle as -1.70 / -4.79 pp. Those are not the differences of the current committed
E-024 and E-017 S3 metrics; they resemble E-022's older gate table. Recomputed with
`stt.eval._compare(a,b,bootstrap=1000,seed=0)`, using the files linked above:

| 0.7.0 fp32 S3 vs | WER Difference, 95% CI (pp) | ETA Difference, 95% CI (pp) |
|---|---|---|
| Forced-IT Whisper S3 | **-19.59 [-24.62,-14.24]** | **+10.42 [+0.28,+21.10]** |
| Whistle S3 | -1.45 [-5.38,+2.62] | -3.66 [-13.11,+5.90] |
| E-022 frozen S3 | +0.26 [-1.37,+1.96] | +1.13 [-2.31,+4.53] |

The S3 ID sets and per-ID `ref_words`/`en_ref` match for the Whisper comparison. Direct-call
E-024 metrics lack a manifest hash envelope; `_compare` itself only checks ID sets, not full
reference text or normalizer identity. These are **audit recomputations from retained counts**,
not new independent experimental wins or replacements for a locked test. In particular, the
nominal ETA classification changes, so neither blindly repeat the card's tie nor announce a
confirmatory gate pass. Reconcile baseline provenance, exact hypotheses and reference hashes in
one regenerated gate table before publication. E-022's own historical gate-table S3 values also
need tracing to the exact baseline it compared, not merely a matching display name.

### Noise Evidence

The following are fp32 checkpoint inference with decoder B, **not int8 browser** measurements.
Each condition uses the first 100 S3 rows (not a random new speaker sample), 589 reference words
and 148 annotated English positions. The two arms receive identical audio within each condition.

| SNR | E-024 Shared Pool: 0.6 -> 0.7 WER | Delta pp, 95% CI | E-025 New Room: 0.6 -> 0.7 WER | Delta pp, 95% CI |
|---|---|---|---|---|
| clean | 22.24 -> 23.94 | +1.70 [-0.34,+3.72] | 22.24 -> 23.94 | same audio/counts |
| 30 dB | 25.98 -> 23.26 | -2.72 [-5.63,+0.17] | 28.52 -> 24.62 | -3.90 [-7.20,-0.83] |
| 20 dB | 27.50 -> 26.83 | -0.68 [-4.13,+2.73] | 38.71 -> 31.92 | -6.79 [-10.84,-3.15] |
| 15 dB | 33.62 -> 27.33 | -6.28 [-10.58,-2.38] | 47.71 -> 36.50 | -11.21 [-14.69,-7.48] |
| 10 dB | 44.82 -> 33.28 | -11.54 [-16.18,-7.43] | 63.33 -> 44.99 | -18.34 [-22.78,-13.40] |
| 5 dB | 50.93 -> 41.26 | -9.68 [-14.50,-5.01] | 81.32 -> 66.72 | -14.60 [-18.84,-10.47] |

Sources: [E-024 comparison](../runs/E-024/comparison.json),
[E-025 result](../runs/E-025/eval/result.json),
[room report](../runs/E-025/room_report.json),
[holdout evaluator](../runs/E-025/holdout_eval.py). Mean deltas -6.18 and -10.06 pp are simple
means at **30/20/15/10 dB only**, not a pooled corpus statistic or a mean including 5 dB; no CI
for the mean is provided. All E-025 ETA-difference CIs include zero.

The room is one 38 s trimmed take from a 78.69 s recording, author-reported different physical
room, recorded after training, VAD speech fraction 0.0. Its new-noise identity is distinct from
the reused speech. Random offsets still share one take, so conditions/utterances are correlated.
RMS mixing uses the whole clip, tiles short noise, scales noise then peak-normalizes mixtures
above one. Nominal SNR is not active-speech SNR or a measured room acoustic condition. A larger
gain on the harsher room does not mean better absolute performance: **36.50% at 15 dB** is worse
than **27.33%** on the pool. Training used all 217 pool clips; a retrospective 80/20 split cannot
repair this without retraining. Shared-pool estimates are potentially optimistic, not rigorous
upper bounds. The card's later "38.4% on the same-pool curve" conflates an earlier audit probe
with the paired E-024 control's **33.62%** at 15 dB.

Clean CV-test improves -0.52 pp [-0.66,-0.38] under the utterance bootstrap, but one seed cannot
separate augmentation benefit from run variance. S3 clean +0.26 pp CI extends to +1.96 pp:
a point-estimate +1 pp adoption rule is not a confidence-bound non-inferiority proof.

## 4. Runtime: Measured Versus Implemented

Actual preprocessing is [features.py](../stt/features.py): periodic Hann, FFT 400, hop 160,
reflect-centered STFT, last frame dropped, Slaney filters, log10 floor 1e-10, per-clip dynamic
range floor 8, `(x+4)/4`. Browser reference is [mel.js](../web/mel.js).
Current [ReleaseMoth](../stt/onnx_runner.py) validates a supplied release, loads ONNX CPU sessions,
binary LM and release filters once, accepts finite mono PCM in [-1,1] at exactly 16 kHz (no
resampling), runs Silero, then segments only if at least two speech runs exist. The no-speech
gate uses the fraction of windows above the VAD window-probability threshold; the fraction must
exceed **.10**. The .10 fraction gate is not the per-window speech-probability threshold.
Each selected span must be 560 samples to 30 s; oversized speech spans raise, not silently crop.
Current [web runner](../web/ort.js) has its own validation/concurrency behavior. Neither runtime
is a streaming implementation; segmentation is not diarization.

[release.py](../stt/release.py) validates config semantics, safe asset paths, declared hashes and
budget; [BinaryBPELM](../stt/lm.py) reads `MOTHLMB1`, so the **current Python release path does not
need pickle, PyTorch or SentencePiece**. Historical experiment scripts do. Runtime dependencies
are numpy/onnxruntime/soundfile/num2words; training/export extras remain separate in
[pyproject.toml](../pyproject.toml). These source improvements do not retroactively change the
historical benchmark or published Space code.

| Evidence / Path | Workload and Conditions | Observed Performance | Interpretation |
|---|---|---|---|
| [E-024 browser/release record](../runs/E-024/release-0.7.0.json) | Historical Apple M5 Chromium/WASM SIMD, 5/10/20 s fixtures, VAD+mel+int8+beam/LM, 10 runs per cell | 4T **45.26/43.17/38.43x**; 1T **24.33/22.91/20.32x** | Actual 0.7.0 recorded speed; the card's opening 47x/24x is not its full current table |
| [E-024 parity](../runs/E-024/parity-wer.json) | Historical 200 S1 clips, browser vs Python int8, different mel front ends | Browser 21.50 vs Python 21.58 WER; **-0.09 pp [-1.19,+1.13]**, 152/200 identical | No detected WER gap, not exact equivalence or unseen-speaker evidence |
| [E-022 parity](../runs/E-022/parity-wer.json) | Earlier E-017 weights, same 200 S1 clips | New decode -0.09 pp **[-1.22,+0.96]**, 154/200 identical; old decode +1.29 **[+0.09,+2.59]** | Historical prose mixed fraction endpoints with pp; do not merge E-022 and E-024 intervals |
| [2026-11-10 native acceptance](../runs/runtime-2026-11-10/RESULT.md) | Windows x86_64 Intel, ORT 1.31 CPU/default threads, unchanged 0.7.0 assets, 104 clips/1,305.6 s, no warmup | VAD+segmentation **24.87x**, no-VAD **36.35x**; load .280 s separately | Local current Python acceptance on reused FLEURS/splices, not new release quality or browser acceptance |

RTF = elapsed seconds / audio seconds; RTFx = reciprocal (larger is faster). State included
stages, loading, warmup, sample duration and threads for each claim. E-003 random-weight speed,
E-010 pre-VAD speed and E-021/E-022 timings are not 0.7.0 timings. The static HF Space's historical
single-thread/approximately 19x assertion is in STATUS, not a new audit of the live endpoint.
The 19.44 MB model asset count excludes actual browser runtime delivery size.

Native acceptance scores **100 speech utterances plus four synthetic controls**. WER is 40.4817%
with VAD vs 39.8213% without, difference **+0.660 pp [-0.456,+1.852]**; no quality gain shown.
ETA is null because all `en_spans` are empty. It uses known public FLEURS and correlated splices,
not natural code-switching. Identical-logit JS/Python checks are 10/10 beam and 10/10 greedy;
they isolate decoders on native ONNX outputs with Python mel, **not browser/WASM acoustics or
JS-feature parity**. Timing excludes IO/hash, scoring, asset load and Node IPC; serial paths,
no warmup and shared host limit speed comparison. The report's four silent/noise controls do not
replace the historical 337-clip nonspeech evaluation.

Historical export verification records fp32 median absolute gap 1.888e-4, worst 1.217e-3 over
80 draws; gates median <=1e-3, worst <=1e-2. Synthetic int8 top-1 agreement 1.0 is not a WER or
cross-platform quality certificate. E-020's +0.32 pp FLEURS int8 effect concerns earlier E-017
weights/decoder; it is not a measurement of 0.7.0 on all long/out-of-domain speech.

## 5. What the Evidence Supports

- A configured 11,575,169-parameter CTC model and a recorded approximately 19.44 MB asset package,
  with retained training configs, logs, hypotheses/counts and many paired comparisons.
- Large decoder-setting improvements on unchanged E-017 weights (E-022), and noise-condition WER
  improvements for E-024, including transfer to one new noise recording (E-025).
- Better Italian filtered-FLEURS and scripted-T performance than the specified Whisper-tiny
  baselines, with poor pure-English FLEURS performance still visible. These are system/decoder
  comparisons, not matched pretraining, capacity, compute or architecture-only experiments.
- Gold benefits observed beyond audio-trained voices on S3, while the aggregate T number is
  optimistic for new speakers. Repeated S3 inspection prevents treating this as a fresh final test.
- Historical browser operation and current native CPU acceptance. Neither proves all browsers,
  microphones, platforms, noise environments or a complete dependency footprint meet the claims.

The evidence **does not** establish architecture-bound gold causality, impossibility of decoder
adaptation, seed-robust small gains, unbiased final-test superiority, spontaneous code-switching
performance, general far-field/noisy-room robustness, statistical equivalence from a "tie", or
performance forecasts for S4, InterCTC, a larger LM or more gold. E-019 is one failed LoRA recipe
with different replay/update/data exposure, not a controlled CTC-versus-decoder experiment.

## 6. Local Reproducibility Limits

At audit time this checkout has **no `release/`**, `models/`, `data/raw/`, `data/shards/`,
`data/tokenizer/`, `recordings/audio/` or E-024 checkpoints. Historical files refer to paths on
the original machine. The augmented LM pickle **is present**, but does not restore the missing
tokenizer, acoustic weights or audio. `stt/assets/` reference assets are not the complete release.
The directory `runs/browser-acceptance-2026-11-10/` is empty: its name is not acceptance evidence.
The native acceptance report used sibling release/evaluation directories; they are outside this
checkout and are not bundled reproduction prerequisites. No release download, live publication
verification, inference rerun or training was performed in this documentation audit.

Saved per-utterance counts allow descriptive arithmetic/CI checks without weights. Saved
hypotheses plus a matching manifest allow scorer reruns. Rebuilding requires the original
tokenizer, model checkpoints and exact input data; re-running the current builder against mutable
scripts is not guaranteed to reproduce an older LM/tokenizer. Some direct-call metrics lack
git/checkpoint/manifest/normalizer hashes, so filenames alone cannot bind them to a release.
The [native acceptance result](../runs/runtime-2026-11-10/RESULT.md) records Windows portability
test failures and Unix-only `resource` imports in training; its historical suite totals are not
claims that all current tests pass here.

## 7. Paper Readiness

**Suitable now for a carefully bounded engineering/negative-results case study; not yet a
confirmatory general code-switching, architecture-causality or deployment-wide superiority claim.**
Concrete deliverables before making those stronger claims:

| Gap | Needed Artifact / Experiment |
|---|---|
| Immutable model identity | Archive all seven 0.7.0 assets at an immutable revision, full hashes/file sums, trusted release manifest, export environment/opset/quantization config; bind checkpoint SHA and source commit to graph |
| Training/tokenizer provenance | Recover original SentencePiece `.model`, vocab, corpus list/hashes and build report; checkpoint hashes, complete resolved configs, environment lock and raw/shard manifest hashes; record actual valid durations, duplicates and exposure vs unique hours |
| LM reproducibility | Freeze exact training transcript/script snapshots, corpus hashes and per-source contributions, tokenizer identity, pruning/backoff and binary exporter version; regenerate and compare binary hash without reading test text |
| Benchmark integrity | Produce one machine-readable table binding every row to model/release, dtype, decoder, full reference/hypothesis hashes, normalizer/scorer version, n/speaker/channel counts and filtering; reconcile S3 gate conflicts and remove copied deltas |
| Multi-seed uncertainty | At least three predeclared seeds for matched E-017 control and E-024 noise arms from the same fixed E-014 initialization; retain every selected checkpoint/log/dev curve/test result, report per-seed deltas, mean and dispersion, not best-seed-only selection. This measures continuation variance; independently seeded upstream training is needed for end-to-end variance |
| Gold/architecture causality | Matched replay/data exposure and validation-selected budgets for gold/no-gold and full/adapter updates; multiple seeds, LR/rank/regularization controls and matched evaluation. Separate added speaker count from duration/repetition. Do not equate 45.3-min LoRA-only training with 50.2-min x50 replay |
| Independent mixed test | Recruit multiple new speakers and channels, hold out rooms/devices, freeze decoder/checkpoint/term taxonomy before recording or revealing references; collect spontaneous natural code-switching as well as read speech; publish consent/access policy and a separate mixed-dev set |
| Leakage audit | Source-level audio hashes, speaker IDs, normalized exact/near-text duplicates across train/dev/test/T/R/pool/A, plus provenance of every test-inspired vocabulary addition. Anchor A overlaps T and cannot be independent dev/test; splices must be split by constituent source audio, not just synthesized clip IDs |
| Upstream contamination | Disclose unknown Whisper/Whistle/Parakeet pretraining overlap with FLEURS/other public corpora; distinguish external-pretraining uncertainty, local audio leakage, tokenizer/LM text leakage and adaptive test reuse. New post-freeze recordings help but do not erase earlier test-guided design |
| Statistical analysis | Predeclare primary endpoints/comparisons and non-inferiority margins; enough independent speakers/rooms for clustered or hierarchical bootstrap, and training-seed variation reported separately. Current iid utterance bootstrap cannot infer a speaker population from one S3 or a room population from one noise take; address multiple testing |
| Noise generalization | Rebuild train/dev/test noise partitions by original source recording before training, retain source hashes; evaluate several independent rooms/transients/reverberation and unseen speech at controlled active-speech-aware SNR; include int8/runtime path and ETA, not only fp32 WER |
| Deployment evidence | Repeat native ARM64/x86 and browser WASM end-to-end with pinned assets/runtime/browser/OS/CPU, speech and nonspeech, warm/cold loads, memory and full raw/HTTP dependency bytes; baseline fp32 versus int8 on matched audio. Retain generated reports and explicit failed/skipped cases |
| Annotation validity | Publish span guidelines, word/term/phrase distinction, bilingual annotator agreement/adjudication, accent/gender/channel metadata subject to consent, rare-term strata and confidence intervals; avoid reporting a 47-item rare subset as 47 independent terms |

Forecasts remain forecasts: E-018's possible 2-3 pp, InterCTC's expected .2-.7 pp reduction,
extra noise/LM gains and E-015's coverage ceiling are not observed improvements. InterCTC's
reported +4.4% step-time probe is a cost measurement, not an accuracy result. No statistical
procedure here converts those estimates into evidence.

## 8. Verification Performed

Read repository guidance and inspected the implementation, scoring/normalization/bootstrap,
E-022/E-024/E-025 scripts/JSON/cards, lineage configs and baseline metrics/meta. Executed source
parameter counting, release file-size arithmetic, benchmark n/word/span checks and seeded S3
paired-count recomputations without writing experiment artifacts. Documentation changes are
restricted to this file and doc 11; experiment JSON, cards and runtime code remain historical
sources with the discrepancies above explicitly preserved.

Verification on this checkout: **207 passed, 2 skipped** across `test_eval`, `test_text`,
`test_model`, `test_release`, `test_release_runtime`, `test_onnx_runner` and `test_lm_binary`;
`ruff check stt tests baselines` passed, local Markdown link targets resolved, and
`git diff --check` passed. This is a targeted source/test check, not a full-suite or release
inference result. The subsequent broader audit ran `ruff check .` successfully and all **7 JS
runtime unit tests** passed. Full `pytest -q` stops during collection at `test_train.py` because
Windows lacks Unix `resource`. With that module explicitly excluded,
`pytest -q --ignore=tests/test_train.py` reports **450 passed, 3 skipped, 5 failed**: four in
`test_baselines_common` (native path separators / shell command quoting with spaces) and one in
`test_sources` (LibriSpeech path separators). These are existing main-platform limitations, not
fixed or suppressed by this documentation change. No browser/WASM end-to-end test was run.
No commit/push or changes to experiment artifacts were made.
