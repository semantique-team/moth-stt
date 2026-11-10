# 12 - Experiment Ledger and Benchmark Evidence

## Scope and Reading Rules

This is an artifact-led audit of the repository at **`main`, `cd1ce1b`**, dated
2026-11-10. It covers every assigned ID **E-000 through E-025**, including IDs without
executed runs, the November audits, runtime acceptance, and the release measurements.
It supplements, rather than silently rewrites, [EXPERIMENTS.md](../EXPERIMENTS.md) and
[the historical decision account](11-evidence-and-decisions.md). Decisions below are
historical adoption decisions, **not blanket certification that every product gate passed**.

Evidence precedence is: retained per-utterance counts/hypotheses and executed code/config;
machine-readable comparisons and release records; contemporary result notes; retrospective
narrative; plans. A config describes a recipe, not proof that every planned evaluation ran.
Where those sources disagree, the disagreement is retained below. Checkpoints, tokenizer
models, audio and mel shards are generally excluded from git; a checkpoint *path* is not a
weight hash. Missing artifacts are not reconstructed by guessing. This audit reads and
cross-checks existing evidence; it does not rerun training or inference.

**Units.** All tables use WER and ETA in **percent**, and differences/CIs in **percentage
points (pp)**, unless explicitly marked otherwise. Lower WER is better; **higher ETA is
better**. JSON generally stores fractions: `0.012` is 1.2 pp, not 0.012 pp. RTF is elapsed
seconds/audio seconds; RTFx is its inverse. MB means decimal bytes/1,000,000; MiB is
bytes/1,048,576. Package size, compressed Python LM size and browser binary LM size are
different quantities.

**Scoring.** [The official scorer](../stt/eval.py) normalizes hypotheses with
`stt.text.normalize`, aligns words by minimum edit distance, and computes corpus WER
from summed substitutions/deletions/insertions and reference words. ETA counts exact
aligned matches at annotated English reference positions (`en_spans`), not semantic
equivalence. Mixed/unspecified language falls back to Italian normalization. Empty
reference rows enter the noise word-count block, not ordinary WER; `overall.wer=0`
on an all-noise file is **not** perfect speech recognition. Unless specified otherwise,
intervals use 1,000 seeded utterance-level bootstrap resamples, seed 0. A CI including
zero is called a **tie**, which means no detected difference, not proven equivalence
or non-inferiority. These intervals do not measure training-seed or speaker-population
uncertainty.

Evidence labels used here:

- **Measured:** retained numerical outputs support the stated measurement.
- **Reported:** a contemporary note gives the value, but an independent raw result is absent.
- **Exploratory:** measured, but used in search, test inspection or a diagnostic subset.
- **Planned / not run:** no executed arm is evidenced; predictions are not results.
- **Unresolved:** available records do not uniquely establish the claimed metadata or conclusion.

## Benchmark Inventory

| Benchmark | Actual population and role | Interpretation constraint / artifact |
|---|---|---|
| MLS-IT training | 40,953 clips, 169.89 h | Sampled from 247.4 h surviving text filters; [build report](../data/manifests/build_report.json) |
| LibriSpeech EN training | 28,370 clips, 100.00 h train-clean-100 | Audiobook English, not spontaneous technical speech |
| Common Voice IT 27 training | 113,118 clips, 167.79 manifest h; about 167.31 cached h | 593 normalized-sentence overlaps removed against evaluation/script hashes; manifest and cache duration accounting differ |
| Extra LS360, E-012 only | 57,198 clips, 199.99 h | Deterministic 200 h sample, not the whole 360 h split |
| Checkpoint-selection dev | MLS-IT dev 1,248 / 40,407 words plus LS EN dev 2,691 / 53,631 words | 3,939 clips, about 10.48 h; unweighted mean of the two corpus WERs, **not pooled WER** |
| Early decoder-selection dev | 200 MLS + 200 LS clips, seed 0 | A subset, not the full-dev macro; E-023 later uses all 3,939 |
| FLEURS-IT test | 697 clips, 16,035 words, 2.74 h | Filtered public test, not the full upstream split; possible baseline pretraining contamination |
| FLEURS-EN test | 525 clips, 11,438 words, 1.39 h | Same caveat; pure-English recognition remains weak for Moth |
| CV-IT dev, full | 14,697 clips, 147,051 words, 24.86 h | Early H2 gate; Whisper forced-IT greedy reference 48.53 |
| CV-IT dev, first 600 | 600 clips, 5,945 words | E-022, E-024 gate, E-015 clean readouts; **not comparable to full CV-dev WER** |
| CV-IT test, full | 14,643 clips, 147,562 words, 25.81 h | Used by early Moth, full Whistle baseline, E-023/E-024 |
| CV-IT test, first 1,000 | 1,000 clips | E-019 LoRA direction check only; 17.59 baseline is not the full-test 20.63 |
| T-200 / T-S1 | 200 clips, 1,163 words, about 481 s | Only recorded mixed test in E-005 through E-014; S1 later supplies gold |
| T-400 | S1 + S2, 400 clips, 2,314 words | E-016 LM/VAD addenda; do not compare its aggregate directly with T-200 or T-600 |
| T-600 | S1 + S2 + S3, 600 clips, 3,488 words, about 23.2 min | S1/S2 are training speakers for E-017/E-024, **400/600 clips**, not 200/600 |
| T-S2 | 200 clips, 1,151 words | Unseen voice/channel for E-016; **seen** voice for E-017 onward |
| T-S3 | 200 clips, 1,174 words | No S3 training audio; shares S1's Mac/microphone. Repeatedly inspected, not a fresh final test |
| T-800 | Planned script: 200 per S1-S4 | **No complete recorded T-800 result**; S4 is absent |
| Anchor A | 60 sentences per recorded speaker, S1-S3 available | Script/channel control asset, not a separate headline quality benchmark here |
| Original nonspeech set | 360 ten-second clips: 120 zeros + 120 MUSAN noise + 120 MUSAN music | 40 min non-music, 20 min music, seed 0, natural levels |
| Expanded nonspeech set | Above plus 97 ten-second room-tone clips | 457 total; 337 non-music = **56.17 min**, not one hour; recording source 974.8 s |
| Controlled noisy speech | First 100 T-S3 clips, 589 words | Synthetic RMS-relative SNR, 30/20/15/10/5 dB; not a natural noisy-speech corpus |
| E-025 noise holdout | Same 100 speech clips + one new 38 s room take | Noise stimulus held out, **speech set reused** |
| Windows runtime sample | 104 reused clips, 1,305.6 audio s | FLEURS IT/EN, artificial splices and four controls; no annotated natural mixed ETA |

Training data filtering is documented in [the manifest builder](../stt/data/build_manifests.py)
and [build report](../data/manifests/build_report.json): deterministic duration sampling,
1-30 s duration, text/charset/digit/speaking-rate filtering, evaluation hashes seeded before
training. FLEURS-IT drops 162 digit-containing and six duration-invalid test rows;
FLEURS-EN drops 121 digit-containing and one charset-invalid test rows. Historical corpus
hours are rounded and distinguish manifest durations from cached decoded durations.

## Shared Model and Training Protocol

The initial Moth is the pretrained Whisper-tiny **encoder**, four transformer blocks,
width 384, six heads, plus a linear CTC projection: **8,027,009 parameters**, not the
39 M parameter full Whisper encoder-decoder. The shared SentencePiece BPE has 1,024
pieces and CTC has 1,025 labels including blank 0. E-006 adds two transformer head
blocks: **11,575,169 parameters**. E-013 alone uses BPE 4,096 / CTC 4,097:
**12,757,889 parameters**. No language token is required by Moth's CTC output.

Features are 16 kHz mono, 80-bin Whisper-style log-mel at 100 Hz, stride-two encoder
output at 50 Hz, generated sinusoidal positions and padding masks. Training caches
are float16; training autocast bf16 is **not** the precision of the released int8 model.
PyTorch evaluations below are the unquantized reference, unless marked ONNX/int8.

Executed Moth training logs record Apple M5/MPS, torch 2.14.1, seed 0. Standard full
training uses AdamW betas `(0.9, 0.98)`, epsilon `1e-8`, weight decay `0.01`, fp32 CTC,
gradient clip 1.0, warm-up then cosine to 5% of peak LR. E-005 onward uses truncated
inputs, batches bounded by 192 audio seconds / 32 clips, full SpecAugment/tempo
augmentation, and macro greedy dev WER for checkpoint selection. The E-005 epoch
augmentation bug is an explicit exception. Checkpoint retention is best plus last
three; periodic saves are about 10-20 min. Reported financial cost is **0 USD external
spend**, not zero electricity, author time or hardware cost.

| Training arm | Initialization | Epochs / completed steps | Head LR / encoder LR; warm-up | Selected dev macro / step | Start SHA in log |
|---|---|---|---|---|---|
| E-004 A | Converted tiny encoder, fresh head; encoder frozen | 3 / 5,871 | 1e-3 / frozen; 500 | 75.769 / 5,871, MLS only | `14ef3f5` |
| E-005 | E-004 padded-arm best, including head | 8 / 72,592 | 1e-3 / 1e-4; 1,500 | 33.772 / 62,000 | `03ecd4b`, later resume SHAs differ |
| E-006 | E-005 best + two random new blocks | 4 / 36,296 | 1e-3 / 5e-5; 500 | 31.273 / 36,296 | `5631c78` |
| E-012 | E-006 best | 2 / 26,028 | 3e-4 / 5e-5; 500 | 29.501 / 26,028 | `72cf112` |
| E-013 | E-006 encoder/blocks, fresh larger projection | 2 / 18,148 | 1e-3 / 5e-5; 500 | 32.788 / 18,148 | `72cf112` |
| E-014 | E-006 best | 2 / 22,494 | 3e-4 / 5e-5; 500 | 30.860 / 22,494 | `50d98b8` |
| E-016 | **E-014 best**, not E-006 or E-016 continuation | 2 / 24,478 | 3e-4 / 5e-5; 500 | 30.904 / 24,000 | `45ad463` |
| E-017 | **E-014 best**, not E-016 | 2 / 26,442 | 3e-4 / 5e-5; 500 | 30.984 / 26,442 | `638217f` |
| E-024 | **E-014 best**, not E-017/0.6.0 | 2 / 26,442 | 3e-4 / 5e-5; 500 | **30.837 / 26,000** | `b5dfe9f` |
| E-015 actual addendum | E-024 best | 1 / 13,252, including 20-step smoke/resume | 3e-4 / 5e-5; 500 | 30.814 / **20** | `9128657` |

Sources: individual `config.yaml` and `train.log.jsonl` linked in the entries below.
The E-024 release record labels best as step 26,442, conflicting with its log; the
selected checkpoint bytes cannot be reconstructed from that label alone. The E-015
step-20 checkpoint is **near baseline after 20 updates**, not literally an untouched
baseline. Dev metrics in this table are greedy selection metrics, not beam+LM results.

## E-000 - Mac Feasibility (2026-10-18)

**Question.** Could the M5 16 GiB machine train the proposed model within a local budget?
**Protocol.** Throwaway random-weight `nn.TransformerEncoderLayer` stand-in, exact
approximately 8.03 M shape, V=1,024; synthetic `[32,80,1000]` mel (32 ten-second clips,
500 output frames, 50 targets each). Forward + CTC + backward + AdamW; fp32 versus bf16
autocast, MPS/CPU CTC settings, torch 2.14.1. No real data, pretrained checkpoint,
decoder or train/dev split.

| Setting as reported | ms/step | Audio seconds/s | Peak memory GiB |
|---|---:|---:|---:|
| fp32, CTC MPS | 578 | 554 | 4.3 |
| bf16, CTC MPS | 394 | 812 | 4.4 |
| fp32, CTC CPU | 577 | 554 | 5.4 |
| bf16, CTC CPU | 432 | 741 | 5.4 |

**Decision / evidence.** Feasibility accepted; MPS bf16 selected. **Reported only** in
[E-000 research log](../EXPERIMENTS.md#e-000--mac-feasibility-benchmark-2026-10-18);
there is no retained `runs/E-000` benchmark or sustained real-data profile. The
projected 7-11 min per 100 h epoch and 5-10 h H2 estimate are projections, not measured
training results. Later real-data memory stalls limit extrapolation from this smoke.

## E-001 - Baselines (2026-10-19, Completed Matrix 2026-10-23)

**Question.** Establish the actual comparison bar, including language forcing and search.
**Protocol.** Evaluation only. OpenAI Whisper tiny, 39 M, fp32 CPU, padded 30 s,
auto language versus forced Italian, greedy/beam 5; Whisper base greedy reference.
Whistle uses cactus-needle auto/forced `it`; Parakeet TDT 0.6B v3 multilingual int8
uses sherpa-onnx. No hotwords. Same filtered FLEURS and normalized T references.
Baseline engine-specific decoding is not an architecture-controlled experiment.

Initial tiny greedy95% CIs: auto IT32.12 [30.96,33.38], EN13.86
[12.83,15.04]; forced-IT IT32.09 [30.91,33.33], EN81.90 [78.63,84.90].
Reported CPU RTFx IT/EN auto53.3/41.8, forced65.5/8.7; T-S1 auto/forced
greedy4.6/7.0 and auto/forced beam3.9/4.1. Whistle reported FLEURS/T
auto121.8/99.4, forced130.0/116.4; Parakeet28.1/29.5. These runners include
different language/search costs and shared-host contention, not matched kernels.
Reported serialized sizes: tiny148.74 MB, base287.3 MB, Whistle16.92 MB,
Parakeet639 MB. Large teacher/baseline files are not the Moth runtime budget.

| System / variant | FLEURS IT | FLEURS EN | T-S1 WER | T-S1 ETA |
|---|---:|---:|---:|---:|
| Tiny auto greedy | 32.12 | 13.86 | 75.75 | 36.72 |
| Tiny forced IT greedy | 32.09 | 81.90 | 62.25 | 25.37 |
| Tiny auto beam 5 | **27.42** | **11.86** | 70.51 | 38.21 |
| Tiny forced IT beam 5 | Not run | Not run | **53.05** | 37.61 |
| Base forced IT greedy | 18.27 | Not run | Not run | Not run |
| Base auto greedy | Not run | 9.88 | Not run | Not run |
| Whistle auto | 11.85 | 11.70 | 36.11 | 50.75 |
| Whistle forced IT | 11.85 | 13.35 | 29.49 | 50.15 |
| Parakeet v3 int8 | 3.65 | 8.17 | 10.66 | 83.88 |

Additional full-CV references: tiny forced-IT **greedy dev 48.53**, Whistle forced-IT
**test 20.63**. Tiny/Parakeet full CV-test and base beam-5 cells are absent. Do not
compare Whistle E-019's first-1,000 CV baseline 17.59 with this full-test 20.63.

**Numerical checks.** Initial greedy tiny auto-minus-forced IT: IT +0.03 pp (tie),
EN -68.05 pp. Corrected E-008 versus tiny auto-beam5 FLEURS-IT: +1.60
[+0.32,+2.83], a loss, not an H2 pass. T-S1 E-008 versus tiny forced-IT beam5:
WER +2.32 [-3.22,+8.02], ETA -16.12 [-25.31,-6.02].

**Decision.** Accept the baselines as comparison anchors; reject forced IT for pure
English. Correct the earlier **32.09 greedy** FLEURS-IT bar to the best measured
**27.42 beam** variant. The full baseline matrix is still incomplete. Tiny auto-beam
has slightly higher T-S1 ETA than forced-IT beam; "best WER variant" does not mean
best on every metric. Public FLEURS may have been seen during baseline pretraining.

**Artifacts / status: measured.** [Baseline runs](../baselines/runs/),
[Whisper auto beam FLEURS](../baselines/runs/whisper-tiny-auto-beam5/),
[T-S1 forced beam](../baselines/runs/whisper-tiny-it-beam5-t/),
[CV tiny chunk metadata](../runs/E-005/baselines/tiny_chunks/meta.json),
[Whisper base results](../runs/E-005/baselines/), and later
[T-600 baseline matrix](../runs/E-017/baselines-t600/).
There is no dedicated `runs/E-001` directory. CPU throughput is indicative, especially
short T runs sharing a host; the browser speed gate is a separate measurement.

## E-002 - Dataset and Cache Build (2026-10-19, CV Added 2026-10-21)

**Question / protocol.** Can the licensed open sources provide the training/evaluation
budget without normalized-sentence leakage? Build evaluation manifests first, seed
overlap hashes, deterministic seed-0 sampling, then filter training. No model,
checkpoint, optimizer or decoder. MLS/LS/FLEURS CC-BY 4.0; CV IT 27 CC0.

**Outcomes.** Initial train 169.89 h MLS + 100.00 h LS = 269.89 h. MLS dev/test
1,248/1,262, LS dev 2,691, FLEURS IT dev/test 334/697 and EN 343/525. Initial cache
291.90 h, 18 float16 shards, about 16 GB, zero decode errors, reported five-clip-per-set
bit-exact raw/mmap checks. CV adds train 113,118 clips/167.79 manifest h, dev
14,697/24.86 h, test 14,643/25.81 h; drops train speaking rate 1,216, overlap 593,
charset 146, empty two, duration one. CV cached train 167.31 h, dev/test 24.83/25.78 h;
total cache about 509.8 h/27 GB. Training mix about **437.7 h**, IT/EN **77/23 by
duration**, not the design shorthand 70/30. Official CV speaker disjointness and raw
audio deletion after parity checks are contemporary reports, not a fresh audio audit.

**Decision / evidence.** Data build accepted. FLEURS IDs use recording filename stems,
not repeated sentence IDs; CV IDs use clip stems, not speaker IDs. Raw MLS/LS deletion
saved disk but later blocks raw-audio augmentation without re-download. Tokenizer
statistics 1.945 tokens/word, 4.70 tokens/s, 48.8% terms within three pieces are reported
in the log/status, but the **current build report has no `tokenizer` key** despite
docs/11's reproduction claim. No synthetic data count is treated as new independent
speech hours.

**Artifacts.** [Build report](../data/manifests/build_report.json),
[manifests](../data/manifests/), [data builders](../stt/data/).
No dedicated `runs/E-002`; original cache/raw checks are partly **reported**, not
replayable from committed audio/shards alone.

## E-003 - Random-Weight Browser Feasibility (2026-10-19)

**Question / protocol.** Can the complete WAV -> JS mel -> int8 WASM -> greedy path
clear 10x real time before training? Random initial 8,027,009-parameter model;
ONNX opset 17, batch one, dynamic time, generated sinusoids, dynamic per-channel QInt8.
Five-second synthetic fixture tiled to 10/20 s; Chromium 156.0.8078.4 headless,
onnxruntime-web 1.30.0 WASM SIMD, M5, COOP/COEP, one/four threads; three warm-ups,
median ten runs. No recognition-quality claim or data split.

**Outcomes.** Acoustic file 8,232,371 B, vocab 23,389 B, filters 64,320 B:
8,320,080 B payload. fp32/PyTorch max gap 3.815e-6, int8 gap 0.0293, top-1 99.7%,
manual JS/Python mel gap 1.07e-6; load 135 ms. RTFx at 5/10/20 s:
four threads **74.01/71.02/64.66**, one thread **37.92/33.05/32.73**.

**Decision / limitations.** Speed accepted; pooling escalation not needed for this
host. Random weights cannot establish useful transcription or trained int8 accuracy.
Cross-platform addendum reports same seeded graph/ORT 1.31.0, macOS ARM64 versus
Docker Linux x86_64: int8 random-model max gap about 0.02 versus 0.85 and top-1
at least 0.98 versus 0.28-0.36. This is not a measured trained-model x86 WER penalty.

**Artifacts / status: measured.** [Config and gate values](../runs/E-003/config.json),
[timing rows](../runs/E-003/rtfx.json); cross-platform result is **reported** in
[E-003 addendum](../EXPERIMENTS.md#e-003--h4-early-int8-export--browser-rtfx-2026-10-19).

## E-004 - Frozen Encoder, Padding and Schedule Control (2026-10-19 to 21)

**Question.** Does a head learn on frozen features, and does 30 s padding matter?
**Protocol.** Frozen pretrained tiny encoder, new `Linear(384,1025)` head, 394,625
trainable parameters. MLS-IT 169.89 h only; 1,248-clip MLS dev, no FLEURS/test selection.
AdamW head LR 1e-3, warm-up 500, cosine to 5%, light frequency masking, MPS bf16,
greedy dev scoring. All arms seed 0.

| Arm | Input / batch seconds, clips | Schedule / stop | Dev WER [95% CI] | Reported wall |
|---|---|---|---|---|
| A | Truncated / 320,64 | 3 epochs, 5,871 steps | 75.769 [75.215,76.302] | 23 min |
| B | Padded30 / 96,8 | 3 epochs, 20,970 steps | 70.787 [70.19,71.41] | 111 min |
| C | Truncated / 96,8 | 1-epoch cosine, 6,990 steps | 77.269 [76.75,77.80] | 11 min |
| C2 | Truncated / 96,8 | **3-epoch cosine**, stopped at 6,990 | 73.755 [73.191,74.312] | 12 min |

**Outcomes / correction.** B-A -4.98 pp [-5.32,-4.64] is measured but confounds batch,
step count and schedule. The first claimed 3.2 pp padding benefit from B versus C
at about 7,000 steps is invalid: C was near the end of its cosine while B was not.
C2 aligns the schedule; at steps 1,500/3,000/4,500/6,000 B versus C2 is
82.03/82.00, 79.71/79.70, 77.69/77.82, 76.42/76.52. At 6,990 B is only
**interpolated** as about 74.06 versus measured C2 73.755, a 0.305 pp gap.
Neither "all differences <0.001" nor exact functional identity follows from that table.
The padded log records **6,990 steps/epoch**, not the7,055 in the original
arm table;6,990 x3 is consistent with the completed20,970 steps.

**Decision.** Learning check accepted; reject the early padding recommendation, use
truncated input for H2 because the controlled curves are close and it saves memory/
compute. Reported A/B/C throughput about 590/310/1,200 audio-s/s and memory
4.3/5.2/1.1 GB are different-batch measurements, not matched architecture speedups.
Padded 320/64 was stopped with 16.9 GB RSS/swapping. C2 is a corrective short control,
not a completed three-epoch replication or formal equivalence test.

**Artifacts / status: measured.** [A config](../runs/E-004/config.yaml),
[A metrics](../runs/E-004/dev.metrics.json), [B](../runs/E-004/arm-padded30/),
[C](../runs/E-004/arm-truncated-96/),
[C2 config, log, hypotheses and metrics](../runs/E-004/arm-truncated-96-sched3/).

## E-005 - Full Fine-Tuning (2026-10-21)

**Question / protocol.** Does unfreezing the initial 8.03 M model on the 437.7 h
CV/MLS/LS mix beat tiny on FLEURS-IT and full CV dev? Initialize **both encoder and
head from E-004 padded best**, then train all parameters. Eight epochs/72,592 steps,
192 s/32 clips, head/encoder LR 1e-3/1e-4, warm-up 1,500. Best macro dev at 62,000:
IT **38.085**, EN **29.459**, macro **33.772**; final macro 33.856.

| Greedy evaluation | WER | Additional result |
|---|---:|---|
| FLEURS-IT | 37.406 | vs tiny IT greedy +5.32 [+4.10,+6.69] |
| FLEURS-EN | 62.441 | vs tiny auto greedy +48.58 [+47.19,+49.99] |
| CV-IT dev full | 37.600 | vs tiny IT greedy -10.93 [-11.33,-10.51] |
| CV-IT test full | 39.478 | Unseen test |
| T-S1, 200 | 65.348 | ETA **16.418**; not yet a training speaker |

**Alternatives / decision.** Last-three checkpoint average gives MLS/LS
38.243/29.483, macro about **33.863**, no improvement over best; keep best, reject
that averaging arm. H2 fails FLEURS-IT even against greedy, and more strongly against
the later beam bar. Full fine-tuning is useful but not a shippable final system.
More epochs alone are **not a separately randomized ablation**; the plateau suggests
limited return, not proof that any longer/LR-adjusted recipe would fail.

**Confounders / cost.** `dataset.epoch` was not advanced, repeating the augmentation
view across epochs; fixed only for future runs. Reported 9 h 22 wall, about eight
training hours, kills/resumes from memory pressure/worktree changes; start/resume
SHAs differ. Zero nonfinite steps at completion. Synthetic feasibility did not
predict sustained memory behavior.

**Artifacts / status: measured.** [Config](../runs/E-005/config.yaml),
[training log](../runs/E-005/train.log.jsonl),
[best evaluations](../runs/E-005/eval/best/),
[averaged evaluations](../runs/E-005/eval/avg/),
[CV baseline](../runs/E-005/baselines/tiny_chunks/).

## E-006 - Two Additional Transformer Blocks (2026-10-22)

**Question / protocol.** Can a larger head close the residual gap within the size cap?
E-005 best encoder/head plus two random transformer blocks, all 11,575,169 parameters
trainable. Same 437.7 h mix; four further epochs/36,296 steps, head/new-block LR
1e-3, encoder 5e-5, warm-up 500. Epoch-varying augmentation now fixed. Best is final:
MLS 35.556, LS 26.990, macro **31.273**.

| Greedy set | E-006 WER | Difference versus E-005 [95% CI] |
|---|---:|---|
| FLEURS-IT | 34.107 | -3.30 [-3.86,-2.69] |
| FLEURS-EN | 61.453 | -0.99 [-1.64,-0.23] |
| CV dev full | 34.447 | -3.15 [-3.33,-2.97] |
| CV test full | 36.156 | -3.32 [-3.51,-3.14] |
| T-S1 | 59.501 | -5.85 [-8.37,-3.28]; ETA **20.000** |

**Decision.** Adopt two blocks as the next checkpoint. Gate still fails: FLEURS-IT
34.11 versus tiny greedy 32.09 (+2.02 [+0.80,+3.32]), or corrected beam 27.42.
This is **capacity + additional training + LR changes + augmentation fix**, not a
capacity-only causal estimate. ETA **increases** 16.42 -> 20.00; prose saying it
"slips" or is "slightly worse" reverses the metric direction. No paired ETA gain
is asserted here without its comparison evidence. Reported training about 5.4 h,
several external restarts, 0 USD.

**Artifacts / status: measured.** [Config](../runs/E-006/config.yaml),
[log](../runs/E-006/train.log.jsonl), [evaluations](../runs/E-006/eval/best/).

## E-007 - Beam Search and Exact-BPE Hotwords (2026-10-22)

**Question / protocol.** Can search/context help without retraining? Frozen E-006
step 36,296. Prefix beam 5, top eight nonblank labels plus blank, true output lengths.
580 normalized general terms, completion bonus 1.0, prefix bonus zero, no LM.
Selection on 200+200 dev clips; beam widths 3/5/8, top-k 8/12, bonuses 0.5/1/2/3,
and prefix 0.05 explored. Beam-5 dev macro 32.48 versus greedy 32.90; beam-8 32.47,
so beam 5 is the cost/quality knee. Hot bonus 1.0 was an a-priori held-out setting,
not selected on T. Full-dev greedy/beam macro 31.27/30.83.

| Set | Greedy | Beam | Beam + hotwords |
|---|---:|---:|---:|
| FLEURS-IT | 34.11 | 33.77 | 33.80 |
| FLEURS-EN | 61.45 | 60.83 | 60.78 |
| CV dev full | 34.45 | 33.98 | 33.98 |
| CV test full | 36.16 | 35.68 | 35.68 |
| T-S1 WER / ETA | 59.50 / 20.00 | 59.07 / 20.90 | 58.73 / 21.79 |

**Decision / limitations.** Keep beam as optional quality mode; no shipped hotword
feature. Beam gains -0.34 IT and -0.62 EN pp are significant; T beam -0.43
[-1.15,+0.18] and hotword-versus-greedy -0.77 [-1.57,0.00] are ties. Exact token
completion cannot reliably recover absent acoustic candidates, but that mechanism
was not separately ablated. **ETA point estimates improve, not worsen**; the
historical rejection wording cannot be justified as an ETA regression. Defensible
rejection: negligible general-domain effect and no demonstrated gate-closing mixed
gain, not "hotwords harm ETA". About three minutes reported evaluation, no training.

**Artifacts / status: measured.** [Decode config](../runs/E-007/config.yaml),
[tuning table](../runs/E-007/tune/summary.md), [beam/hotword results](../runs/E-007/eval/).

## E-008 - BPE Trigram Shallow Fusion (2026-10-22)

**Question / protocol.** Does an edge-size transcript LM improve the frozen E-006
acoustics? Training text only: 182,441 CV/MLS/LS utterances. BPE trigram, stupid
backoff 0.4, bigram/trigram count cutoff two; 195,403 bigrams, 799,918 trigrams,
3.53 MB gzip pickle. Beam 5/top-k 8, no hotwords; score adds alpha times LM log
probability, **no token bonus**. Alpha 0.05/0.1/0.2/0.3/0.5/0.8 dev-subset macro:
30.48/29.37/**28.26**/28.31/31.14/38.98. Full-dev confirmation alpha 0.2/0.3:
27.05/27.18. Keep 0.2 before held-out evaluation.

| Set | Beam + LM WER | Difference versus E-006 greedy [95% CI] |
|---|---:|---|
| FLEURS-IT | 29.024 | -5.08 [-5.61,-4.54] |
| FLEURS-EN | 56.994 | -4.46 [-5.07,-3.90] |
| CV dev full | 28.394 | -6.05 [-6.22,-5.88] |
| CV test full | 30.071 | -6.08 [-6.25,-5.91] |
| T-S1 | 55.374 | -4.13 [-6.26,-2.25]; ETA **21.493** |

**Decision.** Adopt as quality configuration, not an acoustic-only gain. Retract
"H2 passed": IT 29.02 is significantly worse than tiny auto-beam5 27.42,
although better than tiny greedy 32.09; full CV-dev still wins. T ETA 21.49 remains
far below tiny forced-IT beam 37.61. It is not a detected ETA regression versus
greedy 20.00. Later E-022 shows this decoder's length calibration was suboptimal;
historical data/architecture results are conditional on that decoder.

**Artifacts / status: measured.** [Config](../runs/E-008/config.yaml),
[LM builder](../runs/E-008/build_lm.py), [tuning](../runs/E-008/tune/summary.md),
[held-out results](../runs/E-008/eval/w0.2/), [duplicate best outputs](../runs/E-008/eval/best/).
Reported LM build 6.2 s, evaluation about three minutes. The gzip pickle is not
the 5.09 MB browser LM introduced by E-010.

## E-009 - Trained int8 Export (2026-10-23)

**Question / protocol.** Does trained E-006 export faithfully and run fast? Include
both extra blocks, dynamic int8 ONNX, real tokenizer. fp32/PyTorch comparisons on
1/5/10/29 s fixtures; first 100 FLEURS-IT clips for greedy WER. Same M5/headless
Chromium tiling/warm-up/median protocol as E-003. **Greedy only**, no browser LM yet.

**Outcomes.** 11.91 MB acoustic file; about 11.99 MB acoustic/vocab/filter payload.
fp32 maximum gap 2.17e-4, int8 gate top-1 100%, maximum gap 1.72. FLEURS-100
PyTorch/int8 WER **30.92/30.65**, 50/100 identical transcripts. Four-thread
5/10/20 s RTFx **66.40/61.34/54.60**; one thread **29.29/28.33/24.58**;
load about 138 ms.

**Decision / limitations.** Accept export and speed on the measured host. The
100-clip aggregate is not full-test int8 equivalence or E-008 browser speed. Large
logit differences and transcript changes can coexist with close corpus WER.
The E-009 prose's "old 7.9 MB" is inconsistent with E-003's retained 8.232 MB file;
use actual bytes per artifact rather than that informal size comparison.

**Artifacts / status: measured.** [Export config](../runs/E-009/config.yaml),
[100-clip WER check](../runs/E-009/int8_check.json),
[timings](../runs/E-009/rtfx.json), [smoke](../runs/E-009/rtfx-smoke.json).

## E-010 - Browser Beam and Binary LM (2026-10-23)

**Question / protocol.** Port E-008 search to JS without losing useful quality or
the 10x bar. Same E-006 int8 graph; sparse CSR LM with int16 log probabilities,
**5.09 MB browser binary**, beam 5/top-8/alpha 0.2. Complete payload about
17.08 MB. M5 Chromium WASM SIMD, three warm-ups/ten-run medians; all 200 T-S1
audio clips also sent through browser and compared with unquantized Python E-008.

**Outcomes.** RTFx four-thread **65.22/63.25/58.12**, one-thread
**29.96/28.33/25.25** for 5/10/20 s. Decode about 4.3/8.5/15.6 ms.
Reported T-S1 browser/Python WER **56.66/55.37**, difference **+1.29
[-0.34,+3.02]**, tie; ETA **21.19/21.49**, tie; median real-clip RTFx about 65.
Mic capture smoke used Chromium fake audio for 1.5 s, resampled to mono 16 kHz,
not a real-microphone accuracy evaluation.

**Decision / limitations.** Accept browser quality path at the aggregate level,
not exact transcript identity. int8 acoustics, feature implementation and LM
representation differ jointly, so a WER tie alone does not isolate parser parity.
Hotwords are not ported. H2 is **still not passed** against the corrected baseline.
The entry's proposed "next full 200-clip check" is stale: it already describes
that check as completed.

**Artifacts / status.** [Config](../runs/E-010/config.yaml),
[timings](../runs/E-010/rtfx.json), [200 raw browser outputs](../runs/E-010/browser_t200.json),
[packager](../runs/E-010/package_browser.py). Headline paired interval is
**reported** in the research log; the retained browser file contains individual
transcripts/RTFx, not that paired summary. E-022 parity has a coincidentally equal
point difference on **different weights/LM/reference**, not the same experiment.

## E-011 - Nonspeech, Energy Guard, VAD and Room Tone (2026-10-23 to 11-06)

**Question / initial protocol.** Emit zero words on non-music nonspeech; music
reported separately. E-006 greedy versus E-008 beam+LM, plus external baselines,
on 120 ten-second digital silences, 120 MUSAN noises, 120 MUSAN music clips,
seed 0, natural levels (reported median noise/music -20.6/-15.9 dBFS).

| Initial system | Non-music words / nonempty clips (of 240) | Silence words | Music words / nonempty clips (of 120) |
|---|---|---:|---|
| Moth greedy | 24 / 18 | 0 | 250 / 56 |
| Moth beam+LM 0.2 | 259 / 134 | **240** | 146 / 40 |
| Tiny forced IT beam5 | 54 / 13 | 0 | 274 / 26 |
| Whistle forced IT | 264 / 115 | 0 | 554 / 120 |
| Parakeet int8 | **0 / 0** | 0 | 411 / 24 |

**Initial decision.** Reject Moth's unguarded quality path as a safe default for
unknown audio; only Parakeet meets the measured 40-minute non-music zero-word gate.
All 120 silence clips reportedly decode as "non mi" in beam+LM. Acoustic blank
confidence fails as a silence detector: reported mean nonblank 0.125 on zeros
versus 0.108 on speech. A deterministic silence failure implicates the whole
acoustic/search/prior interaction; it is not evidence that the acoustic distribution
is literally all blank or that the LM alone caused it.

**Energy addendum, 2026-10-24.** Guard if maximum normalized mel < -1.0;
digital silence -1.50, weakest measured speech peak +0.10, MUSAN noise +0.64.
Suppresses 120/120 zeros, 0/897 T-S1+FLEURS-IT speech clips; greedy non-music
stays 24, beam falls **259 -> 19**; music unchanged. Accepted for silence,
**not** a loud-noise solution, later superseded.

**VAD addendum, 2026-10-30.** E-016 checkpoint and augmented LM now used, not
E-006. Silero v5 MIT, **2,327,524 B**. 512 new + 64 context samples/call,
state `[2,1,128]`, P>0.5 for speech windows; decode when speech-window fraction
>0.10. A 512-without-context invocation was rejected as erroneous (near-zero
probabilities even on speech). T-400/FLEURS-IT minimum fractions
0.408/0.4187; MUSAN max 0.03834, silence zero. Gates all 240 non-music and
107/120 music; **0/400 T and 0/697 FLEURS** gated, hypotheses unchanged.
E-016 energy -> VAD non-music words greedy **35 -> 0**, beam **31 -> 0**;
music greedy 275 ->154, beam 142 ->98. Accepted, package 0.4.0 about 19.44 MB.
Music is outside the gate, **not exempt from actual VAD filtering**.

**Room-tone addendum, 2026-11-06.** Recorded 974.8 s, not planned 1,200 s;
97 ten-second windows, reported median -69.8 dBFS and no speech windows, all
97 gated. Expanded set 457 clips/337 non-music, zero non-music words with both
decoders, 444/457 total gated. S3 later has minimum fraction **0.194**, so the
4x weakest-speech margin is no longer true for all T. Quiet/far-field speech
outside these recordings may be rejected.

**Artifacts / status: measured.** [Original paths and guards](../runs/E-011/),
[VAD separation JSON](../runs/E-011/vad_guard_summary.json),
[VAD speech beam results](../runs/E-011/vad-speech-beam-lm/),
[expanded noise beam results](../runs/E-011/vad-noise-room-tone-beam-lm/),
[0.4.0 timing/size record](../runs/E-011/browser-0.4.0.json),
[noise report](../data/manifests/noise_report.json),
[S3 VAD check](../runs/E-017/vad_speakers.json).
VAD threshold selection used observed speech/noise separation, not an independent
calibration set. After E-024, some of the nonspeech pool also supplies training
noise; a later zero-word test is no longer wholly unseen-noise evidence.

## E-012 - More Audiobook English (2026-10-23)

**Question / protocol.** Does another 200 h LS English improve mixed ETA?
E-006 best, unchanged V=1,024/two blocks; natural mix grows to about 638 h,
IT/EN about 53/47. Two epochs/26,028 steps, LR 3e-4/5e-5, warm-up 500.
At step 9,350 tempo stretch exceeded 30 s; crop/clamp fix and resume from 9,072.
Best/final greedy MLS/LS macro **29.501** (35.783/23.218).

| Final set | Greedy WER | Beam5 + original LM0.2 WER |
|---|---:|---:|
| FLEURS-IT | 35.485 | 29.791 |
| FLEURS-EN | 56.505 | 51.565 |
| CV test full | 36.604 | 30.384 |
| T-S1 | 65.520 / ETA18.209 | 60.533 / ETA21.493 |

**Decision.** Reject as standalone product lever: versus E-008, EN improves
5.43 pp but T WER worsens **+5.16 [+3.00,+7.37]**, ETA exactly unchanged
(difference CI [-2.93,+2.92]); IT/CV also have worse point estimates. English
volume is useful for English, not demonstrated useful for these mixed terms.
Do not infer that English data in general cannot help; this intervention also
changes source proportions, total exposure and step count. Reported about four
hours training plus ten-minute data build, 0 USD.

**Artifacts / status: measured.** [Config](../runs/E-012/config.yaml),
[log](../runs/E-012/train.log.jsonl), [authoritative final evaluation](../runs/E-012/eval-final/).
[Earlier evaluation](../runs/E-012/eval/) is **8k-checkpoint diagnostic**, not final:
beam IT/EN/CV/T **30.889/54.809/33.044/58.727**, ETA22.687. Mixing that T
with final FLEURS would fabricate a nonexistent configuration.

## E-013 - Larger BPE Vocabulary (2026-10-23)

**Question / protocol.** Does less fragmentation improve English terms?
V=4,096 tokenizer from the same reported corpus/seed; random larger CTC head,
E-006 encoder/two blocks, all **12,757,889** parameters trainable. Same 437.7 h
mix, two epochs/18,148 steps, LR head1e-3/encoder5e-5, warm-up500. No LM:
the 1,024-piece LM is incompatible. Best greedy dev macro **32.788**
(IT37.437/EN28.139), not a beam-selected dev number.

**Outcomes.** Reported tokens/word 1.945 ->1.495, tokens/s 4.70 ->3.61,
terms within three pieces 48.8% ->74.0%. Greedy/beam5 WER: FLEURS-IT
**36.30/36.02**, EN **60.02/59.71**, T-S1 **59.93/59.85**; T ETA
**19.70/20.00**, versus E-006 greedy 20.00. No demonstrated ETA improvement;
Italian regresses versus E-006, English point estimate improves.

**Decision / limits.** Reject this larger-vocabulary standalone recipe; do not
spend the LM-port budget. This does **not** prove BPE size irrelevant, capacity
unimportant, or acoustic data uniquely causal: projection initialization, training
duration and absence of a compatible LM differ. A 4,096-piece run with the later
token-bonus decoder was not measured. Reported about 3.5 h MPS, 0 USD.

**Artifacts / status: measured.** [Config](../runs/E-013/config.yaml),
[log](../runs/E-013/train.log.jsonl), [greedy/beam results](../runs/E-013/eval/).
Tokenizer binary is untracked; original statistics are reported rather than
independently recoverable from the present build report. Research-log dev placement
under "beam" and comments claiming E-012's same head LR are misleading: configs
show **1e-3 here versus 3e-4 in E-012**.

## E-014 - Utterance-Level Mel Splicing (2026-10-24)

**Question / protocol.** Does exposure to an intra-utterance clause switch improve
ETA without gold? Add **110 h / 21,205** IT+EN pairs to the natural mix, about
548 h total. CV/MLS Italian followed by LS100 English, pairs <=27 s, seed0;
concatenate **cached mel**, not waveforms. Drop seams that normalize into a
single corrupted word. E-006 best -> two epochs/22,494 steps, LR3e-4/5e-5,
warm-up500. Best/final greedy dev macro **30.860**.

| Set | Greedy WER | Original LM0.2 beam WER | Beam difference vs E-008 |
|---|---:|---:|---|
| FLEURS-IT | 32.68 | 26.75 | -2.27 [-2.86,-1.72] |
| FLEURS-EN | 60.70 | 56.04 | -0.95 [-1.57,-0.37] |
| CV test full | 36.15 | 30.10 | +0.03 [-0.12,+0.16], tie |
| T-S1 | 61.13 / ETA18.51 | 57.09 / ETA20.60 | WER+1.72 [-0.60,+4.09], ETA-0.90 [-4.23,+2.48], ties |

**Decision.** Keep checkpoint for FLEURS quality; reject the **hypothesis that
this splicing arm raises mixed ETA**. Versus tiny auto-beam5 FLEURS-IT -0.66
[-1.80,+0.47], still tie; T ETA versus forced-IT tiny -17.01 [-25.55,-7.01].
Energy-guarded non-music words 24 greedy/13 beam, music275/138, versus E-008
beam19/146; a count difference alone does not establish a statistical noise tie.

**Confounders.** Added hours repeat constituent training speech, not 110 h of
independent natural switching; clause-level splicing does not test word-level
loanwords. E-012 and E-014 have matching continuation hyperparameters but
different source composition, hours and step counts. Reported 4 h16 wall,
about3 h50 training, memory stall/resume from3,040. A historical E-016 sentence
labels its E-014 initialization "step36,296"; that is E-006's step count.
E-014 itself ends/selects **22,494**.

**Artifacts / status: measured.** [Config](../runs/E-014/config.yaml),
[log](../runs/E-014/train.log.jsonl), [evaluation](../runs/E-014/eval/),
[splicing report](../data/manifests/spliced_it_en_report.json),
[builder](../stt/data/build_spliced.py).

## E-015 - Original Distillation Proposal and Actual Gold Addendum

### Original Pilot (2026-10-24): Not Run

**Question / planned design.** Could Parakeet v3 int8 hard-text pseudo-labels
transfer rare-term mappings? Proposed E-014 student, about100 h new legally
reusable term-rich audio, same two-epoch continuation recipe, human-label
control where available and teacher error on a held-out hour. This is **hard-label
pseudo-labelling**, not soft-logit knowledge distillation. Original target T ETA
at least+5 pp, no WER loss. The motivating T-S1 rare-utterance readout was
Moth E-014 **2/41**, tiny12/41, Whistle17/41, Parakeet32/41; it is not a T-600
measurement of the later 44-term group.

**Status / decision.** **No distillation arm ran.** No acceptable term-rich
audio/provenance decision; T is evaluation-only, existing R has human labels,
public technical speech or TTS would need their own data decision. The projected
3.3 h teacher labelling plus3.5 h training is not consumed compute. Preserve
[the original plan and later addendum](../runs/E-015/PLAN.md) as distinct protocols.

### Executed S1-Only Gold Addendum (2026-11-09): Rejected

**Question / actual protocol.** On 0.7.0 the uncovered target changed: 44 distinct
terms, 47 occurrences/utterances. Add **50 new S1 sentences**, about2.5 min,
separate x20 source about0.84 h, old S1/S2 gold x50 untouched. Planned second
50-sentence S2 voice was **not recorded for this arm**. Continue E-024 best,
one epoch13,252 steps, LR3e-4/5e-5, warm-up500, seed0, standard MPS recipe;
beam12/top10/LM0.5/token2.4 for final readout. Mixture actually gains a small
source; "same589.5 h" is rounded base-mixture shorthand, not exact equality.

| Readout | 0.7.0 | Final E-015 | Difference / interpretation |
|---|---:|---:|---|
| Primary rare utterances | 8/47 =17.02 | 11/47 =23.40 | +6.38 [-4.26,+17.02], tie; target>=50, kill<30 |
| Rare S1/S2/S3 | 6/14,0/15,2/18 | 7/14,2/15,2/18 | Net+1,+2,**0 on S3**; not no individual S3 flips |
| Natural group | 439/817 | 423/817 | -16 hits |
| Gold-only group | 123/179 | 129/179 | +6 hits |
| All term-presence hits | 570/1,043 =54.65 | 563/1,043 =53.98 | -0.67 pp; **not official aligned ETA** |
| Historical gold regression group | 20/27 =74.07 | 19/27 =70.37 | -1 hit |
| T-600 WER | 25.20 | 24.91 | -0.29 [-1.21,+0.63], tie |
| T-S3 WER | 26.92 | 25.55 | -1.36 [-3.16,+0.25], tie |
| FLEURS-IT WER | 20.14 | 20.86 | **+0.72 [+0.32,+1.12]**, worse |
| FLEURS-EN WER | 49.14 | 50.45 | **+1.31 [+0.68,+1.97]**, worse |
| CV dev first600 WER | 19.85 | 19.97 | +0.12 [-0.48,+0.73], tie |

**Selection / decision.** Best logged dev is step20 **30.8136**, then step2,000
33.64, final31.2246. The 20-step smoke was resumed into the full run, and best
never improved. The final checkpoint was deliberately evaluated as the arm that
saw the data, **not the dev-selected model**. Reject final: rare score below30
and both FLEURS regressions exceed0.5 with CIs excluding0. Keep published0.7.0.
Reported prep2 min, training1 h50, eval about3 min, no teacher/downloads/paid spend.

**Measurement caveats.** [The term readout](../runs/E-015/eval_terms.py) checks
normalized **set membership anywhere in the hypothesis**, while official ETA
requires aligned matches. Thus54.65 versus official baseline ETA54.17 is a
definition difference, not numerical corruption. "Never seen" means absent from
the enumerated project fine-tuning text sources, not absent from Whisper pretraining;
the script's natural group is CV+MLS+spliced, not an exhaustive audit of every
possible upstream text or the LM's separate pool/unrecorded-script text. The
regression constant named `TERMS18` actually lists
**16 strings**, and the retained27-occurrence score applies to that implementation.
The advertised+3.7/+2.8 ETA ceiling uses term-presence arithmetic
39/1,043 and29/1,043; treat it as an approximate direct-intervention bound, not an
exact guarantee for the aligned metric. The later script/manifest contains new
sentences informed by T vocabulary: training text is disjoint, but target vocabulary
selection has used test information. This is targeted development, not blind testing.

**Causal limit.** Net-zero S3 rare hits does not prove that one voice can never
transfer; only this small, aggressive recipe failed. LR damage and voice diversity
were not isolated. A two-voice/gentler follow-up is proposed, **not run** here.

**Artifacts / status: measured.** [Actual config](../runs/E-015/config.yaml),
[log](../runs/E-015/train.log.jsonl), [result](../runs/E-015/RESULT.md),
[term JSON](../runs/E-015/eval/terms.json), [final/gate outputs](../runs/E-015/eval/),
[recording protocol](../runs/E-015/RECORDING.md).

## E-016 - First Recorded Gold Speaker (2026-10-29, Later Controls)

**Question / protocol.** Does term-rich gold help beyond synthetic clause splicing?
E-014 best, **635 S1 takes/29.3 min**, x50 index repetition about24.4 effective h,
total572.4 h, gold4.3% duration. Two epochs24,478 steps, LR3e-4/5e-5,
warm-up500. Best step24,000 macro30.904, final30.977. The current R-S1 manifest
grew later to641, then691; those counts are **not the original E-016 training count**.

**Initial T-S1 results, original LM.** Greedy38.09 WER/37.91 ETA;
beam35.51/40.60 versus E-01457.09/20.60. Beam WER **-21.58
[-25.38,-17.99]**, ETA **+20.00 [+13.69,+25.70]**. FLEURS IT26.98,
EN56.22, full CV test30.02, all ties versus E-014. Non-music words worsen
13 ->31 beam,24 ->35 greedy without VAD. Pure-Italian control WER34.7 ->18.1,
English control about51.4 unchanged. Reported term-presence seen/unseen gold
accuracy18.3 ->45.7 /15.8 ->27.7. These controls expose a substantial
voice/channel/task component, not a clean vocabulary-only gain.

**New-speaker control, S2.** Before S2 contributes gold, same original LM:
E-01451.87/20.68 ->E-016**42.22/29.75**, WER-9.64
[-12.55,-6.93], ETA+9.07 [+3.97,+14.61]. Against tiny forced-IT beam
51.69/32.86: WER-9.47 [-14.90,-4.59], ETA-3.12 [-12.46,+5.65], tie.
Transfer exists; matched S1 roughly doubles the observed gain, but an exact causal
speaker fraction cannot be inferred because speakers read different sentences/channels.

**Augmented-LM addendum, 2026-10-30.** Same checkpoint, add training-side R-S1,
R-S2 scripts and pool text to LM; no T/A transcripts. The LM can therefore see
R-S2 **text before R-S2 audio training**. Same backoff/pruning,3.55 MB gzip,
5.116 MB browser binary. Dev alpha0.1/0.2/0.3/0.4 macro28.87/**27.54**/
27.88/29.08. T-400 original ->augmented LM **38.764/35.029 ->36.906/39.099**;
WER-1.86 [-2.55,-1.21], ETA+4.07 [+2.67,+5.61]. S2 becomes40.49/33.43.
FLEURS-IT26.98 ->26.90, EN56.22 ->56.23, CV30.02 unchanged. Adopted0.3.1;
`standup` remains an imperfect contextual example, not a general recognition claim.

**S3/T-600 addendum.** With augmented LM, T-600**38.503/36.050**;
S1**33.448/44.776**, S2**40.487/33.428**, S3**41.567/30.423**.
S3 versus E-01454.770/17.746: WER-13.20 [-16.36,-10.24],
ETA+12.68 [+7.71,+18.03]. S3 shares S1 microphone; not a held-out-channel test.
Against tiny auto on S3, WER improves but **ETA is lower**, not a win on both;
against forced-IT tiny, WER-4.94 [-10.40,+0.34] and ETA-7.61
[-17.02,+1.86] are both ties. T-200/original-LM35.51 must not be compared
directly with T-600/augmented-LM38.50 as a model regression.

**Release / decision.** Adopt E-016 with limitations; use E-017 for the next
cross-speaker test. Release0.3.0 unaugmented LM17.08 MB; fp32 synthetic max
1.050949e-3 at29 s violates old1e-3 gate, real20-clips max5.455e-4,
top1=1; exported with `--no-verify`. int8 FLEURS100 WER29.226/29.226,
44/100 transcripts identical, top1 99.778%, max logit gap12.082.
Full new-checkpoint browser/Python T parity was not measured at that step.
Reported4 h20 wall, memory stall/resume2,695; allocator empty interval50 ->10.

**Artifacts / status: measured.** [Config](../runs/E-016/config.yaml),
[log](../runs/E-016/train.log.jsonl), [evaluation variants](../runs/E-016/eval/),
[export gate JSON](../runs/E-016/export.json),
[LM augmentation builder](../runs/E-008/build_lm_augmented.py),
[dev alpha variants](../runs/E-016/tune/),
[S3 comparisons](../runs/E-017/eval/), [S2 baselines](../baselines/runs/).

## E-017 - Two Gold Speakers, S3 Untrained (2026-11-06 to 07)

**Question / protocol.** Does more gold improve an untrained voice? Reinitialize
from E-014 best, **not E-016**; R-S1 now641 takes plus R-S2 622 valid/623
recorded (exclude0.39 s silent take without renumbering). Total50.2 min,
x50=63,150 rows/41.87 h; mixture589.54 h, gold7.1%. Two epochs26,442
steps, same LR/schedule/batch as E-016, macro greedy checkpoint selection.
Best final macro**30.984** (MLS36.288/LS25.679), 0 nonfinite steps,
reported3 h21 wall, peak6.45 GB, CPU/MPS CTC gap4.89e-6.

| Augmented LM0.2, beam5/top8 | E-016 | E-017 |
|---|---:|---:|
| T-600 WER / ETA | 38.50 /36.05 | **32.77 /42.86** |
| S1, trained both | 33.45 /44.78 | 28.98 /51.94 |
| S2, newly trained | 40.49 /33.43 | 33.88 /41.93 |
| S3, never audio-trained | 41.57 /30.42 | **35.43 /35.21** |
| FLEURS IT / EN WER | 26.90 /56.23 | 26.52 /55.48 |
| CV test full WER | 30.02 | 30.36 |

**Paired outcomes.** T-600 WER-5.73 [-7.14,-4.26], ETA+6.81
[+4.18,+9.55]. S1/S2/S3 WER differences -4.47/-6.60/**-6.13**;
S3 CI[-8.68,-3.51], ETA+4.79 [+0.51,+9.06]. Greedy T-600
37.701/36.433 versus E-01643.263/30.872. VAD nonspeech0 words/337
non-music,444/457 gated; ungated leftovers are music (beam103 words, greedy149).

**Gate and decision.** Adopt E-017 (0.5.0). T-S3 versus forced-IT tiny:
WER-11.07 [-16.28,-5.89], ETA-2.82 [-11.65,+6.56], **strict mixed
gate not passed**. FLEURS-IT26.52 versus tiny27.42: -0.89 [-2.08,+0.19],
tie. On T-600, forced-IT tiny50.72/36.05 is beaten on both; tiny auto68.86/39.50
has ETA difference+3.36 [-2.20,+9.36], **tie**, so the historical claim
"passes against both" is false under the stated two-CI rule. Whistle T-600
29.42/49.28 remains better (+3.35 WER [+1.07,+5.60]); Parakeet10.98/83.60.
Two thirds of T clips are from trained voices.

**Causal/evidence limits.** Positive S3 transfer rules out an explanation solely
based on hearing S3's own voice, not all channel/task adaptation. E-017 versus
E-016 changes **gold amount, sentences (S1 grows635 ->641), voice and channel,
step count**, not speaker count alone. One-seed/three-speaker results cannot prove
"the speaker confound is retired" or identify vocabulary as the sole mechanism.
The claim "best dev of project" is also false: E-01430.860 and E-01630.904
are lower than E-01730.984. E-017 can still be preferable on the product test.

**Release evidence.** [0.5.0 record](../runs/E-017/release-0.5.0.json):
int8 11.91 MB, augmented LM5.12, VAD2.33, total19.44 MB; model SHA prefix
`fe1afcc2`. FLEURS100 top1 99.72%,44/100 identical transcripts, max gap8.59;
these do not establish end-to-end WER parity (E-020 does). Old fp32 verification
still bypassed; E-021 later resolves the measurement gate.

**Artifacts / status: measured.** [Config](../runs/E-017/config.yaml),
[log](../runs/E-017/train.log.jsonl), [gold builder](../runs/E-017/prepare_gold.py),
[eval](../runs/E-017/eval/), [paired T comparisons](../runs/E-017/diffs/),
[full baselines](../runs/E-017/baselines-t600/),
[speaker manifests](../runs/E-017/manifests/).

## E-018 - Scale Gold (Planned, Not Run)

**Question / proposed design.** From E-014 initialization and E-017 recipe,
add R-S4 as third gold voice (G3); optionally about200 more sentences per voice
(G3+) or x100 instead of x50 repetition (G3x). Primary paired T-S3 WER versus
E-017, secondary ETA; success downward WER CI excluding0, stop if third voice
adds<1 pp on S3. G3x is only a diagnostic if diversity versus exposure needs
separation. Estimated about3.5 h per arm, not consumed compute.

**Status / decision.** **None of G3/G3+/G3x ran. No E-018 result, training
config, log or checkpoint exists in `runs/`.** Blocked on S4 recordings and preparation
of R-S4 material. The old plan's R-S4 invariant change is already implemented in current
code (R permits S1/S2/S4); S3 remains excluded from training. S4 test-script review was
completed later, so the plan's "unreviewed" wording is stale, not evidence that
audio exists. The plan's "T-600, three of four speakers trained" wording also
mixes existing600 with prospective800; at this snapshot T-600 is three voices,
two trained. Predictions of2-3 or4-6 pp per extra voice are not results.

**Artifact / status: planned only.** [E-018 protocol](plans/2026-11-07-e018-scale-the-gold.md).
Do not confuse this scaling proposal with E-015's original unexecuted distillation
or its later **executed** gold-term addendum.

## E-019 - Whistle LoRA Side Probe (2026-11-07)

**Question / actual protocol.** Does gold help a different pretrained system
under a cheap adapter recipe? **Exploratory, not Moth.** Whistle training
checkpoint220,618,620 B; rank16/alpha32 LoRA, LR1e-4, batch16 padded30 s,
3 epochs216 steps, seed0, forced `it`. cactus-needle3.2.10, JAX0.12.0.1
**CPU float32** on M5, not MPS. R-S1+R-S2 only1,263 clips; stratified seed0
90/5/5 split: **1,137 train45.3 min,63 val2.5 min,63 test2.4 min**.
Thus45.3 min describes **training**, not the total50.2 min corpus. No natural
CV/MLS/LS replay as in Moth. Validation loss0.825/0.561/0.527 still decreasing.

| Evaluation | L0 / L1 WER | L1-L0 [95% CI] |
|---|---|---|
| T-S3,200, primary | 28.36 /30.66 | +2.30 [-0.68,+5.46], tie ->kill |
| Gold-test63, seen voices | 30.08 /27.10 | -2.98 [-8.38,+1.88], tie |
| FLEURS-IT697 | 11.85 /12.87 | +1.02 [+0.51,+1.56] |
| FLEURS-EN525, forced IT | 13.35 /15.65 | +2.30 [+1.16,+3.26] |
| CV-test **first1,000** | 17.59 /20.25 | +2.66 [+1.74,+3.49] |

S3 ETA52.11 ->47.89 (-4.23 [-10.34,+1.86], tie); gold-test ETA52.58
->60.82, noisy/uncertain gain. L0 reproduces prior S3 hypotheses200/200 exactly.
With VAD, non-music0/337 in both arms; music words217 ->281.

**Decision / not run.** Stop at L1 per predefined kill rule (S3 WER CI covers0).
`--tokens` and new `mix` language-code arms **not run**; full T-600 and
unguarded L1 noise **not run**. Adapter6,786,888 B; exported archive
**29,346,671 B =29.35 MB =27.99 MiB**, base16,919,407 B. Export+VAD
31.67 MB exceeds20 MB. The result note's "28 MB ...21.6 MB over budget"
is arithmetic/unit error; archive alone exceeds by **9.35 decimal MB**.
Reported22 min CPU,0 USD.

**Causal limit.** Reject **this LoRA recipe**, not architecture-independent gold
learning as a universal hypothesis. Different pretrained model, optimizer/update
method, lack of replay, exposure, language forcing and split all differ from Moth.
S3 loss is a **tie**, not significant degradation. Seen-gold improvement is also
a tie. No causal demonstration that an autoregressive decoder "has no mechanism"
to retain generality follows. Historical "architecture-bound" headings overstate
what this single cross-system probe can establish.

**Artifacts / status: measured.** [Full config/environment/bytes](../runs/E-019/config.json),
[split](../runs/E-019/data/split.json), [result with omissions](../runs/E-019/RESULT.md),
[training log](../runs/E-019/train_L1.log), [L1 S3](../runs/E-019/l1-s3/),
[CV subset manifest](../runs/E-019/manifests/cv-first1000.jsonl).

## E-020 - Shipped int8 End-to-End WER (2026-11-07)

**Question / protocol.** Does actual E-017 int8 change corpus quality? Python
ONNX runner, same cached mels, vocabulary, original Python decoders and augmented
LM0.2; compare re-scored E-017 hypotheses on identical manifest. No training,
no browser features or binary-LM port change. Forty-clip greedy smoke matches
PyTorch exactly at41.048 WER/28.767 ETA before full runs.

| Set / decoder | PyTorch | int8 | WER difference [95% CI] | Identity |
|---|---:|---:|---|---|
| T-600 greedy | 37.701 | 38.131 | +0.43 [-0.17,+1.01], tie | 423/600 |
| T-600 beam5+LM0.2 | 32.769 | 32.942 | +0.17 [-0.40,+0.77], tie | 464/600 |
| FLEURS-IT beam5+LM0.2 | 26.523 | 26.848 | **+0.32 [+0.08,+0.57]** | 368/697 |

**Decision / limits.** Keep int8, disclose the small significant FLEURS loss;
ETA T differences -0.10 greedy/-0.38 beam pp are ties. Do not claim quantization
is universally free or extrapolate to later decode/checkpoints. The proposed
length-accumulation explanation is plausible but **uncontrolled**: domain and
duration vary together. This does not isolate length causally. Native ARM-host
evaluation does not certify x86 trained-model WER.

**Artifacts / status: measured.** [Result](../runs/E-020/RESULT.md),
[int8 greedy](../runs/E-020/int8-greedy/), [int8 beam](../runs/E-020/int8-beam-lm/),
[FLEURS](../runs/E-020/int8-fleurs-it/), [reference](../runs/E-020/torch-ref/).

## E-021 - fp32 Export-Gate Audit (2026-11-07)

**Question / protocol.** Was the old1e-3 single-draw failure an export bug?
E-017 fp32 ONNX versus PyTorch, four lengths1/5/10/29 s,20 seeded synthetic
draws per length plus natural-scale/real/padded-real fixtures. Same checkpoint,
not E-016's formerly failing weights. No training/quality optimization.

**Outcomes.** E-017 original gate seed0 max2.43e-4; real clips max3.05e-4,
real padded29 s2.21e-4; natural-scale synthetic1.37e-4. Sweep median about1.8e-4,
worst1.28e-3,1/80 above1e-3, only at29 s. Rebuild0.5.1 verified under new
**median<=1e-3 and worst<=1e-2** gate: median2.127e-4,worst1.472e-3,
3/80 above1e-3. Those are **different sweep/build records**, not interchangeable
counts. Trained int8 byte-identical to0.5.0 (`fe1afcc2...`), greedy T-600
38.131 reproduced. Browser1T25.4/24.2/21.7,4T52.0/51.0/46.5.

**Decision.** Adopt deterministic multi-draw gate and stop `--no-verify` release
bypasses. No new model quality or evidence of every possible input's correctness;
failure tails are empirically bounded on these fixtures. This explicitly changes
the worst-case tolerance, not just repetition count. Release also deduplicates
FLEURS NOTICE attribution; publisher/card checks are integrity work, not an ASR arm.

**Artifacts / status: measured.** [Audit JSON](../runs/E-021/fp32_gap.json),
[result](../runs/E-021/RESULT.md), [verified0.5.1 record](../runs/E-021/release-0.5.1.json).

## E-022 - Decode Recalibration (2026-11-09)

**Question / protocol.** Does cancelling per-token LM cost reduce deletion bias?
Fixed E-017 checkpoint and augmented LM, no new training data/weights. Historical
beam5/top8/LM0.2/token0 versus dev-selected **beam12/top10/LM0.5/token2.4**.
Two-dimensional dev search,200 MLS+200 LS clips; macro31.61 ->27.30. This
intervention changes **bonus, LM weight and search width together**; gains cannot
all be assigned to a single scalar. Zero bonus is bit-exact no-op in30 tested
clips; greedy is unchanged.

| PyTorch set | Before WER / ETA | Frozen WER / ETA | WER difference [95% CI] |
|---|---|---|---|
| T-600 | 32.77 /42.86 | **25.92 /52.64** | -6.85 [-7.94,-5.72] |
| T-S3 | 35.43 /35.21 | **26.66 /47.32** | -8.77 [-11.13,-6.50] |
| T-S1+S2 | 31.46 /46.66 | 25.54 /55.38 | -5.92 [-7.23,-4.52] |
| FLEURS-IT | 26.52 | **20.46** | -6.06 [-6.63,-5.51] |
| FLEURS-EN | 55.48 | 49.63 | -5.85 [-6.55,-5.18] |
| CV-dev **first600** | 24.91 | 19.90 | -5.01 [-5.88,-4.14] |

T ETA gains+9.78 [+7.68,+12.06], S3+12.11 [+8.31,+16.47]. T deletions
12.24 ->8.29%, insertions0.80 ->1.26%. Full CV-test measured for card:
**24.882** (14,643 clips), not first600 CV-dev19.90. int8 T-600
25.946/53.116, S3 27.257/47.324, differences versus PyTorch+0.03/+0.60,
both reported ties. Greedy T-600 remains37.701/36.433.

**Gate / adoption.** FLEURS-IT versus tiny auto-beam5 **-6.95
[-8.02,-5.87]**, now won. T-600 versus forced-IT tiny WER-24.80 and
ETA+16.59 [+10.61,+22.74]; versus Whistle WER-3.50 [-5.85,-1.24],
ETA+3.36 tie. T-S3 versus forced-IT tiny WER-19.51 [-24.18,-14.57],
ETA+8.73 **[-1.49,+18.86]**, still tie; strict held-out mixed gate not
certified. Adopted0.6.0, model/LM byte-identical to0.5.1,19.44 MB payload,
browser20 s one-thread20.28x. No "model was never wrong" inference: successful
decoder calibration does not eliminate remaining acoustic errors.

**Rejected/invalid candidates.** Cheap fallback beam8/top8/LM0.5/token2.4
T-60026.376/51.774, S3 27.257/46.479 was unnecessary because browser speed
passed. Exploratory0.6/3.0 gives T-60024.685/54.650, S3 25.468/49.859,
but was chosen after inspecting S3; **not adopted**, not a clean gate win.
Dev-only re-selection avoids shipping that test-selected cell, but cannot undo
the fact that S3 was inspected during development.

**Browser/Python correction.** With correct augmented LM, E-022 parity file
on200 S1 clips: old decode browser30.782 versus Python29.493, difference
**+1.290 [+0.0855,+2.5884] pp**, **not a tie**; frozen decode browser23.732
versus Python23.818, **-0.086 [-1.2163,+0.9590] pp**, tie,154/200 identical
versus140/200. First+3.27 browser-gap diagnosis packaged the wrong unaugmented
LM and is invalid for the shipped release. The log/docs mix pp point estimates
with fractional CI endpoints and incorrectly identify the old+1.29 with E-010:
same numerical point gap, **different checkpoint, LM, Python precision/reference
and CI**. These measurements do not show byte-identical acoustic execution.

**Noise caveat.** VAD path0 non-music; retained frozen-noise metrics137 music
words. Audit exploratory script records104 ->134 total words with VAD,
all surviving music; unguarded301 ->683 (non-music154 ->444). Different
script/beam conditions explain why those totals should not replace official137.
The bonus increases reliance on VAD, not nonspeech robustness by itself.

**Artifacts / status: measured, with exploratory selection caveat.**
[Plan](../runs/E-022/PLAN.md), [eval implementation](../runs/E-022/eval.py),
[frozen metrics](../runs/E-022/eval/frozen/), [paired deltas](../runs/E-022/diffs/vs-shipped.json),
[gate table](../runs/E-022/gate-table.json), [parity JSON](../runs/E-022/parity-wer.json),
[browser](../runs/E-022/browser.json), [release](../runs/E-022/release-0.6.0.json).
Reported about2 h machine time, no training/paid spend.

## E-023 - LM Retune and Size Tradeoff (2026-11-09)

**Question / protocol.** After recalibration, are weight/backoff/pruning optimal
within the20 MB budget? Fixed E-017 checkpoint, beam12/top10, same augmented
training-text corpus. **Full selection dev3,939**, not earlier400. Phase1
20 weight/bonus cells: weights0.3-0.7, bonuses1.6/2.0/2.4/2.8. Phase2
backoff0.2/0.3/0.4/0.5/0.6. Phase3 bigram/trigram count cutoffs
(1,1),(2,2),(3,3),(2,3). Candidate gate sets evaluated afterward.

**Outcomes.** Exact E-022 cell0.5/2.4 remains best, macro**22.9796**;
backoff macros23.0861/23.0088/**22.9796**/23.0554/23.2273.
Unpruned(1,1) macro**22.3260**, versus(2,2)22.9796,(3,3)23.5407,
(2,3)23.5004. Candidate gzip6.93 MB versus3.55; browser12.31 MB versus5.12,
estimated full package26.63 MB. Gzip savings are not browser download savings.

| Unpruned minus shipped | WER difference [95% CI] |
|---|---|
| T-600 | -0.77 [-1.40,-0.11] |
| T-S3 | **+0.43 [-0.61,+1.62]**, tie |
| FLEURS-IT | -0.72 [-1.01,-0.42] |
| FLEURS-EN | -0.37 [-0.81,+0.04], tie |
| CV-dev **full14,697** | -0.65 [-0.75,-0.55] |

Full-CV dev shipped/candidate23.524/22.874; this is **not** the19.90
first600 score in E-022. T-600 candidate25.143/54.075, S3 27.087/48.451.

**Decision / limits.** Retain shipped LM/parameters; reject unpruned artifact
because it exceeds hard download budget, not because it has no useful quality gain.
No detected S3 gain limits the core-task evidence, but does not prove a larger LM
cannot help code switching generally. This is a local finite-grid optimum, not
global decoder optimality. Reported2 h20 CPU,0 USD, no release change.

**Artifacts / status: measured.** [Sweep](../runs/E-023/sweep.json),
[paired gate differences](../runs/E-023/gate-diffs.json),
[nested gate results](../runs/E-023/eval/lm-1-1/), [result](../runs/E-023/RESULT.md).

## E-024 - Speech Noise Augmentation (2026-11-09)

**Question / actual protocol.** Does replacing half of CV's clean exposure with
noisy speech improve SNR robustness while retaining clean quality? Rerun the
E-017 recipe **from E-014 best**, not continuation from converged0.6.0. Hash
split CV into56,769 clean and56,349 noisy clips; per-ID deterministic SNR10-30 dB,
217 noise clips (120 MUSAN+97 room tone), cycling pool for training. Replace,
do not append another full CV copy. Total mixture nominal589.5 h/266,796 rows,
13,221 steps/epoch,2 epochs26,442, same optimizer/augmentation. One fixed
rendering per selected clip, not newly sampled noise every epoch. Raw CV is
available; MLS/LS raw deletion prevents analogous treatment without re-download.

**Build/selection.** [Build log](../runs/E-024/build.log):83.33 cached noisy h,
four shards,0 errors,1.1 min. Initial manifest duration83.6 h differs from decoded
cache duration. Best logged macro**30.8368 at26,000**; final30.8574 at26,442.
Release record says best step26,442, an **unresolved metadata conflict**. Results
reference `ckpt_best.pt`; no committed weight bytes establish that its contents
were the final-step weights. Reported training about3 h40, evaluation13 min.

**Primary measured comparison.** First100 S3 clips,589 words, recomputed raw-audio
mel, PyTorch CPU models, augmented LM, beam12/top10/0.5/2.4, no VAD in noisy
decode. Same seeded noise sequence per arm, RMS power scaling, tile/clip noise to
speech length, peak normalization if necessary. SNRs use different RNG positions,
not identical noise snippets across all SNRs. Evaluation pool includes room tone;
**not the audit's earlier MUSAN-only curve**.

| Condition | E-017 with E-022 decode | E-024 | Delta [95% CI] |
|---|---:|---:|---|
| Clean100 | 22.241 | 23.939 | +1.70 [-0.34,+3.72], tie |
| 30 dB | 25.976 | 23.260 | -2.72 [-5.63,+0.17], tie |
| 20 dB | 27.504 | 26.825 | -0.68 [-4.13,+2.73], tie |
| 15 dB | 33.616 | 27.334 | **-6.28 [-10.58,-2.38]** |
| 10 dB | 44.822 | 33.277 | **-11.54 [-16.18,-7.43]** |
| 5 dB | 50.934 | 41.256 | **-9.68 [-14.50,-5.01]** |

Mean delta over **four10/15/20/30 dB points**, excluding5 and clean:
**-6.180 pp**, passes predefined mean-gain>=4 rule. Clean T-600
25.917 ->25.201, delta-0.72 [-1.75,+0.26]; full S3 26.661 ->26.917,
+0.26 [-1.37,+1.96]; FLEURS-IT20.461 ->20.143,-0.32 [-0.69,+0.03];
full CV-test24.882 ->24.365,**-0.52 [-0.66,-0.38]**.

**Other readouts / adoption.** Adopted0.7.0. PyTorch T-600**25.201/54.171**,
S3**26.917/48.451**, FLEURS-EN49.143, CV-dev first60019.849.
int8 T-600**25.143/54.650**, S3**27.002/48.451**, both reported WER ties
against unquantized (-0.06/+0.08). These int8 values, not the PyTorch headline,
describe the released graph's T outputs under that runner. VAD noise run0 words
on337 non-music,140 music words, but its metadata says **beam5/top8**, not
the shipped12/10; identify it as a different noise decoder condition.
Browser/Python S1 int8 parity21.496/21.582 WER, difference-0.086
[-1.186,+1.127],152/200 identical; not bit identity. Browser20 s1T20.32x,
4T38.43x. Package remains about19.44 MB, **new acoustic hash**
`7a075da40eb37de6988947efc717824db8b144769628db078e48dcc35b5ab0af`;
LM/vocab/VAD unchanged. Verified fp32 median1.888e-4,worst1.217e-3,1/80>1e-3.

**Rejected alternatives / uncertainty.** All CV noisy removes clean exposure;
append full noisy CV grows mixture to757 h and changes CV share; MLS/LS requires
22 GB re-download. None is a measured rejected-quality arm. Shared217-clip
training/evaluation noise pool means familiar-noise evaluation; E-025 tests one
new room. VAD doesn't solve recognition in background noise, and the speech-
augmentation result is not training on **empty targets**. Clean "non-inferiority"
is a historical acceptance judgment, not a formal CI upper-bound<=1 pp test:
S3 upper bound+1.96 exceeds1, although the point estimate is+0.26 and CI includes0.
Only CV-test of the four tabulated clean sets is significantly better, not "two".
No one-seed proof of regularization or absence of clean tradeoff.

**Artifacts / status: measured.** [Config](../runs/E-024/config.yaml),
[log](../runs/E-024/train.log.jsonl), [mixing implementation](../runs/E-024/noise_eval.py),
[comparison JSON](../runs/E-024/comparison.json), [base/new curve counts](../runs/E-024/eval/),
[gate summary](../runs/E-024/eval/gate/summary.json),
[parity](../runs/E-024/parity-wer.json), [browser](../runs/E-024/browser.json),
[release record](../runs/E-024/release-0.7.0.json).

## E-025 - New-Room Noise Holdout (2026-11-09)

**Question / protocol.** Does E-024's noise improvement transfer beyond the shared
pool? No training/new weights. The proposed post-hoc80/20 split cannot unsee
training noise; all217 clips were cycled and original MUSAN sources deleted.
Record a different physical room **after training**, author-confirmed, built-in
mic,78.69 s; trim40-78 s to38 s to remove earlier loud events. Retained take
RMS-52.6 dBFS,peak-29.1,centroid1,050 Hz,32.7% energy<200 Hz,VAD fraction0,
one50 ms transient. First take rejected for50 Hz hum/93% energy<200 Hz;
untrimmed second take also retained as provenance.

Same first100 S3 speech clips and E-022 decoder as E-024, E-017/E-024 PyTorch
checkpoints. Per-condition seed0+condition-index selects random offsets in the
new take, **same mixed audio for both models**, no VAD in decode. SNR is global
RMS-relative with silence included, not calibrated field sound pressure or a
standard office/cafe condition.

| SNR | E-017 | E-024 | Difference [95% CI] | E-024-pool difference |
|---|---:|---:|---|---|
| 30 dB | 28.523 | 24.618 | **-3.90 [-7.20,-0.83]** | -2.72, tie |
| 20 dB | 38.710 | 31.919 | **-6.79 [-10.84,-3.15]** | -0.68, tie |
| 15 dB | 47.708 | 36.503 | **-11.21 [-14.69,-7.48]** | -6.28 |
| 10 dB | 63.328 | 44.992 | **-18.34 [-22.78,-13.40]** | -11.54 |
| 5 dB | 81.324 | 66.723 | **-14.60 [-18.84,-10.47]** | -9.68 |

**Decision / limits.** Evidence accepted for **one unseen noise recording**;
no model/version change. All WER deltas significant; mean four-point10-30 dB
delta-10.059 versus-6.180 in shared pool. Clean exactly reproduces22.241/23.939
from retained metrics; harness JSON compares rounded22.24/23.94 (about1e-5
residual). ETA differences at **every SNR are ties**; stronger WER robustness
does not establish rare-term gains under noise. Larger benefit does not mean
better absolute quality: 15 dB new-room E-02436.50 versus shared-pool27.33.
Different noise spectra/damage provide different baseline headroom.

Prediction of smaller gains(-3 to-6 at15 dB,-6 to-10 at10 dB) was wrong in
magnitude; retained plan is not retrofitted. One room/short take/reused speech,
trimmed transients, synthetic mixing and one seed limit generalization. This
does not close robustness for arbitrary real rooms, reverberation or competing
speech. Reported recording/analysis minutes, evaluation20 s,0 USD.

**Artifacts / status: measured.** [Plan](../runs/E-025/PLAN.md),
[used-room report](../runs/E-025/room_report.json),
[rejected first take](../runs/E-025/room_report_take1_rejected.json),
[untrimmed second take](../runs/E-025/room_report_take2_untrimmed.json),
[paired result JSON](../runs/E-025/eval/result.json),
[per-condition counts/hypotheses](../runs/E-025/eval/),
[actual offset/mixer implementation](../runs/E-025/holdout_eval.py).

## Non-E Audits and Runtime Runs

### Training Smokes, P0 and InterCTC Cost Probe

**Training feasibility smokes.** [E-005 smoke directories](../runs/E-005/)
retain padded and truncated batch configurations/logs, including192/32 and320/64;
[E-006 smoke](../runs/E-006/smoke/) checks the larger architecture. These are
memory/throughput checks, not separately selected quality models. Reported
unfrozen padded96/8 consumes about11.6 GB/swaps; truncated192/32 about4.2-8.4 GB
and430-520 audio-s/s. Frozen H1 throughput is not directly comparable.

| Smoke / actual cap | Last logged throughput audio-s/s | Last logged memory MB | Outcome |
|---|---:|---:|---|
| E-005 padded30,100 updates |12.383 at100|11,646.7|Completed short check, impractical memory/speed |
| E-005 truncated192/32,200 updates |428.393 at200|4,211.7|Selected workable batch |
| E-005 truncated320/64,60-update cap |338.855 at50|14,306.6|Large-memory configuration not selected |
| E-005 P0 benchmark,250 updates |470.466 at250|4,213.7|No dev selection;10 logged timing windows |
| E-006 smoke,100 updates |301.020 at100|5,339.7|Larger-model execution check |

Last-window values are not run medians and must not be used as controlled
speed ratios. P0's retained timing windows span449.075-536.959 audio-s/s;
its approximately473 value is consistent with the median, whereas477 is a
historical comparison without an independently matched timing arm in that folder.

**P0, 2026-10-22.** [bench-p0](../runs/E-005/bench-p0/) records the small
training benchmark after mmap caching/evaluation housekeeping; contemporary
[optimization review](10-optimization-review.md) and status report about473
versus477 audio-s/s, no meaningful throughput win. Fixing `dataset.epoch` is a
correctness change, **not demonstrated speed/quality gain in an isolated run**.
Historical H2 at42.6k macro35.38 is an interim snapshot, not E-005 final33.77.
Do not compare frozen MLS-only70.79 with bilingual macro35.38 as matched datasets.

**InterCTC cost only, 2026-11-09.** [Cost JSON](../runs/E-024/probe_cost.json)
and [probe code](../runs/E-024/probe_cost.py): MPS, batch32,600 mel frames/clip,
192 audio-s,60 timings/variant, auxiliary weight0.3. Median baseline259.390 ms,
auxiliary270.724 ms, overhead **4.370%**; extra-encoder sensitivity control
499.658 ms/1.926x validates timing sensitivity. Projected3.35 ->3.50 h,+8.8 min
is extrapolation. **No trained InterCTC quality result**, no measured0.2-0.7 pp
gain. Earlier failed timing designs are mentioned in prose but no numerical
results justify incorporating them into a performance average.

### Audit 2026-11-08 (Precursor to E-022)

**Scope.** [AUDIT.md](../runs/audit-2026-11-08/AUDIT.md) describes17 probes
at`ee2f34b`; its statement "nothing committed / no experiment entry yet" is a
historical snapshot. Raw JSON/code retained under
[audit directory](../runs/audit-2026-11-08/) and copied narrative
[E-022 audit account](../runs/E-022/AUDIT-PROBE.md). Not17 independent model
trials; sequential decoding searches on the same weights/data.

| Probe family / useful artifact | Result and disposition |
|---|---|
| Baseline reproduction / [gamma0](../runs/audit-2026-11-08/token-bonus-gamma0.json) | FLEURS IT26.52,EN55.48,CV-first60024.91; establishes old decode reference |
| [Token/boundary bonus sweep](../runs/audit-2026-11-08/token-bonus.json) | At fixedLM0.2, gamma0 ->1: dev31.607 ->29.840, T32.769/42.857 ->31.021/45.446; beta marker bonus also explored, not adopted as a separate feature |
| [Extended bonus sweep](../runs/audit-2026-11-08/token-bonus-ext.json) | Gamma>1 explored across T/FLEURS/CV; exploratory test-visible search, not a confirmatory held-out result |
| [2-D LM/bonus grid](../runs/audit-2026-11-08/lm-gamma-grid.json) and transfer | Saved grid has30 rows, weights0-0.4 and gamma0-2; later cells/beam variants live in scripts/other summaries. Do not claim this JSON alone contains the full0.5/2.4 selection surface |
| [Transfer summary](../runs/audit-2026-11-08/lm-gamma-transfer.json) | At0.5/2.4 narrower search T26.83/51.10,FLEURS IT21.21,EN50.38,CV60020.72; differs from final beam12/top10 T25.92, not contradictory weights |
| [Dev-only candidate](../runs/audit-2026-11-08/dev-only-summary.json) | Final beam12/top10 gives T25.917/52.637,S3 26.661/47.324; beam8 fallback26.376/51.774 and27.257/46.479. Adopted through E-022, not a second training gain |
| [Peeked final candidate](../runs/audit-2026-11-08/final-summary.json) |0.6/3.0 T24.685/54.650,S3 25.468/49.859. Measured but **rejected for test-informed selection** |
| [Noise/bonus interaction](../runs/audit-2026-11-08/noise-token-bonus.json) | VAD-off total301 ->683 words; MUSAN29 ->63,room5 ->141,silence120 ->240,music147 ->239. VAD-on104 ->134 total, all music. Retain VAD; no robustness claim from bonus alone |
| [Gold re-decode](../runs/audit-2026-11-08/gold-deltas.json) | Same new decoder across E-014/E-016/E-017 T-600: **45.499/27.900 ->29.989/46.021 ->25.917/52.637**. Gold gains survive recalibration; marginal WER gains15.51 then4.07 pp, not5.73 under the old decoder |
| [Leading/trailing trim probe](../runs/audit-2026-11-08/trim_test.py) | E-017, beam8/top8/0.5/2.4,300 ms VAD margins; audit reports about-0.7 WER,tie. Script exists; no retained matching numeric trim summary/hypotheses found in that directory. Treat magnitude/CI as **reported**, do not invent an exact interval |
| Beam-width/candidate checks | Scripts `sweep_beam.py`, `sweep_beam_transfer.py`, `beam8_rest.py`, `final_validation.py`, `int8_check.py` remain. Audit reports beam16/top12 worse dev by0.4 pp at60% more decode time; no standalone retained numeric table for that timing claim. Not adopted |

The audit's gold re-decode uses augmented LM consistently, unlike the original
T-S1 E-014/E-016 report. It supports persistence of a gold-associated benefit,
not a constant "points per speaker" law. Audit hypotheses/control descriptions
occasionally mix S1/S2 training roles and future S4; use actual T-600 composition.
Int8/gate comparisons in later E-022 are authoritative for the adopted decode.
Boundary penalties, EMA on later gold checkpoints and BPE2048 remain suggestions,
not executed production changes merely because they occur in this audit.

### Audit 2026-11-09 (Long Clips, Segmentation, Noise)

**Duration hypothesis.** Docs/11 reports FLEURS duration-quartile WER
17.8/19.9/23.1/20.0: nonmonotonic, no simple evidence "longer alone is worse".
No standalone quartile JSON is retained in this audit directory; label those
values **reported descriptive analysis**, not a controlled duration experiment.

**Known-boundary concatenation.** [Long-clip JSON](../runs/audit-2026-11-09/long_clip_probe.json)
and [script](../runs/audit-2026-11-09/long_clip_probe.py): E-017 checkpoint,
augmented LM/E-022 decoder,200 S3 clips arranged into40 groups of five,
about12.479 s/group, artificial gaps. Whole WER **35.945**, true-boundary
separate **26.576**, difference **-9.370 [-12.223,-6.361]**, ETA
39.155 ->47.606. This supports a penalty for this artificial concatenation
protocol; it does not independently identify duration, acoustic seams, mel
normalization or real multi-sentence speech as the sole cause.

**VAD segmentation evidence conflict.** Prose claims about36.6 ->28.5 WER,
80% recovery and CI[-12.22,-3.51] do **not** match the retained known-boundary
JSON above. More importantly,
[retained VAD summary](../runs/audit-2026-11-09/vad_chunk_probe.json) records
**120 groups of one**, margin0.2 s,rule`multi`, whole and VAD WER both
**25.289**, ETA both47.159,delta0,CI[0,0],median one segment. It is a
single-clip no-op control, **not** the40-group/five-clip recovery run.
[VAD script](../runs/audit-2026-11-09/vad_chunk_probe.py) overwrites the same
summary/output paths for each invocation, plausibly explaining loss of the
earlier run; this is not proof of its claimed numbers. Accept that segmentation
code exists, but "80% recovered" remains **reported / not artifact-verified**.
The reported+1.6 to+2.9 pp single-clip trimming penalty also lacks a retained
matching numerical sweep. Current control shows avoiding segmentation when
fewer than two runs is a no-op on that sample, not that all single clips are safe.

**MUSAN-only noise probe.** [Noise JSON](../runs/audit-2026-11-09/noise_curve.json)
and [script](../runs/audit-2026-11-09/noise_curve.py): first100 S3 clips,
E-017 CPU, E-022 decode,120 MUSAN noises only, seeded random sources, raw mel;
VAD measured but not used to suppress hypotheses. Clean22.241;30/20/15/10/5 dB
WER **26.995/32.088/38.370/48.048/58.234**; all conditions significantly
worse than clean, deltas+4.75/+9.85/+16.13/+25.81/+35.99. Deletions
5.60% clean ->24.62% at5 dB; no noisy speech clips rejected by VAD.
Useful to motivate E-024, **not interchangeable with** its217-pool base curve
25.976/27.504/33.616/44.822/50.934. The added room-tone pool changes sources
and RNG source assignments. Docs/11 D32's+5.3/+11.4/+22.6 differences refer
to the latter base curve, not this original diagnostic.

### Runtime Acceptance 2026-11-10 (No New Model)

**Question / protocol.** Can a lightweight Python release loader, binary LM and
decoder execute the **original0.7.0** assets with integrity/parity safeguards?
Windows11 build26300,x86_64 Intel family6/model189,8 logical CPUs,
Python3.12.10,NumPy2.5.4,ORT1.31.0 CPUExecutionProvider/default threads,
Node20.20.2. Pinned release record from`a64f2a1...`, model SHA`7a075da4...`,
normalizer SHA`d74abd1e...`. No training, tuning, weights or published release change.

**Population.**104 previously used FLEURS/spliced/control clips,1,305.6 s;
40 IT,40 EN,20 artificial splices and four synthetic nonspeech controls.
Raw references officially normalized, mixed numeric text Italian fallback,
all`en_spans`empty, **ETA null**. Splices reuse constituent clips and are
correlated. Manifest/audio live in sibling paths, not a fresh independent
natural code-switching benchmark.

| Path | WER | Total measured s | RTF | RTFx |
|---|---:|---:|---:|---:|
| VAD + segmentation | 40.4817 | 52.4988 | 0.04021 | 24.869 |
| No VAD | 39.8213 | 35.9187 | 0.02751 | 36.349 |

VAD-minus-no-VAD WER **+0.660 [-0.456,+1.852] pp**, tie, **no demonstrated
quality gain**.56 speech clips segmented,four controls gated; controls0 words
with VAD versus5 without. VAD path stage seconds:VAD8.843,segmentation0.006,
mel13.339,acoustic9.058,decode21.037; no-VAD9.832/7.502/18.579 for
mel/acoustic/decode. Asset loading0.280 s excluded; no warmup, serial VAD then
baseline, default threads/shared host. File IO/hashing/scoring/Node IPC excluded.
Not comparable as a speed ranking with M5 median browser fixtures.

**Parity / integrity.** Same **native ONNX logit tensors** sent to JS/Python:
beam10/10 and greedy10/10 raw transcripts identical (round-robin IT/EN/spliced,
canonical Python mel,no VAD,real binary LM).459 synthetic paired decoder
fixtures/self-checks passed historically. This tests decoder implementations,
**not browser WASM acoustic or JS feature extraction parity**. Five runtime
payload hashes checked in config; NOTICE/config`hash_verified:false` in
report does not contradict all-seven external-record matching: they are
checked against the pinned E-024 record separately (`all_seven_files_match:true`).
All-seven total**19,440,760 B**, headroom**559,240 B** under20,000,000;
E-024's`package_bytes`19,436,178 counts only five runtime payloads, excluding
NOTICE/config. Both round to19.44 MB.

**Decision / evidence.** Infrastructure acceptance passed on reused sample;
no natural mixed ETA, quality optimization or portability-equivalence claim.
Historical targeted post-run139 passed/2 skipped; Windows suite excluding Unix
`resource` training import446 passed/3 skipped/5 unrelated failures. These are
retained execution results, **not tests rerun by this documentation audit**.

**Artifacts / status: measured.** [Result](../runs/runtime-2026-11-10/RESULT.md),
[full report](../runs/runtime-2026-11-10/report.json),
[VAD official metrics](../runs/runtime-2026-11-10/metrics.vad.json),
[no-VAD metrics](../runs/runtime-2026-11-10/metrics.no_vad.json),
[benchmark](../runs/runtime-2026-11-10/benchmark.py),
[test transcript](../runs/runtime-2026-11-10/validation-tests.txt).
`runs/browser-acceptance-2026-11-10/` is empty in this checkout: **no separate
browser-acceptance result exists**. New mixed-dev/contextual-glossary/LM-pruning
quality experiments in [the runtime plan](plans/2026-11-10-lightweight-it-en-runtime-plan.md)
remain blocked on missing data, not completed by these runtime checks.

## Release and Browser Benchmark Chronology

These are complete feature+acoustic+decode timings; VAD is included only when
stated. M5 headless Chromium/WASM SIMD, three warm-ups and median ten runs,
5/10/20 s fixtures, unless noted. They are **not paired performance ablations**
across dates/host load. Synthetic tone fixtures may be gated after guards were
introduced; consult each record's suite/fixture handling before interpreting
an inference-only or gated duration as full speech transcription throughput.

| Stage | Checkpoint / decode | Payload MB | RTFx4T,5/10/20 s | RTFx1T,5/10/20 s | Evidence |
|---|---|---:|---|---|---|
| E-003 | Random4-block,greedy | 8.32 |74.0/71.0/64.7|37.9/33.1/32.7|[JSON](../runs/E-003/rtfx.json)|
| E-009 | E-006,greedy |11.99|66.4/61.3/54.6|29.3/28.3/24.6|[JSON](../runs/E-009/rtfx.json)|
| E-010 | E-006,beam5/original LM0.2 |17.08|65.2/63.3/58.1|30.0/28.3/25.2|[JSON](../runs/E-010/rtfx.json)|
|0.3.0 | E-016,beam5/original LM,energy guard |17.08|61.0/60.9/53.8|28.5/27.0/23.7|[reported addendum](../EXPERIMENTS.md)|
|0.3.1 | E-016,augmented LM,energy guard |17.11|64.3/62.5/57.3|29.1/27.6/24.5|[reported addendum](../EXPERIMENTS.md)|
|0.4.0 | E-016,augmented LM+VAD |19.44|48.3/46.4/41.8|25.2/24.0/21.5|[record](../runs/E-011/browser-0.4.0.json)|
|0.5.0 | E-017,beam5/LM0.2+VAD |19.44|52.2/50.8/46.3|25.5/24.3/21.8|[record](../runs/E-017/release-0.5.0.json)|
|0.5.1 | Same weights,verified export |19.44|52.0/51.0/46.5|25.4/24.2/21.7|[record](../runs/E-021/release-0.5.1.json)|
|0.6.0 | E-017,beam12/LM0.5/token2.4+VAD |19.44|47.3/44.8/39.1|24.3/23.0/20.3|[JSON](../runs/E-022/browser.json)|
|0.7.0 | E-024,same decode+VAD |19.44|45.3/43.2/38.4|24.3/22.9/20.3|[JSON](../runs/E-024/browser.json)|

The current release is0.7.0. E-025 updates evidence/card, not weights; E-015
actual final is rejected. Static HF Space reports about19x one-thread without
COOP/COEP, not a separate isolated timing table here. Earlier successful runtime
exports/demos and later public publishing do not retroactively convert failed
quality gates into passes.

## Reconciled Discrepancies and Publication Limits

1. **ETA direction reversed in E-006/E-007 and docs/11 D17.**16.42 ->20.00
   and20.00 ->20.90 ->21.79 are improvements in point estimates. Hotwords were
   not demonstrated a useful gate-closing lever, but "ETA worsened" is false.
2. **Padding equivalence overstated.** C2's73.755 versus interpolated B74.06
   differs by0.305 pp, not<0.001 as a fraction. Matching early curves fixes
   the original schedule confound; no formal equivalence test exists.
   The padded steps/epoch7055 in the narrative also conflicts with logged6990.
3. **Wrong baseline bar.** E-008's claimed H2 pass used tiny greedy32.09;
   best measured auto-beam5 is27.42. E-022 later wins this bar20.46, not E-008.
4. **T populations and voices mixed.** T-S1/T-400/T-600 differ; T-600 has
   three speakers, two trained,400/600 seen-voice clips. E-017 passes both
   metrics against forced-IT tiny on aggregate, **not** against auto at that
   stage, and not strict held-out-S3 ETA. Future S4 scripts are not recorded T-800.
5. **Gold experiment not one-variable speaker count.** E-017 also adds gold
   duration, different text/channel, six S1 takes and optimizer steps. Evidence
   supports cross-voice transfer in this setup, not an architecture-causal or
   pure vocabulary effect. S3's shared microphone remains a limitation.
6. **Whistle LoRA is not a controlled architecture comparison.** Its replay,
   optimizer, update type, corpus split/exposure and language forcing differ.
   S3 and seen-gold deltas are ties. "Architecture-bound" is a hypothesis, not
   an established mechanism. The29.35 MB archive was also mislabeled28 MB and
   its budget overage miscomputed in RESULT.
7. **CV subsets omitted.** E-022/E-024/E-015 dev19.90/19.85/19.97 use600,
   E-023 dev23.52 ->22.87 uses14,697; E-019 test17.59 ->20.25 uses1,000,
   original Whistle20.63 uses14,643. Those are not contradictory system scores.
8. **Dev/checkpoint metadata drift.** E-01332.79 is greedy selection, not
   beam. E-014 best is22,494, not36,296. E-01730.98 is not the lowest historical
   dev. E-024 log best26,000 conflicts with release label26,442. E-015 step20
   has20 updates, so "baseline itself" is approximate, not proven byte identity.
9. **Term readout is not aligned ETA.** E-015 set-membership54.65 differs
   legitimately from official54.17; its "18 terms" constant contains16.
   New44-term targeting uses test vocabulary, and a second-voice addendum is
   still proposed. Distillation **never ran**; E-018 **never ran**.
10. **Parity CI units and false identity with E-010.** E-022 old decode
    +1.290 pp CI[+0.0855,+2.5884] is significant, not E-010's tie interval.
    Frozen -0.086 pp CI[-1.2163,+0.9590] is a tie. Fractional JSON intervals
    must all be multiplied by100 when paired with pp point estimates.
11. **Noise curves describe different stimuli.** Audit MUSAN-only38.37 at15 dB,
    E-024 shared-pool33.62, E-025 new-room47.71 are three baseline conditions,
    not three versions of one run. SNR alone does not define perceptual difficulty.
12. **Segmentation claim lacks matching retained result.** Long-clip35.945
    ->26.576 CI[-12.223,-6.361] conflicts with prose36.6/26.6 and a different
    CI. Retained VAD JSON is120 single-clip no-op groups, not80% recovery.
13. **Noise and clean acceptance overstatements.** Expanded non-music56.17 min,
    not60; music is excluded from scoring gate, not exempt from VAD. E-024's
    nonspeech run records beam5, not12. One significant clean-set improvement
    is present, not two; a tie does not prove<=1 pp non-inferiority. RESULT's
    "faster" at30 dB compares WER values, not speed, and should not be quoted.
14. **Sizes and precision must stay explicit.**0.7.0 five-payload sum19,436,178 B
    versus seven-file19,440,760 B are consistent accounting choices. Python gzip
    LM3.55 MB is not browser5.116 MB. PyTorch T-60025.20/54.17 is not int8
    25.14/54.65. Runtime x86 reuse checks do not replace ARM-vs-x86 quality
    comparison on the same natural T set.
15. **Evidence completeness is uneven.** docs/11 says every metrics file has
    checkpoint/decode/hash/SHA, but many `score_rows` outputs lack that wrapper;
    keep adjacent scripts/configs. Build report lacks its advertised tokenizer
    block. Some smoke/timing/trim/quartile/segmentation claims survive only in
    prose. References to absent `train.out` or ignored weights do not create
    committed evidence. Missing metadata stays missing.

**Paper-safe synthesis.** In this single-seed development history, transcript
LMs, targeted gold and decode recalibration produced useful gains; more general
English, larger BPE and simple splicing did not establish the intended rare-term
benefit under their tested recipes. Noise augmentation transferred to one new
room stimulus, with large absolute residual errors. The model remained small
and fast on measured hosts. These are bounded observations about these systems,
datasets and recipes, not a demonstration that CTC is causally superior to
autoregressive architectures or that data coverage is the only bottleneck.

Remaining publication threats: repeated test inspection/targeted development,
one seed per training configuration, utterance rather than speaker/source
bootstrap, small number of speakers/channels/noise recordings, public-benchmark
contamination uncertainty, many comparisons without familywise correction,
filtered benchmark populations, synthetic mixtures/concatenations, incomplete
baseline cells, and untracked raw/checkpoint dependencies. No strict natural
mixed held-out ETA certification should be inferred from runtime acceptance.

## Verification of This Ledger

This documentation audit parsed **305 `*.metrics.json` files** under `runs/`
and `baselines/runs/`. For nonempty-reference results, reported WER equals
`(sub + del + ins) / ref_words`; retained per-utterance counts sum to aggregate
counts, and aligned English hit/reference sums reproduce reported ETA. No
arithmetic mismatch was found. This verifies stored score consistency, not
audio/reference correctness, provenance independence or model execution.

All26 numbered-ID sections are present, including explicit not-run entries;
all177 relative Markdown links resolve in this checkout. Training-log best/final
events, selected configs, release bytes/hashes, scorer/readout definitions and
diagnostic mixer code were inspected directly. ASCII/whitespace checks passed.
No inference, training, browser suite or repository test suite was rerun for this
documentation-only change. Temporary audit helpers were kept outside the repository.
Only this ledger was authored for this task; `EXPERIMENTS.md` was left unchanged.
Other documentation may change concurrently; discrepancy descriptions refer to
the historical statements and artifacts audited, not a promise that those prose
errors remain uncorrected in every later revision.
