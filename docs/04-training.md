# 04 — Training

## 1. Where training runs, and the evidence

**Primary: the author's MacBook Pro** — Apple M5, 16 GiB unified memory (17,179,869,184 bytes), ≈ 108 GiB free
disk, macOS, `uv` available, no `ffmpeg` and no PyTorch installed at the time of the benchmark.

### Feasibility benchmark (2026-10-18)

A throw-away script in a scratch virtual environment (not in any repository, deleted after the session):
PyTorch **2.14.1** from PyPI, device `mps`, a **stand-in with the model's exact shape and random weights**
(Whisper-tiny encoder shape + CTC head, V = 1024, 8.03 M parameters, built from `nn.TransformerEncoderLayer`),
full fine-tuning steps (forward, CTC loss, backward, AdamW) on **32 synthetic clips of 10 s per step**
(mel `[32, 80, 1000]`, 500 encoder frames, 50 targets each).

| Setting | CTC loss on | ms/step | Audio-seconds per second | Audio-hours per hour | Peak GPU memory |
|---|---|---|---|---|---|
| fp32 | MPS | 578 | 554 | ≈ 554 | 4.3 GiB |
| bf16 autocast | MPS | **394** | **812** | ≈ **812** | 4.4 GiB |
| fp32 | CPU | 577 | 554 | ≈ 554 | 5.4 GiB |
| bf16 autocast | CPU | 432 | 741 | ≈ 741 | 5.4 GiB |

Findings:

- **CTC loss forward *and* backward ran natively on MPS** with no CPU fallback (no `PYTORCH_ENABLE_MPS_FALLBACK`
  set; an unimplemented op would have raised). This changes the earlier assumption that MPS lacked `ctc_loss`.
  Upstream, [#187716](https://github.com/pytorch/pytorch/pull/187716) (forward) was merged in June 2026; the backward
  PR [#188187](https://github.com/pytorch/pytorch/pull/188187) was still open when last checked, so which change
  delivered the backward pass in 2.14.1 was not verified — only that it works there. Pin the torch version.
- The CTC loss cost is negligible (MPS and CPU placement give the same step time).
- bf16 autocast is ≈ 1.4× faster than fp32 and ran without errors; fp16 is avoided (a third-party script reported NaNs with fp16 on MPS).
- Peak memory ≈ 4.4 GiB for 320 audio-seconds per step: comfortable on 16 GiB.
- One epoch over 100 h of audio ≈ 7–11 minutes at these throughputs.

**Not yet verified** (to be checked in the first training runs): behavior with the real Whisper weights; a real data
loader; numerical parity of MPS vs CPU CTC loss on a small batch (a test is specified); sustained thermal throttling
over hours; throughput with key-padding masks and real length distributions.

### Time estimates

| Run | Audio processed | Pure compute at 550–810 audio-h/h | With loader/eval overhead (estimate) |
|---|---|---|---|
| H1 (frozen encoder, ≈ 100 h, 3 epochs, forward-only encoder) | 300 audio-h | < 1 h | < 1 h (padded-input variant ≈ 2×) |
| **H2** (≈ 440 h × 8 epochs) | ≈ 3,500 audio-h | ≈ 4.3–6.4 h | **≈ 5–10 h** |
| H3 (≈ 440 h-equivalent × 1–2 epochs) | ≈ 440–880 audio-h | ≈ 0.5–1.6 h | ≈ 1–3 h |
| Whisper-base fallback (H2-sized) | ≈ 3,500 audio-h at ≈ 2.6× the FLOPs | ≈ 11–17 h | ≈ 13–25 h (extrapolation) |

## 2. Environment

- Python ≥ 3.12 managed by `uv`; `pyproject.toml` pins `torch==2.14.*`, `numpy`, `sentencepiece`, `soundfile`,
  `num2words`, `onnx`, `onnxruntime`, `pytest`, `ruff`. `transformers` is a *dev* dependency used only by the
  weight-conversion parity test.
- One `--device {mps,cuda,cpu}` flag; the training code contains no other device-specific branches.
- bf16 autocast for the encoder/head matmuls; `log_softmax` and CTC in fp32.
- Seeds fixed for data order and initialization; MPS CTC backward uses atomics and is **non-deterministic** — runs
  are repeatable in distribution, not bit-exact. Say so in `EXPERIMENTS.md`.
- Keep the machine awake (`caffeinate -i`), on power, with other heavy apps closed.

## 3. Batching

- **Duration buckets:** sort by duration into buckets; each batch has ≤ **320 audio-seconds** in total (the benchmarked
  load) and ≤ 64 clips; shuffle batches, not clips within a bucket's tail.
- Encoder gets a key-padding mask; CTC gets true `T_enc` and label lengths.
- If the Mac shows memory pressure, lower the audio-seconds budget before touching anything else.

## 4. Augmentation (on cached mel)

| Augmentation | Setting (starting point) |
|---|---|
| SpecAugment, frequency | 2 masks, width ≤ 15 of 80 bins |
| SpecAugment, time | up to 5 masks (1 per 2 s of audio), width ≤ 40 frames (0.4 s), total masked ≤ 20% |
| Mel-domain tempo stretch | factor ∼ U(0.9, 1.1) by time interpolation |
| Mixing | none in v0.1 |

H1 uses light SpecAugment only; H2/H3 use the full set.

## 5. The experiment ladder

Each experiment gets an `EXPERIMENTS.md` entry (hypothesis, configuration, dataset, parameters, hardware, metrics,
result, conclusion) and a `runs/<id>/` directory with config, metrics and logs.

### H0 — baselines (no training)

- **Question:** where do existing models stand on our test sets under our `normalize()`?
- **Run:** Whisper tiny and base (auto vs forced Italian), Cactus Whistle (auto vs forced `it`, with and without
  keywords), Parakeet TDT v3 (offline, with and without hotwords). Details in `05-evaluation.md` §7.
- **Outputs:** the baseline table and the real numbers for every relative bar below. Cost: 0.

### H1 — frozen encoder + linear head (go/no-go probe)

- **Hypothesis:** the pretrained Whisper-tiny encoder exposes features from which a linear CTC head can read
  Italian, and unpadded (truncated) input is not catastrophically worse than 30 s-padded input.
- **Config:** encoder frozen; `Linear(384, V_ctc)` trained; ≈ 100 h Italian (Common Voice + MLS); 3 epochs;
  peak LR 1e-3; light SpecAugment. Two inference/training modes: **truncated** and **padded to 30 s** (CTC sees only real frames).
- **Measure:** dev WER (greedy) per mode with confidence intervals; CTC loss curve; time.
- **Gate (provisional; calibrate with H0):** the head learns (dev WER clearly below the trivial/blank baseline and
  still falling at the end), and truncated is not worse than padded by more than the noise interval. If truncated
  fails badly, H2 trains with padded 30 s windows (slower browser inference) and the decision is recorded.

### H2 — full fine-tune on the IT+EN mix

- **Hypothesis:** fine-tuning the whole encoder with CTC beats Whisper-tiny on Italian dev/test sets and
  gives usable English.
- **Init:** pretrained encoder + the H1 head (equivalently: freeze the encoder for the first 1,000 steps).
- **Config (starting points, to tune on dev):**

| Parameter | Value |
|---|---|
| Data | ≈ 340 h IT + ≈ 100 h EN, mixture weights IT 0.7 / EN 0.3 by audio time |
| Optimizer | AdamW, betas (0.9, 0.98), eps 1e-8, weight decay 0.01 |
| Peak LR | head 1e-3, encoder 1e-4 (layer-wise decay optional) |
| Schedule | linear warm-up 1,500 steps → cosine to 5% of peak |
| Batch | ≈ 320 audio-seconds (≈ 5,000 steps/epoch at 440 h) |
| Epochs | 8 (≈ 40k steps); early-stop on dev |
| Gradient clipping | 1.0 |
| Precision | bf16 autocast, fp32 CTC |
| Averaging | average the last 3–5 checkpoints before final evaluation |

- **Measure:** dev WER every 2,000 steps (IT sets and EN sets separately; selection metric = macro-average); final: FLEURS
  IT/EN, Common Voice IT test, mixed test (T), noise set, quick browser RTFx via an early export.
- **Gate:** beats Whisper-tiny (H0 number) on Italian FLEURS and Common Voice dev. If not → try, in order:
  more epochs / LR tuning, extra head blocks (`02` §7), BPE size 512/2048; if still not → **Whisper-base fallback**
  (needs a decision to relax the size ceiling to ≈ 25 MB, `02` §7).

### H3 — code-switch data

- **Hypothesis:** adding mixed-sentence audio improves WER and ETA on the mixed test set without hurting
  FLEURS IT/EN beyond the noise interval.
- **Config:** continue from the H2 checkpoint, 1–2 epochs, LR 5e-5 (encoder) / 3e-4 (head); sampling mixture
  (by audio time) H3a: IT 0.5 / EN 0.2 / gold 0.3; H3b: IT 0.4 / EN 0.15 / gold 0.2 / spliced 0.25 (starting points):

| Arm | Extra training data |
|---|---|
| H3a | Gold recordings only (S1+S2 script R) |
| H3b | Gold + utterance-level splicing |
| H3c | Gold + word-level splicing or TTS (only if H3b falls short) |

- **Measure:** mixed test per category and per speaker group (seen vs held-out S3), ETA, FLEURS regression.
- **Gate:** keep a data source only if it improves the mixed-set WER or ETA beyond the noise interval.

### H4 — export check (can run as early as M3)

Export a random-weight model to int8 ONNX and measure browser RTFx **before** training is finished, to de-risk the
speed bar (`06`). After H2/H3: int8 vs fp32 parity.

## 6. Checkpointing, logging, resume

- Checkpoint every ≈ 20 minutes of wall-clock (model, optimizer, scheduler, scaler state, data-loader position, RNG
  states). Keep the last 3 plus the best by selection metric. Resume restarts the same run ID mid-epoch.
- Training log: one JSON line per 50 steps (loss, LR, grad norm, audio-seconds/s, memory); dev line per evaluation.
- Checkpoints and mel shards are `.gitignore`d; final weights go to the Hugging Face repo. Run configs and metrics
  JSON are committed.
- Anomaly guards: abort on non-finite loss for > 20 consecutive steps; log and skip a batch whose CTC loss is `inf`
  (counted — `zero_infinity` handles the gradient).
- Parity guard at the start of every run: the same small batch's CTC loss on MPS and CPU agrees within 1e-3 (fp32).

## 7. Cloud fallback (only if the Mac proves insufficient)

| Platform | How | Limits |
|---|---|---|
| **Kaggle** | Upload mel shards as a Kaggle Dataset; run `train.py --device cuda`; write checkpoints to `/kaggle/working` and download them. | 30 GPU h/week, sessions ≤ 12 h, ≈ 20 GB working disk, phone verification. The 26 GB mel cache does not fit in one disk — shard into several datasets or use a smaller mixture. |
| **Colab** | Mount Drive; same script. Free tier is dynamic; Pro costs ≈ 10 USD per 100 compute units. | Sessions end unpredictably; checkpoint often. |

Cost guardrail: log compute units/dollars spent in each `EXPERIMENTS.md` entry; **stop at 8 USD** and review. The
budget is expected to stay unspent.
