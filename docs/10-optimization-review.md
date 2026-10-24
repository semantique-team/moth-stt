# 10 — Optimization review: architecture, training, decoding, export

> **Date:** 2026-10-21
> **Scope:** review of the current Moth architecture and code against the project constraints
> (Whisper-tiny encoder + CTC, ≈ 8 M parameters, IT+EN code-switching, browser/WASM, ≈ 8–10 MB int8),
> with a prioritized list of code-level optimizations and recent STT literature that is actually
> applicable. This document does **not** change the model or the training recipe by itself.
> **Status:** living review; each item becomes an `EXPERIMENTS.md` entry only when it is run.
> **Context:** H2 (`E-005`) was training on the author's M5 while this review was written. The live
> numbers below are a snapshot, not a benchmark. Nothing here was run on MPS while H2 was training.

---

## 1. TL;DR

1. **The architecture is sound.** A pretrained Whisper-tiny encoder + a linear CTC head is the right
   compromise for 8 M parameters. The H1 → H2 jump (frozen 0.7079 → full fine-tune 0.3538 dev macro)
   shows that full fine-tuning is where the gain is; do not freeze the encoder or switch to LoRA.
2. **Do not touch the running H2.** It is at ~42.6 k / 72.6 k steps, dev macro 0.3538, still
   descending. Changing `train.py`, `dataset.py` or the model now would invalidate the comparison.
3. **There is a real augmentation bug**: `MelDataset` seeds SpecAugment/tempo with `self.epoch`,
   but `train.py` never updates `dataset.epoch`. Every epoch sees the same augmented view of each
   clip. Fix it for H3/future runs, not for the running H2.
4. **The training loop leaves throughput on the table**: single-threaded loading, `np.load(...,
   mmap_mode="r")` per clip, two/three copies per clip, no prefetch, manual attention with an fp32
   softmax, and a dev eval in fp32 with per-utterance CPU argmax. The measured 450–530 audio-s/s is
   ~20–40 % below the 590 (H1 truncated 320 s/64) and 812 (synthetic benchmark) references.
5. **The cheapest quality wins** are: checkpoint averaging/EMA, hotword beam + a small LM at decode
   time, and InterCTC (training-only). CR-CTC is the strongest CTC-specific training upgrade but
   costs 2× compute; use it as a short fine-tune from H2, not a full rerun.
6. **The export graph has clear headroom**: 8.02 MB int8 + 0.12 MB fp32, 25 `MatMulInteger`, 19
   `DynamicQuantizeLinear`, 2 `ConvInteger`, and **8 fp32 `MatMul` for attention**. QKV fusion,
   static QDQ, and keeping `conv1` fp32 are the three experiments to run with real H2 weights.
7. **Capacity escalation should not start with Whisper-base.** Try a conv head, then +2 blocks, then
   **25 Hz + 2 blocks** (more depth, less compute, ≈ 12 MB int8). Whisper-base (20.35 M, 20.5–23.5 MB)
   is the last resort because it breaks the 20 MB ceiling.
8. **Noise augmentation is the biggest robustness gap** for a browser/mic product. Oído's 2026
   results show noise/reverb augmentation moving average WER from 8.9 to 6.2 across 14 conditions.
   This is a data-pipeline project, not a one-line change; do it if the noise set is a gate.
9. **Do not over-invest in BPE size.** The 2026 German tokenization study finds small/inconsistent
   WER differences between BPE-512/1024/2048. Test 2048 only if ETA is the blocker.
10. **The 10× browser bar is already comfortable** (65–74× at 4 threads, 33–38× at 1 thread with
    random weights). WebGPU, 4-bit, and streaming are v0.2+ concerns, not v0.1.

---

## 2. H2 snapshot (measured, not a benchmark)

Source: `runs/E-005/train.log.jsonl` in the main checkout (`/Users/tancredi/Desktop/semantique/moth-stt`).

| Item | Value |
|---|---|
| Run / config | `E-005`, H2 full fine-tune, 437.7 h (CV IT 167.8 + MLS IT 169.9 + LS EN 100) |
| Init | H1 padded-arm best checkpoint (`runs/E-004/arm-padded30/checkpoints/ckpt_best.pt`) |
| Batch | truncated, 192 audio-s / 32 clips |
| Schedule | 8 epochs, 9,074 steps/epoch, 72,592 steps; warm-up 1,500; cosine to 5 % |
| LR | head 1e-3, encoder 1e-4; AdamW (0.9, 0.98), wd 0.01; clip 1.0 |
| Precision | bf16 autocast; fp32 CTC; MPS, torch 2.14.1 |
| Step at snapshot | 42,600 / 72,592 (epoch 4/8) |
| Dev macro WER | **0.3538** (IT 0.3955, EN 0.3122) |
| Throughput | 450–530 audio-s/s in the log windows |
| Memory | 3.2–5.3 GB driver-allocated |
| Best checkpoint | `ckpt_best.pt` at 0.3538 |

The dev curve is still going down (24 k: 0.3922 → 42 k: 0.3538). The final H2 gate is **FLEURS-IT and
Common Voice IT dev**, not this MLS/LS dev; do not compare the 0.3538 with the Whisper-tiny FLEURS
baseline (0.3209) directly.

H1 (frozen encoder, linear head only) ended at **0.7079** dev WER; H2 full fine-tuning is at
**0.3538** by epoch 4. That is the strongest argument for the current recipe.

---

## 3. Architecture assessment

### 3.1 What is already right

| Choice | Why it is right |
|---|---|
| Whisper-tiny encoder pretrained + CTC | Reuses acoustic knowledge that would cost 10⁴–10⁵ h to learn from scratch; drops the decoder that causes language-token and hallucination problems. |
| Truncated input (true length) | E-004 addendum: padded and truncated are equivalent at a matched schedule; truncated uses 4× less memory. |
| Fixed sinusoid positions, sliced | Length-agnostic, no checkpoint table, matches Whisper. |
| bf16 autocast + fp32 CTC | Correct precision split. |
| Mel cache float16 + mmap + duration buckets | Good base; 291.9 h in 16 GB; bounded-memory builder. |
| 1024 shared BPE | 1.945 tokens/word, 4.70 tokens/s; 50 Hz / 4.7 ≈ 10.6 frames per token, no CTC length pressure. |
| int8 dynamic ONNX, 8.23 MB | Size and speed bars are met with margin. |
| Bootstrap CIs, paired tests, one normalizer | Rigorous evaluation. |

### 3.2 Where the time goes, and where it is lost

The encoder is ~0.8 GFLOP per audio-second: 4 blocks at 50 Hz (≈ 0.71 GFLOP/s) plus the conv stem
(≈ 0.06 GFLOP/s); attention is small for short clips and grows quadratically for 20–30 s clips.

The H2 throughput gap has four plausible contributors, in order:

1. **Loader**: single-threaded `MelDataset.__getitem__`, `np.load` per clip, multiple numpy copies,
   no overlap with the MPS step.
2. **Attention**: manual `q @ k.T` + fp32 softmax + `weights @ v`, with fp32 score tensors; no fused
   SDPA kernel.
3. **CTC/head layout**: `logits` in `[B,T,V]`, then `.float()`, `log_softmax`, then
   `.transpose(0,1).contiguous()` — a ~98 MB copy per batch.
4. **Dev evaluation**: fp32 forward, per-utterance CPU argmax, 3,939 utterances every 2,000 steps.

All four are addressable without changing the model's function.

---

## 4. Code-level optimizations

### 4.1 Model and training

#### 4.1.1 SDPA instead of manual attention

`stt/model.py::_Block._attention` currently does:

```python
scores = q @ k.transpose(-1, -2)
if keep_mask is not None:
    scores = scores.masked_fill(~keep_mask, torch.finfo(scores.dtype).min)
weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
merged = (weights @ v).transpose(1, 2).reshape(batch, frames, width)
```

q and k are each scaled by `head_dim ** -0.25`, which is exactly SDPA's default
`scale = head_dim ** -0.5`. So the math is already SDPA-equivalent. Replace with:

```python
attn = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=keep_mask,          # bool [B, 1, 1, T], True = participate
    scale=head_dim ** -0.5,
    dropout_p=0.0,
)
merged = attn.transpose(1, 2).reshape(batch, frames, width)
```

Expected: lower attention memory (no fp32 score tensor), fewer kernels, and fused kernels on MPS
when the mask allows. Caveat: with a padding mask, MPS may fall back to the math kernel; even then,
removing the explicit fp32 upcast is a win. Add a parity test (math vs SDPA, fp32 and bf16; logits
and CTC loss). Export caveat: the legacy TorchScript ONNX exporter may not support SDPA; keep a
`math` path for export or force `SDPBackend.MATH` during export.

#### 4.1.2 Fused QKV projection

Today: three `Linear(384, 384)` (query, key, value; key has no bias). Replace with one
`Linear(384, 1152)` and zero bias for the key slice:

```python
self.qkv = nn.Linear(width, 3 * width)
# weight = cat([query.weight, key.weight, value.weight], dim=0)
# bias   = cat([query.bias, zeros_like(key_bias), value.bias], dim=0)
q, k, v = self.qkv(x).chunk(3, dim=-1)
```

Same parameters, one GEMM instead of three, fewer ONNX `MatMulInteger` nodes (12 → 4) and better
weight/cache behavior on WASM. Requires a one-time checkpoint conversion in
`stt/convert_whisper.py` (and for H1/H2 checkpoints). Expected gain: small on GPU, larger on
browser/CPU.

#### 4.1.3 Time-major CTC head

Current training path:

```python
logits = self.head(hidden)                       # [B, T, V]
log_probs = F.log_softmax(logits.float(), dim=-1)
loss = criterion(log_probs.transpose(0, 1).contiguous(), targets, ...)
```

The transpose copies `[T, B, V]` ≈ 98 MB per batch (32 × 750 × 1025 × 4). Transpose **before** the
head, where the width is 384:

```python
hidden_t = hidden.transpose(0, 1).contiguous()   # [T, B, 384] ≈ 37 MB
logits_t = self.head(hidden_t)                   # [T, B, V]
log_probs = F.log_softmax(logits_t, dim=-1, dtype=torch.float32)
loss = criterion(log_probs, targets, out_lengths, target_lengths)
```

Saves ~60 MB and one kernel per batch. Keep the existing `[B,T,V]` forward for export and add a
`time_major=True` path (or `forward_ctc`) for training. Also use
`F.log_softmax(..., dtype=torch.float32)` instead of `.float()` + `log_softmax`.

#### 4.1.4 EMA / checkpoint averaging

The H2 plan already says “average the last 3–5 checkpoints before final evaluation”. Do it:
average the float tensors of the last 3–5 checkpoints (or the top-K by dev) and evaluate the average.
It is free and reduces the noise of a single MPS checkpoint.

For future runs, add a `ModelEMA`:

```python
ema = copy.deepcopy(model).eval()
for p in ema.parameters():
    p.requires_grad_(False)
# after optimizer.step():
for e, p in zip(ema.parameters(), model.parameters()):
    e.mul_(decay).add_(p.detach(), alpha=1 - decay)   # decay ≈ 0.999
```

Evaluate both raw and EMA on dev; save the EMA in the checkpoint. Standard practice in ASR recipes
(icefall/Zipformer); expected 1–3 % relative WER, occasionally more on noisy CTC training.

#### 4.1.5 InterCTC (cheap, training-only)

Attach an auxiliary CTC loss at an intermediate block (e.g. after block 2 of 4):

```python
# training only; the auxiliary head is not shipped
aux_logits = aux_head(hidden_after_block_2)              # [B, T, V]
loss = (1 - lam) * ctc(final_log_probs, ...) + lam * ctc(
    F.log_softmax(aux_logits.float(), dim=-1).transpose(0, 1), ...
)
```

`lam ≈ 0.3`. No inference cost, small training cost (one extra `[B,T,V]` log-softmax + CTC loss).
Evidence: Lee & Watanabe, ICASSP 2021 (WSJ 9.9 % WER, AISHELL-1 5.2 % CER); Granite 5.0 TurboCTC
uses intermediate/final CTC weights 0.2/0.8. For a 4-layer encoder the gain may be smaller than in
deep encoders, but the experiment is cheap.

#### 4.1.6 CR-CTC (stronger, 2× compute)

Consistency-Regularized CTC (Yao et al., ICLR 2025): two independently augmented views of the same
mel, two CTC distributions, a consistency loss between them, with larger time masking than standard
SpecAugment. LibriSpeech Zipformer: **2.51 → 2.12 test-clean, 6.02 → 4.62 test-other** (16–23 %
relative). The recipe:

- apply tempo/warp **once** before splitting, so the two views have the same length;
- apply independent frequency/time masks to each view;
- `L = 0.5 * (L_ctc(a) + L_ctc(b)) + alpha * L_CR(a, b)`, with a stop-gradient on one branch;
- `alpha` to tune (≈ 0.1–0.5); time masking increased ~2.5×.

Cost: 2× forward/backward. Use it as a **short fine-tune from H2** (1–2 epochs), not a full 8-epoch
rerun. It also pairs naturally with the `dataset.epoch` fix, because it needs genuinely different
views.

#### 4.1.7 Capacity variants (only if H2 misses the gate)

| Variant | Params | Compute (est.) | int8 size (est.) | When |
|---|---|---|---|---|
| Current | 8.03 M | 0.8 GFLOP/s | 8.2 MB | baseline |
| Conv head (2 depthwise-separable layers, kernel 15) | +0.3–0.9 M | +5–10 % | 8.5–9 MB | first try; cheapest capacity/local context |
| +2 transformer blocks at 50 Hz | 11.6 M | +45 % | ≈12 MB | if more depth is needed |
| **25 Hz + 2 blocks** | 11.6 M | **≈ −25 %** vs current | ≈12 MB | more depth, less attention cost; best capacity/compute trade |
| Whisper-base | 20.35 M | +160 % | 20.5–23.5 MB | last resort; breaks the 20 MB ceiling |

The 25 Hz variant is safe for BPE: the subsampling literature shows word-piece CTC can sustain
6.25–25 Hz (characters fail below 25 Hz). At 4.7 tokens/s and 25 Hz there are ~5.3 frames/token,
still comfortable. It needs a pooling layer after the encoder, a new 25 Hz position table for the
extra blocks, and a short fine-tune. It is a better escalation than Whisper-base if H2 is close.

#### 4.1.8 Smaller knobs

- **Weight decay on LayerNorm/bias**: exclude them from AdamW weight decay (param groups). Small,
  standard, low risk.
- **Layer-wise LR decay** for H3 (0.8–0.9 from top to bottom) if full fine-tuning on ~1 h of gold
  data overfits.
- **Dropout 0.05–0.1** in attention/MLP for H3 only; H2 on 438 h does not need it.
- **Blank-bias init** for a from-scratch head (log blank prior); not needed for H2 (init from H1).
- **Gradient clipping**: 1.0 is active (grad norms 4–14). It is a valid choice; if H2 plateaus,
  test 5.0, but do not change mid-run.

### 4.2 Data pipeline

#### 4.2.1 The `dataset.epoch` bug

`stt/data/dataset.py::MelDataset.__getitem__` seeds the augmentation RNG with:

```python
rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + position)
```

`stt/train.py` updates `state.epoch` but **never** `dataset.epoch`. Therefore each clip receives the
same tempo factor and the same masks in every epoch. The docstring says the training loop sets it;
the code does not. Fix for H3/future runs:

```python
state.epoch = epoch
dataset.epoch = epoch          # <-- missing today
```

and on resume, before the loop: `dataset.epoch = state.epoch`. Add a test: the same position must
produce different augmented mels at epoch 0 and epoch 1. **Do not apply this to the running H2**:
epochs 5–8 would use a different augmentation distribution from epochs 1–4, and a crash/resume would
silently pick up the change.

#### 4.2.2 Cache open memmaps

`stt/data/build_mel.py::read_clip` calls `np.load(..., mmap_mode="r")` for every clip. With 32
clips/batch and ~2.5 batches/s that is ~80 opens/s, each re-reading the `.npy` header. Cache the
memmap per shard in `MelDataset`:

```python
self._shards: dict[str, np.memmap] = {}
array = self._shards.get(entry["shard"])
if array is None:
    array = np.load(self.shards_dir / entry["shard"], mmap_mode="r")
    self._shards[entry["shard"]] = array
```

The number of shards is small (≈ 11 for the training set), so file-handle pressure is not a concern.

#### 4.2.3 Remove redundant copies

`read_clip` converts float16 → float32; `collate` calls `np.ascontiguousarray(mel, dtype=np.float32)`
and then copies into the batch tensor. At least two full copies per clip, plus the copies inside
`tempo_stretch` / `mask_freq` / `mask_time`. Simplify:

- remove `np.ascontiguousarray` in `collate`; `torch.from_numpy` accepts a non-contiguous array and
  the assignment into `batch[position, :, :frames]` handles strides;
- apply augmentation to one copy (`out = mel.copy()` once, then masks in place);
- optionally read the float16 block and write it directly into the float32 batch
  `[B, 80, T_max]` with one convert+transpose.

Expected: lower CPU time and memory bandwidth in the loader; no accuracy change.

#### 4.2.4 Prefetch / overlap

The training loop currently blocks on data preparation while MPS is idle. Add a bounded prefetch of
1–2 batches (a `ThreadPoolExecutor(max_workers=1)` + `queue.Queue(maxsize=2)`, or
`DataLoader(num_workers=2)` with a collate function). Keep the per-item RNG and batch order
deterministic; only the *time* of preparation changes. This is the single most likely throughput win
for H2-like runs.

#### 4.2.5 Frame-aware batch budget

`plan_batches` caps total audio seconds and clips, but attention memory is O(T²) per clip. A batch of
6 × 30 s clips has ~4.7× the attention mass of 32 × 6 s clips at the same total seconds. Add a budget
on `sum(T_i²)` (or `max(T_i)`) in addition to `max_seconds`/`max_clips`: stable memory and larger
batches for short clips. This may allow 256–320 s batches without the current 3–5 GB swings.

#### 4.2.6 Augmentation details

- `fill = out.min()`: the original SpecAugment masks with 0 after normalization (the mean), not the
  clip minimum. The current fill is much more aggressive (near silence). Run an ablation:
  min-fill vs zero-fill.
- **Noise augmentation is absent.** For a browser/mic product this is the biggest robustness gap.
  Oído (2026) reports that adding MUSAN noise, music, babble and simulated RIR changed average WER
  over 14 noise/reverb conditions from 8.9 to 6.2 (with an LM). Implementation options: keep raw
  audio for a subset and mix on the fly, or build a noisy-mel cache for a fraction of the data and
  sample it with probability p. Licenses (MUSAN/RIR) must be verified before use.

### 4.3 Evaluation and decoding

#### 4.3.1 Eval: inference mode, autocast, device argmax

`_evaluate` currently runs fp32 (no autocast) and calls `greedy(log_probs[offset], ...)`, which does
`.detach().cpu().numpy()` for every utterance; the argmax then runs on CPU. Copying `[T, 1025]` per
utterance is several GB per dev evaluation. Better:

```python
with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16):
    log_probs, out_lengths = model(...)
ids = log_probs.argmax(dim=-1)                 # [B, T] on device
for offset, row in enumerate(rows):
    hyps[row["id"]] = greedy_ids(ids[offset, :int(out_lengths[offset])], pieces)
```

- `torch.inference_mode()` instead of `no_grad()`;
- bf16 for periodic dev (the model is trained bf16); fp32 for the final report if desired;
- argmax on device, transfer only the ids;
- consider a fixed stratified subset (e.g. 1,000 utterances) for periodic evals and the full set only
  at the end. The 95 % CI widens, but selection remains usable and the run saves tens of minutes.

#### 4.3.2 Greedy → beam + hotword + small LM

Greedy is the functional weak point of CTC. The plan already includes hotword beam search; add:

- **CTC prefix beam search** (beam 8) with a hotword trie and per-token bonus; decode-time only, no
  retraining. Targeted at ETA and rare English terms.
- **Small LM in shallow fusion**: Oído uses a 1.3 M-parameter GRU LM and gains **10–13 % relative**
  (3.70 → 3.33 clean, 8.23 → 7.15 other on LibriSpeech). A 1–1.5 M LM at int8 is ~1–1.5 MB; the total
  goes from 8.2 to ~9.5–10 MB, still under the 20 MB ceiling. A 1-layer GRU or a 2-layer small
  transformer is enough; a large LM is not.
- **Blank/VAD guard** for the noise set: average blank probability or an energy VAD can suppress
  music/noise output; tune on the noise set, not on FLEURS. *Done for silence (2026-10-24): the
  energy guard in `stt/no_speech.py` and `web/mel.js` removes all 240 silence words; MUSAN noise
  still needs augmentation with empty targets (E-011 addendum).*

### 4.4 Export and browser

Inspected `release/model.int8.onnx` (random weights, E-003):

| Item | Value |
|---|---|
| File size | 8,232,371 bytes (8.23 MB) |
| Weights | 8.02 MB int8 + 0.12 MB fp32 |
| Nodes | 505 |
| `MatMulInteger` | 25 (4 blocks × q/k/v/out/mlp1/mlp2 = 24 + head) |
| `DynamicQuantizeLinear` | 19 |
| `ConvInteger` | 2 (both convs quantized) |
| fp32 `MatMul` | 8 (the two attention matmuls per block: QKᵀ and score·V) |
| `LayerNormalization` | 9 |
| `Softmax` / `LogSoftmax` | 4 / 1 |
| `Constant` / `Cast` / `Unsqueeze` / `Transpose` | 103 / 51 / 49 / 21 |

#### 4.4.1 Fuse QKV

See §4.1.2. In ONNX this reduces 12 `MatMulInteger` to 4 and removes two dequant/rescale paths per
block. Likely the best inference-side win per unit of effort.

#### 4.4.2 Static QDQ vs dynamic int8

The graph has 19 `DynamicQuantizeLinear` nodes, and the 8 attention `MatMul` remain fp32. Options:

- **Dynamic (current)**: no calibration; activation scales computed at runtime; larger overhead;
  attention stays fp32.
- **Static QDQ**: calibrate on ~200 dev clips; often faster; can quantize more of the graph, including
  attention inputs; accuracy risk. Measure **WER on dev/mixed**, not only max-abs logits.
- **Keep `conv1` fp32**: +0.28 MB; Oído keeps the first conv fp32; test whether it closes an int8
  WER gap. The first conv sees the mel directly and is the most sensitive to input quantization.
- **Per-frame scales**: Oído shows that per-frame activation scales make int8 essentially lossless
  (0.02/0.12 WER); ORT's `DynamicQuantizeLinear` is per-tensor. This is a plausible reason for a
  larger gap on real weights; static calibration or a custom graph may help.

The export gate should be **WER-based**, not only logit-based: compare int8 vs fp32 greedy transcripts
on dev and the mixed set, then browser RTFx. The E-003 addendum already warns that x86 int8 kernels
differ from ARM64; measure on the target platform.

#### 4.4.3 Graph simplification and opset

103 `Constant`, 51 `Cast`, 49 `Unsqueeze` are exporter noise. `onnxsim.simplify` can reduce node count;
verify that dynamic axes and the `[1,80,T]` input contract survive. Keep opset 17 for ORT Web
compatibility unless a specific fusion (e.g. `Gelu` at opset 20, `Attention` at 23) is proven to work
in the target browser. Do not raise the opset just because it is newer.

#### 4.4.4 Sinusoids and JS front-end

- In-graph `Range/Sin/Cos` computes 576 k sin/cos for 30 s; ~2 % of the encoder cost. A precomputed
  fp16 table would add 1.1 MB; not worth it now.
- `web/mel.js` uses a direct 201 × 400 DFT (~8 M MAC/s, <3 % of the encoder cost on a laptop). A
  400-point exact FFT (mixed-radix or Bluestein) is a later optimization for low-end phones only.
- Threads: 4 threads 65–74×, 1 thread 33–38×. The 10× bar is comfortable; WebGPU is not needed for
  v0.1.

---

## 5. Research notes: what is applicable to Moth

The following is a curated subset; numbers are from the cited papers and are not directly comparable
across datasets/models. Preprints are marked.

| Work | What it shows | What to borrow | What not to borrow |
|---|---|---|---|
| **CR-CTC** — Yao et al., ICLR 2025, [arXiv 2410.05101](https://arxiv.org/abs/2410.05101) | Consistency between two augmented views; LibriSpeech 2.51→2.12 clean, 6.02→4.62 other (Zipformer) | The consistency loss + stronger time masking; short fine-tune from H2 | The 2× cost for a full rerun |
| **InterCTC** — Lee & Watanabe, ICASSP 2021, [arXiv 2102.03216](https://arxiv.org/abs/2102.03216) | Auxiliary CTC at an intermediate layer; no inference overhead | Aux head after block 2, λ≈0.3 | — |
| **Self-conditioned CTC** — Nozaki & Komatsu, 2021, [arXiv 2104.02724](https://arxiv.org/abs/2104.02724) | Feeding intermediate predictions back improves CTC | Only if InterCTC helps | Extra complexity if not needed |
| **OWSM-CTC** — Peng et al., ACL 2024, [arXiv 2402.12654](https://arxiv.org/abs/2402.12654) | Encoder-only CTC at 1 B, 3–4× faster than AED, 180 k h | Validation that encoder-only CTC is a viable direction | The 1 B scale |
| **Granite 5.0 TurboCTC** — IBM, 2026 preprint, [arXiv 2609.20104](https://arxiv.org/abs/2609.20104) | 470 M Conformer-CTC; pyramidal subsampling, block-diagonal attention, intermediate CTC | Intermediate CTC (0.2/0.8) and 25/12.5 Hz subsampling ideas | The 470 M size and block-diagonal attention |
| **Oído** — 2026 preprint, [arXiv 2610.13447](https://arxiv.org/abs/2610.13447) | 13 M Conformer-CTC, int8 on an ESP32-S3: 3.70/8.23 LibriSpeech; int8 PTQ 0.02/0.12 WER loss with per-frame scales; 1.3 M GRU LM → 10–13 % relative; noise aug 8.9→6.2 | Small LM, noise augmentation, first conv fp32, per-frame scales | Replacing Whisper with a from-scratch Conformer |
| **Moonshine** — 2025, [arXiv 2509.02523](https://arxiv.org/abs/2509.02523) | 27 M monolingual model, compute proportional to input length; pseudo-label recipe | Data/pseudo-label strategy for v0.2 | The encoder-decoder instead of CTC |
| **Distil-Whisper** — 2023, [arXiv 2311.00430](https://arxiv.org/abs/2311.00430) | KD + pseudo-labels: 5.8× faster, 51 % fewer params, within 1 % WER | Distillation from a larger teacher if H2 misses the gate | Doing it before v0.1 is closed |
| **uDistil-Whisper** — NAACL 2025, [arXiv 2407.01257](https://arxiv.org/abs/2407.01257) | Label-free filtering for KD; 25–50 % compute/memory reduction | Filtering strategy for unlabeled Italian audio | — |
| **Whisper fine-tuning strategies (Pashto)** — 2026 preprint, [arXiv 2604.06507](https://arxiv.org/abs/2604.06507) | Whisper-base CV20: full FT 21.2 % vs frozen encoder 36.0 % vs LoRA 54.6 % | Full fine-tuning for a shallow encoder | Freezing/LoRA as a quality shortcut |
| **Subsampling** — 2022, [arXiv 2210.07189](https://arxiv.org/abs/2210.07189) | BPE CTC can sustain 6.25–25 Hz; characters fail below 25 Hz | 25 Hz pooling if capacity/speed needs it | 12.5 Hz with a 1024 BPE |
| **German tokenization** — 2026 preprint, [arXiv 2610.11646](https://arxiv.org/abs/2610.11646) | BPE-512/1024/2048: small, inconsistent WER differences | Treat BPE size as a low-priority ablation | Expecting large gains from vocabulary size alone |
| **LAL / CAMEL** — [arXiv 2403.05887](https://arxiv.org/abs/2403.05887), [arXiv 2412.12760](https://arxiv.org/abs/2412.12760) | Frame-level LID and language-wise CTC help code-switching | An auxiliary LID loss for H3 if the mixed set is the bottleneck | MoE/large architectures |
| **Conformer / FastConformer** — [arXiv 2005.08100](https://arxiv.org/abs/2005.08100), [arXiv 2305.05084](https://arxiv.org/abs/2305.05084) | Conv + attention is parameter-efficient; 8× downsampling is 2.8× faster | A zero-init depthwise conv adapter per block, or 25 Hz pooling | Replacing the pretrained encoder from scratch |

### 5.1 What this means for Moth

- **CTC is not a dead end.** OWSM-CTC, Granite, and Oído all show encoder-only CTC is competitive
  and much faster than AED/transducer alternatives.
- **The two training upgrades worth testing are InterCTC (cheap) and CR-CTC (strong, 2×).**
- **The best decode upgrade is beam + hotwords + a small LM**, not an autoregressive decoder.
- **The best robustness upgrade is noise augmentation.**
- **The best “if the gate fails” upgrade is distillation/pseudo-labeling**, not Whisper-base first.
- **BPE size and 12.5 Hz are secondary levers.**

---

## 6. Prioritized roadmap

### P0 — safe to implement now (worktree, CPU tests only)

| Item | Where | Effort | Effect | Risk |
|---|---|---|---|---|
| Fix `dataset.epoch` + test | `stt/train.py`, `stt/data/dataset.py` | 1 h | Augmentation diversity across epochs | Must not be applied to the running H2 or its main checkout |
| Cache memmaps | `stt/data/dataset.py`, `read_clip` | 1 h | Less open/header overhead | Low |
| Remove redundant copies | `stt/data/dataset.py::collate`, `augment` | 2–3 h | Less CPU/memory bandwidth | Low |
| Prefetch 1–2 batches | `stt/train.py` | 2–4 h | 10–25 % throughput | Medium: keep RNG/order deterministic |
| Eval: `inference_mode`, autocast, device argmax, optional subset | `stt/train.py`, `stt/decode.py` | 2–4 h | Faster dev evals; tens of minutes per run | Low; final eval full/fp32 |

### P1 — after H2, measure on real weights

| Item | Where | Effort | Effect | Risk |
|---|---|---|---|---|
| SDPA + parity test | `stt/model.py` | 3–4 h | 10–30 % step time/memory on long clips | Export path must keep a math fallback |
| Time-major head + `log_softmax(dtype)` | `stt/model.py`, `stt/train.py` | 2–3 h | ~60–100 MB less per batch | Low; add a parity test |
| Fused QKV + checkpoint conversion | `stt/model.py`, `stt/convert_whisper.py` | 3–4 h | 3–10 % inference; fewer ONNX nodes | Checkpoint conversion |
| EMA / checkpoint averaging | `stt/train.py` | 2 h | 1–3 % relative WER | Low |
| Hotword beam | `stt/decode.py`, `web/decode.js` | 1–2 days | ETA/rare words | Decode-only; test against Python reference |
| Static QDQ + `conv1` fp32 vs dynamic | `stt/export.py` | 1 day | Faster/possibly more accurate int8 | Calibration and accuracy gate |
| Real-weight export + browser RTFx/WER | `stt/export.py`, `web/` | 1 day | Closes the int8 gate | Must be on the target platform |

### P2 — H3 / next training run

| Item | Where | Effort | Effect | Risk |
|---|---|---|---|---|
| InterCTC | `stt/model.py`, `stt/train.py` | 4–6 h | 1–3 % relative (est.) | Low; training-only |
| CR-CTC short fine-tune | `stt/train.py`, `stt/data/dataset.py` | 1–2 days | 5–15 % relative (est.) | 2× compute; needs two views |
| Frame-aware batch budget | `stt/data/dataset.py::plan_batches` | 2–3 h | Stable memory/larger batches | Low |
| Weight-decay param groups, layer-wise LR, dropout for H3 | `stt/train.py`, `stt/model.py` | 2–4 h | Small/regularization | Low |
| Noise augmentation | data pipeline | 2–5 days | Large under noise | Data/licenses/storage |
| Small LM (GRU 1–1.5 M) + shallow fusion | new model + decode | 3–5 days | 5–15 % relative | Size/latency/browser |

### P3 — v0.2 / only if the gate fails

| Item | Why |
|---|---|
| Conv head / +2 blocks / 25 Hz+2 blocks | Capacity escalation cheaper than Whisper-base |
| Whisper-base | Accuracy fallback; relax the 20 MB ceiling to ≈25 MB |
| Distillation from Parakeet/Whisper-small | Biggest quality lever; 438 h + unlabeled Italian; GPU cost |
| BPE 2048 | Only if ETA is the blocker; low expected WER effect |
| JS 400-point FFT | Only for low-end phones; mel is <3 % on a laptop |
| WebGPU, 4-bit QAT, streaming, punctuation | v0.2+, not needed for the 10× bar |

---

## 7. Benchmark protocol (so the numbers mean something)

Every change gets one variable at a time and an `EXPERIMENTS.md` entry.

### Throughput

- Same config, same data order, 200 steps after a 20-step warm-up; report median audio-s/s and peak
  memory; no dev eval inside the window.
- Compare against H2's 450–530 audio-s/s and H1's ≈590 (320 s/64, frozen).
- Run on power with `caffeinate -i`; note thermal state.

### Quality

- Same init, same data, same schedule; one change at a time.
- Selection: dev macro (IT+EN); final: FLEURS IT/EN, CV IT dev/test, mixed T (when recorded), noise.
- Paired bootstrap over utterances; call a tie when the CI includes 0.
- Do not tune on T; T is scored once per candidate.

### Export

- WER int8 vs fp32 on dev and mixed, not only max-abs logits.
- Browser RTFx on 5/10/20 s clips, 1 and 4 threads; transcript parity with Python on the 20-clip
  e2e set.
- Measure on the target platform: x86 int8 kernels differ from ARM64 (E-003 addendum).

---

## 8. Can this be done while H2 is training?

Short answer: **yes for implementation and CPU-only tests in this worktree; no for real MPS training
runs, and no for merging into the main checkout until H2 finishes.**

Why editing this worktree is safe:

- H2 runs from the main checkout and has already imported its Python modules; editing files in a
  different worktree cannot affect the running process.
- This worktree is a separate directory/branch. It is currently behind main: it lacks the H2 commits
  (`empty_cache_every`, `dev_en`, `_evaluate_all`, the optimizer-state fix). P0 patches should
  therefore target **main's version** (a fresh worktree from main, or update this one after H2), not
  this older `train.py`, otherwise the patch will conflict.

| Activity | Now? | Where | Notes |
|---|---|---|---|
| Write this review | done | worktree | No effect on H2 |
| Implement P0 + unit tests | yes | worktree from main | CPU only; do not run on real data/MPS |
| Merge/commit P0 into main | **no** | — | AGENTS.md: no commit without request; a crash/resume would pick up `dataset.epoch` and change epochs 5–8 |
| Implement InterCTC/CR-CTC + configs/tests | yes | worktree | Training-only code; CPU smoke with random tensors |
| Run InterCTC/CR-CTC on real data | **no** | — | Needs MPS; would contend with H2 (3–5 GB, 16 GB total) and slow/risk the run |
| Implement static QDQ + calibration reader | yes | worktree | CPU; use random/H1 weights for smoke |
| Run export on H2 weights / browser WER+RTFx | **no** | — | Wait for H2; the real numbers are the gate |
| Full pytest | yes, targeted | worktree | Avoid heavy data/MPS tests; run CPU unit tests |

The main risk is not code editing; it is **resuming H2 from a modified main checkout**. If H2 crashes
and restarts after `dataset.epoch` is fixed in the main tree, epochs 5–8 use a different augmentation
distribution from epochs 1–4. Keep all changes in the worktree until H2 is finished.

---

## 9. What not to do

- Do not change H2 mid-run; do not merge P0 into the main checkout before it finishes.
- Do not freeze the encoder or use LoRA on a 4-layer tiny encoder; the evidence says full FT wins.
- Do not jump to Whisper-base before trying conv head / +2 blocks / 25 Hz+2 blocks.
- Do not use 12.5 Hz with a 1024 BPE.
- Do not over-invest in BPE size without an ETA signal.
- Do not use fp16 on MPS (known NaNs); bf16.
- Do not use `torch.compile` blindly on MPS with dynamic shapes and CTC; measure first.
- Do not add a large LM; 1–1.5 M parameters is enough.
- Do not forget licenses for MUSAN/RIR, teachers, and CS-FLEURS (CC-BY-NC, eval-only).

---

## 10. Open questions / decisions needed

1. **After H2:** is checkpoint averaging enough, or do we add EMA for H3?
2. **H3 priority:** InterCTC (cheap) first or CR-CTC (strong, 2×) first?
3. **Noise set:** is it a ship gate? If yes, noise augmentation moves up to P1.
4. **LM:** do we accept +1–1.5 MB and a second browser session for 10–13 % relative WER?
5. **Capacity:** if H2 misses the gate, conv head or 25 Hz+2 blocks first?
6. **Worktree:** create a fresh worktree from main for P0/P1, or update this one after H2?

---

## 11. References

- Yao et al. **CR-CTC: Consistency regularization on CTC for improved speech recognition.** ICLR 2025.
  <https://arxiv.org/abs/2410.05101>
- Lee & Watanabe. **Intermediate Loss Regularization for CTC-based Speech Recognition.** ICASSP 2021.
  <https://arxiv.org/abs/2102.03216>
- Nozaki & Komatsu. **Relaxing the Conditional Independence Assumption of CTC-based ASR by
  Conditioning on Intermediate Predictions.** 2021. <https://arxiv.org/abs/2104.02724>
- Peng et al. **OWSM-CTC: An Open Encoder-Only Speech Foundation Model for Speech Recognition,
  Translation, and Language Identification.** ACL 2024. <https://arxiv.org/abs/2402.12654>
- IBM. **Design of the IBM Granite 5.0 TurboCTC ASR Model.** 2026 preprint.
  <https://arxiv.org/abs/2609.20104>
- Varela Sánchez & Martín-Coello Juárez. **Oído: Open-Vocabulary Speech Recognition on a $5
  Microcontroller without a Neural Accelerator.** 2026 preprint. <https://arxiv.org/abs/2610.13447>
- **Moonshine: Speech Recognition for Live Transcription and Voice Commands.** 2025.
  <https://arxiv.org/abs/2509.02523>
- Gandhi et al. **Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling.**
  2023. <https://arxiv.org/abs/2311.00430>
- Waheed et al. **uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation in Low-Data
  Regimes.** NAACL 2025. <https://arxiv.org/abs/2407.01257>
- Rahman. **Fine-tuning Whisper for Pashto ASR: strategies and scale.** 2026 preprint.
  <https://arxiv.org/abs/2604.06507>
- **On Compressing Sequences for Self-Supervised Speech Models.** 2022.
  <https://arxiv.org/abs/2210.07189>
- **Phonologically Informed Tokenization for German Speech Recognition: A Cross-Domain Study.**
  2026 preprint. <https://arxiv.org/abs/2610.11646>
- Liu et al. **Aligning Speech to Languages to Enhance Code-switching Speech Recognition.** TASLP
  2025. <https://arxiv.org/abs/2403.05887>
- **CAMEL: Cross-Attention Enhanced Mixture-of-Experts and Language Bias for Code-Switching Speech
  Recognition.** 2025. <https://arxiv.org/abs/2412.12760>
- Gulati et al. **Conformer: Convolution-augmented Transformer for Speech Recognition.** 2020.
  <https://arxiv.org/abs/2005.08100>
- Rekesh et al. **Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition.**
  2023. <https://arxiv.org/abs/2305.05084>
