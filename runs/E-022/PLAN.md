# E-022 — decode recalibration: a per-token bonus and an LM weight that was doing two jobs

> **Written 2026-11-09, prediction first, code second.** This file is committed *before*
> `stt/decode.py` changes, on purpose: the audit that motivates it (`runs/E-022/AUDIT-PROBE.md`)
> already measured the outcome, so this is **not a blind pre-registration**. What is fixed here is
> the *decision rule* and the *arm*, not a guess. The honest framing is "replication protocol": the
> arm was chosen on dev, frozen, and is now re-run through the repository's official harness so the
> numbers live in `EXPERIMENTS.md` like every other result.

## Hypothesis

`stt/decode.py::beam_search` adds `lm_weight * lm.score(...)` to every emitted token and **nothing
subtracts it back**. An LM log-probability is negative, so each extra token costs score: the decoder
is biased toward emitting **fewer** tokens. Evidence that this is real and not a story:

| Measurement (T-600, shipped decode, E-017 checkpoint) | Value |
|---|---|
| Deletions, % of reference words | **12.2** |
| Insertions, % of reference words | **0.8** |
| Clips whose **first** word is deleted | **13.3%** |
| Clips whose **last** word is deleted | **0.5%** |
| Clips losing a word whose hypothesis is *shorter* than the reference | 0.886 hyp/ref words |

Two decode-only knobs correct it, and neither touches a weight:

- **`gamma`** — a constant added once per emitted token (the standard CTC length/insertion fix);
- **`lm_weight`** — re-tuned upward, which only became visible once `gamma` stopped the LM from also
  doing length control.

The deeper diagnosis is the sweep design, not the model: E-008 tuned `alpha` in one dimension while
`alpha` controlled **both** language modelling and output length, so its 1-D optimum was the optimum
of neither.

## Prediction (recorded before the code change)

Frozen arm: **beam 12 / top-k 10 / `lm_weight` 0.5 / `gamma` 2.4 / `beta` 0**, chosen on dev only
(`mls_it_dev` 200 + `ls_en_dev` 200 — never T, never S3).

| Set | shipped (beam 5 / lm 0.2 / γ 0) | predicted | predicted Δ WER |
|---|---|---|---|
| T-600 | 32.77 / ETA 42.86 | **25.9 ± 1.0** / ETA **52.6 ± 1.5** | −6.8 [−8.0, −5.7] |
| T-S3 (held-out speaker) | 35.43 / 35.21 | **26.7 ± 1.5** / ETA **47.3 ± 2.0** | −8.8 [−11.1, −6.5] |
| FLEURS-IT | 26.52 | **20.5 ± 0.6** | −6.1 |
| FLEURS-EN | 55.48 | **49.6 ± 0.8** | −5.9 |
| CV-IT dev | 24.91 | **19.9 ± 0.8** | −5.0 |
| Noise, non-music words (VAD on) | 0 | **0** | — |

**Decision rule, fixed now:** adopt if (a) every set's WER CI excludes 0 in the improving direction,
(b) T-S3 moves at least as much as T-600 (i.e. the gain is not speaker adaptation), (c) the
non-music noise count stays 0, and (d) the browser 1-thread RTFx on 20 s audio stays **≥ 10×**
(`docs/05` §5). If (d) fails, fall back to the pre-registered cheaper arm
**beam 8 / top-k 8 / `lm_weight` 0.5 / `gamma` 2.4** (dev 26.90; T-600 predicted −6.4), which is
already measured and keeps most of the gain.

**Explicitly not predicted**: the ship gate passing. The audit's peeked arm
(`lm_weight` 0.6 / `gamma` 3.0) does win the T-S3 ETA half, but that cell was chosen after looking
at S3, which violates the "never tune against S3" rule. It is reported in the audit as a
self-correction and is **not** the arm adopted here. Winning the ETA half honestly needs the
dev-only arm plus a better acoustic model, not a nudge from the test speaker.

## Method

1. Implement `gamma` (and `beta`, defaulted to 0) in `stt/decode.py::beam_search`, keeping every
   existing call site valid.
2. Mirror it in `web/decode.js::beamSearch` (the two decoders must stay twins, as VAD already is).
3. Add `tests/test_decode.py` cases: `gamma = 0` reproduces the old output bit-for-bit; a positive
   `gamma` can only reorder beams toward longer hypotheses; `beta` normalises monotonically.
4. Run the official evaluation with the frozen arm: `runs/E-022/eval.sh`, covering greedy,
   beam+LM on T-600 / T-S3 / FLEURS IT/EN / CV-IT dev, the noise set with the VAD guard, and the
   int8 artifact through `stt/onnx_runner.py`.
5. Browser: Playwright RTFx at 1 and 4 threads on 5/10/20 s clips, then rebuild `release/`.

## Data / hardware / compute

Decode-only: **no training, no new data, no checkpoint change**. E-017 `ckpt_best.pt` (step 26,442)
stays byte-identical; the int8 artifact must stay `fe1afcc2…`. Audio is the existing mel cache
(`data/shards/`, 44 GB) and the shipped ONNX graph. Hardware: the M5 16 GB, CPU decode.

Estimated cost: **~1 h 45 of machine time**, ~3 h of author time including code and the browser run.
**0 USD.**

## What would make this a failure

- Any set whose CI includes 0 (the change would be noise, and the audit over-read it).
- T-S3 moving **less** than T-600 (would mean the gain is speaker/channel adaptation).
- Browser 1-thread RTFx under 10× on 20 s clips (then the cheap arm ships instead).
- Non-music noise words above 0 (the bonus rewards emitting tokens; the VAD must remain load-bearing).

## Artifacts (to create)

```text
runs/E-022/
  AUDIT-PROBE.md      # the seventeen probes and every number behind this entry
  eval.sh             # the official evaluation, frozen arm
  eval/               # metrics + hyp per set, greedy and beam+LM
  diffs/              # paired vs E-017 (shipped decode) on every set
  browser.json        # Playwright RTFx, 1 and 4 threads
  RESULT.md           # measured vs predicted, and the decision
```

## Out of scope

- Retuning anything that needs training (`gamma` is a decode scalar, not a learned parameter).
- The peeked `0.6 / 3.0` cell (it may be validated later; it cannot be adopted from this run).
- Any change to the acoustic model, the LM, the tokenizer or the data.