# 11 — Evidence, decisions and negative results

> **Why this file exists.** The design docs (`01`–`10`) record what we *intended*; `EXPERIMENTS.md`
> records what we *ran*. This file records what we **decided, rejected and corrected**, with the
> artifact and the command behind every claim. It is written so that a paper about the failures and
> the choices can be assembled from it without re-deriving anything. Entry numbers refer to
> `EXPERIMENTS.md`; `E-0NN` artifacts live in `runs/E-0NN/`.

> **Audit: 2026-11-10, main `cd1ce1b`.** This is a historical decision log, not a certificate
> that the interpretations in every experiment report/model card are established. See
> [13 - Current Model and Paper Readiness](13-current-model-and-paper-readiness.md) for the exact
> model, benchmark matrix, provenance conflicts and missing evidence. Release 0.7.0 assets are
> **not bundled in this checkout**; current main runtime code is not necessarily published code.

## 1. How any number is verified

The canonical text rule is [`stt.text.normalize`](../stt/text.py). The
[scorer](../stt/eval.py) normalizes hypotheses but reads references from the existing manifest
`norm` field without re-normalizing it. WER is `sum(S+D+I)/sum(reference words)`; ETA is the share
of annotated English **word positions** in `en_spans` aligned as exact matches, not distinct-term
accuracy. Standard CIs use 1,000 utterance resamples with replacement, seed 0, recomputing corpus
ratios; paired differences use identical sampled indices for both systems. Sorted zero-based
indices 25 and 975 provide the nominal 95% interval. Standalone `score_rows` supplies a WER CI,
not an ETA CI; `_compare` supplies difference CIs for WER and, where defined, ETA.

JSON WER/ETA points and CI endpoints are **fractions**. Percentage tables multiply **both points
and endpoints by 100**; differences are percentage points (pp), not relative percent.
`insertion_rate` is already per 100 reference words. A CI including zero means no difference was
detected by this procedure, **not equivalence or non-inferiority**.

```bash
# score a hypothesis file (writes metrics.json with per-utterance counts)
uv run python -m stt.eval --manifest <manifest.jsonl> --hyp <hyp.jsonl> --out <metrics.json>
# paired difference against another metrics.json (same manifest): WER and ETA CIs
uv run python -m stt.eval --manifest <manifest.jsonl> --hyp <hyp.jsonl> \
    --vs <other.metrics.json> --out <diff.json>
```

Provenance is not uniform: the CLI adds manifest/hypothesis/normalizer hashes, git SHA and an
optional config **path**, not embedded checkpoint/decode settings. Direct `score_rows` callers,
including E-022/E-024/E-025, can omit that envelope; some clean-gate artifacts contain counts
without hypothesis files. Inspect the actual artifact and companion script/config. Re-scoring
saved hypotheses is distinct from re-decoding or retraining with missing audio/weights/shards.

## 2. Decision log (execution phase)

The design decisions D1–D13 are in `docs/README.md`. These are the decisions taken while running,
each with the evidence that forced it.

| # | Date | Decision | Evidence | Alternative rejected |
|---|---|---|---|---|
| D14 | 2026-10-21 | H2 trains on **truncated** input, not padded 30 s | E-004 matched-schedule final values 0.7375 vs interpolated ≈0.7406 differ by ≈0.0031 (0.31 pp), **not <0.001**; no large padding advantage established. Earlier attribution was schedule-confounded | `padded30`: higher measured memory/compute |
| D15 | 2026-10-21 | Keep **Common Voice IT 27** in the mix (437.7 h: 337.7 IT / 100 EN) | E-002 addendum: 167.8 h after filters, speaker-disjoint splits, 593 train/test sentence overlaps removed | Dropping CV (≈170 h of Italian lost) |
| D16 | 2026-10-22 | Keep **+2 head blocks** (11.58 M, ~12 MB int8) | E-006: FLEURS-IT 37.41 → 34.11, CV dev 37.60 → 34.45, T 65.35 → 59.50; all paired CIs exclude 0 | Staying at 4 blocks (E-005 plateaued) |
| D17 | 2026-10-22 | Historical beam quality mode; exact-BPE hotwords **not shipped** | E-007: beam −0.34…−0.62 WER pp; T ETA 20.0 → 20.9 → 21.8% for greedy/beam/hotwords. The old claim that increasing accuracy is worse had its direction reversed; this still does not establish a useful hotword gain | Hotword bonus as a product feature |
| D18 | 2026-10-22 | Adopt the **BPE trigram LM (α = 0.2)** as the quality decoder | E-008: −5.08 FLEURS-IT, −6.05 CV dev, −4.13 T (all significant) | Acoustic-only greedy (kept as fast default) |
| D19 | 2026-10-23 | **Correct the H2 bar** to the best H0 variant (Whisper-tiny auto+beam5, 27.42), retracting the E-008 "gate passed" claim | E-001 addendum: 32.09 was the greedy variant; Moth 29.02 is +1.60 [+0.32, +2.83] worse | Keeping the greedy bar (would have been self-serving) |
| D20 | 2026-10-23 | Ship the **17.08 MB** package (11.91 MB int8 + 5.09 MB LM) with browser beam+LM | E-009/E-010: 58–65× RTFx on 4 threads, 25–30× on 1, T200 parity vs Python is a tie | Greedy-only 12 MB package (worse WER, same size class) |
| D21 | 2026-10-23 | **Reject A** (300 h English) | E-012: FLEURS-EN −5.4 but T +5.16 worse [3.00, 7.37] and **ETA unchanged** (21.49) | Adopting it for the English gain |
| D22 | 2026-10-23 | **Reject B** (BPE V=4096, 12.76 M) | E-013: terms ≤3 pieces 48.8% → 74.0%, but ETA 20.0 → 20.0, FLEURS-IT +1.9 worse | Adopting it for the shorter sequences |
| D23 | 2026-10-24 | Add a **no-speech energy guard**, not a VAD; greedy stays the default on unknown input | E-011 addendum: silence words 240 → 0 (120/120 clips), 0/897 speech clips touched | Silero VAD (~2 MB, out of v0.1 scope) |
| D25 | 2026-10-30 | Adopt **Silero VAD** (MIT, 2.33 MB) as the no-speech guard, replacing the energy guard; package goes to ~19.4 MB | Separability on the full sets: speech fraction min 0.194 (S3; S1/S2 0.408/0.410, FLEURS-IT 0.419) vs noise max 0.038, silence 0.000; threshold 0.10 gates 100% of non-music and 0% of speech, and the guard takes non-music words **35/31 → 0** (greedy/beam) with T-400 and FLEURS-IT hypotheses bit-identical (`runs/E-011/vad_guard_summary.json`, `runs/E-017/vad_speakers.json`) | Noise training with empty targets (would train on the evaluation set without a MUSAN re-download and split), energy/flatness VAD (measured, no separation) |
| D26 | 2026-11-07 | **Keep gold instances that fail validation out of the manifest instead of renumbering** (R-S2: 622 of 623) | The builder maps cue→file by position; renaming would shift every later cue. The excluded take is 0.39 s of silence, so nothing is lost; the count is reported | Dropping the whole set, or shifting the cue indices |
| D27 | 2026-11-07 | **Adopt E-017 historically**, superseded by E-024 in 0.7.0; ETA gate unresolved | E-017 vs E-016: T-600 WER −5.73 [−7.14, −4.26], ETA +6.81 [+4.18, +9.55]; S3 WER −6.13 [−8.68, −3.51], ETA +4.79 [+0.51, +9.06]. Transfer beyond trained voices is observed, not equality of effects or removal of speaker/channel confounds (§3.3) | Keeping E-016; waiting for more gold |
| D28 | 2026-11-07 | **Stop exploratory Whistle LoRA at L1**, not a Moth release | S3 +2.30 WER [−0.68, +5.46] (no detected gain); FLEURS-IT/EN and first 1,000 CV-test clips regress. This rejects adoption of this recipe, **not** architecture-independent transfer in general. H3/H4 unrun (§3.4) | Continuing past the kill; presenting LoRA as Moth. [Protocol](plans/2026-11-07-whistle-lora-probe.md), [result](../runs/E-019/RESULT.md) |
| D24 | 2026-10-24 | **Reject H3b splicing as an ETA lever**; keep E-014 as the best acoustic checkpoint | E-014: FLEURS-IT −2.27, FLEURS-EN −0.95 (both significant), CV/T/noise ties, T ETA −0.90 (tie) | More splicing, word-level splicing (too early) |
| D38 | 2026-11-11 | **E-029 QAT is negative; do not fund the full arm, keep 0.7.0** | Custom MPS STE (per-channel int8 weights + uint8 activations) is mechanically viable (0 non-finite, 1.1–1.6× step time) and closes the dev INT8-vs-fp32 gap from 0.21 to 0.05, but the best probe (500 steps, frozen encoder) is 25.34 / 53.21 on T-600, 10/47 rare, 428/817 natural; the T-600, rare and natural gates fail, and its own fp32 control (24.97 / 54.07, natural 434) is already below E-028 fp32 | Funding a longer QAT arm expecting the dev gap closure to transfer; publishing E-028 INT8 or E-029 as 0.8.0 without the product gates |
| D37 | 2026-11-11 | **Fund E-029 QAT** as the next experiment; do not ship 0.8.0 before it | E-028 fp32 passes, but INT8 T-600 is 25.49 / 53.50 vs 0.7.0 INT8 25.14 / 54.65 (ties), natural 441 → 428; post-training variants are exhausted (default best in budget, head-fp32 over 20 MB, static QDQ broken). PyTorch fake-quant is unsupported on MPS; E-029 needs a custom straight-through quantizer | Assuming QAT will recover; publishing the fp32 (53.9 MB) or the INT8 tradeoff without a product decision |
| D36 | 2026-11-11 | **Do not publish E-028 INT8 as 0.8.0**; keep 0.7.0 | E-028 fp32 is a real candidate (rare presence 17.0 → 27.7%, T-600 24.66 / 54.65, T-S3 25.55 / 49.01), but the shipped INT8 artifact gives back 13 natural terms and 1.15 T-600 ETA vs 0.7.0 INT8; dev is 31.01 vs 30.81 and the 16-term gold check 20/27 → 18/27 | Publishing a point regression because the fp32 gate passed; calling the fp32 artifact deployable (53.9 MB) |
| D35 | 2026-11-09 | **Do not adopt E-015 first gold addendum**, keep 0.7.0 | 50 S1 sentences x20, one epoch: presence 8/47 → 11/47 (+6.4 [−4.3, +17.0] pp), S3 2/18 unchanged; natural hits 439 → 423; FLEURS-IT +0.72 / EN +1.31 pp regress. E-026 correction: aligned ETA on the same 47 is 19/81 → 25/81 (23.5% → 30.9%), but full T-600 ETA is 565/1043 → 557/1043 (54.17% → 53.40%). Dev best is near-start step 20; no useful new checkpoint selected | Final checkpoint; repeating this recipe. Follow-up S2 plus gentler LR changes two variables and cannot isolate either cause |
| D34 | 2026-11-09 | **Keep ETA as a gate** and fund **E-015 as a gold addendum**, not Parakeet | 0.7.0 term-presence diagnostic: natural 53.7%, gold-only 68.7%, rare 8/47 = **17.0%** across 44 missing term strings. Historical `TERMS18` group actually contains **16 strings**, 20/27 = **74.1%**. These presence readouts are not official aligned ETA. Direct ETA ceiling **+2.8-3.7 pp** is an estimate, not an observed gain; historical target revised to >=50% rare, ETA +2-4, WER not worse | WER-first; blocked Parakeet distillation; retaining unreachable +5 target |
| D33 | 2026-11-09 | **Add one unseen-noise-recording control**, no weights change | E-025 uses 38 s of new ambience on the same first 100 S3 clips: gains −3.90/−6.79/−11.21/−18.34/−14.60 pp at 30/20/15/10/5 dB; all WER CIs exclude zero, all ETA CIs include zero. Mean −10.06 pp over 30/20/15/10 dB; absolute WER 36.50 at 15 dB. Shared-pool concern narrowed, general robustness not certified | A post-hoc 80/20 split cannot make an already-trained checkpoint source-disjoint; retraining with a source-level split is needed |
| D32 | 2026-11-09 | **Noise augmentation on half of CV-IT**, release **0.7.0** | E-024: fixed per-clip hash-selected SNR 10–30 dB, same ≈589.5 h mixture; gains −6.28 at 15 dB, −11.54 at 10 dB, −9.68 at 5 dB. Clean T-600 −0.72, S3 +0.26 (ties), CV-test −0.52 pp (CI excludes zero). This is not a confidence-bound non-inferiority proof or an isolated decoder/acoustic diagnosis | Noising all CV; adding a full noisy copy (changes source share); noising MLS/LS (raw audio absent) |
| D31 | 2026-11-09 | **Keep the LM as shipped** (weight 0.5 from E-022, backoff 0.4, count pruning ≥ 2, 5.12 MB browser binary) after a full re-tune; **do not** buy the bigger LM | E-023: on full dev (3,939 clips, 10× E-022's selection set) the optimum is exactly E-022's cell, and 0.4 stays the best backoff. The unpruned LM (pruning ≥ 1) is **0.65 dev points better** and wins the gate sets (T-600 −0.77 [−1.40, −0.11], FLEURS-IT −0.72, CV-IT dev −0.65) — but it is a **12.31 MB** browser binary, i.e. a **26.63 MB** package against the 20 MB ceiling (D4), and it is a **tie on the held-out speaker** (+0.43 [−0.61, +1.62]). The ceiling costs general Italian WER, not the code-switching claim the project exists to make | Relaxing the 20 MB ceiling for ~0.7 WER (buys an Italian model, not a code-switching one; D4 is a product constraint, not a quality knob); pruning harder to save 1.8 MB at +0.56 dev points (the LM is 5.12 MB of 19.44 and is paying its way) |
| D29 | 2026-11-07 | **Make the fp32 export gate a seeded sweep** (median ≤ 1e-3, worst ≤ 1e-2 over 20 draws per length) instead of one random draw, and rebuild the release **with verification on** (0.5.1) | E-021: the gate fixture's typical gap is 1.8e-4, but 1 draw in 80 crosses 1e-3, always at 29 s; the value depends on (checkpoint, seed, length) — E-017 gives 2.43e-4 at seed 0 and 1.28e-3 at seed 2. Real speech measures 2.2–3.1e-4 with top-1 1.0, so the export was never wrong. The rebuilt 0.5.1 passes the new gate (median 2.127e-4, worst 1.472e-3, 3/80 over) and produces a **byte-identical** int8 artifact (`fe1afcc2…`) | Keeping the single draw and `--no-verify` (would keep shipping unverified packages for a ~1% coin flip); loosening the threshold to 2e-3 (hides a real regression instead of measuring the distribution) |
| D30 | 2026-11-09 | **Adopt decode recalibration**: bonus 2.4, LM weight 0.5, beam 12/top-k 10; 0.6.0, unchanged weights | E-022: T-600 −6.85 [−7.94, −5.72], S3 −8.77 [−11.13, −6.50], FLEURS-IT −6.06, EN −5.85, CV-dev **600-clip slice** −5.01 pp. This is a joint decoder-setting intervention, not an isolated bonus ablation or proof the acoustic model has no deficiencies | Peeked `0.6/3.0` cell rejected; shipped cell re-selected on dev, which does not erase prior S3 exposure (§3.5) |

## 3. Negative results

Historical ablations. "Tie" means a nominal paired 95% CI includes zero, not proof of no change.
Percentage differences/CIs below are pp. E-022/E-024 CV-dev results use the first **600 clips**,
not the full 14,697-clip dev set. Historical adoption does not by itself establish causality.

| Lever | Experiment | What it actually changed | Verdict | Evidence |
|---|---|---|---|---|
| More epochs alone | E-005 | Plateau after ~62k steps; best checkpoint not the last; last-3 averaging did not help | Not a lever | `runs/E-005/eval/best/` |
| +2 head blocks | E-006 | FLEURS-IT −3.30, CV −3.15, T −5.85 (all significant) | **Kept** | `runs/E-006/eval/best/` |
| CTC beam search | E-007 | −0.34 FLEURS-IT, −0.62 FLEURS-EN, −0.43 T (tie) | Kept as optional | `runs/E-007/eval/beam/` |
| Exact-BPE hotword bonus | E-007 | ≈ 0 on FLEURS/CV; T WER −0.77 (tie); ETA 21.8% vs greedy 20.0%, **higher**, not worse | **Not adopted**; useful term gain unestablished | `runs/E-007/eval/beam_hot*/` |
| BPE trigram LM (α=0.2) | E-008 | −5.08 FLEURS-IT, −4.46 FLEURS-EN, −6.05 CV dev, −4.13 T | **Kept** | `runs/E-008/eval/w0.2/`, `eval/best/` |
| 200 h more English | E-012 | EN −5.4, **T +5.16 worse**, **IT +0.77 worse**, ETA 0 | **Rejected** | `runs/E-012/eval-final/` |
| BPE V=4096 | E-013 | Sequences −23% tokens, ETA 0, IT/T slightly worse | **Rejected** | `runs/E-013/eval/` |
| IT+EN utterance splicing | E-014 | FLEURS-IT −2.27, FLEURS-EN −0.95, **ETA −0.90 (tie)** | Rejected **as an ETA lever**; kept as quality | `runs/E-014/eval/` |
| Energy guard on silence | E-011 add. | Silence words 240 → 0, noise words 19 (unchanged) | **Replaced by Silero VAD** (D25): 0 non-music words, 0/240 clips | `runs/E-011/guard-*/`, `runs/E-011/vad-*/` |
| Silero VAD v5 as the guard | E-011 add. | Non-music words 31 → **0** both decoders; 0 gated on T-400 and FLEURS-IT; +2.33 MB | **Kept** (0.4.0) | `runs/E-011/vad-*/`, `runs/E-011/vad_guard_summary.json` |
| Gold from S1+S2 | E-017 | T-600 38.50 → 32.77 WER, 36.05 → 42.86 ETA; S3 −6.13 WER / +4.79 ETA, equality across speakers untested | **Kept historically**, superseded by E-024 | `runs/E-017/eval/`, `runs/E-017/diffs/` |
| **Per-token bonus + LM weight re-tuned** | **E-022** | T-600 **−6.85** [−7.94, −5.72], held-out **S3 −8.77** [−11.13, −6.50], FLEURS-IT −6.06, FLEURS-EN −5.85, CV −5.01; deletions 12.2% → 8.3%, insertions 0.8% → 1.3% | **Kept** (0.6.0, decode-only, weights unchanged) | `runs/E-022/eval/`, `runs/E-022/diffs/` |
| **Noise augmentation (half of CV-IT)** | **E-024** | Controlled-SNR curve: **15 dB −6.28** [−10.58, −2.38], **10 dB −11.54** [−16.18, −7.43], **5 dB −9.68** [−14.50, −5.01]; 30 dB −2.72 and 20 dB −0.68 (ties). Clean: T-600 −0.72, T-S3 +0.26 (ties), CV-IT test **−0.52** (significant) | **Kept** (0.7.0) | `runs/E-024/eval/`, `runs/E-024/comparison.json` |
| **Noise gain on one unseen room recording** | **E-025** | 30/20/15/10/5 dB gains −3.90/−6.79/−11.21/−18.34/−14.60; all WER CIs exclude zero, ETA CIs do not | **Transfer observed**, not broad robustness certification; no new weights | `runs/E-025/eval/result.json`, `runs/E-025/room_report.json` |
| Gold term addendum (50 S1 ×20) | E-015 | Rare **17.0% → 23.4%** (8/47 → 11/47, +6.4 [−4.3, +17.0]); **S3 unchanged** (2/18); natural hits 439 → 423; `TERMS18` group (16 strings) 74.1% → 70.4%; T-600 −0.29 tie, T-S3 −1.36 tie, **FLEURS-IT +0.72 / EN +1.31 significantly worse**; dev at step20 30.81 → 33.64 → 31.22 | **Killed** (0.7.0 stays); data reusable, proposed next arm S2 + gentler recipe | `runs/E-015/RESULT.md`, `runs/E-015/eval/terms.json` |
| LM re-tune (weight, backoff, pruning) | E-023 | Optimum is **exactly E-022's cell** on 10× the selection set; backoff 0.4 best. The unpruned LM is −0.65 dev but **12.31 MB** → 26.63 MB package, and a **tie on the held-out speaker** | **Nothing adopted** (does not fit); D31 | `runs/E-023/sweep.json`, `runs/E-023/gate-diffs.json` |
| Whistle LoRA on gold source | E-019 | S3 +2.30 WER [−0.68, +5.46] (tie); FLEURS-IT +1.02, EN +2.30, first 1,000 CV-test +2.66 (CIs exclude zero); matched gold-test −2.98 (tie) | **Killed at L1**; architecture causality untested | `runs/E-019/RESULT.md`, `runs/E-019/l1-s3/` |
| int8 quantization of the shipped model | E-020 | T-600 +0.43 WER greedy / +0.17 beam+LM (both ties); FLEURS-IT +0.32 [+0.08, +0.57] | **Kept** (the artifact is honest on our speakers); FLEURS caveat for the card | `runs/E-020/RESULT.md` |
| fp32 export gate of 1e-3, single draw | E-021 | Typical gap **1.8e-4**, but 1/80 draws crosses 1e-3, always at 29 s; real audio ≤ 3.05e-4 | **Gate is the problem, not the export**: replaced by a seeded sweep (D29) | `runs/E-021/RESULT.md`, `runs/E-021/release-0.5.1.json` |

### 3.2 A matched speaker can inflate apparent ETA gains

E-016 (2026-10-29) fine-tuned E-014 on 29 min of S1's recorded gold, repeated 50x, and improved
the mixed set from **57.09% WER / 20.60% ETA to 35.51% / 40.60%** (paired CIs exclude 0), while
FLEURS-IT, FLEURS-EN and CV test were ties. Read alone, that is a breakthrough.

The controls say otherwise:

- the **pure-Italian control** (`control_it`, no English words) improved from **34.7% to 18.1%
  WER**: the model got better at S1's Italian, not just at English terms;
- the **pure-English control** was unchanged (51.4%);
- terms **present** in R-S1 went 18.3% → 45.7% ETA (+27.4), terms **absent** 15.8% → 27.7%
  (+11.9) — a real term effect, but on top of a larger speaker effect;
- nothing transferred to the **external domains** (FLEURS-IT, FLEURS-EN, CV-IT: all ties); within
  the same task, the cross-speaker transfer is real and measured in the next paragraph.

The test speaker was a training speaker. **Never present a matched-speaker mixed-set gain as
unseen-speaker generalization**; report it with an audio-untrained slice and a no-English control.

The follow-up measured the size of the confound: S2's 200 T clips (S2 never contributed training
audio) moved **51.87 → 42.22 WER** and **20.68 → 29.75 ETA** under E-016, both significant. The
matched speaker roughly **doubles** the apparent gain (S1: −21.6 / +20.0) but does not create it.
S3 — the speaker intended to be held out for E-017 — replicated it in a zero-shot read
(2026-10-30): **54.77 → 41.57 WER** and **17.75 → 30.42 ETA** (beam+LM; both CIs exclude 0), again
about 60% of the S1 gain. Caveat: S3 recorded on the **same Mac and microphone as S1**, so part of
its gain may be channel adaptation; S2 (external recorder) is the cleaner speaker test, and E-017
(training on S1+S2 gold, testing on S3) is the one that averages two channels.
The rule stands — report the matched-speaker number and the unseen-speaker number side by side,
never one alone.

### 3.3 Gold gains transfer to one audio-untrained speaker

E-017 (2026-11-07) trained the same recipe on gold from **two** speakers (S1 + S2, 50.2 min) and
tested on **S3**, which never contributed audio. Against E-016 (one gold speaker), the paired gains
with beam+LM were:

| Speaker | Role under E-017 | Δ WER | Δ ETA |
|---|---|---|---|
| S1 | trained (matched) | −4.47 [−6.72, −2.41] | +7.16 [+3.26, +11.45] |
| S2 | trained (matched) | −6.60 [−8.87, −4.33] | +8.50 [+3.84, +13.50] |
| **S3** | **never trained** | **−6.13** [−8.68, −3.51] | **+4.79** [+0.51, +9.06] |

The observation supports transfer beyond the trained voices, conditional on one run and these
clips. It does not show equal speaker effects (no interaction/equivalence test), identify what
gold teaches, or eliminate adaptation as one component of the aggregate gain. Speaker, channel,
vocabulary and task familiarity remain entangled: S1/S3 share a microphone, there is only one
audio-untrained speaker, and S3 results were already inspected in the E-017 pre-check. Report
matched and audio-untrained slices together; the confound is not retired, and S3 is not a locked,
untouched final test.

Not tested (do not claim either way): distillation from Whistle/Parakeet, spontaneous gold
code-switch speech, noise augmentation with empty targets, word-level splicing, InterCTC/CR-CTC.
Scripted gold was tested in E-016/E-017; listing all H3a gold as untested was stale.

### 3.4 One Whistle LoRA recipe failed to demonstrate transfer

E-019 (2026-11-07) used R-S1/R-S2 as a Whistle LoRA probe, **not a matched architecture ablation**.
[Config](../runs/E-019/config.json): 1,137 train / 63 validation / 63 test clips, 45.3 min training,
rank 16, alpha 32, LR 1e-4, three epochs, seed 0, forced `it`. Moth E-017 instead uses all 1,263
gold takes (50.2 min), x50, mixed with hundreds of hours of replay, full fine-tuning, and a
different objective/tokenizer/decoder. Whistle's 220,618,620-byte training checkpoint is not its
16,919,407-byte deployed `.cact`. It led historical Moth by 3.35 WER pp on T-600. The probe found:

| Set | Whistle base | Whistle + LoRA | Δ WER | 95% CI |
|---|---|---|---|---|
| **T-S3 (held-out speaker)** | 28.36 | 30.66 | **+2.30** | [−0.68, +5.46] |
| Gold test (both speakers trained) | 30.08 | 27.10 | −2.98 | [−8.38, +1.88] |
| FLEURS-IT | 11.85 | 12.87 | **+1.02** | [+0.51, +1.56] |
| FLEURS-EN | 13.35 | 15.65 | **+2.30** | [+1.16, +3.26] |
| CV-IT test, first 1,000 clips | 17.59 | 20.25 | **+2.66** | [+1.74, +3.49] |

The S3 point estimate worsens but its CI includes zero; the matched-speaker gold-test gain is
also not significant. Read-speech regressions support stopping this recipe. They do **not** show
that the gold effect is architecture-bound, that a decoder cannot retain generality, or that
speaker fit caused the regressions. Replay, update method, schedule and starting point differ;
one seed of one LoRA recipe cannot isolate architecture.

The secondary questions are therefore unanswered by choice: the plan's kill criterion fired at L1,
so `mix` (H3) and `--tokens` (H4) were never run. The probe was "less ours" by construction
(D28); its negative result is the useful part, and it belongs to the paper as a boundary on how
far this **recipe** generalises, not a boundary on decoder architectures or fine-tuning in general.

### 3.1 Historical vocabulary-coverage diagnosis (E-014)

Measured 2026-10-24 (scripts in the repo, numbers reproducible from the manifests):

- T annotates 94 English terms; **47 never occur in the training text** (CV IT + MLS IT + the
  E-014 spliced set). The missing ones are the hard categories: `backlog, cfo, cpu, crm, deadline,
  deploy, erp, kpi, merge, milestone, okr, onboarding, paypal, phishing, url, warranty`.
- **18 of them were absent from every recording script** and existed only in T. Fixed the same
  day: 35 sentences added to R-S1 and 54 to R-S2, so both R scripts now cover all 18.
- On the **41 T utterances containing a rare term**, term accuracy is Parakeet **32/41 (78%)**,
  Whistle 17/41 (41%), Whisper-tiny 12/41 (29%), **Moth E-014 2/41 (5%)**.

These are historical E-014/T-era coverage counts, not current 0.7.0 coverage. They motivate
targeted audio but do not prove coverage is the sole cause or that adding such audio suffices.
E-015 later tested a gold addendum and failed adoption criteria (§3.8); its earlier distillation
plan is not evidence that distillation was executed.

### 3.5 The biggest gain in the project came from the decoder, not the model

E-022 (2026-11-09) started as an audit of the finished project and found that roughly a fifth of
the reported WER was a **decoding** artefact. `beam_search` charged the LM log-probability on every
emitted token and inserted no counterweight, so with `lm_weight = 0.2` the search preferred
hypotheses that were too short. The evidence was in the aggregate: deletions were **12.2%** of
reference words against **0.8%** insertions, and the first word of a clip vanished **13.3%** of the
time against **0.5%** for the last.

The intervention changes two decode scalars **and beam/top-k budgets**. On the same checkpoint,
with **no training** (mixed rows show WER / ETA in percent):

| Set | before | after | Δ WER |
|---|---|---|---|
| T-600 | 32.77 / 42.86 | **25.92 / 52.64** | −6.85 [−7.94, −5.72] |
| T-S3 (held-out) | 35.43 / 35.21 | **26.66 / 47.32** | −8.77 [−11.13, −6.50] |
| FLEURS-IT | 26.52 | **20.46** | −6.06 [−6.63, −5.51] |

Three lessons, and each is uncomfortable in a different way.

1. **The gain was larger on audio-untrained S3 than on S1/S2** (−8.77 vs −5.92). Together with
   unchanged weights and the deletion/insertion shift this supports a decoder-level explanation;
   the speaker ordering alone does not identify a mechanism or certify independent generalization.
2. **A 1-D sweep cannot optimise a parameter that has two jobs.** E-008 tuned `lm_weight` over
   `{0.05 … 0.8}` and found 0.2, because a larger weight both improved the language model and
   suppressed tokens; the curve's shape was the sum of those two effects, and its optimum was the
   optimum of neither. Nothing was wrong with the sweep's execution, only with its dimension.
3. **FLEURS-IT, the bar that had blocked the project since E-008 and forced the D19 correction,
   was won by fixing the decoder** (−6.95 [−8.02, −5.87]), not by any of the data or architecture
   experiments that came before. Five experiments (E-012, E-013, E-014, E-016, E-017) were run
   against a metre that was miscalibrated by about 6 points.

**The self-correction inside the self-correction**: the strongest cell found (`lm_weight` 0.6 /
`gamma` 3.0) also turned the S3 ETA tie into a win, but it was selected **after** looking at S3.
It is reported in `runs/E-022/AUDIT-PROBE.md` §1.4 and deliberately **not** adopted. The shipped
cell was re-selected on dev. That is better than adopting the peeked cell, but does not erase the
exposure: S3/T and public test results repeatedly informed audits, vocabulary additions and
adoption decisions. A new locked speaker/channel-disjoint test is needed for confirmatory claims.

**A second finding, and a correction to it**: the audit's first pass claimed a browser/Python WER
gap of +3.27 [1.74, 4.88]. That was an **artifact of the auditor's own packaging**: the packager's
default LM is the unaugmented one, while every release since 0.3.1 ships the augmented one and
overrode the default by hand. With the correct LM, on the 200 S1 clips, browser minus Python
int8 WER is **+1.29 pp [+0.09, +2.59]** at the old decode and **−0.09 pp [−1.22, +0.96]** at
E-022 (154/200 vs 140/200 identical). Endpoints come from
[`parity-wer.json`](../runs/E-022/parity-wer.json), converted from fractions; the old mixed-unit
display understated the interval by 100x. E-010 used an earlier model/comparison; its tie is not
the same statistical result. CI inclusion of zero is not proof of transcript equivalence.

### 3.6 The model was deaf to noise, and nobody had measured it

E-024 (2026-11-09) started from a complaint, not a hypothesis: the demo produced poor transcripts
when tested with a real microphone. Two candidate causes were tested against existing artifacts,
and the first one was **wrong**:

- **clip duration alone** — not supported by this descriptive check. FLEURS-IT WER by quartile is 17.8 / 19.9 / 23.1 /
  20.0: not monotonic, and the longest quartile is better than the third. MLS and LibriSpeech have
  14–15 s medians, so the model had seen long clips all along.
- **several sentences in one pass** — degradation and benefit from known-boundary splitting are
  observed. The narrative's 9.4 pp / 80% automatic-VAD recovery is not substantiated by the retained
  segmentation artifacts; see [the audit reconciliation](12-experiment-ledger.md). Automatic VAD
  segmentation exists in code, but its implementation alone is not that quality measurement.
- **noise** — degradation measured. The earlier audit probe reported
  26.99/32.09/38.37/48.05/58.23 WER at 30/20/15/10/5 dB. Do not mix it with the paired E-024
  control below: [`eval/base/result.json`](../runs/E-024/eval/base/result.json) reports
  **25.98/27.50/33.62/44.82/50.93**, against 22.24 clean. Control deletions rise from 5.6% to
  18.5% at 5 dB; substitutions/insertions also occur. This does not isolate a unique cause or
  exclude decoder effects. Synthetic RMS SNR is not a validated equivalent of an office/bar/street.

Half of Common Voice IT was then re-rendered through noise at 10–30 dB, with the mixture held at
exactly the same 589.5 h so that "noise" could not be confused with "more CV". Result, shipped as
0.7.0:

| SNR | before | after | Δ |
|---|---|---|---|
| clean | 22.24 | 23.94 | +1.70 (tie) |
| 30 dB | 25.98 | 23.26 | −2.72 (tie) |
| 20 dB | 27.50 | 26.83 | −0.68 (tie) |
| **15 dB** | 33.62 | **27.33** | **−6.28** |
| **10 dB** | 44.82 | **33.28** | **−11.54** |
| **5 dB** | 50.93 | **41.26** | **−9.68** |

Three lessons.

1. **The measurement existed to be made and had not been.** The project had known since
   `docs/10` §4.2.6 that noise augmentation was its biggest robustness gap; what nobody had done was
   put a number on the gap, so it never competed for budget against levers that looked more
   interesting. One afternoon of measurement turned "we should probably do noise one day" into a
   measured degradation that then outranked other proposed work. The paired E-024 15 dB control
   is +11.38 pp versus clean, not the earlier probe's approximately 16 pp.
2. **The prediction was right on the mean and wrong on the shape.** `PLAN.md` predicted −3 to −6 at
   20 dB and −4 to −10 at 10 dB; measured −0.68 and −11.54. The gain is concentrated where the noise
   is worst. A model at 20 dB was already holding on; what it could not do was survive 10 dB. That
   is recorded as a wrong prediction, not reframed after the fact.
3. **A cheap descriptive check weakened a plausible hypothesis.** The non-monotonic quartiles
   do not establish that duration never matters; they do not support duration alone as the
   explanation for this failure. Controlled matched-content duration tests remain unrun.

The noise used for training and this curve came from the same 217 clips, so the estimate could be
optimistic for new noise; it was **not a mathematical upper bound**. The per-source
control (train on 80% of the noise, evaluate on the other 20%) turned out to be impossible to run —
all 217 clips were seen, and the source audio was deleted. §3.7 replaces it with a room recorded
after training; that adds one unseen noise recording, not independent speech or a general bound.

### 3.7 The noise gain was not an artifact of the shared pool

The 0.7.0 card said the noise gain was an **upper bound**, because training and evaluation drew from
the same 217 clips. The obvious control — an 80/20 split — cannot be run: the training build cycled
all 217 (`pool[noise_index % len(pool)]`), and the source MUSAN audio was deleted after the noise
set was built, so there is no way to hold out a source recording. E-025 therefore recorded a **new
room after training** — a different physical room from the room tone already in the training pool,
confirmed by the author, and spectrally distinct (the training room tone is 91% below 120 Hz; this
take is broadband) — and re-ran the same 100 held-out-speaker clips at the same SNRs, with both
checkpoints decoding the same mixed audio.

| SNR | 0.6.0 | 0.7.0 | Δ | same pool (E-024) |
|---|---|---|---|---|
| 30 dB | 28.52 | **24.62** | **−3.90 [−7.20, −0.83]** | −2.72, tie |
| 20 dB | 38.71 | **31.92** | **−6.79 [−10.84, −3.15]** | −0.68, tie |
| 15 dB | 47.71 | **36.50** | **−11.21 [−14.69, −7.48]** | −6.28 |
| 10 dB | 63.33 | **44.99** | **−18.34 [−22.78, −13.40]** | −11.54 |
| 5 dB | 81.32 | **66.72** | **−14.60 [−18.84, −10.47]** | −9.68 |

Three lessons.

1. **Transfer was observed to one newly recorded room.** Memorization of those 217 recordings
   alone cannot explain this observation. It is not a population-wide noise claim. The gain is
   **larger**, and the reason matters: the new room is harsher at the same nominal SNR (0.6.0 scores
   47.7 at 15 dB there against 33.6 on the pool), so the baseline has more headroom. A gain at fixed
   nominal SNR is **not a property of the model alone**; it depends on how damaging the noise is.
   The absolute number is worse (36.5 at 15 dB), and both halves have to be quoted together.
2. **The first recording was rejected for the right reason.** It was dominated by 50 Hz mains hum
   from the charger — 93% of the energy below 200 Hz, and after a 200 Hz high-pass only 4% of the
   energy was left. It would have measured electrical hum, not a room. The spectrum check caught it;
   without that check the experiment would have produced a precise number about the wrong stimulus.
3. **The prediction was wrong in the same direction as E-024's.** I predicted attenuation (−3..−6
   at 15 dB); measured −11.21. Two experiments, two conservative misses. Anchoring to the previous
   curve assumes the new stimulus is equally damaging, which is exactly what a held-out stimulus is
   not.

### 3.8 The first gold addendum did not teach the rare terms

The measured gap was real: 0.7.0 gets **17.0%** on the 44 terms that appear in no training text,
against 74.1% on the historical `TERMS18` gold group (the actual constant contains **16 strings**,
not 18). E-015 recorded 50 S1 sentences covering all 44
terms and added them at x20 to the 589.5 h mixture, one epoch from 0.7.0 at the E-017 learning
rates. The result was a **kill**:

| Readout | 0.7.0 | E-015 | Δ |
|---|---|---|---|
| 47 rare utterances | 8/47 (17.0%) | 11/47 (23.4%) | +6.4 [−4.3, +17.0] |
| S1 / S2 / S3 rare | 6/14, 0/15, 2/18 | 7/14, 2/15, 2/18 | +1, +2, **0** |
| Natural terms | 439/817 | 423/817 | −16 |
| Historical `TERMS18` group (16 strings) | 20/27 (74.1%) | 19/27 (70.4%) | −1 |
| FLEURS-IT / EN | 20.14 / 49.14 | 20.86 / 50.45 | **+0.72 / +1.31** |

The 2026-11-11 diagnostic **E-026** adds the missing canonical readout. On the same 47 rows the
aligned ETA (81 English occurrences) is **19/81 = 23.5% → 25/81 = 30.9%**, so the rare-subset gain
was larger than presence suggested; but full T-600 aligned ETA fell **565/1043 = 54.17% →
557/1043 = 53.40%**, so the arm gave back more on natural terms than it gained on the rare ones.
The verdict is unchanged, and the next arm must treat aligned ETA as primary and presence as a
diagnostic.

The dev curve is **30.81 after 20 smoke updates, 33.64 at step 2,000, 31.22 at the end**.
The model was pulled three points away and spent the whole epoch recovering; the best checkpoint by
dev is step 20, a near-start checkpoint, **not an untouched baseline**. The final-checkpoint
term readout does not describe that selected checkpoint; no useful replacement was adopted.

Three lessons.

1. **This recipe did not meet adoption criteria.** The dev excursion motivates a gentler update
   but does not isolate learning rate, exposure, seed or data quality as the cause.
2. **Rare-term transfer was not demonstrated.** S1 +1, S2 +2, S3 **0** on small denominators does
   not prove one voice is intrinsically insufficient or that adding S2 will work. Changing voice
   count and learning rate together cannot separately estimate either effect.
3. **A failed arm still buys information.** The 50 clips, the x20 source and the readout are
   reusable; the next arm changes S2 + the learning rate, not the data. The kill rule was fixed
   before the run and it fired on both criteria; the bar is not moved after the fact.

### 3.9 Contextual bias did not recover the rare terms

E-026 showed that 41/47 rare T references survive top-k 10, so E-027 tested the opt-in
`ContextualBias` on the published 0.7.0 release with a pre-registered 24-term work/meeting glossary
from `data/en_terms.txt`. At bonus 1.0 / prefix 0.25 it changed T-600 WER by **−0.03 points**
(95% CI [−0.12, +0.06]) and aligned ETA by **+0.10** ([−0.20, +0.42]); the 47 rare rows did not
move at all (ETA **22.22% → 22.22%**, presence 8/47). The irrelevant-glossary control produced a
small non-specific change (+1.23 rare ETA, CI [0.00, +3.49]), so there is no glossary-specific
effect. Zero bonus is byte-identical to baseline, and Italian controls show **0 → 0** glossary
insertions: the mechanism is safe but not the missing lever.

Rejected by the pre-registered gate (≥ +1.0 aligned-ETA point). A future bias needs a product
glossary validated on a real mixed-dev, not one mined from T failures; the six top-k-pruned rows
(`startup`, `prototype`, `chromecast`, `charger`, `exam`, `mondays`) need acoustic/data work.
`runs/E-027/RESULT.md`.

### 3.10 The S2 arm is the first one that passes the gates

E-028 repeated the E-015 gold question with **S2 as a second voice** (50 new sentences, 2.1 min,
x20) and a **gentler fine-tune** (head 1e-4 / encoder 1e-5 instead of 3e-4 / 5e-5). It passes the
pre-registered gates:

| Readout | 0.7.0 | E-028 | Δ |
|---|---:|---:|---|
| Rare presence (47) | 8/47 (17.0%) | **13/47 (27.7%)** | +10.6 [+2.1, +21.3] |
| Rare aligned ETA (81) | 19/81 (23.5%) | **25/81 (30.9%)** | +7.4 [0.0, +15.4] |
| Natural terms | 439/817 | 438/817 | −1 |
| T-600 WER / ETA | 25.20 / 54.17 | **24.66 / 54.65** | −0.54 tie / +0.48 tie |
| T-S3 WER / ETA | 26.92 / 48.45 | **25.55 / 49.01** | −1.36 tie / +0.56 tie |
| FLEURS-IT / EN | 20.14 / 49.14 | 20.47 / 49.38 | **+0.33** significant / +0.24 tie |

The rare gain is no longer confined to the training voices: **S1 6/14 → 8/14, S2 0/15 → 2/15,
held-out S3 2/18 → 3/18**. The natural terms are preserved (−1) and the first arm's FLEURS
regression is roughly halved (IT +0.33, EN +0.24). The 16-term gold check drops 20/27 → 18/27
(`milestone`, `paypal`).

The caveat is the dev: **31.01% vs 30.81%**, still worse than 0.7.0, although the final checkpoint is
the best inside the run. This is the A1 problem in practice: greedy IT/EN dev does not select for
the mixed product. A real mixed-dev is still needed before a product release.

The INT8 artifact, which is what the browser ships, does not repeat the fp32 gains. The package is
19,440,760 bytes and browser acceptance passes 4/4, but T-600 INT8 is 25.49 / 53.50 against
0.7.0 INT8 25.14 / 54.65 (both ties), rare presence improves 8/47 → 12/47, and natural presence
falls 441 → 428. The default dynamic quantization is the best variant inside 20 MB; per-tensor,
reduce-range, QUInt8 and static QDQ are worse or broken, and head-fp32 costs 20.62 MB for only
0.08 dev points. **0.8.0 is not published**; cheap post-training quantization is exhausted, so the
next route is QAT or keeping 0.7.0. `runs/E-028/QUANTIZATION.md`.

## 4. Self-corrections

Recorded because a paper about failures should include the failures of the authors, not only of the
models.

1. **E-004: padding vs truncation.** The first reading claimed a 3.2-point padding effect. The two
   arms also differed in the LR schedule (1 vs 3 epochs), so the control C2 was run at the matched
   schedule: 0.7375 vs an interpolated ≈0.7406. This does not establish formal equivalence, but
   removes the evidence for the initial large padding advantage. H2 switched to
   truncated input (`EXPERIMENTS.md`, E-004 addendum).
2. **E-008: "gate passed".** The claim was made against Whisper-tiny **greedy** (32.09). The best
   H0 variant is auto+beam 5 at **27.42**, against which Moth E-008 is +1.60 [+0.32, +2.83], i.e. a
   significant loss. The claim is corrected in the E-001 addendum and inside the E-008 entry.
3. **E-012: intermediate numbers.** `runs/E-012/eval/` was produced from the 8k checkpoint used to
   check the crash fix; `runs/E-012/eval-final/` is authoritative. Both are committed, and the
   E-012 entry says which is which.
4. **E-022: an arm chosen with the test set in view.** The audit's first pass picked
   `lm_weight` 0.6 / `gamma` 3.0, which wins the S3 ETA tie, after looking at S3 — a violation of
   the "never tune against S3" rule. The shipped arm was re-selected using dev alone; the peeked
   cell is documented and not adopted (§3.5, `runs/E-022/AUDIT-PROBE.md` §1.4).
5. **E-022: the auditor's browser/Python claim was an artifact.** The audit reported a +3.27 WER
   browser gap because it packaged the **unaugmented** LM (`runs/E-010/package_browser.py`
   defaults to `lm_bpe3.pkl.gz`; every release since 0.3.1 overrode it by hand). With the correct
   LM the old decode is +1.29 pp [+0.09,+2.59], a different comparison from E-010 despite the
   matching rounded estimate; the E-022 decode is a **tie** with higher
   transcript identity. The default is fixed in the same change. Recorded because the auditor
   fell for the repository's own footgun, and the lesson is that a surprising result should be
   checked against the build recipe before it is believed.

6. **E-025: predicted attenuation, measured amplification.** `PLAN.md` said the gain would shrink
   to −3..−6 at 15 dB and −6..−10 at 10 dB; measured −11.21 and −18.34, with 30 and 20 dB also
   significant against a predicted tie. The direction was right and the magnitude wrong at every
   point, because the new room is harsher than the training pool at the same nominal SNR, so the
   baseline has more headroom. The prediction is kept in `runs/E-025/PLAN.md`, not rewritten.

An earlier, smaller correction: the H2 bar in E-005/E-006 was the greedy number (32.09), not the
best variant; those entries are annotated, not rewritten, so the history of the bar is visible.

## 5. Threats to validity (for the paper)

The 0.7.0 card's S3 cross-system deltas do not match the retained E-024/baseline count files.
Doc 13, "Cross-System Gate Conflict", records the recomputed CIs and the resulting change in
nominal ETA classification. Reconcile this provenance conflict before quoting a current gate
verdict; repeated test reuse prevents turning a new audit calculation into a confirmatory win.

- **One seed per configuration.** All training runs use seed 0. The quoted CIs are utterance-level
  bootstrap intervals: they describe sampling noise over utterances, **not** training variance.
  Seed-to-seed variation is not measured and is likely comparable to some of the smaller deltas
  (E-007's beam gain, E-014's FLEURS-EN gain).
- **The mixed set T has three speakers** (S1 + S2 + S3, 600 of the 800 planned sentences,
   2026-11-06). S4 is unrecorded; its scripts were subsequently
   [reviewed](../recordings/scripts/audit-2026-11-07.md), but the four-speaker T still does not
   exist. **S3 is audio-untrained**, but repeatedly
  inspected, not a locked final test; S1 and S2 contribute gold. S1 and S3 recorded on the same
  Mac/microphone, so for those two the speaker is held out but the channel is not — the remaining
  confound after E-017.
- **Gold is small and repeated.** 50.2 min (R-S1 641 + R-S2 622 takes) repeated ×50 = 41.87 h, i.e.
  7.1% of the training audio. Each sentence is seen ~100 times over 2 epochs. A different repetition
  factor would change the result and was not measured.
- **FLEURS is public.** Whisper and Whistle may have seen it during pretraining; Whisper is trained
  on 680k h of web audio. FLEURS numbers are comparable across systems but not contamination-free.
  T is project-recorded, but test-driven term additions and repeated evaluation make it a
  development-exposed comparison, not contamination-free confirmatory evidence. Audit audio,
  text, tokenizer and LM exposure separately.
- **Coverage gaps in the baseline matrix.** Whisper on CV-IT test was never run (informational, the
  gate uses CV dev); Whistle's CV-IT row exists, Moth's CV-IT dev/test exists.
- **The noise set is 56 min of non-music, not 60**: room tone is **16.2 min**, not 20 — the
  recording stopped early (974.8 s of 1,200 s). The 97 room-tone clips are real room audio
  (−69.8 dBFS median, 0 speech windows, max window probability 0.42) and the VAD gates **97/97**.
   Music is reported separately and excluded from the zero-word acceptance criterion, not exempt
   from VAD. Since D25 the Silero VAD gates **all** 337 non-music clips
  (0 words) and 89% of the music clips; the gate is measured on synthetic MUSAN noise plus one own
  room, not on real far-field noise. On T-600 the weakest speech clip is **0.194** (`T-0033`, S3)
  — 1.9× the 0.10 gate, not the 4× of S1/S2/FLEURS — so a very quiet or reverberant speaker could
  still fall under it.
- **Historical int8 gates were ARM64 measurements.** E-003's ~40x discrepancy used random
  weights, not a measured 0.7.0 WER penalty. Later [x86 acceptance](../runs/runtime-2026-11-10/RESULT.md)
  runs 0.7.0, but lacks paired fp32 acoustics and natural mixed ETA; the quality matrix remains open.
- **ETA uses hand-annotated `en_spans`**, not a runtime lookup of the 580-term list. That list
  influences tokenizer/LM/data design; span coverage defines scoring. ETA weights word occurrences,
  not distinct terms, and does not penalize unrelated English insertions.
- **No correction for multiple comparisons.** Roughly ten gate comparisons were run; individual CIs
  are 95% and should be read with that in mind.
- **The LM is trained on training transcripts only** (CV/MLS/LS plus the training-side mixed
  scripts R-S1/R-S2/pool, never dev/test/T); its size (5.12 MB browser binary) is part of the
  approximately **19.44 MB** runtime-asset package, excluding browser engine/dependencies. For
  0.7.0 the seven recorded file sizes sum to **19,440,760 B**, not the stale aggregate
  `package_bytes: 19,436,178`. See doc 13 for exact scope/hashes.

## 6. Reproduction index

This indexes artifacts/recipes, **not locally available prerequisites**. This checkout lacks
`release/`, `models/`, `data/{raw,shards,tokenizer}/`, recorded audio and training checkpoints.
`data/manifests/build_report.json` has **no `tokenizer` key**: the statistics advertised below
survive as historical prose, not in that report. Some commands require Unix shell/MPS or scripts
importing Unix-only `resource`. See doc 13 before attempting a reproduction.

| Claim | Artifact | Regenerate with |
|---|---|---|
| Run configs and seeds | `runs/E-0NN/config.yaml` | — |
| Training curves and throughput | `runs/E-0NN/train.log.jsonl` | `uv run python -m stt.train --config runs/E-0NN/config.yaml` |
| Headline WER/ETA per set | `runs/E-0NN/eval/*/*.metrics.json` (+ `.hyp.jsonl`) | `runs/E-0NN/eval.sh` (E-014) or `runs/E-005/eval_sets.py` |
| Paired differences and CIs | `*.vs-*.json` next to the metrics | `uv run python -m stt.eval --vs ...` |
| Noise / silence behaviour (VAD, 0.4.0) | `runs/E-011/{vad-noise-greedy,vad-noise-beam-lm}/noise.metrics.json`, `runs/E-011/vad_guard_summary.json` | `runs/E-011/vad_eval.sh` (or `runs/E-005/eval_sets.py --sets noise --vad-guard`) |
| Noise / silence behaviour (energy guard, legacy) | `runs/E-011/{moth-*,guard-*}/noise.metrics.json` | `runs/E-005/eval_sets.py --sets noise --no-speech-guard` |
| LM value of the gold vocabulary | `runs/E-008/lm_bpe3_aug.pkl.gz`, `runs/E-016/eval/t400-aug-lm/` | `runs/E-008/build_lm_augmented.py`, then decode with `--lm` |
| Browser RTFx and parity | `runs/E-009/rtfx.json`, `runs/E-010/browser_t200.json` | `web/` Playwright suite, `runs/E-010/package_browser.py` |
| Release rebuild with E-016 (v0.3.0) | `runs/E-016/export.json`, addendum in `EXPERIMENTS.md` | `stt/export.py --checkpoint runs/E-016/checkpoints/ckpt_best.pt --head-blocks 2`, then `runs/E-010/package_browser.py` |
| Release 0.4.0 with the VAD | `runs/E-011/browser-0.4.0.json` (sizes, hashes, browser RTFx) | `stt/export.py --version 0.4.0` (copies the VAD), then `runs/E-010/package_browser.py --lm runs/E-008/lm_bpe3_aug.pkl.gz`; `web/` Playwright suite |
| Tokenizer statistics | `data/manifests/build_report.json` (`tokenizer`) | `uv run python -m stt.data.build_tokenizer ...` |
| Data build reports | `data/manifests/build_report.json`, `data/shards/*/build_report.json` | `uv run python -m stt.data.build_manifests`, `build_mel` |
| Spliced set (E-014) | `data/manifests/spliced_it_en_report.json` | `uv run python -m stt.data.build_spliced --hours 110` |
| Gold set and its x50 oversampling (E-016) | `recordings/manifests/R-S1.jsonl`, `runs/E-016/prepare_gold_oversample.py` | `build_manifest.py build --kind R --speaker S1`, then `build_mel`, then the prepare script |
| S3 zero-shot cross-speaker read (E-017 pre-check) | `runs/E-017/{eval,baselines,manifests}/`, `runs/E-017/vad_speakers.json` | `runs/E-017/zero_shot_s3.sh` |
| Cross-speaker gold (E-017) | `runs/E-017/{config.yaml,train.log.jsonl,eval/,diffs/}` | `runs/E-017/prepare_gold.py --sets R-S1,R-S2 --factor 50`, then `stt.train --config runs/E-017/config.yaml`, then `runs/E-017/eval.sh` |
| Noise set (E-011) | `data/manifests/noise_report.json` | `uv run python -m stt.data.build_noise ...` |
| Whistle LoRA probe (E-019) | `runs/E-019/{RESULT.md,config.json,data/,l0/,l1-s3/,l0-sets/,l1-sets/}` | `uv run python runs/E-019/build_jsonl.py`, then `runs/E-019/train_L1.sh` (throwaway `.venv-whistle`), then `baselines/whistle_needle.py --weights runs/E-019/whistle-lora-L1.cact` |
| int8 WER of the shipped model (E-020) | `runs/E-020/{RESULT.md,int8-greedy/,int8-beam-lm/,int8-fleurs-it/,torch-ref/}` | `uv run python runs/E-020/eval_onnx.py --sets recorded_t --out ...` (`stt/onnx_runner.py`), then `stt.eval --vs` the re-scored E-017 hypotheses |
| Decode recalibration (E-022) | `runs/E-022/{PLAN.md,RESULT.md,eval/,diffs/,gate-table.json,parity-wer.json,browser.json}` | `uv run python runs/E-022/eval.py --arm frozen --int8`, then `runs/E-022/diffs.py`; parity: `runs/E-022/parity.py --clips 200`, `cd web && npx playwright test tests/parity.spec.ts`, `runs/E-022/parity_wer.py` |
| Browser/Python parity (E-022) | `runs/E-022/parity-wer.json` | as above; the JS mel is `web/mel.js`, the Python one `stt/features.py` |
| Noise augmentation (E-024) | `runs/E-024/{RESULT.md,comparison.json,release-0.7.0.json}`, `data/manifests/cv_it_train_{clean,noisy}50.jsonl` | `uv run python runs/E-024/build_noisy_mel.py` (1 min), then `stt.train --config runs/E-024/config.yaml`, then `runs/E-024/noise_eval.py --checkpoint ... --out ...` for both arms and `runs/E-024/compare.py` |
| Noise hold-out on an unseen room (E-025) | `runs/E-025/{PLAN.md,RESULT.md,room_report.json,eval/result.json}` | record in a noisy room: `uv run python runs/E-025/record_room.py --seconds 90`, trim events with `--trim-start/--trim-end`, then `uv run python runs/E-025/holdout_eval.py --noise data/raw/noise/room_holdout/<take>_trimmed_16k.wav --out runs/E-025/eval` |
| Gold term addendum (E-015) | `runs/E-015/{RESULT.md,terms.json in eval/,PLAN.md,RECORDING.md}` | record: `runs/E-015/RECORDING.md`; source: `runs/E-015/prepare_terms.py`, `stt.data.build_mel`, `runs/E-017/prepare_gold.py --sets R-S1-terms --factor 20`; train: `stt.train --config runs/E-015/config.yaml`; readout: `runs/E-015/eval_terms.py` |
| LM tuning (E-023) | `runs/E-023/{sweep.json,gate-diffs.json,eval/}` | `uv run python runs/E-023/sweep.py --phase all`, then `runs/E-023/gate.py --lm-weight 0.5 --gamma 2.4 --lm runs/E-023/lm-prune-1-1.pkl.gz` |
| fp32 export gap (E-021) | `runs/E-021/{RESULT.md,fp32_gap.json}`, `runs/E-021/release-0.5.1.json` | `uv run python runs/E-021/fp32_gap.py`; release: `stt/export.py --version 0.5.1` (verified), then `runs/E-010/package_browser.py` |

## 7. Open questions, in decision order

1. **Re-scope the mixed gate — decided 2026-11-09 (D34).** The use case needs rare English terms
   used in Italian, so **ETA stays a gate**; the WER-first option is out. The remaining work is a
   short memo documenting the choice and the realigned phases, not a new debate.
2. **E-015 v2 — the first arm failed, the next changes two variables.** The 50 S1 sentences moved
   the 47 rare utterances **17.0% → 23.4%** on presence (not significant), with **S3 unchanged**
   (2/18), and the E-017 learning rates shocked the dev (30.81 → 33.64); FLEURS-IT/EN regressed.
   E-026: aligned ETA on the same 47 is **23.5% → 30.9%**, but full T-600 ETA **54.17% → 53.40%**;
   presence and ETA are not interchangeable. The data is reusable. The next arm adds **S2 as the
   second voice** (the 50 sentences are already imported, `--start 624`) and uses a **gentler
   fine-tune** (head 5e-5 / encoder 1e-5, or a frozen encoder), with the same kill rule and aligned
   ETA primary. D35; `runs/E-015/RESULT.md`, `runs/E-026/RESULT.md`.
3. **Gold scaling (E-018)** needs new speaker/channel data. Historical marginal deltas −5.73
   and −4.07 depend on decoder/recipe; a third speaker's 2–3 pp gain is a **forecast, not a
   measurement or upper bound**. S4 audio is absent; its scripts were subsequently reviewed.
4. **InterCTC** — training-only auxiliary CTC at an intermediate block; **+4.4% step time**
   measured in a probe; forecast **0.2–0.7 pp WER reduction**, not an observed quality gain.
5. **Noise**: E-024/E-025 show improvements on synthetic mixtures of 100 reused S3 clips,
   including one unseen ambience take. Multi-room/source-disjoint evaluation, new speech,
   transients and multiple seeds remain. Speech-plus-noise is not an empty-target experiment.
6. **Whistle LoRA** is closed as a product experiment; the architecture/data causal question
   remains open (§3.4). `mix` and `--tokens` were never run after the kill. Never presented as Moth.
7. **Not now, and why** — LM 4-gram (0.56 MB of headroom under the 20 MB ceiling), 25 Hz (an
   architecture change, not a knob), word timestamps (a feature, not accuracy), `en_terms`
   expansion (cannot fix terms the acoustics never saw). Reasons and costs: `STATUS.md` § "Da fare".
