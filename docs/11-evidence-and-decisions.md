# 11 — Evidence, decisions and negative results

> **Why this file exists.** The design docs (`01`–`10`) record what we *intended*; `EXPERIMENTS.md`
> records what we *ran*. This file records what we **decided, rejected and corrected**, with the
> artifact and the command behind every claim. It is written so that a paper about the failures and
> the choices can be assembled from it without re-deriving anything. Entry numbers refer to
> `EXPERIMENTS.md`; `E-0NN` artifacts live in `runs/E-0NN/`.

## 1. How any number is verified

All text goes through `stt.text.normalize` (the one text rule). All WER/ETA numbers use the same
scorer (`stt/eval.py`), with a seeded paired bootstrap (1,000 resamples, seed 0) over utterances.

```bash
# score a hypothesis file (writes metrics.json with per-utterance counts)
uv run python -m stt.eval --manifest <manifest.jsonl> --hyp <hyp.jsonl> --out <metrics.json>
# paired difference against another metrics.json (same manifest): WER and ETA CIs
uv run python -m stt.eval --manifest <manifest.jsonl> --hyp <hyp.jsonl> \
    --vs <other.metrics.json> --out <diff.json>
```

Every `metrics.json` carries the checkpoint path, the decode configuration, the manifest sha256 and
(where produced by `stt.eval`) the git SHA. Hypothesis files are committed next to the metrics, so a
disputed sentence can be inspected directly.

## 2. Decision log (execution phase)

The design decisions D1–D13 are in `docs/README.md`. These are the decisions taken while running,
each with the evidence that forced it.

| # | Date | Decision | Evidence | Alternative rejected |
|---|---|---|---|---|
| D14 | 2026-10-21 | H2 trains on **truncated** input, not padded 30 s | E-004 addendum: at a matched LR schedule the two inputs differ by <0.001 (0.7375 vs ≈0.7406), the earlier 3.2-point "padding effect" was the schedule | `padded30`: 4× memory, ~2.5× compute for the same function |
| D15 | 2026-10-21 | Keep **Common Voice IT 27** in the mix (437.7 h: 337.7 IT / 100 EN) | E-002 addendum: 167.8 h after filters, speaker-disjoint splits, 593 train/test sentence overlaps removed | Dropping CV (≈170 h of Italian lost) |
| D16 | 2026-10-22 | Keep **+2 head blocks** (11.58 M, ~12 MB int8) | E-006: FLEURS-IT 37.41 → 34.11, CV dev 37.60 → 34.45, T 65.35 → 59.50; all paired CIs exclude 0 | Staying at 4 blocks (E-005 plateaued) |
| D17 | 2026-10-22 | Beam search is an **optional quality mode**; exact-BPE hotwords are **not shipped** | E-007: beam −0.34…−0.62 WER; hotwords ≈ neutral on FLEURS/CV and **worse ETA** on T (21.8 vs 20.0) | Hotword bonus as a product feature |
| D18 | 2026-10-22 | Adopt the **BPE trigram LM (α = 0.2)** as the quality decoder | E-008: −5.08 FLEURS-IT, −6.05 CV dev, −4.13 T (all significant) | Acoustic-only greedy (kept as fast default) |
| D19 | 2026-10-23 | **Correct the H2 bar** to the best H0 variant (Whisper-tiny auto+beam5, 27.42), retracting the E-008 "gate passed" claim | E-001 addendum: 32.09 was the greedy variant; Moth 29.02 is +1.60 [+0.32, +2.83] worse | Keeping the greedy bar (would have been self-serving) |
| D20 | 2026-10-23 | Ship the **17.08 MB** package (11.91 MB int8 + 5.09 MB LM) with browser beam+LM | E-009/E-010: 58–65× RTFx on 4 threads, 25–30× on 1, T200 parity vs Python is a tie | Greedy-only 12 MB package (worse WER, same size class) |
| D21 | 2026-10-23 | **Reject A** (300 h English) | E-012: FLEURS-EN −5.4 but T +5.16 worse [3.00, 7.37] and **ETA unchanged** (21.49) | Adopting it for the English gain |
| D22 | 2026-10-23 | **Reject B** (BPE V=4096, 12.76 M) | E-013: terms ≤3 pieces 48.8% → 74.0%, but ETA 20.0 → 20.0, FLEURS-IT +1.9 worse | Adopting it for the shorter sequences |
| D23 | 2026-10-24 | Add a **no-speech energy guard**, not a VAD; greedy stays the default on unknown input | E-011 addendum: silence words 240 → 0 (120/120 clips), 0/897 speech clips touched | Silero VAD (~2 MB, out of v0.1 scope) |
| D25 | 2026-10-30 | Adopt **Silero VAD** (MIT, 2.33 MB) as the no-speech guard, replacing the energy guard; package goes to ~19.4 MB | Separability on the full sets: speech fraction min 0.194 (S3; S1/S2 0.408/0.410, FLEURS-IT 0.419) vs noise max 0.038, silence 0.000; threshold 0.10 gates 100% of non-music and 0% of speech, and the guard takes non-music words **35/31 → 0** (greedy/beam) with T-400 and FLEURS-IT hypotheses bit-identical (`runs/E-011/vad_guard_summary.json`, `runs/E-017/vad_speakers.json`) | Noise training with empty targets (would train on the evaluation set without a MUSAN re-download and split), energy/flatness VAD (measured, no separation) |
| D26 | 2026-11-07 | **Keep gold instances that fail validation out of the manifest instead of renumbering** (R-S2: 622 of 623) | The builder maps cue→file by position; renaming would shift every later cue. The excluded take is 0.39 s of silence, so nothing is lost; the count is reported | Dropping the whole set, or shifting the cue indices |
| D27 | 2026-11-07 | **Adopt E-017 as the shipped checkpoint**; gold works and generalises across speakers, but the ETA half of the mixed gate is still not won | E-017 vs E-016 on T-600: WER −5.73 [−7.14, −4.26], ETA +6.81 [+4.18, +9.55]; on **S3, which never contributed audio**, WER −6.13 [−8.68, −3.51] and ETA +4.79 [+0.51, +9.06] — the same gain as on the two trained speakers (S1 −4.47, S2 −6.60), so it is not speaker adaptation. Gate on S3 vs Whisper forced-`it`: WER −11.07 [−16.28, −5.89] **won**, ETA −2.82 [−11.65, +6.56] **tie** | Keeping E-016 (worse everywhere, and its cross-speaker claim rests on S2 alone); waiting for more gold before adopting |
| D28 | 2026-11-07 | **Run E-019, a Whistle LoRA probe, as an exploratory side artifact** — explicitly "less ours", never presented as Moth. D5 (own model, not a Whistle fine-tune) is **unchanged** for the Moth deliverable. **Executed the same day: killed at L1** — on held-out S3 +2.30 WER [−0.68, +5.46] (tie), FLEURS-IT/EN and CV-IT all significantly worse; only the two trained speakers improve. H2 (data, not architecture) rejected. H3/H4 not run, per the kill criterion | The wait for S4 (E-018's blocker) is the cheapest window; Whistle leads Moth by +3.35 WER / 6.42 ETA on T-600, and the probe answers two things Moth's CTC cannot: whether a decoder-based code-switch model gains from our gold, and whether a new `mix` language code expresses intra-clip switching. Protocol and kill criteria: `docs/plans/2026-11-07-whistle-lora-probe.md`. Result: `runs/E-019/RESULT.md`, `EXPERIMENTS.md` E-019, §3.4 | Presenting the LoRA as Moth; skipping the probe; running it on T (leakage) |
| D24 | 2026-10-24 | **Reject H3b splicing as an ETA lever**; keep E-014 as the best acoustic checkpoint | E-014: FLEURS-IT −2.27, FLEURS-EN −0.95 (both significant), CV/T/noise ties, T ETA −0.90 (tie) | More splicing, word-level splicing (too early) |
| D33 | 2026-11-09 | **Keep 0.7.0's noise claim, and upgrade it from "upper bound" to "measured on an unseen room"** (the weights do not change) | E-025: the 80/20 split proposed in STATUS was invalid — all 217 clips were in training, and the source MUSAN audio was deleted, so no partition at the source level was possible. Replaced by a room recorded **after** training (38 s of steady, broadband ambience; VAD speech 0.0). On the same 100 S3 clips the gain **transfers and is larger**: 30 dB −3.90 [−7.20, −0.83], 20 dB −6.79 [−10.84, −3.15], 15 dB −11.21 [−14.69, −7.48], 10 dB −18.34 [−22.78, −13.40], 5 dB −14.60 [−18.84, −10.47]; mean −10.06 over 10–30 dB against −6.18 on the pool. Clean reproduced E-024 exactly (22.24 / 23.94, within 1e-5), which is the harness check | An 80/20 split of the 217 clips (leaky by construction: every clip was seen); keeping the upper-bound caveat unmeasured; treating the larger gain as better absolute performance — the room is harsher, so 0.7.0 is 27.3 at 15 dB on the pool and **36.5** here |
| D32 | 2026-11-09 | **Train with noise augmentation on half of Common Voice IT**, and ship it as **0.7.0** | E-024: on a controlled SNR curve, mixed noise costs the pre-0.7.0 model **+5.3 WER at 20 dB, +11.4 at 15 dB, +22.6 at 10 dB**; at 15 dB it scored worse than Whisper-tiny does on clean audio. Half of CV-IT re-rendered through noise at 10–30 dB, with the mixture held at the same 589.5 h, buys **−6.28 at 15 dB, −11.54 at 10 dB, −9.68 at 5 dB** (mean −6.18 over 10–30 dB), and the clean gate passes (T-600 −0.72, T-S3 +0.26, CV-IT test **−0.52** significant). Population size: the build took 1.1 min because only the noisy half is computed | Not training on noise (the model drops words under noise, it does not mis-hear them, so this is a distribution gap and not a decoder problem); noising all of CV-IT (removes clean Italian, and clean WER is a gate); adding a noisy copy of all CV (mixture 757 h, CV share 28.5% → 44.3%, confounding "noise" with "more CV"); noising MLS/LS (raw audio deleted, 22 GB re-download) |
| D31 | 2026-11-09 | **Keep the LM as shipped** (weight 0.5 from E-022, backoff 0.4, count pruning ≥ 2, 5.12 MB browser binary) after a full re-tune; **do not** buy the bigger LM | E-023: on full dev (3,939 clips, 10× E-022's selection set) the optimum is exactly E-022's cell, and 0.4 stays the best backoff. The unpruned LM (pruning ≥ 1) is **0.65 dev points better** and wins the gate sets (T-600 −0.77 [−1.40, −0.11], FLEURS-IT −0.72, CV-IT dev −0.65) — but it is a **12.31 MB** browser binary, i.e. a **26.63 MB** package against the 20 MB ceiling (D4), and it is a **tie on the held-out speaker** (+0.43 [−0.61, +1.62]). The ceiling costs general Italian WER, not the code-switching claim the project exists to make | Relaxing the 20 MB ceiling for ~0.7 WER (buys an Italian model, not a code-switching one; D4 is a product constraint, not a quality knob); pruning harder to save 1.8 MB at +0.56 dev points (the LM is 5.12 MB of 19.44 and is paying its way) |
| D29 | 2026-11-07 | **Make the fp32 export gate a seeded sweep** (median ≤ 1e-3, worst ≤ 1e-2 over 20 draws per length) instead of one random draw, and rebuild the release **with verification on** (0.5.1) | E-021: the gate fixture's typical gap is 1.8e-4, but 1 draw in 80 crosses 1e-3, always at 29 s; the value depends on (checkpoint, seed, length) — E-017 gives 2.43e-4 at seed 0 and 1.28e-3 at seed 2. Real speech measures 2.2–3.1e-4 with top-1 1.0, so the export was never wrong. The rebuilt 0.5.1 passes the new gate (median 2.127e-4, worst 1.472e-3, 3/80 over) and produces a **byte-identical** int8 artifact (`fe1afcc2…`) | Keeping the single draw and `--no-verify` (would keep shipping unverified packages for a ~1% coin flip); loosening the threshold to 2e-3 (hides a real regression instead of measuring the distribution) |
| D30 | 2026-11-09 | **Adopt a per-token bonus in beam search and re-tune the LM weight with it** (`token_bonus` 2.4, `lm_weight` 0.5, beam 12 / top-k 10), shipped as **0.6.0** with the weights unchanged | E-022: the shipped decode inserted no counterweight for the per-token LM cost, so it deleted 12.2% of reference words against 0.8% insertions and dropped the first word of 13.3% of clips. With the bonus: T-600 **−6.85** [−7.94, −5.72], held-out **T-S3 −8.77** [−11.13, −6.50] (larger on the unseen speaker, so not adaptation), FLEURS-IT **−6.06**, FLEURS-EN −5.85, CV-IT dev −5.01 — every CI excludes 0, from 1 h 45 of decode and **no training**. `token_bonus=0` is a bit-exact no-op, greedy is untouched, and the int8 artifact stays byte-identical. **FLEURS-IT finally beats Whisper-tiny auto+beam 5** (−6.95 [−8.02, −5.87]), which the D19 correction had made the blocking bar | Retraining to fix a decoder bias (the CTC loss is length-neutral, so the model was never wrong); adopting the `0.6/3.0` cell that also wins the S3 ETA tie (it was selected after looking at S3, breaking the held-out rule — see the E-022 self-correction); leaving the LM weight at 0.2 (it was tuned under the broken calibration, so its 1-D optimum was the optimum of neither job) |

## 3. Negative results

The paper's core. "No change" means a paired 95% CI that includes 0 on the ship-gate sets.

| Lever | Experiment | What it actually changed | Verdict | Evidence |
|---|---|---|---|---|
| More epochs alone | E-005 | Plateau after ~62k steps; best checkpoint not the last; last-3 averaging did not help | Not a lever | `runs/E-005/eval/best/` |
| +2 head blocks | E-006 | FLEURS-IT −3.30, CV −3.15, T −5.85 (all significant) | **Kept** | `runs/E-006/eval/best/` |
| CTC beam search | E-007 | −0.34 FLEURS-IT, −0.62 FLEURS-EN, −0.43 T (tie) | Kept as optional | `runs/E-007/eval/beam/` |
| Exact-BPE hotword bonus | E-007 | ≈ 0 on FLEURS/CV; T WER −0.77 (tie) but **ETA worse** | **Rejected** | `runs/E-007/eval/beam_hot*/` |
| BPE trigram LM (α=0.2) | E-008 | −5.08 FLEURS-IT, −4.46 FLEURS-EN, −6.05 CV dev, −4.13 T | **Kept** | `runs/E-008/eval/w0.2/`, `eval/best/` |
| 200 h more English | E-012 | EN −5.4, **T +5.16 worse**, IT −0.8, ETA 0 | **Rejected** | `runs/E-012/eval-final/` |
| BPE V=4096 | E-013 | Sequences −23% tokens, ETA 0, IT/T slightly worse | **Rejected** | `runs/E-013/eval/` |
| IT+EN utterance splicing | E-014 | FLEURS-IT −2.27, FLEURS-EN −0.95, **ETA −0.90 (tie)** | Rejected **as an ETA lever**; kept as quality | `runs/E-014/eval/` |
| Energy guard on silence | E-011 add. | Silence words 240 → 0, noise words 19 (unchanged) | **Replaced by Silero VAD** (D25): 0 non-music words, 0/240 clips | `runs/E-011/guard-*/`, `runs/E-011/vad-*/` |
| Silero VAD v5 as the guard | E-011 add. | Non-music words 31 → **0** both decoders; 0 gated on T-400 and FLEURS-IT; +2.33 MB | **Kept** (0.4.0) | `runs/E-011/vad-*/`, `runs/E-011/vad_guard_summary.json` |
| Gold data from two speakers (S1+S2) | E-017 | T-600 38.50 → **32.77** WER, 36.05 → **42.86** ETA; on held-out S3 −6.13 WER / +4.79 ETA, the same as on the trained speakers | **Kept** (best model) | `runs/E-017/eval/`, `runs/E-017/diffs/` |
| **Per-token bonus + LM weight re-tuned** | **E-022** | T-600 **−6.85** [−7.94, −5.72], held-out **S3 −8.77** [−11.13, −6.50], FLEURS-IT −6.06, FLEURS-EN −5.85, CV −5.01; deletions 12.2% → 8.3%, insertions 0.8% → 1.3% | **Kept** (0.6.0, decode-only, weights unchanged) | `runs/E-022/eval/`, `runs/E-022/diffs/` |
| **Noise augmentation (half of CV-IT)** | **E-024** | Controlled-SNR curve: **15 dB −6.28** [−10.58, −2.38], **10 dB −11.54** [−16.18, −7.43], **5 dB −9.68** [−14.50, −5.01]; 30 dB −2.72 and 20 dB −0.68 (ties). Clean: T-600 −0.72, T-S3 +0.26 (ties), CV-IT test **−0.52** (significant) | **Kept** (0.7.0) | `runs/E-024/eval/`, `runs/E-024/comparison.json` |
| **Noise gain on an unseen room** | **E-025** | A room recorded after training: **30 dB −3.90** [−7.20, −0.83], **20 dB −6.79** [−10.84, −3.15], **15 dB −11.21** [−14.69, −7.48], **10 dB −18.34** [−22.78, −13.40], **5 dB −14.60** [−18.84, −10.47]; mean −10.06 vs −6.18 on the pool; clean reproduced exactly (22.24 / 23.94) | **Caveat closed** (0.7.0, card updated in place, no new weights) | `runs/E-025/eval/result.json`, `runs/E-025/room_report.json` |
| LM re-tune (weight, backoff, pruning) | E-023 | Optimum is **exactly E-022's cell** on 10× the selection set; backoff 0.4 best. The unpruned LM is −0.65 dev but **12.31 MB** → 26.63 MB package, and a **tie on the held-out speaker** | **Nothing adopted** (does not fit); D31 | `runs/E-023/sweep.json`, `runs/E-023/gate-diffs.json` |
| LoRA on Whistle with the same gold | E-019 | On held-out S3 **+2.30 WER** [−0.68, +5.46] (tie, kill); FLEURS-IT +1.02, FLEURS-EN +2.30, CV-IT +2.66 — **all significant worse**; only the two trained speakers improve (−2.98, tie) | **Killed at L1**; H2 (data, not architecture) rejected | `runs/E-019/RESULT.md`, `runs/E-019/l1-s3/` |
| int8 quantization of the shipped model | E-020 | T-600 +0.43 WER greedy / +0.17 beam+LM (both ties); FLEURS-IT +0.32 [+0.08, +0.57] | **Kept** (the artifact is honest on our speakers); FLEURS caveat for the card | `runs/E-020/RESULT.md` |
| fp32 export gate of 1e-3, single draw | E-021 | Typical gap **1.8e-4**, but 1/80 draws crosses 1e-3, always at 29 s; real audio ≤ 3.05e-4 | **Gate is the problem, not the export**: replaced by a seeded sweep (D29) | `runs/E-021/RESULT.md`, `runs/E-021/release-0.5.1.json` |

### 3.2 A matched speaker can fabricate a +20-point ETA

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

The test speaker was a training speaker. **Never report a mixed-set gain measured on a speaker
that contributed training audio**, and keep a no-English control in the test set: it is what made
the confound visible.

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

### 3.3 The confound is retired: a held-out speaker gains as much as the trained ones

E-017 (2026-11-07) trained the same recipe on gold from **two** speakers (S1 + S2, 50.2 min) and
tested on **S3**, which never contributed audio. Against E-016 (one gold speaker), the paired gains
with beam+LM were:

| Speaker | Role under E-017 | Δ WER | Δ ETA |
|---|---|---|---|
| S1 | trained (matched) | −4.47 [−6.72, −2.41] | +7.16 [+3.26, +11.45] |
| S2 | trained (matched) | −6.60 [−8.87, −4.33] | +8.50 [+3.84, +13.50] |
| **S3** | **never trained** | **−6.13** [−8.68, −3.51] | **+4.79** [+0.51, +9.06] |

An unseen speaker improves as much as the seen ones. Whatever the gold teaches, it is not "this
person's voice": adaptation to a speaker the model never heard is not a mechanism. Note the
asymmetry is in the *right* direction — S1, with the most gold audio, gains the least, which is
consistent with the gold teaching vocabulary-and-task rather than speaker traits.

Consequence for how results are reported: the matched-speaker number is still the optimistic one,
but the held-out number no longer collapses, so the gold claim can be stated without the E-016
caveat. The remaining confound is **channel**, not speaker (S1 and S3 share a microphone; S2 does
not, and gains the most).

Not tested (do not claim either way): distillation from Whistle/Parakeet, gold code-switch data
(H3a), noise augmentation with empty targets, word-level splicing, InterCTC/CR-CTC.

### 3.4 The gold effect is architecture-bound: a Whistle LoRA loses on the same data

E-019 (2026-11-07) ran the external check the gold result needed: the **same** 45.3 min of gold
(R-S1 + R-S2, never T) as a LoRA on Cactus Whistle, a 220 MB decoder model that is 3.35 WER
points *ahead* of Moth on T-600. If the gold effect were a property of the data, it should move
Whistle the same way it moved Moth. It does not:

| Set | Whistle base | Whistle + LoRA | Δ WER | 95% CI |
|---|---|---|---|---|
| **T-S3 (held-out speaker)** | 28.36 | 30.66 | **+2.30** | [−0.68, +5.46] |
| Gold test (both speakers trained) | 30.08 | 27.10 | −2.98 | [−8.38, +1.88] |
| FLEURS-IT | 11.85 | 12.87 | **+1.02** | [+0.51, +1.56] |
| FLEURS-EN | 13.35 | 15.65 | **+2.30** | [+1.16, +3.26] |
| CV-IT test | 17.59 | 20.25 | **+2.66** | [+1.74, +3.49] |

Where Moth's CTC gained **−6.13 WER** on S3 with the same data (D27), Whistle's decoder **loses
2.3**, and every out-of-gold set degrades significantly. The adapter does adapt to the two trained
voices; it buys speaker fit at the cost of the base model's generality. A LoRA on a decoder with
one language token per clip has no mechanism to keep what it knew.

The secondary questions are therefore unanswered by choice: the plan's kill criterion fired at L1,
so `mix` (H3) and `--tokens` (H4) were never run. The probe was "less ours" by construction
(D28); its negative result is the useful part, and it belongs to the paper as a boundary on how
far the gold claim generalises — **to our architecture**, not to fine-tuning in general.

### 3.1 Why the ETA does not move: it is vocabulary coverage

Measured 2026-10-24 (scripts in the repo, numbers reproducible from the manifests):

- T annotates 94 English terms; **47 never occur in the training text** (CV IT + MLS IT + the
  E-014 spliced set). The missing ones are the hard categories: `backlog, cfo, cpu, crm, deadline,
  deploy, erp, kpi, merge, milestone, okr, onboarding, paypal, phishing, url, warranty`.
- **18 of them were absent from every recording script** and existed only in T. Fixed the same
  day: 35 sentences added to R-S1 and 54 to R-S2, so both R scripts now cover all 18.
- On the **41 T utterances containing a rare term**, term accuracy is Parakeet **32/41 (78%)**,
  Whistle 17/41 (41%), Whisper-tiny 12/41 (29%), **Moth E-014 2/41 (5%)**.

This reframes the earlier failures: E-012 (more English) and E-014 (splicing) added English audio
whose vocabulary was audiobook/web prose, not modern tech terms; neither could teach the terms.
It also identifies the data that would: **audio where those words are actually spoken**. See
`runs/E-015/PLAN.md` for the distillation protocol, which is blocked on exactly that asset.

### 3.5 The biggest gain in the project came from the decoder, not the model

E-022 (2026-11-09) started as an audit of the finished project and found that roughly a fifth of
the reported WER was a **decoding** artefact. `beam_search` charged the LM log-probability on every
emitted token and inserted no counterweight, so with `lm_weight = 0.2` the search preferred
hypotheses that were too short. The evidence was in the aggregate: deletions were **12.2%** of
reference words against **0.8%** insertions, and the first word of a clip vanished **13.3%** of the
time against **0.5%** for the last.

The fix is two decode scalars. On the same checkpoint, with **no training**:

| Set | before | after | Δ WER |
|---|---|---|---|
| T-600 | 32.77 / 42.86 | **25.92 / 52.64** | −6.85 [−7.94, −5.72] |
| T-S3 (held-out) | 35.43 / 35.21 | **26.66 / 47.32** | −8.77 [−11.13, −6.50] |
| FLEURS-IT | 26.52 | **20.46** | −6.06 [−6.63, −5.51] |

Three lessons, and each is uncomfortable in a different way.

1. **The gain was larger on the held-out speaker than on the training speakers** (−8.77 vs −5.92),
   which is the signature of a bias being removed rather than a model being fitted. Any result
   whose gain is biggest on the speakers it trained on should be distrusted more than this one.
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
It is reported in `runs/E-022/AUDIT-PROBE.md` §1.4 and deliberately **not** adopted. The arm that
shipped is the one dev alone selected. A result that needs the test set to look good is not a
result, and the repository's "never tune against S3" rule exists precisely because the temptation
is real.

**A second finding, and a correction to it**: the audit's first pass claimed a browser/Python WER
gap of +3.27 [1.74, 4.88]. That was an **artifact of the auditor's own packaging**: the packager's
default LM is the unaugmented one, while every release since 0.3.1 ships the augmented one and
overrode the default by hand. With the correct LM, on 200 clips: shipped decode **+1.29**
[0.0009, 0.0259] — the same point estimate E-010 measured — and the **E-022 decode is a tie at
−0.09 [−0.012, +0.010] with *higher* transcript identity** (154/200 vs 140/200). The wider beam
improves parity rather than hurting it. `docs/11` §3.5 carries the corrected numbers; §4 records
the auditor's error next to the project's own.

### 3.6 The model was deaf to noise, and nobody had measured it

E-024 (2026-11-09) started from a complaint, not a hypothesis: the demo produced poor transcripts
when tested with a real microphone. Two candidate causes were tested against existing artifacts,
and the first one was **wrong**:

- **clip duration** — falsified. On FLEURS-IT the WER by duration quartile is 17.8 / 19.9 / 23.1 /
  20.0: not monotonic, and the longest quartile is better than the third. MLS and LibriSpeech have
  14–15 s medians, so the model had seen long clips all along.
- **several sentences in one pass** — confirmed, and fixed by splitting at pauses (9.4 WER points,
  80% recovered by the VAD that already shipped).
- **noise** — confirmed, and it was the big one. Measured on 100 held-out-speaker clips with MUSAN
  mixed at a controlled SNR: **26.99 at 30 dB, 32.09 at 20 dB, 38.37 at 15 dB, 48.05 at 10 dB,
  58.23 at 5 dB**, against 22.24 clean. Every condition significant. Deletions went from 5.6% of
  reference words to **24.6%**: the model did not mis-hear words under noise, it **dropped** them,
  which is the signature of a training set with no noise in it. At 15 dB — an ordinary noisy office
  — it scored worse than Whisper-tiny does on clean audio.

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
   16-point hole at 15 dB, which then outranked everything else on the list.
2. **The prediction was right on the mean and wrong on the shape.** `PLAN.md` predicted −3 to −6 at
   20 dB and −4 to −10 at 10 dB; measured −0.68 and −11.54. The gain is concentrated where the noise
   is worst. A model at 20 dB was already holding on; what it could not do was survive 10 dB. That
   is recorded as a wrong prediction, not reframed after the fact.
3. **A plausible hypothesis was falsified in ten minutes.** Clip duration felt like the obvious
   explanation for the bad transcript and cost one query on existing artifacts to kill. The
   expensive mistake is not testing the cheap one first.

The limit as stated at the time: the noise used for training and the noise used for this curve came
from the same 217 clips, so the gain was an **upper bound** for an unfamiliar room. The per-source
control (train on 80% of the noise, evaluate on the other 20%) turned out to be impossible to run —
all 217 clips were seen, and the source audio was deleted. §3.7 replaces it with a room recorded
after training, and the bound holds.

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

1. **The caveat was worth testing, and the gain is real.** It is not a memory of those 217
   recordings: it transfers to a room that did not exist in any training artifact. In fact it is
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

## 4. Self-corrections

Recorded because a paper about failures should include the failures of the authors, not only of the
models.

1. **E-004: padding vs truncation.** The first reading claimed a 3.2-point padding effect. The two
   arms also differed in the LR schedule (1 vs 3 epochs), so the control C2 was run at the matched
   schedule: 0.7375 vs ≈0.7406 → equivalent. The conclusion was retracted and H2 switched to
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
   LM the shipped decode is +1.29 — E-010's number — and the E-022 decode is a **tie** with higher
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

- **One seed per configuration.** All training runs use seed 0. The quoted CIs are utterance-level
  bootstrap intervals: they describe sampling noise over utterances, **not** training variance.
  Seed-to-seed variation is not measured and is likely comparable to some of the smaller deltas
  (E-007's beam gain, E-014's FLEURS-EN gain).
- **The mixed set T has three speakers** (S1 + S2 + S3, 600 of the 800 planned sentences,
  2026-11-06). S4 is unrecorded and its 200 sentences are unreviewed, so the final four-speaker T
  does not exist yet. **S3 is held out from training** (it recorded only T and A, never R), so it
  is a valid generalisation test; S1 and S2 contribute gold. S1 and S3 recorded on the same
  Mac/microphone, so for those two the speaker is held out but the channel is not — the remaining
  confound after E-017.
- **Gold is small and repeated.** 50.2 min (R-S1 641 + R-S2 622 takes) repeated ×50 = 41.87 h, i.e.
  7.1% of the training audio. Each sentence is seen ~100 times over 2 epochs. A different repetition
  factor would change the result and was not measured.
- **FLEURS is public.** Whisper and Whistle may have seen it during pretraining; Whisper is trained
  on 680k h of web audio. FLEURS numbers are comparable across systems but not contamination-free.
  T, being ours and unrecorded by others, is the cleaner comparison — it is also the harder one.
- **Coverage gaps in the baseline matrix.** Whisper on CV-IT test was never run (informational, the
  gate uses CV dev); Whistle's CV-IT row exists, Moth's CV-IT dev/test exists.
- **The noise set is 56 min of non-music, not 60**: room tone is **16.2 min**, not 20 — the
  recording stopped early (974.8 s of 1,200 s). The 97 room-tone clips are real room audio
  (−69.8 dBFS median, 0 speech windows, max window probability 0.42) and the VAD gates **97/97**.
  Music is reported, never gated. Since D25 the Silero VAD gates **all** 337 non-music clips
  (0 words) and 89% of the music clips; the gate is measured on synthetic MUSAN noise plus one own
  room, not on real far-field noise. On T-600 the weakest speech clip is **0.194** (`T-0033`, S3)
  — 1.9× the 0.10 gate, not the 4× of S1/S2/FLEURS — so a very quiet or reverberant speaker could
  still fall under it.
- **int8 was gated on ARM64 only.** On x86_64 the same int8 file deviates ~40× more (E-003
  addendum), so the int8 quality gate on the target platform is still open.
- **ETA depends on a curated term list** (580 terms in `data/en_terms.txt`); T has hand-annotated
  `en_spans`. A different term list would move ETA in ways this project did not measure.
- **No correction for multiple comparisons.** Roughly ten gate comparisons were run; individual CIs
  are 95% and should be read with that in mind.
- **The LM is trained on training transcripts only** (CV/MLS/LS plus the training-side mixed
  scripts R-S1/R-S2/pool, never dev/test/T); its size (5.12 MB browser binary) is part of the
  **19.44 MB** 0.4.0 package (11.91 MB int8 + 5.12 MB LM + 2.33 MB VAD + 0.08 MB features/vocab).

## 6. Reproduction index

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
| LM tuning (E-023) | `runs/E-023/{sweep.json,gate-diffs.json,eval/}` | `uv run python runs/E-023/sweep.py --phase all`, then `runs/E-023/gate.py --lm-weight 0.5 --gamma 2.4 --lm runs/E-023/lm-prune-1-1.pkl.gz` |
| fp32 export gap (E-021) | `runs/E-021/{RESULT.md,fp32_gap.json}`, `runs/E-021/release-0.5.1.json` | `uv run python runs/E-021/fp32_gap.py`; release: `stt/export.py --version 0.5.1` (verified), then `runs/E-010/package_browser.py` |

## 7. Open questions, in decision order

1. **Re-scope the mixed gate** — the top decision now that E-025 is closed and E-018 is dead. On
   held-out S3 the WER is **won** and ETA is a **tie** (+8.73 [−1.08, +20.00]); the question is
   whether the gate is WER-first with ETA secondary, or whether ETA stays a target and E-015 is
   funded. Evidence: `runs/E-024/MODEL_CARD.md` and §5 below.
2. **Distillation from Parakeet** — the teacher gets **78%** of the rare terms right against Moth's
   5% (§3.1), so the signal exists. Success criterion **ETA +5** with WER not worse; cost 3.3 h of
   labeling + 3.5 h of fine-tune. Blocked on audio provenance (gold / CC0-BY / TTS), not compute:
   `runs/E-015/PLAN.md`.
3. **Gold code-switch data (E-018)** — confirmed by E-017, but the scaling question needs S4, who
   was never recorded. The marginal return of the second speaker fell from −5.73 to −4.07, so a
   third is worth roughly 2–3 WER at best. Treat as dead unless S4 appears.
4. **InterCTC** — training-only auxiliary CTC at an intermediate block; **+4.4% step time**
   measured, expected **+0.2–0.7 WER**. The cheapest size-neutral accuracy lever; not yet run.
5. **Noise: what remains** — E-024 (training) and E-025 (holdout) closed the "noisy speech"
   question for CV-IT and one unseen room. Remaining: noise on MLS/LS (22 GB, disk-bound) and a
   second room (evidence only, no model change). The old "noise with empty targets" item is
   **closed**: E-024 augmented speech, which is what the transcript needed.
6. **Whistle LoRA probe** (E-019, D28) — **answered and closed**: a decoder-based code-switch model
   does **not** gain from the same gold (S3 +2.30 WER, tie; FLEURS/CV significantly worse), so the
   gold effect is architecture-bound (§3.4). `mix` and `--tokens` were never run because the kill
   criterion fired. Result: `runs/E-019/RESULT.md`. Never presented as Moth.
7. **Not now, and why** — LM 4-gram (0.56 MB of headroom under the 20 MB ceiling), 25 Hz (an
   architecture change, not a knob), word timestamps (a feature, not accuracy), `en_terms`
   expansion (cannot fix terms the acoustics never saw). Reasons and costs: `STATUS.md` § "Da fare".
