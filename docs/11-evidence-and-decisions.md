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
| LoRA on Whistle with the same gold | E-019 | On held-out S3 **+2.30 WER** [−0.68, +5.46] (tie, kill); FLEURS-IT +1.02, FLEURS-EN +2.30, CV-IT +2.66 — **all significant worse**; only the two trained speakers improve (−2.98, tie) | **Killed at L1**; H2 (data, not architecture) rejected | `runs/E-019/RESULT.md`, `runs/E-019/l1-s3/` |
| int8 quantization of the shipped model | E-020 | T-600 +0.43 WER greedy / +0.17 beam+LM (both ties); FLEURS-IT +0.32 [+0.08, +0.57] | **Kept** (the artifact is honest on our speakers); FLEURS caveat for the card | `runs/E-020/RESULT.md` |
| fp32 export gate of 1e-3, single draw | E-021 | Typical gap **1.8e-4**, but 1/80 draws crosses 1e-3, always at 29 s; real audio ≤ 3.05e-4 | **Gate is the problem, not the export**; switch to a seeded sweep | `runs/E-021/RESULT.md` |

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
| fp32 export gap (E-021) | `runs/E-021/{RESULT.md,fp32_gap.json}` | `uv run python runs/E-021/fp32_gap.py` |

## 7. Open questions, in decision order

1. **Distillation from Parakeet (639 MB)** — the teacher gets **78%** of the rare terms right
   against Moth's 5% (§3.1), so the signal exists. Blocked on audio containing those terms:
   `runs/E-015/PLAN.md` lists the three possible sources and the provenance decision each needs.
2. **Gold code-switch data (H3a)** — the original hypothesis, now **confirmed** by E-017: gold works
   and generalises across speakers. The open part is **how far it scales** (E-018, blocked on S4's
   recordings and the review of its 200 sentences).
3. **Re-scope the mixed gate** — ETA as a secondary metric, "Italian-first with frequent English
   terms", if 1 and 2 are out of budget. Cheaper to decide after E-018: the gap to Whistle is now
   3.35 WER, not 14.
4. **Noise augmentation with empty targets** (E-011 follow-up 2) — the VAD closes the "no speech
   in, no words out" gate, but noisy *speech* still degrades the transcript; this remains the
   principled fix and is independent of the gate.
5. **Whistle LoRA probe** (E-019, D28) — **answered and closed**: a decoder-based code-switch
   model does **not** gain from the same gold (S3 +2.30 WER, tie; FLEURS/CV significantly worse),
   so the gold effect is architecture-bound (§3.4). `mix` and `--tokens` were never run because
   the kill criterion fired. Result: `runs/E-019/RESULT.md`. Never presented as Moth.
