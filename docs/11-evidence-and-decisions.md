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
| Energy guard on silence | E-011 add. | Silence words 240 → 0, noise words 19 (unchanged) | **Kept** | `runs/E-011/guard-*/` |

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
- nothing transferred to other speakers (all three external sets are ties).

The test speaker was a training speaker. **Never report a mixed-set gain measured on a speaker
that contributed training audio**, and keep a no-English control in the test set: it is what made
the confound visible.

The follow-up measured the size of the confound: S2's 200 T clips (S2 never contributed training
audio) moved **51.87 → 42.22 WER** and **20.68 → 29.75 ETA** under E-016, both significant. The
matched speaker roughly **doubles** the apparent gain (S1: −21.6 / +20.0) but does not create it.
The rule stands — report the matched-speaker number and the unseen-speaker number side by side,
never one alone.

Not tested (do not claim either way): distillation from Whistle/Parakeet, gold code-switch data
(H3a), noise augmentation with empty targets, word-level splicing, InterCTC/CR-CTC.

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
- **The mixed set T is two speakers now** (S1 + S2, 400 of the 800 planned sentences, 2026-10-30),
  read in quiet rooms. S3 and S4 are unrecorded, so the *held-out* speaker for E-017 does not exist
  yet, and every number measured before 2026-10-30 used the 200 S1 clips only.
- **FLEURS is public.** Whisper and Whistle may have seen it during pretraining; Whisper is trained
  on 680k h of web audio. FLEURS numbers are comparable across systems but not contamination-free.
  T, being ours and unrecorded by others, is the cleaner comparison — it is also the harder one.
- **Coverage gaps in the baseline matrix.** Whisper on CV-IT test was never run (informational, the
  gate uses CV dev); Whistle's CV-IT row exists, Moth's CV-IT dev/test exists.
- **The noise set is 40 min of non-music, not 60**: room tone (20 min) is not recorded. Music is
  reported, never gated. The guard fixes digital silence only; MUSAN noise still produces 13–24
  words.
- **int8 was gated on ARM64 only.** On x86_64 the same int8 file deviates ~40× more (E-003
  addendum), so the int8 quality gate on the target platform is still open.
- **ETA depends on a curated term list** (580 terms in `data/en_terms.txt`); T has hand-annotated
  `en_spans`. A different term list would move ETA in ways this project did not measure.
- **No correction for multiple comparisons.** Roughly ten gate comparisons were run; individual CIs
  are 95% and should be read with that in mind.
- **The LM is trained on training transcripts only** (CV/MLS/LS), never on dev/test/T; its size
  (3.53 MB pickled, 5.09 MB browser binary) is part of the 17.08 MB package.

## 6. Reproduction index

| Claim | Artifact | Regenerate with |
|---|---|---|
| Run configs and seeds | `runs/E-0NN/config.yaml` | — |
| Training curves and throughput | `runs/E-0NN/train.log.jsonl` | `uv run python -m stt.train --config runs/E-0NN/config.yaml` |
| Headline WER/ETA per set | `runs/E-0NN/eval/*/*.metrics.json` (+ `.hyp.jsonl`) | `runs/E-0NN/eval.sh` (E-014) or `runs/E-005/eval_sets.py` |
| Paired differences and CIs | `*.vs-*.json` next to the metrics | `uv run python -m stt.eval --vs ...` |
| Noise / silence behaviour | `runs/E-011/{moth-*,guard-*}/noise.metrics.json` | `runs/E-005/eval_sets.py --sets noise --no-speech-guard` |
| Browser RTFx and parity | `runs/E-009/rtfx.json`, `runs/E-010/browser_t200.json` | `web/` Playwright suite, `runs/E-010/package_browser.py` |
| Tokenizer statistics | `data/manifests/build_report.json` (`tokenizer`) | `uv run python -m stt.data.build_tokenizer ...` |
| Data build reports | `data/manifests/build_report.json`, `data/shards/*/build_report.json` | `uv run python -m stt.data.build_manifests`, `build_mel` |
| Spliced set (E-014) | `data/manifests/spliced_it_en_report.json` | `uv run python -m stt.data.build_spliced --hours 110` |
| Gold set and its x50 oversampling (E-016) | `recordings/manifests/R-S1.jsonl`, `runs/E-016/prepare_gold_oversample.py` | `build_manifest.py build --kind R --speaker S1`, then `build_mel`, then the prepare script |
| Noise set (E-011) | `data/manifests/noise_report.json` | `uv run python -m stt.data.build_noise ...` |

## 7. Open questions, in decision order

1. **Distillation from Parakeet (639 MB)** — the teacher gets **78%** of the rare terms right
   against Moth's 5% (§3.1), so the signal exists. Blocked on audio containing those terms:
   `runs/E-015/PLAN.md` lists the three possible sources and the provenance decision each needs.
2. **Gold code-switch data (H3a)** — the only untested hypothesis, and the plan's original one.
   Costs recording hours: R-S1 is at 76/600 and T has S1 only.
3. **Re-scope the mixed gate** — ETA as a secondary metric, "Italian-first with frequent English
   terms", if 1 and 2 are out of budget.
4. **Noise augmentation with empty targets** (E-011 follow-up 2) — independent of the gate and
   needed before shipping a demo that accepts arbitrary audio.
