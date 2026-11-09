# E-015 — distillation pilot (planned, **not started**)

Status: **blocked on data**, not on compute. This file is the protocol and the evidence that led to
it; it becomes an `EXPERIMENTS.md` entry only when a run exists.

## Why this experiment exists: the ETA failure is vocabulary coverage

Measured 2026-10-24 on the committed artifacts:

- T annotates **94 distinct English terms**. **47 of them never appear anywhere in the training
  text** (CV IT + MLS IT + the E-014 spliced set) — including exactly the categories where ETA
  collapses: `backlog, cfo, cpu, crm, deadline, deploy, erp, kpi, merge, milestone, okr, onboarding,
  paypal, phishing, url, warranty`.
- **18 of those terms were absent from every recording script** (they existed only in T). Fixed on
  2026-10-24 by adding 35 sentences to R-S1 and 54 to R-S2 (all 18 terms covered).
- On the **41 T utterances that contain a rare term**, term accuracy:

| System | Rare terms correct |
|---|---|
| Parakeet TDT 0.6B v3 int8 | **32/41 (78%)** |
| Cactus Whistle | 17/41 (41%) |
| Whisper-tiny beam 5 | 12/41 (29%) |
| **Moth E-014** | **2/41 (5%)** |

- This also explains E-014: the spliced set concatenated LibriSpeech audiobook English, which
  carries no modern tech vocabulary. It improved FLEURS (read Italian) and could not move ETA.

**Conclusion:** the model cannot emit words whose acoustics it has never seen; the bottleneck is
audio **content**, not model capacity, BPE size, English volume or switch structure.

## Hypothesis

Teacher transcripts on audio that actually contains these terms transfer the acoustic mapping to
the student, at the price of the teacher's noise (22% wrong on the rare terms).

## Teacher choice

Parakeet TDT 0.6B v3 int8 (`baselines/parakeet_sherpa.py`, 639 MB, ≈30× real time). Not Whistle
(41% on the rare terms) and not Whisper-tiny (29%). The teacher returns **text**, so this is
hard-label distillation / pseudo-labelling, not soft-target KD; the vocabularies never need to
align.

## Data: the actual blocker

The teacher needs audio where those terms are spoken, and we have none that may be trained on:

- **T** has 41 such utterances but is evaluation-only, by project rule.
- **R-S1** (76 takes) has human transcripts — no teacher needed.
- **Public Italian corpora** (Common Voice) contain the frequent loanwords (`file`, `test`,
  `computer`, `meeting`) but not the rare ones; that is why the gap is concentrated in
  `brand`/`acronym`/`single_word`.

Options, in order of defensibility:

1. **Gold recordings** — human labels, no teacher, so this is E-016, not E-015. The 35+54 new
   term sentences make the recordings cover all 18 missing terms.
2. **Legally reusable Italian tech speech** (e.g. CC0/BY sources) — needs an explicit provenance
   decision before any compute; an Apache-2.0 model should not be trained on NC/ND material.
3. **TTS synthesis** (H3c) of new term-rich sentences — cheapest and fully legal, but introduces
   voice artifacts; the design already lists H3c as the fallback.

**No source, no E-015.** The decision is the user's; nothing here should be started on a
questionable corpus just to have a number.

## Protocol (ready for when a source exists)

- **Student**: E-014 best (`runs/E-014/checkpoints/ckpt_best.pt`, 11.58 M).
- **Intervention**: add the pseudo-labelled audio to the E-014 mix (same 547.5 h baseline), same
  2-epoch continuation schedule and LRs (head 3e-4 / encoder 5e-5) as E-012/E-014, so the only
  variable is the new data.
- **Controls**: (a) the same audio with human labels where available — the upper bound;
  (b) the teacher's error rate on a held-out 1 h of the new audio, for context.
- **Success**: ETA on T **≥ +5 points** with the WER difference CI not worse than 0; secondary
  readout = term-level accuracy on the 18 missing terms (currently 2/41 overall for E-014).
- **Cost**: Parakeet labels 100 h in ≈3.3 h; fine-tune ≈3.5 h; no model-size change.
- **Artifacts**: `runs/E-015/{config.yaml,train.log.jsonl,eval/}`; new manifest + mel shards under
  `data/`.

---

## Addendum (2026-11-09) — the baseline moved, and the target is narrower

This plan was written before E-017 added the S1+S2 gold (41.87 h, x50) and before 0.7.0. The
"5%" baseline above is E-014's, and it is obsolete. Measured on the **0.7.0** hypotheses
(`runs/E-024/eval/gate/t600.hyp.jsonl`), on the 600 recorded T clips, by provenance of each
annotated English term:

| Term group | Occurrences | Correct | Accuracy |
|---|---|---|---|
| Present in the natural training text (CV + MLS + spliced) | 817 | 439 | **53.7%** |
| Present **only in the gold** (R-S1/R-S2) | 179 | 123 | **68.7%** |
| **Never seen** in any training text (natural or gold) | 47 | 8 | **17.0%** |

The sanity check holds: overall term accuracy is 570/1043 = **54.65%**, against the card's ETA
54.17. On the **18 terms** that were added to the gold on 2026-10-24, 0.7.0 gets **20/27
utterances = 74.1%**, not 5%. **The gold already fixed the original target.**

What remains is a different set: **44 distinct terms, 47 occurrences in 47 T utterances, that
appear in no training text at all** (`airpods, baskin, bitbucket, cadbury, charger, checkout,
chromecast, delivery, domain, dyson, exam, folder, hosting, invision, ipad, kfc, logout, mondays,
oral, popup, prototype, reset, restart, roomba, scaleup, screenshot, selfie, sharing, shutdown,
shutterstock, skills, skyscanner, smartwatch, solving, starbucks, startup, subway, swipe,
tripadvisor, twinings, unfollow, upload, wallpaper, website`). The model gets **8/47 = 17%**.

### Revised expectation (important)

The 47 never-seen occurrences are **4.5%** of all 1,043 annotated term occurrences. Fixing only
them cannot buy +5 ETA:

- at the teacher's 78% (37/47): **+2.8 ETA**;
- at 100% (47/47): **+3.7 ETA** (hard ceiling of this intervention alone).

The original success criterion (**ETA ≥ +5**) is therefore unreachable by design if the new audio
covers only these terms. Revised:

- **primary**: rare-term accuracy on the 47 utterances from **17% to ≥50%** (target ≥70%);
- **secondary**: ETA **+2 to +4**, WER not worse than +0.5 (CI);
- **regression check**: the 18 gold-covered terms must stay **≥70%** (currently 74.1%).

### Revised design: gold, not Parakeet

The gold path is cheaper and better than 100 h of pseudo-labelling: the model already learns from
gold (the 18 terms went 5% → 74%), the labels are human, and no teacher-noise ceiling applies.
Parakeet remains the fallback if the gold addendum does not transfer.

- **Data**: ~100 new sentences, **50 for S1 and 50 for S2**, distinct wording, each covering all 44
  terms once per speaker (plus six two-term sentences per speaker). Same canonical script rules as
  R-S1/R-S2; reviewed and recorded by the author.
- **Mix**: add the new gold to E-017's 589.5 h mixture with **x20 oversampling** (~1.7 h of
  effective audio for 100 sentences at ~3 s), not x50: the existing gold is 41.87 h and the new
  sentences are highly repetitive, so x50 would overfit 100 templates.
- **Recipe**: continue from `runs/E-024/checkpoints/ckpt_best.pt` (0.7.0), 1 epoch (13,221 steps,
  ~1 h 50) with the E-017 LRs (head 3e-4, encoder 5e-5), then evaluate. One epoch, not two: the
  intervention is small and the baseline is already trained.
- **Readout**: the 47 rare-term utterances (primary), T-600/S3 clean WER, FLEURS-IT, CV-IT test;
  paired bootstrap as always.
- **Kill**: rare-term accuracy <30%, or clean WER worse by >0.5 with a CI excluding 0.

### Cost

100 sentences: ~15–20 min of recording. Build + fine-tune: ~2 h. Evaluation: ~15 min. **No
teacher, no 100 h labelling, no new downloads.**
