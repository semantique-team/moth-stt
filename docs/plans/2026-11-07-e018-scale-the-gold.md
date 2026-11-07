# Plan — E-018: scale the gold (the third speaker and more sentences)

> Written 2026-11-07, after E-017 (D27) and E-019 (D28). **Status: not started — blocked on S4.**
> This plan is a handoff: it says what to run, in which order, and what would make it a failure,
> so the run can start the day S4's audio lands. Live state: `STATUS.md`.

## Why this experiment exists

E-017 answered the question it was built for: gold works **and** generalises to a speaker who never
contributed audio (S3: WER −6.13 [−8.68, −3.51], ETA +4.79 [+0.51, +9.06]). The remaining question
is not "does it work" but **"how far does it scale"**, and the evidence says the answer is more
speakers, not more epochs:

| Evidence | Number |
|---|---|
| E-016, one gold speaker (S1) | T-600 38.50 WER / 36.05 ETA |
| E-017, two gold speakers (S1+S2) | T-600 **32.77 / 42.86** (Δ −5.73 WER [−7.14, −4.26]) |
| Gain on the held-out speaker, per extra speaker | ~6 WER points |
| E-019 (the same gold on Whistle's decoder) | **+2.30 WER on S3** — the effect does not transfer across architectures |
| Gold share of the E-017 mixture | 50.2 min × 50 = 41.9 h = **7.1%** of 589.5 h |
| ETA on S3 vs Whisper forced-`it` | −2.82 [−11.65, +6.56] → **tie, the mixed gate still fails on ETA** |

So there are two separate open questions and E-018 must answer **only the first**:

1. **Scale** — does a third speaker plus more sentences per speaker keep paying, or does the gain
   flatten at ~6 points per speaker? (primary)
2. **ETA** — does more gold finally move the ETA gate on a held-out speaker? (secondary, and it is
   the half of the mixed gate that has never been won)

## The blocker, precisely

S4 must record T and A. Before that, its 200 sentences need the author's review, and two of them
carry a decision the audit flagged on 2026-10-24:

- `T-0714` "ho aperto un ticket per il bug del login" — category `single_word`, three English words
- `T-0778` "il workspace è condiviso con il team" — category `single_word`, two English words

Either re-categorise them `switch` (accepting a 2-row deviation from the T quotas) or keep only the
primary term annotated. **No recorded data is affected**: both are in the unreviewed S4 block.

An S4 **R** script does not exist yet. `recordings/generate_scripts.py` refuses `--kind R` for any
speaker other than S1 and S2 (`script_path`, line ~99: *"S3/S4 are held out from training"*), which
is a **code invariant that E-018 must intentionally change** — see Step 2.

## Design

**One variable against E-017: the gold set.** Everything else — init E-014 best, natural + spliced
mix, head 3e-4 / encoder 5e-5, warm-up 500, 2 epochs, bf16, truncated input, augmentation full,
batch 192 s / 32 clips — stays byte-identical to `runs/E-017/config.yaml`. A different recipe would
confound "more gold speakers" with "different training".

**Arms.** Run them in this order and stop at the first one that answers the question:

| Arm | Gold | Question it answers | Cost |
|---|---|---|---|
| **G3** | R-S1 + R-S2 + R-S4 (three speakers) | does a third speaker pay like the second? | ~3 h 30 |
| **G3+** | G3 + ~200 more sentences per speaker (R-S1b, R-S2b, R-S4b) | is it the speaker count or the amount of speech? | +~1 h recording, ~3 h 30 training |
| **G3×** | G3 at ×100 repetition instead of ×50 (gold ≈ 14%) | is it repetition or diversity? | ~3 h 30, no new audio |

G3 is the default; G3+ only exists if the author records extra sentences. **G3× is a trap to name
explicitly**: E-016/E-017 already oversample ×50, so "more gold exposure" has been tested and is not
the same lever as "more gold speakers". Run it only if G3's gain is smaller than E-017's and the
explanation must be separated.

**Data rules, non-negotiable (unchanged from every other run in this repo):**

- Train on R-S1 + R-S2 + R-S4 only. **T and A are never read** for training, validation, token
  mining or hyper-parameter choice.
- Validation stays `mls_it_dev` (as in E-017). S4's gold is training data, not validation data.
- The honest test speaker is **S3** (`runs/E-017/manifests/T-S3.jsonl`), which still contributes no
  audio to any training run. Report T-600 too, with the caveat that three of its four speakers are
  now training speakers — **state the ratio explicitly** in the entry.
- S4's T (200 clips) is a **seen** speaker in G3 and must be reported as such, never mixed into the
  held-out claim.

**The metric that decides.** Primary: **T-S3 WER**, paired bootstrap against E-017, 1000 resamples,
seed 0, through `stt.eval`. Secondary: T-S3 ETA, then the ship-gate sets (FLEURS-IT/EN, CV test,
noise with VAD). A gain that appears only on T-600 is not a gain: it is a training speaker.

## Steps

1. **Author: review the S4 block** (`uv run recordings/generate_scripts.py review --kind T`,
   200 rows), decide `T-0714`/`T-0778`, then `mark-reviewed`.
2. **Author + agent: relax the R-speaker invariant** in `recordings/generate_scripts.py` so S4 can
   get an R script, and generate it (~600 sentences, same categories and quotas as R-S1/R-S2, no
   reuse of T/pool sentences — the importer already enforces dedupe). This is the one code change
   E-018 needs, and it must be a separate commit with a test asserting S3 still *cannot* have one
   (S3 is the held-out speaker for E-018 as well).
3. **Author: S4 records T (200), A (60) and R-S4 (~600)** — roughly 2 h of sessions with the guided
   recorder; `record.py` resumes automatically.
4. **Agent: data build** — manifests, `build_mel` for R-S4, then a gold set that repeats the three
   sets at the same ×50 factor as E-017 (`runs/E-017/prepare_gold.py --sets R-S1,R-S2,R-S4`).
   Verify: 0 missing ids, 0 overlapping ids, total gold hours and share of the mixture recorded in
   `config.yaml` exactly like E-017's comment block.
5. **Agent: smoke 50 steps**, then the full run with `caffeinate -i`, checkpoint every 10 min.
6. **Agent: evaluation** via a new `runs/E-018/eval.sh` modelled on `runs/E-017/eval.sh` — greedy
   and beam+LM on T-600, the S3 slice, S4's slice, FLEURS-IT/EN, CV test, noise with VAD — then
   every paired difference against `runs/E-017/eval/`.
7. **Agent: documents** — `EXPERIMENTS.md` entry E-018, a `docs/11` decision (D29) if the scaling
   result is worth keeping, `STATUS.md`, and the model card / release rebuild **only if** the gate
   moves.

## Acceptance / kill criteria

- **Success (scale confirmed)**: T-S3 WER paired CI against E-017 excludes 0 **and points down**.
  ETA is reported but does not decide.
- **The good outcome**: the same, *and* the S3 ETA CI excludes 0 upward — that would be the first
  time the ETA half of the mixed gate is won on a held-out speaker.
- **Negative result worth keeping (plateau)**: the S3 CI covers 0 while T-600 improves. That means
  the gain is speaker adaptation again, and it belongs in `docs/11` §3 next to E-019.
- **Kill the scaling story**: if a third speaker adds < 1 WER point on S3, stop after G3; do **not**
  run G3+ or G3×. Write the plateau, re-scope the gate (`docs/11` §7.3), and stop buying gold.
- **Never**: compare a number across two different manifests; report T-600 alone as the result;
  tune epochs/LR against S3.

## Risks and open questions

- **Three speakers, one microphone.** S1 and S3 share the Mac's mic; S2 used an external device. If
  S4 records on yet another device, the gold spans three channels and the S3 test still shares one
  of them. Measure S4's channel and state it; the channel confound was already noted in E-017 and
  is not removed by more speakers.
- **Diminishing returns are the default hypothesis.** One speaker → two gave ~6 WER points on S3.
  Two → three will probably give less, because E-016's gain was already known to be partly
  speaker adaptation (D27, §3.2). Budget for a plateau; that is a legitimate result.
- **Recording time is the real cost.** ~2 h of the author's time for S4's R script, plus the review.
  The training is a single unattended 3.5 h run.
- **E-019 says the gold is architecture-bound.** Do not cite E-018 as evidence that "gold data
  helps speech models" in general; it is evidence about **this** CTC model, and `docs/11` §3.4 must
  be cited next to it.
- **The ETA gate may still not move.** E-017 bought +4.79 ETA on S3, and the mixed gate needs more.
  If G3's ETA gain is again ~5 points, the gate remains a tie and the decision in `docs/11` §7.3
  (re-scope, ETA as secondary) becomes the honest next move rather than a fourth gold run.

## Artifacts (to create)

```text
runs/E-018/
  config.yaml             # E-017's config with the gold arm swapped, one variable
  prepare_gold.py         # thin wrapper over runs/E-017/prepare_gold.py --sets R-S1,R-S2,R-S4
  train.log.jsonl
  eval.sh
  eval/                   # metrics + hyp per set
  diffs/                  # paired vs E-017 on every set
  RESULT.md
```

Checkpoints, mel shards and raw audio stay out of git, as always.

## Explicitly out of scope

- Re-tuning the recipe (LR, epochs, augmentation, batch shape): that is a different experiment.
- Distillation (E-015) and noise augmentation: independent, still blocked on their own decisions.
- A fourth speaker: decide after G3, not now.