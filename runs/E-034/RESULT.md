# E-034 — RESULT: trigram coverage is a real dev lever, and it does not fit

> Executed 2026-11-12 from the E-028 checkpoint, decode-only. Pre-registered in `PLAN.md` before
> the run. **No arm advances.** Dev only: T, T-S3, FLEURS, the noise pool and the browser path
> were not read, and nothing is published or recommended for shipment.

## Verdict

1. **Relaxing trigram pruning helps dev for real**: `(2,1)` is **−0.598 pp** macro WER and
   `(1,1)` **−0.644 pp**, both with paired CIs excluding zero. The cheap intermediate arm is not
   a budget escape: `(2,1)` costs **12,034,004 B** (package **26,358,301 B**) for about 93% of the
   `(1,1)` gain. Both exceed the **20 MB** ceiling and the **25 MB** tier (binary limit
   10,675,703 B).
2. **Term weighting is saturated at ×3**: raising the 143 term-containing sentences to ×5 on
   unchanged pruning is a **tie** (+0.0019 pp, CI [−0.0083, +0.0120]) and changes coarse-probability
   bytes without changing size (5,381,992 B, package unchanged).
3. Nothing advances under the pre-registered rule (lower macro, CI upper < 0, binary
   ≤ 5,675,703 B). The useful output is the **budget arithmetic for a follow-up**, not a candidate.

`(1,1)` at **12,313,440 B / 26,637,737 B package** reproduces E-023's rejected arm exactly,
independently of that run; E-023 is the prior evidence that the gain also shows on some gate sets
(T-600 −0.77, FLEURS-IT −0.72). E-034 does **not** re-read those sets.

## Readout (full dev, 3,939 clips / 10.48 h, decoder E-030 frozen)

| Arm | Bigrams / trigrams / weight | Dev macro | Δ vs control (pp) | 95% CI (pp) | IT | EN | Binary B | Package B |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `A3_shipped` | ≥1 / ≥2 / ×3 | 22.8494 | — | — | 27.879 | 17.820 | 5,381,992 | 19,706,289 |
| `N1_2_1` | ≥2 / ≥1 / ×1 | **22.2511** | **−0.598** | [−0.724, −0.457] | 27.201 | 17.302 | 12,034,004 | 26,358,301 |
| `N2_1_1` | ≥1 / ≥1 / ×1 | **22.2054** | **−0.644** | [−0.768, −0.503] | 27.094 | 17.316 | 12,313,440 | 26,637,737 |
| `N3_1_2_terms5` | ≥1 / ≥2 / ×5 | 22.8513 | +0.002 | [−0.008, +0.012] | 27.871 | 17.831 | 5,381,992 | 19,706,289 |

The control reproduces the retained E-031 sweep **byte for byte**: the 3,939 `dev.hyp.jsonl` lines
are identical and the macro is 22.8494, so this sweep introduces no harness drift.

`N1` vs `N2` differ only in bigram pruning: keeping all bigrams adds 279,436 B and 0.046 pp. Both
extra-coverage arms keep the same 2,280,427 trigrams, i.e. **+1,477,990 trigram entries over A3
for +6.65 MB, about 4.5 B per entry**.

## What this does and does not establish

- **Does**: on standard Italian/English read dev, more LM coverage is worth ~0.6 points, and the
  shipped term weight is not under-tuned. The dev gain is far outside E-031's own arm-to-arm band
  (−0.011 to −0.097 pp), so it is not noise on this set.
- **Does not**: transfer to the product. Dev is MLS-IT and LibriSpeech-EN read speech; the mixed
  T/FLEURS readouts of a larger LM were not run here, and E-023's gate deltas came from a
  different corpus and decoder state. A 12 MB LM also changes browser load time, memory and
  first-token latency, none of which is measured here.
- A dev gain is a selection signal, not a product claim, and this set has informed every previous
  selection round.

## Follow-up the numbers point to

Only **293,711 B** of headroom remains, which at ~4.5 B/entry buys about **65,000 of the
1,477,990 extra trigram entries (4.4%)**. A budgeted arm is therefore a selection problem, not a
pruning-threshold problem: keep `(1,2)` and add single-count trigrams ranked by something
defensible (e.g. the count of their preceding bigram, or token-level coverage of rare terms) until
the binary hits the byte limit. That arm must be pre-registered against A3 on dev and, if it
advances, read once on T/FLEURS/noise before any claim. It is **not** authorized or executed here.

## Artifacts

`runs/E-034/` — `PLAN.md`, `build_variants.py` + `build.json` (sizes, hashes, entry counts; the
binaries and pickles are reproducible and gitignored), `dev_sweep.py`,
`dev/dev-only/{A3_shipped,N1_2_1,N2_1_1,N3_1_2_terms5}/{dev.hyp.jsonl,dev.metrics.json}`,
`dev/dev-only/sweep.json`, `decide.py` + `decision.log`, `dev_sweep.log`.

Reproduction order: the control loads `runs/E-031/build/frozen_1_2_terms.pkl.gz`, which is
gitignored — rebuild E-031 first (`uv run python runs/E-031/build_lm.py`, corpus already present
or reproduced with `runs/E-031/reproduce_corpus.py`), then `runs/E-034/build_variants.py`, then
`runs/E-034/dev_sweep.py --name <new-name>`. `decide.py` needs no audio.

Cost: build 3 variants 18 s, four dev decodes ~16 min, no training, no download, 0 USD.
