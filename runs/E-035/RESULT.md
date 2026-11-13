# E-035 — RESULT: the last 293,711 bytes buy a small, real dev gain

> Executed 2026-11-13 from the E-028 checkpoint, decode-only. Pre-registered in `PLAN.md` before
> the build. **The arm advanced on dev under the pre-registered rule**, but the later product
> readout (`READOUT.md`) is a **tie and the candidate is not adopted**: 0.8.1 remains the release.
> This report keeps the dev result as measured, not rewritten after the readout.

## Verdict

**`B1_budgeted` advances.** Adding **73,427 single-count trigrams** to the shipped
`(1,2)` + term-×3 LM — ranked by the frequency of their preceding bigram and fitted exactly into
the 20 MB package — lowers dev macro WER from **22.8494%** to **22.7849%**: **−0.0645 pp**, paired
95% CI **[−0.1144, −0.0134]**. Pooled WER agrees (**−0.0627 pp**, CI [−0.1082, −0.0181]), both
languages move the same way, and the binary fits the budget with **3 B** to spare. The
pre-registered rule required a lower point estimate, a paired CI upper bound below zero and a
package at or below 20,000,000 B; all three hold.

This is a **dev advance, not an adoption**: the product readout on T-600, T-S3, FLEURS-IT/EN, the
47 rare rows, the natural set and the noise pool — plus browser parity and latency — is a separate
step and was **not run here**.

## Readout (full dev, 3,939 clips / 10.48 h, decoder E-030 frozen)

| Arm | Trigram entries | Dev macro | Δ vs control (pp) | 95% CI (pp) | IT | EN | Pooled WER | Binary B | Package B |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `A3_shipped` | 802,437 | 22.8494 | — | — | 27.879 | 17.820 | 22.1421 | 5,381,992 | 19,706,289 |
| `B1_budgeted` | 875,864 | **22.7849** | **−0.0645** | **[−0.1144, −0.0134]** | 27.802 | 17.768 | **22.0794** | 5,675,700 | **19,999,997** |

Per-language deltas: **IT −0.0767 pp**, **EN −0.0522 pp**; pooled **−0.0627 pp**
[−0.1082, −0.0181]. No language regresses.

## The comparison is genuinely paired and drift-free

- The control was **rebuilt from the frozen E-031 corpus** in this run and reproduces the shipped
  binary **byte for byte**: **5,381,992 B**, sha256 `3be1177b…f16f1`. The build would have stopped
  before decoding otherwise.
- The control's 3,939 hypotheses are **identical to E-034's retained `A3_shipped` file**
  (`0` differing clips), so the harness did not drift between the two experiments.
- Both arms share the checkpoint (`runs/E-028/checkpoints/ckpt_best.pt`), the decoder (beam 12 /
  top-k 10 / LM 0.55 / bonus 2.7), the `MelDataset`, the batch plan and the session; only the LM
  binary differs.
- Paired per-utterance bootstrap within language, 1,000 resamples, seed 0.

## What the 73,427 entries did

- They are **4.97%** of the 1,477,990 extra trigram entries E-034 measured at −0.598 pp, and they
  capture **10.8%** of that dev gain. As a rough cross-arm ratio, not a decomposition, the
  frequency ranking is **better than proportional**: single-count continuations of frequent
  contexts are unusually valuable per byte.
- **442 of 3,939 hypotheses changed** (11.2%): **170 improved**, **116 worsened**, 156 changed
  wording without changing the error count. Net **−59 errors** (20,822 → 20,763) over 94,038
  reference words; **IT −31** errors on 40,407 words and **EN −28** on 53,631. The gain is broad,
  not one outlier clip.
- The selection fell at contexts occurring **≥ 698** times and touched **1,359** contexts. No new
  trigram context was created, so every added entry cost exactly 4 B:
  `5,381,992 + 4 × 73,427 = 5,675,700 B`, package **19,999,997 B**.
- Build **8.8 s**; decodes **243.8 s** (control) + **255.4 s** (arm).

## What this does and does not establish

- **Does**: inside the 20 MB ceiling, the headroom is not dead weight. A frequency-ranked slice of
  single-count trigrams is worth a small but statistically supported dev gain, and the arm is a
  shippable-size candidate (`lm_bpe3.bin` sha256 `458441ab…`).
- **Does not**: say anything about the product sets, browser load time/memory/latency, or the
  int8 acoustic path. Dev is MLS-IT/LS-EN read speech that has informed every previous selection
  round, so this is a selection signal, not a generalization claim. Dev has no `en_spans`, so ETA
  is not available here; that is a property of the set, not a missing run.
- The **3 B** headroom is exact arithmetic, not a safety margin: any additional required asset
  would force the arm to be trimmed before release.
- The ranking is a heuristic, not entropy pruning; a rank-all/swapping arm, four-grams, a different
  binary layout or a lower/higher cap are separate, unexecuted questions.

## Product readout (executed 2026-11-13)

The single readout pre-registered in `READOUT-PLAN.md` was run after this result: T-600, T-S3,
FLEURS-IT/EN, the rare 47, the term groups, noise and same-logits JS/Python parity, both arms on
the int8 0.8.1 assets. **All seven primary product metrics tie** (T-600 WER +0.0287 pp, T-S3 WER
−0.0852 pp, FLEURS-IT +0.0062 pp, EN −0.0437 pp, rare aligned ETA 38/81 in both, T-600 ETA
identical); natural/gold term counts are identical; the only absolute-gate miss is `noise_music`
(+2 words on 120 clips); decoder parity is exact. With no significant product gain the adoption
rule fails and **`B1` is not shipped: 0.8.1 remains the release**. The full numbers and the
interpretation are in [`READOUT.md`](READOUT.md).

## Artifacts

`runs/E-035/` — `PLAN.md`, `build_budgeted.py`, `build.log`, `build.json` (control and arm sizes,
hashes, entry counts, selection stats), `dev_sweep.py`, `dev_sweep.log`,
`dev/budgeted-1/{A3_shipped,B1_budgeted}/{dev.hyp.jsonl,dev.metrics.json}` +
`dev/budgeted-1/sweep.json`, `decide.py`, `decision.json`, this report, and the `EXPERIMENTS.md`
entry. The pickles and browser binaries are reproducible from the pinned corpus and **gitignored**;
they remain on this machine.

Reproduction order: `uv run python runs/E-035/build_budgeted.py`, then
`uv run python runs/E-035/dev_sweep.py --name budgeted-1`, then
`uv run python runs/E-035/decide.py`. No training, no download, 0 USD.
