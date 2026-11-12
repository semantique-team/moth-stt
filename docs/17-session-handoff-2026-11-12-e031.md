# 17 - Session handoff, 2026-11-12: E-031 LM candidate, E-032 rejected

**Purpose.** Resume document for the second half of 2026-11-12, after the 0.8.0 handoff
([doc 16](16-session-handoff-2026-11-12.md)). It records the two experiments, the published
**0.8.1 release**, the browser and noise checks, the new mixed-dev tooling and the remaining
blocker. Read this first, then `STATUS.md`, then the experiment reports.

## TL;DR

- **Published model is 0.8.1** (HF `semantique-team/moth-stt`, rev `1257327e…`;
  Space `semantique-team/moth-demo`, rev `9dfe70df…`, `RUNNING`), released 2026-11-12.
- **0.8.1 is a decode-only update.** Same acoustic int8 graph as 0.8.0; only `lm_bpe3.bin` and the
  version changed. Package **19,706,289 bytes**. 0.8.0 remains the historical comparison.
- **E-031 adopted:** frozen-corpus LM, all bigrams + trigrams ≥2, term sentences ×3.
  - T-600 **24.08 / 55.23** (0.8.0 24.71 / 54.17; WER −0.63 [−1.11, −0.20], ETA +1.06 [+0.09, +2.04]).
  - T-S3 **24.96 / 51.27** (ties vs 25.21 / 50.14).
  - FLEURS-IT **20.16**, EN **49.81** (ties).
  - Rare aligned ETA **46.9% (38/81)** vs 28.4% (23/81), **+18.5 [+9.6, +31.0]**; presence 24/47.
  - Natural **432/817** (gate ≥431), gold 127/179, 16-term 19/27.
  - **The 0.8.0 rare-gate exception is closed.**
- **E-032 rejected:** 50 new S1 rare-term sentences (six pruned terms ×4), gentle fine-tune from
  E-028. Rare 32.1% (0.8.0 LM) / 51.9% (E-031 LM), but T-S3 26.75 / 47.61, FLEURS worse, natural
  424 / 415; five pre-registered gates fail. The data remain reusable.
- **Browser:** real Chromium acceptance **4/4** on 0.8.1; JS↔Python **decoder parity exact**
  on identical logits. The 3 s fixture differs by 2 words with **both** the 0.8.0 and 0.8.1 LM, so
  the residual is the browser acoustic path, not the new LM.
- **Noise:** full E-024 curve re-run with the new LM on the same checkpoint, decoder, 100 S3 clips
  and seed; **all conditions tie**, mean 5–30 dB **29.47 vs 30.05**, no regression.
- **Blocker:** independent natural mixed-dev + a new-voice rare readout. **No new speaker is
  available**, so this cannot proceed yet. Do not publish 0.8.1 as if it had independent evidence.

## Repository state

- E-031/E-032 work is committed and pushed through `e16aadc`; the 0.8.1 publication record and
  the noise/publish scripts are committed with this handoff.
- `ruff check .` clean; `pytest -q` **888 passed, 1 skipped**.
- **HF and Space now serve 0.8.1**: model rev `1257327e…` (release `f40b691d…`, card
  `455267ca…`), Space rev `9dfe70df…`, `RUNNING`; remote config `0a5ca224…`, model `7bbfad7e…`,
  LM `3be1177b…`.
- New/changed tracked files: `recordings/import_manifest.py`, `tests/test_import_manifest.py`,
  `recordings/scripts/R-S1.jsonl` (50 appended rows), `recordings/manifests/R-S1.jsonl` (+50),
  `recordings/manifests/R-S1-terms3.jsonl`, `recordings/manifests/recorded_text.sha256.json`,
  `docs/guides/mixed-dev-and-quality-experiments.md`, `EXPERIMENTS.md`, `STATUS.md`, `docs/11`,
  `docs/13`, `docs/16`, this file, `runs/E-024/noise_eval.py`, `runs/E-022/publish_space.py`,
  `runs/E-024/space/README.md`, and `runs/E-031/`, `runs/E-032/`.
- Gitignored/local: checkpoints, `runs/E-031/release/`, `runs/E-032/release/`, mel shards, audio.

## E-031 — intermediate LM inside 20 MB

**Question:** E-023 showed `(1,1)` is better but 12.31 MB. Does `(1,2)` (all bigrams, trigrams
≥2) recover the gain in budget, especially now that the frozen corpus contains the E-015/E-028
term sentences the 2026-10-30 LM never saw?

**Method:** `stt.data.freeze_lm_corpus` on CV/MLS/LS train + R-S1/R-S2/pool, with every T/A
script and manifest and every IT/EN dev/test manifest as explicit exclusions. 186,003 sentences,
939 duplicates collapsed, 36,065 exclusions, no overlap error. Arms: shipped control, `(2,2)`
rebuild, `(1,2)`, `(1,2)` + term sentences ×3. Selection on full dev (3,939 clips), decoder fixed
at 0.55/2.7. Bytes first.

| Arm | Dev macro | Δ vs control | 95% CI | Binary |
|---|---:|---:|---|---:|
| shipped | 22.9463 | — | — | 5,116,464 |
| frozen (2,2) | 22.9352 | −0.010 | [−0.031, +0.015] | 5,099,304 |
| frozen (1,2) | 22.8541 | −0.084 | [−0.140, −0.035] | 5,378,740 |
| **frozen (1,2) + terms ×3** | **22.8494** | **−0.091** | **[−0.147, −0.040]** | **5,381,992** |

All nine pre-registered gates pass; rare ETA and T-600 WER improve with CIs excluding zero. Report:
`runs/E-031/RESULT.md`; plan: `runs/E-031/PLAN.md`; frozen corpus + receipt:
`runs/E-031/frozen/`; build: `runs/E-031/build.json`; gate: `runs/E-031/gate.json`.

## E-032 — S1 acoustic addendum (rejected)

50 reviewed S1 sentences (`R-S1-0692`–`0741`), 2.8 min, ×20 = 0.93 h; six pruned terms ×4; gentle
fine-tune from E-028, 13,314 steps, dev **31.06%** (E-028 31.01%), final checkpoint selected.

| vs baseline | Rare aligned ETA | T-600 | T-S3 | FLEURS-IT/EN | Natural |
|---|---|---|---|---|---|
| 0.8.0 LM | **32.1% (26/81)**, +3.7 [0.0, +9.0] | 25.23 / 53.21 | **26.75 / 46.76** | 20.66 / 50.47 | **424** |
| E-031 LM | **51.9% (42/81)**, +4.9 [0.0, +12.0] | 24.83 / 53.88 | **26.41 / 47.61** | 20.54 / 50.28 | **415** |

Five pre-registered gates fail; the rare CI touches zero. Report: `runs/E-032/RESULT.md`; data
provenance: `runs/E-032/DATA.md`; plan/config/scripts in `runs/E-032/`.

## 0.8.1 release (published 2026-11-12)

- Bundle: `runs/E-031/release/` (0.8.0 assets + new LM), **19,706,289 B**, valid under
  `stt.release --budget-bytes 20000000` (293,711 B margin).
- Model sha **`7bbfad7e…`** (unchanged); LM sha **`3be1177b0ca9f18343efb8d00f0d96365226595c9de32d992720aa4e0b0f16f1`**
  (5,381,992 B); config version **0.8.1**; remote config `0a5ca224…`.
- Card: `runs/E-031/MODEL_CARD.md`; record: `runs/E-031/release-0.8.1.json` (`published: true`,
  HF/Space revisions recorded).
- HF model rev `1257327e…` (release `f40b691d…`, card `455267ca…`); Space rev `9dfe70df…`,
  `RUNNING`.
- Browser acceptance: `runs/E-031/browser-acceptance/RESULT.md` (4/4).
- Parity: `runs/E-031/browser-parity/RESULT.md`; decoder-level exact match; fixture difference
  identical with 0.8.0.
- Noise: `runs/E-031/noise-e031-lm/RESULT.md`; all conditions tie, mean 5–30 dB **29.47 vs 30.05**.

## Mixed-dev tooling

`recordings/import_manifest.py` builds a `natural_mixed_dev` / `split=dev` manifest from audio +
transcripts, enforces mono/1–30 s/non-silent/clipping, normalization, categories and `en_spans`,
refuses exact duplicates and reports near duplicates. 7 tests; documented in the mixed-dev guide.
It does not grant consent or freeze roles; those remain human steps.

## Blocked: independent data

**The one thing that blocks the next evidence level is a new speaker.** Record 2–3 new voices,
20–30 min each, natural Italian with English terms, consent and pseudonymous IDs, roles/splits
frozen before inspecting errors. Follow
`docs/guides/mixed-dev-and-quality-experiments.md`. Until then:

- do not present 0.8.1 as validated on an independent voice;
- do not tune E-032 or another recipe on T;
- do not treat the T/S3 evidence as population generalization.

## Next steps, in order

1. **Mixed-dev indipendente** as soon as a new speaker is available; then one new-voice rare
   readout. This is the missing evidence for the card's main limitation. **Blocked: no speaker.**
2. **Paper/docs:** fold E-031/E-032 into the narrative; keep E-032 as a negative result; state that
   E-031 is decode-only and that natural terms are close to the gate.

## Do not

- Re-run QAT, quantization variants or contextual bias (E-029/E-028/E-027).
- Re-run E-032 as-is on T; the data are for a future arm with independent selection.
- Call the browser fixture difference an LM regression: the 0.8.0 control shows the same diff.
- Claim independent-voice evidence for 0.8.1.
