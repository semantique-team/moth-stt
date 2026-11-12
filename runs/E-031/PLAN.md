# E-031 — an intermediate trigram LM inside the 20 MB ceiling

**Status:** pre-registered 2026-11-12, before building the frozen corpus and before any decode.
One question, no sweep of decoder weights. No training, no checkpoint change.

## Why this arm exists

E-023 measured the price of the 20 MB ceiling: keeping every bigram and trigram (`(1,1)`) is
**−0.65 dev points** and significantly better on T-600/FLEURS-IT/CV, but its browser binary is
**12.31 MB**, so the package would be 26.63 MB. Post-hoc arms between `(2,2)` and `(1,1)` were
never tested. E-031 asks whether **all bigrams + trigrams with count >= 2 (`(1,2)`)** recovers a
useful share of that gain inside the budget.

Two facts changed since E-023:

1. The shipped LM was built on 2026-10-30, **before** the 50 S1 and 50 S2 rare-term sentences
   existed. A rebuilt corpus therefore includes training-side term text the shipped LM never saw.
2. The 0.8.0 decoder is fixed at LM 0.55 / bonus 2.7 (E-030). The ceiling headroom is
   **559,240 B** (`20,000,000 - 19,440,761`), i.e. an LM binary up to **5,675,704 B**.

A byte probe on the historical text (`runs/E-031/probe_lm_bytes.py`) gives:

| Arm | Browser binary | Fits |
|---|---:|---|
| shipped (2,2) | 5,116,464 B | yes |
| **(1,2)** | **5,396,484 B** | **yes (+279,820 B)** |
| (2,1) | 12,034,620 B | no |
| (1,1) | 12,313,440 B | no |

## Frozen corpus

Built with `stt.data.freeze_lm_corpus` (the protocol in
`docs/guides/mixed-dev-and-quality-experiments.md`): CV-IT, MLS-IT, LS-EN train transcripts plus
the R-S1, R-S2 and pool scripts, **never T/A**, with every T/A script and manifest, every IT/EN
dev/test manifest, and the non-speech set named as explicit exclusions. Output and receipt are
committed under `runs/E-031/frozen/`; the corpus is deduplicated by normalized text, so this is a
corpus-controlled comparator, **not** a byte-identical rebuild of the shipped LM.

**Declared absence:** no natural mixed-dev and no independent holdout exist at this date. They are
not silently substituted; no term-quality claim rests on them in this experiment.

## Arms

| Arm | Text | Counts | Browser bytes |
|---|---|---|---:|
| **A0 control** | shipped LM (`release/lm_bpe3.bin`) | shipped | 5,116,464 |
| **A1 corpus** | frozen corpus | (2,2) | to measure |
| **A2 bigrams** | frozen corpus | (1,2) | to measure |
| **A3 terms** | frozen corpus, term rows x3 | (1,2) | to measure |

A3 repeats at build time every frozen sentence containing one of the 44 never-seen terms
(whole-word, after `stt.text.normalize`). Duplicates do not add n-gram types under (1,2) and (2,2)
beyond the same text, so A3 is expected to stay within a few KB of A2; it is rejected at the byte
gate if it does not.

## Decision rule (frozen now)

1. **Byte gate first.** Any arm whose exported `lm_bpe3.bin` makes the package exceed
   **20,000,000 B** with the other 0.8.0 assets unchanged is rejected before decoding.
2. **Dev selection.** Full dev only: `mls_it_dev` (1,248) + `ls_en_dev` (2,691), macro WER,
   decoder fixed at beam 12 / top-k 10 / LM 0.55 / bonus 2.7, fp32 E-028 checkpoint.
   An arm **advances** only if its dev macro is lower than A0 **and** the paired 95% utterance
   bootstrap CI for the WER difference excludes zero.
3. **A2 vs A3.** A3 advances in preference to A2 when its dev macro is within **+0.10** of A2
   (non-inferior on general text), because the arm exists to move rare terms; otherwise A2.
4. **One gate readout**, int8 0.8.0 + the candidate browser LM, decoded once:
   T-600, T-S3, FLEURS-IT/EN, the 47 rare rows, natural 817, gold-only. No selection on T/S3.
5. **Adoption gates**, all required: T-600 WER <= 25.21 (+0.5 vs 24.71) and ETA >= 53.67;
   T-S3 WER <= 25.71 and ETA >= 49.64; FLEURS-IT <= 20.75 / EN <= 50.31; natural >= 431;
   rare aligned ETA >= 28.4%; package <= 20,000,000 B; browser LM loads and decodes with the
   same hypotheses as Python on a frozen sample; no non-speech regression on the E-011 set.
6. **Adopt only if** all gates pass **and** at least one of {rare aligned ETA > 28.4%,
   T-600 WER < 24.71, FLEURS-IT < 20.25} has a paired CI excluding zero. Otherwise keep 0.8.0
   and record a negative.

## Cost

No training, no downloads, 0 USD. Build is seconds; dev decode is the CPU-bound part (E-023 scale:
~7 min per arm on this machine). Gate readout only for the single advancing arm.

## Artifacts

`runs/E-031/` (`PLAN.md`, `probe_lm_bytes.py`, `probe/`, `freeze.sh`, `frozen/`, `build_lm.py`,
`build.json`, `dev_sweep.py`, `sweep.json`, `gate.py`, `eval/`, `RESULT.md`).
