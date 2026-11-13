# E-036 — RESULT: the conservative fine-tune does not rescue the S1 addendum

> Executed 2026-11-13. Two arms trained from E-028, exported to int8, and read once against the
> committed 0.8.1 baseline with the pre-registered rule in `PLAN.md`. **Both arms are rejected.
> The acoustic model stays at 0.8.1; there is no candidate for the new-voice readout.**

## Verdict

**Interpretation correction (2026-11-13, no new inference):** this readout decodes cached
mel clips without VAD or segmentation. Silence emission describes that ungated acoustic/
decoder path, not the full shipped runtime, which has a VAD guard. E-039 identifies the
120 silence files as one byte-identical signal repeated 120 times, not independent trials.
The historical rejection is unchanged. Music word counts below match the retained
`eval/frozen-1/readout.json` and `eval/last2-1/readout.json`; no JSON or gate was recomputed.

The protection worked where it was aimed — WER on T-600, T-S3 and FLEURS is a **tie** for both
arms — but it did not deliver the rare-term target and it introduced an ungated silence regression:

- **Neither arm reaches 42/81 rare aligned ETA**: `frozen` gets **39/81** (+1), `last2` **41/81**
  (+3), both with a paired CI whose lower bound is exactly zero. The pre-registered target was
  42/81, the point level E-032 reached; 39–41 is insufficient and is not rounded up.
- **Both arms emit `x` on all 120 silence clips**, where 0.8.1 emits nothing. Non-music
  non-empty clips jump from 146 to 272; the `noise_non_music` and `noise_music` gates fail. This
  is a deterministic emission on the repeated ungated signal, not a full-runtime measurement.
- **`last2` also loses T-S3 ETA**: 51.27 → 49.86, paired CI [−2.76, −0.28] pp, a significant
  regression. `frozen` keeps T-S3 ETA flat.

The pre-registered rule required all absolute gates, the 42/81 rare target and no significant
regression. Both arms fail at least two conditions. **0.8.1 remains the release.**

## Training And Dev

| Arm | Trainable params | Steps | Best dev (step) | Final dev | Non-finite |
|---|---:|---:|---:|---:|---:|
| `E-036-frozen` | 3,942,785 | 13,314 | 30.9890 (4,000) | 30.9955 | 0 |
| `E-036-last2` | 7,490,945 | 13,314 | 30.9683 (8,000) | 31.0172 | 0 |

Dev selection (greedy, no LM, 3,939 clips; E-028 initial decoded in the same session at
**31.0101**):

| Comparison | Δ macro (pp) | Paired 95% CI (pp) | Dev guard |
|---|---:|---|---|
| `E-036-frozen` vs initial | **−0.0242** | [−0.0806, +0.0335] | pass |
| `E-036-last2` vs initial | **−0.0381** | [−0.1048, +0.0263] | pass |

Both are dev **ties**. The guard (not worse than the initial by >0.50 pp) passes, so both arms went
to the product readout as planned. The `ckpt_best` re-decode differs from the trainer's in-run
value by 0.003–0.004 pp (MPS nondeterminism on a few utterances); the paired comparison uses the
re-decoded hypotheses from one session, not the trainer's aggregates. The KD term stayed at ~2–6%
of the CTC loss; the teacher was E-028, hash `01b83cc3…`.

## Product Readout (int8, A3 LM, versus 0.8.1)

### `E-036-frozen`

| Metric | 0.8.1 | Frozen | Δ (pp) | 95% CI (pp) |
|---|---:|---:|---:|---|
| T-600 WER | 24.0826 | 24.3406 | +0.2580 | [−0.1985, +0.7048] |
| T-600 ETA | 55.225 | 55.417 | +0.192 | — |
| T-S3 WER | 24.9574 | 25.1278 | +0.1704 | [−0.4198, +0.7519] |
| T-S3 ETA | 51.268 | 51.268 | 0.000 | [−1.136, +1.176] |
| FLEURS-IT | 20.1621 | 20.2869 | +0.1247 | [−0.0192, +0.2782] |
| FLEURS-EN | 49.8077 | 49.9126 | +0.1049 | [−0.1473, +0.3267] |
| Rare aligned ETA | 38/81 | **39/81** | +1.23 | [0.00, +4.35] |
| Rare target-only | 24/47 | 25/47 | +1 | — |
| Natural terms | 432/817 | **435/817** | +3 | — |

### `E-036-last2`

| Metric | 0.8.1 | Last2 | Δ (pp) | 95% CI (pp) |
|---|---:|---:|---:|---|
| T-600 WER | 24.0826 | 24.2259 | +0.1433 | [−0.3178, +0.6592] |
| T-600 ETA | 55.225 | 55.321 | +0.096 | — |
| T-S3 WER | 24.9574 | 25.2129 | +0.2555 | [−0.3431, +0.8410] |
| T-S3 ETA | 51.268 | **49.859** | **−1.408** | **[−2.762, −0.280]** |
| FLEURS-IT | 20.1621 | 20.2495 | +0.0873 | [−0.1200, +0.2882] |
| FLEURS-EN | 49.8077 | 49.8426 | +0.0350 | [−0.2856, +0.3543] |
| Rare aligned ETA | 38/81 | **41/81** | +3.70 | [0.00, +8.64] |
| Rare target-only | 24/47 | **27/47** | +3 | — |
| Natural terms | 432/817 | **434/817** | +2 | — |

### Noise: the new failure

| Group | 0.8.1 non-empty / words | Frozen | Last2 |
|---|---:|---:|---:|
| silence (120) | **0 / 0** | **120 / 120** (`x`) | **120 / 120** (`x`) |
| room tone (97) | 97 / 188 | 97 / 194 | 97 / 190 |
| MUSAN (120) | 49 / 70 | 55 / 78 | 55 / 71 |
| music (120) | 84 / 299 | 86 / 293 | 88 / 305 |
| non-music total (337) | 146 / 258 | **272 / 392** | **272 / 381** |

The candidate emits the single token `x` on **every** silence clip. The frozen arm shows it too,
so the behavior comes from the trainable head stack, not from encoder drift. The `noise_non_music`
and `noise_music` gates fail for both arms.

## Gates

| Check | Frozen | Last2 |
|---|---|---|
| package 19,706,289 B ≤ 20,000,000 | pass | pass |
| T-600 WER/ETA, T-S3 WER, FLEURS, natural | pass | pass |
| T-S3 ETA paired non-regression | pass | **fail** |
| rare ≥ 42/81 | **fail (39)** | **fail (41)** |
| noise non-music / music | **fail** | **fail** |
| decoder parity exact | pass | pass |
| no significant primary regression | pass | **fail** |

## What This Establishes

- **Freezing plus replay plus a weak KL teacher preserves WER but not silence behaviour.** The
  output stack learns to emit `x` on silence from the S1 fine-tune; the frozen encoder proves the
  encoder is not the source of this observed ungated emission. This is not a test of the
  shipped VAD guard; weight anchoring alone does not preserve the ungated silence behavior.
- **The rare target is not reached.** `last2` gets closer (41/81, target-only 27/47) but still
  misses 42/81, and it buys that with a significant T-S3 ETA loss. The six acoustically pruned
  terms still need a change that this recipe does not provide.
- **E-032's rare gain is not recovered by protection.** E-032 reached 42/81 only with a full
  fine-tune that regressed T-S3/FLEURS/natural; E-036 avoids those WER regressions but loses the
  rare gain and adds a silence hallucination.
- **0.8.1 is frozen.** No candidate, no new-voice readout, no release change.

## Limits

One seed per arm, one KD weight/temperature, one S1 mixture. T and S3 are reused evidence; two
thirds of T are S1/S2, so even a rare pass would not establish a new voice. The noise set is the
E-011 pool; the readout is one pass and was not used to tune anything. The browser acoustic path
was not measured; only same-logits decoder parity (exact) and the package size were checked.

## Artifacts

`runs/E-036/` — `PLAN.md`, `frozen/` and `last2/` configs, `train.log.jsonl`, `train.out`,
checkpoints (gitignored), `dev/select-1/`, `release-<arm>/` exports (gitignored), `export-<arm>.log`,
`eval/<arm>-1/` candidate hypotheses/metrics, `terms.json`, `parity.json`, `readout.json`,
`readout-<arm>.log`, and this report. Reproduction: `train_all.sh`, `dev_select.py`,
`stt.export` (commands in `readout.py`), then `readout.py --arm <arm> --name <arm>-1`.

Cost: ~1 h 47 per arm, two exports, two readouts; no downloads, 0 USD.
