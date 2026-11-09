# E-022 — RESULT: the decode recalibration

> Executed 2026-11-09. Plan and prediction: `PLAN.md` (written before the code change).
> Audit that motivated it: `AUDIT-PROBE.md`. Decision rule: adopted only if every WER CI moved,
> T-S3 moved at least as much as T-600, non-music noise stayed 0, and 1-thread browser RTFx
> stayed ≥ 10×.

## Verdict

**Adopted.** All four criteria passed. Shipped as **release 0.6.0**, weights unchanged.

| Criterion | Required | Measured | |
|---|---|---|---|
| (a) every set's WER CI excludes 0, improving | yes | 6/6, all CIs below 0 | **pass** |
| (b) T-S3 gain ≥ T-600 gain (not speaker adaptation) | yes | −8.77 vs −6.85 | **pass** |
| (c) non-music noise words | 0 | 0 (all 337 clips gated) | **pass** |
| (d) browser 1-thread RTFx, 20 s | ≥ 10× | **20.2×** | **pass** |

## Measured vs predicted

Prediction from `PLAN.md` (dev-only arm, frozen before the code change):

| Set | shipped | predicted | **measured** | Δ WER [95% CI] | Δ ETA [95% CI] |
|---|---|---|---|---|---|
| T-600 | 32.77 / 42.86 | 25.9 ± 1.0 | **25.92 / 52.64** | **−6.85** [−7.94, −5.72] | +9.78 [+7.68, +12.06] |
| T-S3 (held-out) | 35.43 / 35.21 | 26.7 ± 1.5 | **26.66 / 47.32** | **−8.77** [−11.13, −6.50] | +12.11 [+8.31, +16.47] |
| T-S1+S2 (training) | 31.46 / 46.66 | — | **25.54 / 55.38** | −5.92 [−7.23, −4.52] | +8.72 |
| FLEURS-IT | 26.52 | 20.5 ± 0.6 | **20.46** | −6.06 [−6.63, −5.51] | — |
| FLEURS-EN | 55.48 | 49.6 ± 0.8 | **49.63** | −5.85 [−6.55, −5.18] | — |
| CV-IT dev | 24.91 | 19.9 ± 0.8 | **19.90** | −5.01 [−5.88, −4.14] | — |

Every prediction landed inside its stated interval. The arm was chosen on dev and did not move
when it met the test sets, which is what the plan was betting on.

**Deletion/insertion shift**, T-600: deletions **12.2% → 8.3%** of reference words, insertions
**0.8% → 1.3%**. That is the mechanism, visible in the aggregate rather than argued from a story:
mass moved from deletions to insertions and the total fell.

### Controls that make the number believable

- **`shipped` arm reproduces E-017 to four decimals** (T-600 32.77, FLEURS-IT 26.52, T-S3 35.43):
  the harness was not changed by the experiment.
- **Greedy is untouched**: 37.70 on T-600, identical to E-017. `token_bonus` and `lm_weight`
  only exist in the beam path, so the fast default cannot have been affected.
- **`token_bonus=0` is a bit-exact no-op**: 0/30 real clips changed output, and
  `tests/test_decode.py` pins it. Every existing call site keeps its behaviour.
- **The int8 artifact is byte-identical** to 0.5.1 (`fe1afcc2…`): the change is decode-only, so
  it costs zero bytes. Package still **19.44 MB** of the 20 MB ceiling.
- **int8 parity**: T-600 25.95 / ETA 53.12, T-S3 27.26 / 47.32 — the gain survives quantization
  and is marginally *larger* on the held-out speaker than in PyTorch, i.e. no quantization cliff.

## Ship gate, at this decode

| Comparison | shipped decode | **E-022 decode** |
|---|---|---|
| **FLEURS-IT** vs Whisper-tiny auto+beam5 (27.42, the D19 bar) | +1.60, **failed** | **−6.95 [−8.02, −5.87], WON** |
| T-600 vs Whisper-tiny forced-`it` | WER −17.95, ETA +6.81 | **WER −24.80, ETA +16.59 [+10.61, +22.74], both WON** |
| T-600 vs **Whistle** | +3.35 worse | **WER −3.50 [−5.85, −1.24], WON**; ETA +3.36, tie |
| **T-S3** vs Whisper forced-`it` | WER −10.73 won, ETA −3.38 tie | **WER −19.51 WON**, ETA +8.73 [−1.49, +18.86], **tie** |
| T-S3 vs Whistle | +7.07 worse | −1.70 [−5.70, +2.13], tie |
| FLEURS-IT vs Whistle | +14.67 worse | +8.61 [7.74, 9.42], still behind |

**FLEURS-IT — the bar that has blocked this project since E-008 and forced the D19 self-correction
— is won, and by 7 points.** On the mixed set Moth beats Whisper on both metrics and Whistle on
WER. The **ETA half on a held-out speaker is still a tie** (+8.73 points, CI crosses 0), so the
strict reading of `docs/05` §5 is still not satisfied there; the point estimate moved from −3.4 to
+8.7, which is the right direction but not a certificate on 200 utterances.

**Do not cite the peeked arm.** `lm_weight 0.6 / gamma 3.0` would flip that tie to a win, but it
was selected after looking at S3 (`AUDIT-PROBE.md` §1.4). It is not adopted and must not be quoted
as if it were the result.

## Cost

- **Training: none.** No weight, no data, no checkpoint change. The E-017 checkpoint is reused
  byte-for-byte.
- Machine time: **~1 h 45** (eval 4 min, parity + browser 15 min, export 6 s, sequencing and
  re-runs included).
- **USD 0.**
- Browser: 1-thread RTFx 24.3 / 22.9 / **20.2×** on 5/10/20 s (was 25.4 / 24.2 / 21.7 at the
  shipped decode) — a ~7% cost, and 2× the bar.

## Code changed

- `stt/decode.py::beam_search` — new `token_bonus` parameter (default 0), documented with the
  mechanism.
- `web/decode.js::beamSearch` — the twin, same semantics.
- `web/ort.js` — passes `config.decode.token_bonus` through.
- `runs/E-010/package_browser.py` — `--beam/--top-k/--lm-weight/--token-bonus` instead of
  hard-coded settings.
- `stt/export.py` — the exported `config.json` no longer advertises `hotword_bonus`, a key no
  runtime ever read, and no longer contradicts the packager on the default beam width.
- Tests: 4 new decoder tests (including the LM-length-bias regression), 2 new JS parity tests,
  `test_export.py` updated. **294 Python tests, 9 Playwright, all green; `ruff` clean.**

## What the parity measurement first got wrong, and then right

The browser/Python comparison was run **twice**, and the first answer was wrong for a reason worth
recording.

The first pass measured **+3.27 WER** [1.74, 4.88] at the 0.5.1 decode and +3.87 at E-022, and
concluded that the two mel front-ends diverge and that a wider beam widens the gap. That number was
an artifact of my own packaging: `runs/E-010/package_browser.py` defaults to the **unaugmented** LM
(`lm_bpe3.pkl.gz`), while every release since 0.3.1 ships the augmented one and overrode the
default by hand. I had not overridden it, so the browser was scoring with an LM that had no prior
for the recorded vocabulary while Python had the right one.

With the correct LM in `release/lm_bpe3.bin` (200 S1 clips, shipped int8 graph):

| Arm | identical transcripts | browser WER | Python WER | Δ |
|---|---|---|---|---|
| 0.5.1 decode | 140/200 (70%) | 30.78 | 29.49 | **+1.29 [0.0009, 0.0259]** |
| **E-022 decode** | **154/200 (77%)** | **23.73** | 23.82 | **−0.09 [−0.012, +0.010], tie** |

Three things follow:

1. **E-010's "T200 parity is a tie" was right**, and it survives a bigger sample: the point
   estimate is the same +1.29 they measured. The CI now barely excludes 0 at the old decode, but
   at the shipped decode it is a clean tie with **higher** transcript identity (77% vs 70%).
2. **The wider beam improves parity rather than hurting it**: more identical transcripts and a
   smaller gap. A beam that can express longer hypotheses is less likely to be flipped by a
   1e-6 mel difference.
3. **The packager footgun is real and is now closed**: the default pointed at the wrong LM for
   eleven months and every release fixed it by hand. `LM_PICKLE` now points at the augmented one,
   and 0.6.0's record hashes it as `dcf1d59f…`, the same bytes 0.5.1 shipped.

## Next

1. **Re-tune the LM** with the bonus in place (~20 min, CPU): `runs/E-008/tune/` tuned `alpha` under
   the broken calibration, so its optimum may move — and a smaller LM would free package budget
   under the 20 MB ceiling for capacity.
2. **E-018 (gold, S4)** unchanged, with the revised expectation of ~4–6 WER points per speaker
   (`AUDIT-PROBE.md` §Tier 3): the second speaker's marginal return measured −4.07 at the fixed
   decode, against −5.73 before it.
3. **Publish** (`runs/E-021/upload_hf.py`, dry run by default): 0.6.0 passes its guard. Still
   blocked on the HF token and the namespace decision.
4. Optional, low priority: the E-022 decode widens the search, so **noisy speech** is where an
   over-long hypothesis would now show up — noise augmentation with empty targets (`docs/11` §7.4)
   remains the principled fix and is still unrun.