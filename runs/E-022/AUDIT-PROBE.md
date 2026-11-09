# Audit — 2026-11-08

A read of the project at `ee2f34b` (290 tests green, `ruff` clean), plus seventeen decode-only probes
run against the shipped checkpoint. **Nothing here changes the model, the data, or the training
recipe**, and nothing was committed. Every number below is reproducible from the scripts in this
directory.

The audit had one question: *where is the cheapest remaining gain?* It found one, it is large, and it
is not in the model — it is in the decoder's calibration.

---

## 1. The finding: the decoder is mis-calibrated, and it has been costing ~7 WER points

`beam_search` (`stt/decode.py`) adds the LM log-probability per emitted token
(`lm_weight * lm.score(...)`) but has **no counterweight for that cost**. An LM log-probability is
negative, so every additional token makes the hypothesis worse. With `lm_weight = 0.2` the decoder is
therefore systematically biased toward **emitting fewer tokens**, which is exactly what the error
profile shows: on T-600 with the shipped decode, deletions are **12.2%** of reference words against
**0.8%** insertions, and the model drops the *first* word in 13.3% of clips but the last in 0.5%.

Two standard CTC knobs fix it and neither requires retraining:

- **per-token bonus** `gamma`, added once per emitted token (the classic CTC "insertion/length"
  correction; in ``sweep_token_bonus.py`` as `beam_with_bonus`);
- **LM weight**, re-tuned upward now that the length bias is cancelled.

### 1.1 The 2-D sweep (selection set: `mls_it_dev` 200 + `ls_en_dev` 200, never T)

| `lm_weight` | `gamma` | dev WER | deletion | insertion |
|---|---|---|---|---|
| **0.2 (shipped)** | **0.0** | **0.3161** | 0.079 | 0.018 |
| 0.3 | 1.6 | 0.2855 | 0.049 | 0.030 |
| 0.4 | 2.0 | 0.2784 | 0.051 | 0.027 |
| **0.5** | **2.4** | **0.2730** | 0.051 | 0.025 |
| 0.6 | 2.8 | 0.2756 | 0.056 | 0.026 |
| 0.8 | 3.2 | 0.2856 | 0.071 | 0.022 |

Dev WER moves **31.6% → 27.3%** with no training. The optimum is a broad basin, not a spike.

### 1.2 It transfers to every set, including the held-out speaker

Beam 12 / top-k 10, `lm_weight` 0.5 / `gamma` 2.4 — **the cell dev alone picks**:

| Set | shipped decode | candidate | Δ WER [95% CI] | Δ ETA [95% CI] |
|---|---|---|---|---|
| **T-600** (mixed) | 32.77 / 42.86 | **25.92 / 52.64** | −6.85 [−7.94, −5.72] | +9.78 |
| **T-S3** (held-out speaker) | 35.43 / 35.21 | **26.66 / 47.32** | −8.77 [−11.13, −6.50] | +12.11 |
| FLEURS-IT | 26.52 | **20.46** | −6.06 [−6.63, −5.51] | — |
| FLEURS-EN | 55.48 | **49.63** | −5.85 [−6.55, −5.18] | — |
| CV-IT dev | 24.91 | **19.90** | −5.01 [−5.88, −4.14] | — |

Every CI excludes 0. The gain is *larger* on the held-out speaker than on the training speakers
(−8.77 vs −6.85), which is the opposite of an overfitting signature.

It also survives quantization: on the shipped `release/model.int8.onnx`, T-S3 goes **35.95 → 27.26
WER** and ETA **34.08 → 47.32**.

### 1.3 What it does to the ship gate

This is the part that matters for the project's scope. The mixed gate has failed on ETA since E-005,
and on FLEURS-IT since E-008 (D19). At the **dev-only** candidate, on the same checkpoint
(`gate-table-dev-only.json`):

| Comparison | shipped | candidate (dev-only) |
|---|---|---|
| FLEURS-IT vs Whisper-tiny auto+beam5 (27.42 — the D19 bar) | +1.60, gate fails | **−6.95 [−8.02, −5.87], won** |
| T-600 vs Whisper-tiny forced-`it` WER | −17.95 won | **−24.80 won** |
| T-600 vs Whisper-tiny forced-`it` **ETA** | +6.81, won | **+16.59 [+10.61, +22.74], won by more** |
| T-600 vs **Whistle** (the stretch bar) | +3.35 worse | **−3.50 [−5.85, −1.24], WER won**; ETA +3.36, tie |
| **T-S3** (held-out speaker) vs Whisper forced-`it` WER | −10.73 won | **−19.51 won** |
| **T-S3** vs Whisper forced-`it` **ETA** | −3.38 tie | +8.73 [−1.69, +18.61], **still a tie** |
| T-S3 vs Whistle | +7.07 worse | −1.70 [−5.43, +2.03], **tie** |
| Noise, non-music words (with VAD) | 0 | **0** |

Reading this honestly: FLEURS-IT — the bar that has blocked the project since E-008 — **is finally
won, decisively**. On the mixed set Moth now beats Whisper on both metrics with a wide margin and
beats Whistle on WER. **The ETA half on a held-out speaker is still a tie** (+8.73 points but the CI
crosses 0), so the mixed gate at the strict reading of `docs/05` §5 is still not passed there — the
point estimate moved from −3.4 to +8.7, which is the direction the project wanted, but 200 utterances
cannot certify it.

Two caveats that must travel with this: (a) three of T's four speakers are training speakers, so
T-600 is the optimistic read and T-S3 the honest one; (b) Moth is still 8.6 WER behind Whistle on
FLEURS-IT, so "beats Whistle" is scoped to the code-switching set.

The peeked arm (§1.4, `lm_weight 0.6 / gamma 3.0`) does push T-S3 ETA past the bar
(+11.27 [+1.06, +21.99], won). That is precisely why it cannot be used as the headline: the arm that
makes the gate pass is the arm that looked at the test speaker. Validate it on dev first, or report
it as peeking.

### 1.4 A self-correction, recorded because it happened

A first pass selected `lm_weight 0.6 / gamma 3.0` (T-600 24.68, T-S3 25.47, T-S3 ETA gate **won**)
— but that cell was **not in the dev grid**; it was picked after looking at T-S3, which violates the
project's rule *"never tune epochs/LR against S3"* (`docs/plans/2026-11-07-e018-scale-the-gold.md`,
"Never" list). The numbers above are therefore the **dev-only** cell. The 0.6/3.0 row is in
`final-summary.json` and is ~1.2 WER better, but **it must not be adopted without its own dev
validation** — and if it is adopted, it must be reported as tuned-with-a-peek, not as a clean result.
This is the third self-correction of this kind in the project's history (`docs/11` §4); the pattern is
that a tempting number on S3 is exactly the one to distrust.

### 1.5 Cost

Decode-only, no package growth. Python decode RTFx drops from ~280× to ~180× on these sets (still
20× the bar). Browser cost is the open question: beam 12 / top-k 10 raises the CPython-side decode
time ~40%, and the JS `beamSearch` is the same algorithm. **This needs a browser RTFx measurement
before shipping** (`web/` Playwright suite); if 1-thread RTFx falls under ~10× on 20 s clips, ship the
beam-8 arm (`dev 0.2690`, T-600 −6.39) instead — it keeps most of the gain.

---

## 2. Why nobody caught this: the tuning surface had one dimension

E-008 tuned `alpha` over `{0.05 … 0.8}` and kept 0.2 (`runs/E-008/tune/summary.md`). The curve was
flat-to-worse above 0.3 **precisely because** the missing token bonus made a larger alpha suppress
tokens. `alpha` was being asked to do language modelling *and* length control at once, and the sweep
could not see the second job. The lesson is in the sweep design, not in the model: **when one scalar
controls two effects, its 1-D optimum is not the optimum of either.**

---

## 3. The residual error, now that the deletion bias is gone

At the candidate decode, T-600 errors decompose as substitution 63.2%, deletion 32.0%,
insertion 4.9% (was 60.2 / 37.4 / 2.4 with the shipped decode: the bonus moves mass from deletions
to insertions, which is the intended direction). The three structural causes, in order of size:

1. **Substitutions inside English terms.** 304/596 ETA errors are a wrong word (`bug`→`ung`,
   `milestone`→`primaestone`, `jira`→`gira`), 79 are truncations (`merge`→`merg`, `update`→`date`),
   33 are terms absorbed into a longer token (`map`→`roadmap`). This is capacity + acoustic
   coverage, not decoding.
2. **Word-boundary loss.** 18 hypotheses on T-600 are two reference words glued (`login`←`log`+`in`,
   `timeline`←`time`+`line`, `l'indirizzourl`). The `▁` marker is a normal vocabulary piece, so CTC is
   free to skip it. A boundary-aware penalty in the beam would recover ~36 words (1% WER) for free.
3. **Italian function-word deletion.** 12.1% of Italian function words are deleted vs 7.8% of content
   words, concentrated at the start of the clip. Trimming the leading room tone with the shipped VAD
   was tested (`trim_test.py`): **no gain** (−0.7 WER, tie), so the onset theory is wrong and this is
   acoustic, not a preprocessing bug.

---

## 4. What else is worth doing, ranked by expected value per unit of author time

### Tier 1 — cheap, decode-only, validated above

1. **Adopt the token bonus + re-tuned LM weight** (this report §1). ~3 h of work: implement in
   `stt/decode.py` and `web/decode.js`, re-tune on dev, one browser RTFx run, one release rebuild.
   Expected **−6 to −9 WER / +9 to +12 ETA** on every set. This is the single largest lever found
   in the project's history that needs no new data.
2. **Boundary penalty for `▁`** (§3.2). Decode-only, ~1 WER point, one afternoon. Rides along with 1.
3. **Ship the release** (§1.3): FLEURS-IT — the bar that blocked the project since E-008 — is now
   won decisively, and the mixed set beats Whisper on both metrics. The publisher is ready
   (`runs/E-021/upload_hf.py`); it is blocked only on the HF token and namespace.

### Tier 2 — training, cheap, one variable each (~3.5 h per run, unattended)

4. **InterCTC** (auxiliary CTC mid-encoder, λ≈0.3, ~30 lines in `model.py`/`train.py`, no inference
   cost). Listed in `docs/10` P2 and never run. Expect 1–3% relative.
5. **EMA / checkpoint averaging on the decoder side** — `docs/10` says averaging the last 3 did not
   help E-005, but that was before gold data changed the loss landscape; re-test on E-017's 3 kept
   checkpoints (free, one eval).
6. **Re-run the LM alpha sweep with the bonus in place.** The LM is 5.12 MB of the 19.44 MB package
   and was tuned under the broken calibration; a freshly tuned LM may now be worth a *larger* weight,
   or a smaller/cheaper one may now suffice. Also free.

### Tier 3 — data, and the honest limits

7. **E-018 (gold from S4)** is still the right *data* experiment, and the audit **checked** whether
   the decoder fix eats its rationale. It does not: re-decoding E-014, E-016 and E-017 at the fixed
   decode (`gold_deltas.py`), the gold gain is intact —

   | Comparison (T-600) | shipped decode | fixed decode |
   |---|---|---|
   | E-014 → E-016 (first gold speaker) | −14.99 WER / +15.72 ETA | **−15.51 / +18.12** |
   | E-016 → E-017 (second gold speaker) | −5.73 WER / +6.81 ETA | **−4.07 / +6.62** |
   | E-014 → E-017 (all gold) | −20.73 WER / +22.53 ETA | **−19.58 / +24.74** |

   So gold is real, not a decode artefact, and E-018 keeps its justification. But note the
   **marginal** return of the second speaker drops from 5.7 to 4.1 WER points once the decoder is
   fixed, which is mild evidence for the plateau the E-018 plan already budgets for. The plan's
   "~6 points per speaker" estimate should now be read as "~4–6".
8. **Distillation (E-015)** stays blocked on provenance and is now lower priority: the decoder fix
   bought more than the pilot's go/no-go threshold (<2 ETA points) in one afternoon.
9. **Noise augmentation with empty targets** (`docs/10` §4.2.6, `docs/11` §7.4) is still the
   principled fix for *noisy speech*. The VAD closes the gate, not the robustness gap. Note the
   audit found the token bonus **more than doubles** words on non-speech when the VAD is off
   (301 → 683, and digital silence goes 120 → 240): the bonus makes the VAD load-bearing, which
   strengthens the case for the augmentation rather than weakening it. With the VAD on, the
   non-music count stays **0** in both arms; the 104 → 134 delta is music, which is never gated.

### Tier 4 — worth knowing, not worth doing now

10. **BPE 2048**: `data/en_terms.txt` has 589 terms and only 5 are single-piece; `deploy` is 4 pieces,
    `kickoff` 5. E-013 (V=4096) killed the ETA hypothesis, but it was evaluated **under the broken
    decoder**, where emitting more pieces was penalized. Worth one cheap re-read of E-013's
    hypotheses at the new decode before dismissing it.
11. **Leading-silence trimming**: tested, no gain (§3.3). Close it.
12. **Beam 16 / top-k 12**: +0.4 WER over beam 12 on dev, 60% more decode time. Not worth it.

---

## 5. Correctness and hygiene issues found in the code

None of these are urgent; all are small and would fit the next housekeeping commit.

1. **`release/config.json` ships dead keys.** `decode.hotword_bonus: 2.0` is written by
   `stt/export.py:382` but no hotword index is ever loaded (`web/ort.js` has no hotword path), so the
   key advertises a feature the runtime does not have. Either wire it or drop it.
2. **`stt/export.py:382` defaults `beam: 8` while `runs/E-010/package_browser.py:35` overrides it to
   5.** The shipped config says 5; the exporter's default says 8. Harmless today, a trap later.
3. **`beam_search` has no length/normalization knobs at all** — which is the root cause of §1.
   Adding `gamma` (and optionally `beta`) as defaulted parameters keeps every existing call site
   valid.
4. **`web/decode.js` cannot express the fix** as written: `lmBonus` is per-token-only and there is no
   hook for a constant per-token term. It needs the same two parameters, and the two decoders must
   stay twins (the project already tracks this for VAD).
5. **The dev grid in `runs/E-008/tune/summary.md` is recorded but not reproducible as a script** —
   the tuning loop lived in an ad-hoc session. Given §1, a `runs/E-0NN/tune_decode.py` that emits the
   2-D grid would make the next calibration a one-command job.
6. **`stt/eval.py::_speaker_group` hardcodes `S1+S2` / `S3+S4`.** Adding S4 (E-018) silently keeps S4
   in the "held out" group, which is the correct default but is not stated anywhere in the code.
7. **`STATUS.md` line 3 says "262 test verdi"; the suite is now 290.** Trivial, and exactly the kind
   of drift that makes a status file unreliable.

---

## 6. Reproduce

```bash
# 2-D grid on dev (the selection step)
uv run python runs/audit-2026-11-08/sweep_lm_gamma.py

# transfer of the leading cells
uv run python runs/audit-2026-11-08/sweep_transfer.py

# the dev-only candidate on every set, PyTorch and int8
uv run python runs/audit-2026-11-08/dev_only_candidate.py

# the ship gate at the candidate parameters
uv run python runs/audit-2026-11-08/gate_table.py

# the negative result (trimming) and the noise interaction
uv run python runs/audit-2026-11-08/trim_test.py
uv run python runs/audit-2026-11-08/noise_check.py

# the gold deltas re-read at the fixed decode
uv run python runs/audit-2026-11-08/gold_deltas.py
```

Artifacts: `lm-gamma-grid.json`, `lm-gamma-transfer.json`, `dev-only-summary.json`,
`final-summary.json`, `gate-table.json`, `noise-token-bonus.json`, and hypothesis + metrics files
under `eval/`, `final/` and `dev-only/`.

**Nothing in this audit is an `EXPERIMENTS.md` entry yet.** Per the repository protocol a finding
becomes an experiment when it is run as one, with a config and an ID; the natural next ID is
**E-022 — decode recalibration**, and it should be written *before* the code changes, so the
prediction above can be scored against the result.