# E-019 — Whistle LoRA probe: result

> Exploratory side artifact, **not Moth** (D28). D5 is unchanged. Measured 2026-11-07 on the
> host recorded in `config.json` (MacBook Pro Mac17,2, Apple M5, 10 cores, 16 GiB, CPU-only JAX).

## Verdict: killed at L1

The plan's kill criterion is "if L1's S3 CI covers 0, stop before L2/L3". It does:

| Set | L0 base | L1 + LoRA | ΔWER | 95% CI | Sig. |
|---|---|---|---|---|---|
| **T-S3 (primary, held-out speaker)** | 28.36 | **30.66** | **+2.30** | [−0.68, +5.46] | tie → **kill** |
| Gold test (S1+S2, speakers seen) | 30.08 | 27.10 | −2.98 | [−8.38, +1.88] | tie |
| FLEURS-IT | 11.85 | 12.87 | +1.02 | [+0.51, +1.56] | **worse** |
| FLEURS-EN | 13.35 | 15.65 | +2.30 | [+1.16, +3.26] | **worse** |
| CV-IT test (first 1000 clips) | 17.59 | 20.25 | +2.66 | [+1.74, +3.49] | **worse** |

ETA on S3 falls 52.11 → 47.89 (−4.23, CI [−10.34, +1.86], tie). Noise with our VAD stays at
**0 non-music words / 337 clips** in both arms (music 217 → 281, reported apart).

## What it means

The direction is unambiguous and it is the same shape E-016 found for Moth: the adapter improves
the speakers it was trained on (−2.98 WER, +8.25 ETA on the gold test split) and **degrades
everything else**, including a held-out speaker and both FLEURS sets, where the loss is
statistically significant. 45 min of gold on a 220 MB decoder model buys speaker adaptation, not
code-switch ability.

This is a **negative result worth recording** (docs/11 §3): it is the external check the plan
wanted in `Why run it` point 2 — and the answer is that the gold effect is *not* architecture
independent. Moth's CTC gained on S3 with the same data (E-017: −6.13 WER); Whistle's decoder
loses 2.3. The likely reason is that LoRA on a decoder with one language token per clip can only
re-tune the acoustic-to-text mapping toward two voices, and has no mechanism to keep the base's
generality.

H2 (data, not architecture) is **rejected** in this form. H3 (`mix` language code) and H4
(`--tokens`) were never run: the plan forbids them after a kill, and the kill is clean.

## Cost, for the record

| Item | Value |
|---|---|
| Training | 3 epochs × 72 steps = 216 steps, **22 min** wall on CPU |
| Throughput | ~5.7 s/step at batch 16 (16 clips `_arrays` pads to 30 s, so the cost is fixed) |
| Adapter | `lora-L1.safetensors` **6.5 MB** (above the plan's ~1 MB committable threshold) |
| Export | `whistle-lora-L1.cact` **28 MB** vs base 16.92 MB — **21.6 MB over the 20 MB budget** on its own |
| Base checkpoint | 220.6 MB `.safetensors`, downloaded once, outside the repo |
| Val loss | 0.825 → 0.561 → 0.527 over the 3 epochs (still falling, underfit) |

The export size alone would have blocked shipping this arm as a 20 MB package with our 2.33 MB
VAD (28 + 2.33 = 30.3 MB). That is a second, independent reason not to continue.

## Baseline sanity check (plan step 3)

Re-running L0 on T-S3 with this environment reproduces E-017 **exactly**: hypotheses
bit-identical (200/200), WER 28.3646 / ETA 52.1127, sub/del/ins 228/80/25. Only per-clip
latencies differ. So the JSONL builder, the audio path and the loader are right, and L1's
difference is the adapter.

## Artifacts

```text
runs/E-019/
  build_jsonl.py, config.json, data/{train,val,test}.jsonl, data/split.json
  train_L1.sh, train_L1.log
  lora-L1.safetensors        6.5 MB   (gitignored: above the ~1 MB threshold)
  whistle-lora-L1.cact       28 MB    (gitignored)
  l0/                        L0 re-run on T-S3 + metrics
  l1-s3/                     L1 on T-S3 + metrics + paired diff vs L0
  gold-test-{base,l1}/       the 63-clip gold test split, both arms
  l0-sets/, l1-sets/         FLEURS-IT and FLEURS-EN, both arms
  l0-cv/, l1-cv/             CV-IT first 1000 clips, both arms
  l0-noise/, l1-noise/       noise with our VAD, both arms
  manifests/                 gold-test.jsonl, cv-first1000.jsonl
```

Every number above comes from `stt.eval` with a paired bootstrap at seed 0 (1000 resamples).
The CLI's own `wer 29.9%` printed by `needle whistle finetune` is **not** used anywhere: it has
its own normalizer and its own test split.

## What was deliberately not measured

The plan's step 5 lists T-600 and "noise with and without the VAD" as well. Once the S3 kill
fired, spending ~5 min of compute on an arm that is losing everywhere was not worth it, so:

- **T-600 was not run.** It would mostly re-measure the two trained speakers (400 of 600 clips)
  and the caveat already applies. Given the S3 result, its expected sign is negative.
- **Noise was measured with the VAD only** (the shipping path). Without the guard both arms emit
  words on music; the L0 number for that is already in E-011 (264 words), and the LoRA is worse on
  music with the guard (281 vs 217), so the without-guard direction is not in doubt.
- **CV-IT uses the first 1000 clips**, not all 14,643, as a cheap direction check.
- **L2 (`--tokens`) and L3 (`mix`) were not run**: the plan forbids them after a kill.