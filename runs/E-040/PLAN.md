# E-040 — Multi-seed replication of the E-036 arms

**Status:** pre-registered 2026-11-13, **before** the first training step of any seed. Training
authorized explicitly by the owner for this experiment only.

## Question

E-036 rejected two conservative fine-tunes (`frozen`, `last2`) on a **single seed** (seed 0). Its
verdict rests on point differences that are small next to dev noise: dev 30.9859% and 30.9720% vs
31.0101% (ties), rare 39/81 and 41/81 against a 42/81 target, and `x` emitted on all 120 silence
clips. The closure plan names multi-seed as the remaining closure evidence. This experiment asks:

> Are the E-036 failure modes properties of the **recipe**, or run-to-run variation that a different
> seed would remove? How large is the seed-to-seed spread of the same recipe on the same data?

It is **dispersion evidence about a rejected recipe**, not a new candidate search. The shipped
model stays 0.8.1 and no run from this experiment replaces it without a separate pre-registered
readout.

## Design

Exact replication of `runs/E-036/{frozen,last2}/config.yaml`, changing **only** `seed` and `run`:

| Arm | Trainable parameters | Recipe |
|---|---:|---|
| `frozen` | 3,942,785 | encoder, conv front-end, final norm frozen; head blocks + CTC head move |
| `last2` | 7,490,945 | last two encoder blocks + head blocks + CTC head move |

- Mixture, augmentation, optimizer, schedule, batch policy and distillation (E-028 teacher,
  weight 20, T=2) are byte-identical to E-036; only `seed` differs.
- Seeds **1, 2, 3**. Seed 0 already exists as E-036 and counts as the fourth replicate: its configs
  and fingerprint are recorded there, and it is included in every summary as `seed 0 (E-036)`.
- Each run trains one epoch to the same step count as E-036 (13,314 steps) with the same
  checkpoint policy; checkpoints stay out of git.
- Data mixture includes the same gold/term sources as E-036, so the comparison is seed-only.

## Measurements

Per run, at the best dev checkpoint exactly as E-036 selects it:

1. **Dev**: best macro WER over MLS-IT + LS-EN (the E-036 selection metric).
2. **Product readout** with the shipped INT8 path and the A3 LM (beam 12 / top-k 10 / LM 0.55 /
   bonus 2.7): T-600, T-S3, FLEURS-IT, FLEURS-EN, rare aligned ETA (81 span tokens), natural/gold
   term counts, and the silence/noise behaviour (non-empty clips, `x` count) on the same 120-clip
   silence probe used by E-036.
3. **Per-seed dispersion**: mean, min, max and range of dev macro WER, T-600 WER/ETA, rare aligned
   ETA and silence non-empty clips across the four replicates per arm.

## Decision Rule (pre-committed)

- **No adoption from this experiment.** A run that looks good is reported as a replicate, and any
  adoption requires its own PLAN written before looking at its product readout.
- The E-036 verdict is **confirmed** if the failure modes reproduce across seeds (rare below target
  or silence `x` present in a majority of replicates); it is **downgraded to inconclusive** if the
  E-036 point differences sit entirely inside the seed spread while the failure modes disappear in
  most seeds. Both outcomes are recorded, no run is cherry-picked.
- Seed spread is reported as a range over replicates, never as a confidence interval: four
  replicates do not estimate a variance.

## Cost And Execution Order

~1 h 47 per run on the M5; six runs ≈ **11 h** of machine time, one run at a time, started only
after the previous one finishes. Preflight (`--check-config`) before each start; logs and metrics
committed, checkpoints gitignored.

Order: `seed1/frozen`, `seed1/last2`, `seed2/frozen`, `seed2/last2`, `seed3/frozen`,
`seed3/last2`. If a run is interrupted, resume the same run ID; never re-init.

## Artifacts

`runs/E-040/seed{1,2,3}/{frozen,last2}/` — `config.yaml`, `train.log.jsonl`, `train.out`,
`checkpoints/` (gitignored), `readout/` (hypotheses + metrics for the product sets), plus a
top-level `RESULT.md`, `summary.json` and the `EXPERIMENTS.md` entry.
