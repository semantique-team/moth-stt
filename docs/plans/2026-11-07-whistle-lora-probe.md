# Plan — E-019: Whistle LoRA probe (a second model, "less ours")

> Written 2026-11-07 at the author's request. **Status: not started. This document is the handoff
> for a new session; no code has been written for it.**
>
> Context: E-017 closed the gap to Whistle to **+3.35 WER** on T-600 (32.77 vs 29.42) and left the
> ETA gap at −6.42. The next Moth step (E-018, scale the gold with S4) is blocked until S4 records.
> The author wants to use the wait to probe the **other** direction: fine-tune Whistle, to have a
> second model available and to learn what a decoder-based code-switch model does on our data.
>
> **This does not change D5.** `docs/01` §3 constraint 1 ("own model, not a fine-tune of Cactus
> Whistle") still governs the *Moth* deliverable. E-019 is an **exploratory side artifact**,
> explicitly "less ours", and it must never be presented as Moth or as satisfying the project's
> owning constraints. If it ever ships, it ships under its own name and card, stating the
> derivation from `Cactus-Compute/whistle` (Apache-2.0).

## Why run it

Three reasons, in decreasing order of value:

1. **A second model on the shelf.** On T-600 Whistle is 3.35 WER points ahead of Moth and 6.4 ETA
   points ahead. A LoRA on our own gold is the cheapest possible way to find out whether that
   ordering is a ceiling or a starting point, and to have a fallback artifact if Moth's next step
   stalls.
2. **A clean external check on our own claim.** If a LoRA on 50 min of the same gold moves Whistle
   by a similar amount to what it moved Moth (+6 WER, E-017), then the gold effect is a property of
   the *data*, not of our architecture — a stronger, more general statement for the paper.
3. **Two knobs we cannot test on Moth.** Whistle has (a) a **language token per clip** and
   (b) a **keyword/token biasing** interface with a vocabulary-extension flag. Our CTC has neither.
   Testing them tells us what we are giving up by not having a decoder, and whether a "mix" language
   code can express intra-clip code-switching.

## What is already known (measured, this repo)

| System | T-600 WER / ETA | T-S3 (held out) WER / ETA | FLEURS-IT | FLEURS-EN | CV-IT | Non-music words | Size |
|---|---|---|---|---|---|---|---|
| Whistle forced `it` | **29.42 / 49.28** | **28.36 / 52.11** | **11.85** | **13.35** | **20.63** | 264 / 240 clips | 16.9 MB |
| Moth E-017 | 32.77 / 42.86 | 35.43 / 35.21 | 26.52 | 55.48 | 30.36 | **0 / 337** | 19.44 MB |

Source artifacts: `runs/E-017/baselines-t600/whistle-it/`, `runs/E-017/baselines/whistle-it/`
(T-S3), `baselines/runs/whistle-it{,-cv,-noise}/`. Whistle is **Apache-2.0**, one 16.9 MB `.cact`
file, 8-layer encoder + 8-layer laddered decoder (width 512), 2–4-bit Cactus quants
(`docs/09` research notes). Its weakness on our task is structural: **one language token per
clip**, and a decoder that can emit text on music (554 words on the music half of the noise set).

Verified on this machine (2026-11-07, read-only):

- `needle whistle finetune --help` exposes exactly what we need: `train/validation/test` JSONL of
  `{"audio", "text", "language"}`, `--epochs --batch-size --lr --lora-rank --lora-alpha --seed
  --checkpoint-dir --out`, plus **`--tokens`** ("file of tokens to add to the vocabulary") and
  **`--language`** ("a new code **adds that language**").
- `needle whistle build --lora LORA --out X.cact` merges the adapter and exports for a platform
  (`wasm` is a supported target).
- The base weight is cached at `~/.cache/cactus-needle/whistle/2.0.0/whistle.cact` (`.cact`,
  engine format). The **`.safetensors` base needed by `finetune` is *not* cached** and will be
  downloaded on first use — size unknown, to be recorded.
- **JAX/Flax/Optax are not installed** in `.venv-baselines` (they are the training path of
  `needle`). Install them in a throwaway environment, never as a project dependency.

## Hypotheses

- **H1 (transfer).** A LoRA on R-S1+R-S2 improves Whistle on the T-S3 slice by a paired CI that
  excludes 0. Direction expectation: WER down; ETA up.
- **H2 (data, not architecture).** The size of the improvement is in the same range as Moth's
  E-016→E-017 gain on S3 (−6.13 WER / +4.79 ETA). If it is, the gold effect is a property of the
  data.
- **H3 (language token).** A new language code for mixed clips (e.g. `mix`) beats forcing `it` on
  code-switched audio, because it stops the model from resolving the English terms toward Italian
  phonotactics. This is the hypothesis our CTC cannot express and is the main scientific reason to
  run the probe.
- **H4 (tokens).** Passing the 580 terms of `data/en_terms.txt` to `--tokens` improves ETA more
  than the LoRA alone.

## Protocol (mirrors E-017 exactly, so the numbers are comparable)

**Data rules, non-negotiable:**

- Train on **R-S1 (641 takes) + R-S2 (622 takes)** only. `T` is the evaluation set and is **never**
  read for training, validation, token mining or hyper-parameter choice. `A` is a subset of T and
  is likewise excluded. Same rule as every experiment in this repo.
- Validation: a seeded 5% split **of the gold** (same speakers), because a validation set drawn
  from T would be leakage. The `test` argument of `needle whistle finetune` gets a second gold
  split, not T.
- The honest test set is the **S3 slice** (`runs/E-017/manifests/T-S3.jsonl`), exactly as for
  E-017: S3 contributed no audio to any training. Report T-600 too, but with the caveat that S1 and
  S2 are training speakers.

**Arms (one variable each, in this order; stop and reassess after each):**

| Arm | What | Why |
|---|---|---|
| L0 | Whistle, no adapter | Already measured — reference, no run needed |
| **L1** | LoRA, `--language it`, defaults elsewhere (rank 16, α 32) | the minimal probe |
| L2 | L1 + `--tokens data/en_terms.txt` | H4 |
| L3 | LoRA with a new code `mix` on the same gold | H3 |

Do **not** tune epochs/LR against S3; if a schedule must be chosen, choose it on the gold
validation split and say so.

## Steps

1. **Environment** (throwaway, never a project dependency):
   ```bash
   uv venv .venv-whistle --python 3.12
   uv pip install --python .venv-whistle/bin/python "cactus-needle[jax]" soxr soundfile
   ```
   Verify `needle whistle finetune --help` runs and record the versions in the run dir.
2. **Build the JSONL** from the two gold manifests (script to write:
   `runs/E-019/build_jsonl.py`). One line per clip:
   `{"audio": "<absolute or repo-relative wav>", "text": "<normalized text>", "language": "it"}`.
   Use `stt.text.normalize` output (the `norm` field) so every system is scored on the same text
   rule. Split: `train` 90%, `val` 5%, `test` 5%, seeded, stratified by speaker so both speakers
   appear in all three. Write the split file so the run is reproducible.
3. **Baseline re-run on the exact same clips** (cheap sanity check that the JSONL loader and the
   audio path are right): `needle whistle compare` on one clip, then L0 through
   `baselines/whistle_needle.py` on the S3 slice if the numbers do not match
   `runs/E-017/baselines/whistle-it/`.
4. **L1**: `needle whistle finetune train.jsonl val.jsonl test.jsonl --epochs … --out runs/E-019/lora-L1.safetensors`.
   Record wall time, host CPU/GPU, peak memory, and the val metric per epoch.
5. **Evaluate** every arm on: T-S3 (primary), T-600 (with caveat), FLEURS-IT/EN, CV-IT test,
   noise with and without the VAD. Scoring goes through `stt.eval` with paired bootstrap, exactly
   like every other number in the repo — never trust the CLI's own WER.
6. **L2 / L3** only if L1 shows a signal on S3 (paired CI excluding 0).
7. **Export**: `needle whistle build --lora … --out whistle-lora.cact --platform wasm` and record
   the file size. If it is to be usable in the browser, add our VAD the way 0.4.0/0.5.0 do; the
   budget is 16.9 + 2.33 = 19.2 MB, still under 20.
8. **Documents**: `EXPERIMENTS.md` entry E-019 (artifacts, all metrics, the negative-result table if
   it fails), `docs/11` §3 if the result is a negative result worth keeping, a model card stub
   stating the Whistle derivation and Apache-2.0, and a **NOTICE** entry for the base weights.

## Acceptance / kill criteria

- **Signal**: H1 confirmed if the S3 paired WER CI excludes 0 and points down.
- **Interesting**: H3 confirmed if `mix` beats forced `it` on S3 by a paired CI excluding 0. This is
  the outcome most worth having, because it is knowledge our CTC cannot produce.
- **Kill**: if L1's S3 CI covers 0, stop before L2/L3, write the negative result, and do not spend
  more time. A LoRA that does not move a held-out speaker on 50 min of gold is not worth more gold.
- **Never**: present a LoRA result as a Moth result, or compare a LoRA number against a Moth number
  measured on a different manifest.

## Risks and open questions

- **JAX on Apple Silicon.** `needle`'s training path is JAX; MPS support varies by version. If it
  falls back to CPU, the LoRA may be slow but 1,263 clips is small — measure, do not assume.
- **`.safetensors` base download size** is unknown and must be recorded; it is not part of any
  shipped artifact.
- **Language token semantics.** "A new code adds that language" is documented but its effect on
  intra-clip switching is not: Whistle's design is one token per clip, so `mix` may simply mean
  "train a bucket", not "switch inside the clip". L3 is a probe of exactly that.
- **License and naming.** Apache-2.0 permits the derivative, but the artifact must carry the notice
  and a card that names the derivation. It also needs a name that is not "Moth" to avoid confusing
  the two lines; that is an author decision.
- **Contamination.** Whistle may have seen FLEURS in pretraining (`docs/11` §5 already says so for
  the baselines). LoRA results on FLEURS remain comparable across systems but not contamination-free;
  T is the only clean set and S3 the only clean speaker.
- **This is not E-018.** E-018 (scale the gold: S4 + more sentences per speaker) stays the priority
  for Moth and is blocked on S4's recordings. E-019 must not consume S4's review time.

## Artifacts (to create)

```text
runs/E-019/
  build_jsonl.py          # manifests -> train/val/test JSONL, seeded, speaker-stratified
  config.json             # versions, arms, hyper-parameters, data counts, host
  lora-L1.safetensors     # adapter (small, committable if < ~1 MB)
  eval/{l1,l2,l3}/        # metrics.json + hyp.jsonl + paired diffs per set
  compare.md              # the table: Whistle, Whistle+LoRA, Moth E-017, per set
docs/plans/2026-11-07-whistle-lora-probe.md   # this file
```

Checkpoints, base weights and the `.cact` build stay out of git (`.gitignore` already covers
`*.onnx`; add the pattern if the adapter is large).