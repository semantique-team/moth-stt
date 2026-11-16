# Data card — recorded sets (moth-stt)

Status: **complete and frozen (2026-11-16).** The T, R and A sets are recorded for
S1, S2 and S3; S4 was never recorded and is not part of the release. All three
speakers chose **Option A (CC-BY 4.0)** and confirmed it on 2026-11-16. The
canonical dataset, canonical manifests and SHA-256 values are in the sibling
repository `../moth-cs-dataset` (v1.0.0); this file is the model-repository
summary. Audio never enters git.

| Field | Value |
|---|---|
| Speakers | S1 (the author), S2, S3 (**held out** from all training; S3 reads T and A only) |
| Sets | T = test, 600 clips (200/speaker); R-S1 = 741 training clips (641 base + 50 term v1 + 50 v3); R-S2 = 672 training clips (622 base + 50 term); A = 180 anchor clips (60 T sentences × 3 speakers) |
| Duration | T 23.2 min · R-S1 34.9 min · R-S2 22.7 min · A 7.0 min · **total 87.8 min (1.46 h)** |
| Consent | S1, S2, S3: **Option A — CC-BY 4.0**, confirmed 2026-11-16; signed forms held offline, not in git |
| Recording conditions | 48 kHz mono 16-bit WAV; S1 T on 2026-10-19 and R-S1 on 2026-10-19/23; S2 T on 2026-10-30, R-S2 on 2026-10-30, term addendum on 2026-11-11; S3 T and A on 2026-10-30. **Devices were not logged for every session**; do not infer them |
| Format | WAV 16-bit mono, 48 kHz; files `S<x>_<NNNN>.wav`; session logs `recordings/audio/<set>/<speaker>/session.log.jsonl` |
| Retakes / skips | S1: 12 retakes over 200 T takes, 0 skips (session log). no S2 session log (T, R-S2, anchor); the addendum builder report says 50/50 files accepted, 0 errors. S3: session log retained |
| Speaker characteristics | **S1 has rhotacism ("r moscia"): a non-standard /r/.** T-S1 is both a test speaker and R-S1 (S1) training audio (×50 in E-016), so T-S1 is a **matched** speaker and any ETA/WER gain there is optimistic for speakers with a standard /r/. S2/S3 provide the cross-speaker check; S3 is held out. Speakers record naturally; no correction is applied |
| Scripts | `recordings/scripts/*.jsonl`; all T rows human-reviewed on 2026-10-19; R-S2 addendum rows 0624–0673 recorded 2026-11-11 |
| Dataset licence | **CC-BY 4.0** for audio and transcripts; code Apache-2.0. The consent form asks that the voice not be used for TTS or biometric identification; CC-BY does not encode that restriction — see the dataset card's ethical-use section |
| Publication | `semantique-team/moth-cs` v1.0.0 (prepared 2026-11-16; see `../moth-cs-dataset`) |

## Excluded files

- 15 WAV in `recordings/audio/S1_alt_2026-10-19T1724`: a superseded alternate
  take session; the canonical T takes are in `recordings/audio/S1`.
- `recordings/audio/R-S2/S2/S2_0001.wav`: an unlabelled take, rejected at import
  and absent from `R-S2.jsonl`.

Both are documented with byte sizes in
`../moth-cs-dataset/manifests/excluded.jsonl`.

## One header repair

`S2_0624.wav` had `RIFF`/`data` chunk sizes of `0xFFFFFFFF` (an ffmpeg streaming
artefact). The canonical dataset copy repairs the header to the real file sizes
without touching the samples. Original and repaired SHA-256 values are in
`../moth-cs-dataset/releases/v1.0.0.json`.

## Canonical copy

The standalone dataset repository contains the audio, the canonical manifests
(`train.jsonl`, `test.jsonl`, `anchor.jsonl`, `all.jsonl`), the checksums and the
verification script. The legacy manifests in this checkout are the historical
record used by the experiments; they keep the old `<speaker>/<file>` paths and
must be resolved with the per-set audio root. Do not publish them as-is.
