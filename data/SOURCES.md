# Data sources and retention

The frozen checkout keeps the **manifests**, the **recorded-set metadata**, the
**tokenizer** and the **noise/room-tone evaluation inputs**. The multi-gigabyte
raw corpora and the mel shards were deleted on 2026-11-16 (see
`docs/20-archive-handover.md`); they are re-downloadable, but the exact revision
of each source is part of the experiment record below.

| Source | Version / revision | Licence | Local path before deletion | Retained? |
|---|---|---|---|---|
| Mozilla Common Voice IT | 27.0 (`cv_it_27`) | CC0 | `data/raw/cv_it_27/` (11 GB) | no — re-download 27.0 |
| MLS Italian | OpenSLR 94 | CC-BY 4.0 | `data/raw/mls_it/` | no — re-download OpenSLR 94 |
| LibriSpeech | `train-clean-100` + `dev-clean` | CC-BY 4.0 | `data/raw/librispeech/` | no — only the metadata was left |
| FLEURS IT/EN | HF `google/fleurs` snapshot used 2026-10-19 | CC-BY 4.0 | `data/raw/fleurs/` (1.7 GB) | no — HF revision was **not** recorded (gap) |
| MUSAN | OpenSLR 17, `noise` + `music` | CC-BY 4.0 | `data/raw/musan/` | no — derived noise clips retained |
| Own room tone | recorded 2026-11-06 | authors' own | `data/raw/room_tone.wav` | **yes** |
| Noise evaluation set | built by `stt/data/build_noise.py` | mixed (see above) | `data/raw/noise/` | **yes** |
| Recorded T/R/A sets | S1–S3, 2026-10-19 → 2026-11-11 | CC-BY 4.0 | `recordings/audio/` | **yes**, canonical copy in `moth-cs-dataset` |

## Checksums and versions actually verified

- LibriSpeech `dev-clean` and `train-clean-100`: MD5 checked by
  `data/raw/fetch.sh` against the OpenSLR list (`fetch.log`, 2026-10-19).
- MLS Italian: MD5 checked against the OpenSLR list (`fetch.log`, 2026-10-19).
- Common Voice IT 27.0: downloaded manually after accepting the terms; the
  archive checksum was **not** retained. The version is pinned by the
  `cv_it_27` directory name and the manifests.
- FLEURS: downloaded through the Hugging Face cache; the snapshot revision was
  **not** recorded. The filtered test manifests and all hypotheses are
  retained, so the reported scores can be re-scored, but the exact audio cannot
  be guaranteed to rebuild bit-for-bit from a future FLEURS revision.
- Own room tone: 16 kHz mono 16-bit, 974.795 s,
  SHA-256 `0eac8410928f2053ec447020f963557fa70186e769aea54cac9e2609ee886260`.

## What is needed to retrain

1. Re-fetch CV 27.0 IT, MLS Italian (OpenSLR 94), LibriSpeech
   `train-clean-100`/`dev-clean`, FLEURS IT/EN (evaluation only) and MUSAN
   (noise/music).
2. Restore the recorded sets from `semantique-team/moth-cs-dataset` (or the
   private backup) into `recordings/audio/`.
3. Rebuild the mel shards with `stt/data/build_mel.py`.
4. A training checkpoint is **not** in this checkout; only the exported 0.8.1
   ONNX is on Hugging Face. The release checkpoint (`E-028/ckpt_best.pt`) is in
   the private backup, not in git.

The exact commands and the environment are in
`docs/guides/repository-workflows.md` and `docs/20-archive-handover.md`.
