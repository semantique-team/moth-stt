# Data card — recorded sets (moth-stt)

Status: **in progress** — the T set is complete for S1, S2 and S3 (600/600); S1's R-S1 is
complete (641 base + 50 term-addendum clips, 691 valid rows); S2's R-S2 is complete (622 base +
50 term-addendum clips, 672 valid rows, 2026-11-11); S4 was never recorded. The per-speaker
consent is still not signed. Audio never enters git; it lives on Hugging Face once the dataset
is published (`docs/03-data.md` §7, §11).

| Field | Value |
|---|---|
| Speakers | S1 (the author), S2, S3 (**held out** from all training; S3 reads T only) |
| Sets | T = test, 600 sentences (200/speaker); optional anchor A = 60 T sentences read by all; R-S1 / R-S2 = training, 600–1000 each |
| Consent | per speaker: signed on `____`; release choice: `A (CC-BY 4.0)` / `B (private)` — fill per speaker (**pending for S1, S2, S3**) |
| Recording conditions | device per speaker (laptop mic / phone), room, date range — fill (**S1: 2026-10-19, 48 kHz mono WAV; S2 addendum: 2026-11-11, 48 kHz mono WAV; devices to confirm**) |
| Format | WAV 16-bit mono, native rate; files `S<x>_<NNNN>.wav` (or `.m4a`/`.flac`); session logs `recordings/audio/<speaker>/session.log.jsonl` |
| Retakes / skips | per the session logs — summarize per speaker (**S1: 12 retakes over 200 takes; 0 skips. S2 addendum: session log not returned; 50/50 files accepted by the builder, 0 errors**) |
| Speaker characteristics | **S1 has rhotacism ("r moscia"): a non-standard /r/.** This matters because T-S1 is the test speaker *and* R-S1 (S1) is training audio (×50 in E-016): T-S1 is therefore a **matched** speaker, so any ETA/WER gain measured there is optimistic for speakers with a standard /r/. The natural corpora (438 h) keep the standard realisation in the base model. Mitigations: S2/S3 recordings with **S3 held out** (E-017), and a per-word error split on /r/-containing terms. Speakers record naturally; no correction is applied. |
| Scripts | `recordings/scripts/*.jsonl`, commit `d48532d` (T and R-S1), `df31eb7` (pool); R-S2 addendum rows 0624-0673 (2026-11-09; recorded 2026-11-11); all T rows human-reviewed on `2026-10-19` |
| Dataset license decision | CC-BY 4.0 proposed for publishable parts; final decision at M8 (`docs/08` §2) |
| Publication | Hugging Face dataset repository — link to fill at M8 |
