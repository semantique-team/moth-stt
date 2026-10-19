# Data card — recorded sets (moth-stt)

Status: **not started** — fill this in as soon as the first consent is signed and the first
session is recorded. Audio never enters git; it lives on Hugging Face once the dataset is
published (`docs/03-data.md` §7, §11).

| Field | Value |
|---|---|
| Speakers | S1 (the author), S2, S3 (**held out** from all training; S3 reads T only) |
| Sets | T = test, 600 sentences (200/speaker); optional anchor A = 60 T sentences read by all; R-S1 / R-S2 = training, 600–1000 each |
| Consent | per speaker: signed on `____`; release choice: `A (CC-BY 4.0)` / `B (private)` — fill per speaker |
| Recording conditions | device per speaker (laptop mic / phone), room, date range — fill |
| Format | WAV 16-bit mono, native rate; files `S<x>_<NNNN>.wav` (or `.m4a`/`.flac`); session logs `recordings/audio/<speaker>/session.log.jsonl` |
| Retakes / skips | per the session logs — summarize per speaker |
| Scripts | `recordings/scripts/*.jsonl`, commit `__________`; all T rows human-reviewed on `____` |
| Dataset license decision | CC-BY 4.0 proposed for publishable parts; final decision at M8 (`docs/08` §2) |
| Publication | Hugging Face dataset repository — link to fill at M8 |
