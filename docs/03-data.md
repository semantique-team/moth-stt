# 03 — Data

## 1. Principles

1. **v0.1 uses human-transcribed audio only.** No pseudo-labels, no teacher model (D8). Parakeet stays a baseline.
2. **Open licenses only** (CC0, CC-BY, or the authors' own recordings). Every dataset gets a line in `NOTICE`.
3. **Evaluation data is never trained on** — not even the FLEURS *train* split. Leakage is prevented by construction
   (separate manifests, hash-based sentence dedupe against every test/dev text).
4. **One text rule.** All transcripts pass through the single `normalize()` function (`05-evaluation.md` §1) before
   tokenizer training, training and scoring.
5. **Code-switching is learned from mixed data** (recordings first, synthetic later and only if it helps).

## 2. Sources

| Role | Source | License | Amount used (proposal) | Notes |
|---|---|---|---|---|
| Train, Italian | **Common Voice IT 27.0** | CC0 | ≈ 170 h of the 364 validated h (430 h total) | Many speakers/microphones, short read prompts, limited text diversity. Official train split only. Drop clips whose transcript has digits. |
| Train, Italian | **MLS (Multilingual LibriSpeech) Italian** | CC-BY 4.0 | ≈ 170 h of ≈ 214 h cleaned | Audiobook read speech, longer clips (10–20 s), lowercase transcripts without punctuation. Different domain from Common Voice on purpose. |
| Train, English | **LibriSpeech `train-clean-100`** (CC-BY 4.0) *or* a **Common Voice EN** slice (CC0) | see left | ≈ 100 h | Common Voice EN carries accent metadata; if an *Italian-accented English* slice exists it is the better match for how Italians pronounce English words. **To verify at M2.** |
| Train, mixed | **Recorded mixed IT+EN sentences** (author, friend; script R) | own, CC-BY 4.0 proposed | ≈ 1–3 h gold | §7. Used in H3. |
| Dev | Common Voice IT dev, MLS IT dev, LibriSpeech dev-clean, FLEURS IT/EN **dev**, and a 100-sentence mixed-dev holdout | as above | ≈ 2,000 utterances | Model selection only. |
| Test | **FLEURS IT** and **FLEURS EN** test (CC-BY 4.0), **Common Voice IT** official test, **the recorded mixed test set (script T)** | as above | see §8 | Final numbers only. |
| Noise | Own room tone + digital silence + a permissively licensed noise/music set (e.g. MUSAN; **license to verify** before use) | — | 1 h | §9. |

Not used in v0.1 (backlog): VoxPopuli-IT (≈ 53 h cleaned), Granary/YODAS/YouTube-Commons pseudo-labeled Italian,
parliamentary or podcast data, TTS-synthesized audio.

Total training audio for H2: **≈ 440 h** (≈ 340 h IT + ≈ 100 h EN). Final mixture weights are fixed by dev results.

## 3. Filters and cleaning

Applied while building manifests; every dropped utterance is counted by reason in a build report.

| Filter | Rule |
|---|---|
| Duration | 1.0 s ≤ d ≤ 30.0 s (30 s is the positional-table limit). Training prefers d ≤ 20 s for bucketing efficiency; longer clips are kept but rare. |
| Charset | After `normalize()`, any character outside the charset → utterance dropped. |
| Digits | Transcripts containing digits are dropped from training (the test scripts spell numbers as words). |
| Speaking rate | Characters per second outside a plausible range (initial guess 4–30, set from the histogram) → dropped; catches truncated clips and wrong transcripts. |
| CTC feasibility | `len(tokens) ≤ T_enc` (and room for repeated tokens) or dropped. |
| Sentence overlap | Normalized transcript hash present in any dev/test text → dropped from training. For Common Voice, official splits already separate speakers and sentences; the hash check is a second guard. |
| Empty/corrupt audio | Decode failure, all-zero audio, or clipping > 1% of samples → dropped. |

## 4. Manifests

One JSONL file per split, produced by `stt/data/` and the only input to training and evaluation:

```json
{"id": "cvit27-000123", "audio": "path/or/shard-ref", "text": "raw transcript", "norm": "normalized transcript",
 "duration": 4.37, "source": "cv_it_27", "split": "train", "speaker": "abc123",
 "lang": "it", "category": null, "en_spans": []}
```

- `lang` is `it`, `en` or `mix`.
- `category` and `en_spans` are filled only for the recorded mixed sets (§7): `en_spans` is a list of
  `[start_word, end_word)` index pairs into `norm.split()`.
- Speaker IDs are stable and, for the recorded set, are `S1`, `S2`, `S3`.

## 5. Mel cache

Features are computed once, not per epoch, because the Mac's CPU would otherwise become the bottleneck.

| Item | Value |
|---|---|
| Content | `features.log_mel()` output, float16, shape `[80, T]` |
| Size | 3,600 s × 100 frames × 80 × 2 B = **57.6 MB per hour** → ≈ 26 GB for 450 h (the Mac has ≈ 108 GB free; 13 GB if stored as int8) |
| Layout | Shards of ≈ 1–2 GB: a flat float16 `.npy` plus an index (`id`, offset, frames); memory-mapped by the data loader |
| Raw audio | Stream or download per split, compute mel, **delete raw audio**; keep nothing but manifests + shards |
| Augmentation | Applied on the cached mel at load time (SpecAugment; mel-domain tempo stretch of 0.9–1.1 by time interpolation — an approximation of speed perturbation that does not shift pitch). Waveform-domain augmentation (noise, reverb) is out of scope for v0.1. |

Tooling needed on the Mac: `ffmpeg` for Common Voice mp3 (it is not installed at the time of writing; `brew install ffmpeg`). FLAC (MLS/LibriSpeech) can be read with `soundfile`.

## 6. Mixed (code-switch) text

**Purpose:** (a) scripts for the recordings, (b) tokenizer training text, (c) later TTS or splicing sources.

- **Generation:** LLM-written sentences in controlled categories, then **every test sentence is reviewed by hand**
  by the author for naturalness and pronounceability. Italian orthography; English terms in standard English
  spelling; Italianized forms are written as Italians write them ("deployato", "forwardare").
- **Domains:** work and tech (meetings, deploys, tickets, calendars), everyday life (apps, streaming, shopping),
  brands and apps, acronyms.
- **Categories (also the evaluation categories):**

| Category | Example | Test quota (of 600) |
|---|---|---|
| `single_word` — one English word inside an Italian sentence | "ho spostato il meeting a domani" | 150 |
| `phrase` — multi-word English phrase | "serve un quick update prima della call" | 100 |
| `brand` — app/brand/product name | "apri Spotify e metti la playlist" | 100 |
| `acronym` — spoken-letter or word-like acronym | "il KPI del trimestre" | 60 |
| `italianized` — English stem with Italian morphology | "abbiamo deployato ieri sera" | 80 |
| `switch` — clause-level switch | "va bene, let's talk about it domani mattina" | 60 |
| `control_it` — pure Italian | — | 30 |
| `control_en` — pure English | — | 20 |

- **Annotation:** each sentence carries `category` and `en_spans` (English-origin words, including Italianized
  forms) in the manifest — the basis of ETA (`05`).
- **Size:** ≈ 600 test sentences (script T), ≈ 600–1,000 sentences per training speaker (script R), and a pool of
  ≈ 3–5k further sentences for tokenizer text and later synthesis.
- **Hygiene:** exact and near-duplicate removal across T / R / pool (normalized-text hash + word 3-gram
  Jaccard > 0.8); T text never appears anywhere else. Provenance is logged (generator, date, reviewer).

## 7. Recordings

**Speakers:** S1 (the author), S2 (the friend), S3 (a third person, **held out** from every training use).

| Script | Content | Who reads | Use |
|---|---|---|---|
| **T** (test) | 600 unique sentences, stratified by category so each speaker gets 200 (≈ 1 h total) | S1: 200, S2: 200, S3: 200 | Test only. Once H3 trains on S1/S2's script-R recordings, S1/S2 rows measure *seen speaker, unseen text* and S3 rows *unseen speaker*; before H3 (H0–H2) all three are unseen speakers. Report both groups. |
| **A** (anchor, optional) | 60 sentences from T read by all three | S1, S2, S3 | Separates speaker effect from sentence effect. |
| **R** (train) | 600–1,000 sentences, disjoint from T | S1, S2 (S3 never) | H3 training (≈ 1–1.5 h each); 100 sentences held out as mixed-dev. |

**Protocol**

- Quiet room; record in the same conditions the model will see: at least one speaker through the **browser demo
  path** on a laptop mic, one through a phone. Native 48/44.1 kHz is fine; the pipeline resamples.
- Format: WAV, 16-bit mono (16 kHz or native); file name `S1_0001.wav`; one sentence per file; light silence trim only.
- Read naturally at conversational speed; pronounce English words **the way you would in conversation** (accented or
  not); no re-reading to "fix" a word that came out Italianized; retakes only for flubs; the number of retakes is logged.
- Time: ≈ 6 s per sentence → about 2 h of sessions per 600 sentences including retakes.
- **Consent and licensing:** each speaker consents in writing to the intended release (CC-BY 4.0 for the dataset if
  published; private otherwise). The decision is recorded in the data card.
- The test recordings can start **immediately**, in parallel with everything else (the milestone M1 depends on them).

## 8. Dev and test sets (summary)

| Set | Size | Used for |
|---|---|---|
| Mixed test (T) | 600 utterances, 3 speakers | Primary metric, per-category and per-speaker tables |
| FLEURS IT test / FLEURS EN test | a few thousand utterances each (full test splits) | Italian-only and English-only regression bars |
| Common Voice IT test | official split | Comparison with published Italian figures |
| Noise set | 1 h | Hallucination test |
| Dev | ≈ 2,000 utterances across IT/EN/mixed | Checkpoint selection |

## 9. Noise set (silence/hallucination test)

- 20 min digital silence, 20 min room tone (own recordings, including fan/keyboard), 20 min permissively licensed
  noise and music (license verified before use). Cut into 10 s clips.
- Reported separately: **non-music noise/silence** (hard bar: zero words) and **music** (reported, not gated — the
  Whistle model card also warns that music can still come out as text).

## 10. Deferred data work (documented so it is not lost)

| Idea | Sketch | Trigger to do it |
|---|---|---|
| Utterance-level splicing | Concatenate an Italian and an English clip; transcript is the concatenation. Cheap; gives clause-level switches only. | First thing H3 tries. |
| Word-level splicing | Force-align clips with the H2 model; cut an English word from an English clip, insert it at a word boundary of an Italian carrier sentence with a 20 ms crossfade and loudness match. | Only if H3 shows gold + utterance splicing is not enough. |
| TTS synthesis | Synthesize mixed sentences with an open TTS that has IT+EN voices (the shelved TTS plan is the long-term source). Per-language segments joined; beware prosody seams and license of generated audio. | Only if it improves the mixed set beyond the noise interval. |
| Pseudo-labeling | Parakeet v3 (or NVIDIA's Granary pipeline) labels untranscribed Italian audio; filter by agreement with a second model. | Only for a v0.2 data scale-up. |
| More English-accent data | Common Voice EN Italian-accent slice or similar. | If ETA lags after H3. |

## 11. Licenses and attribution (to place in `NOTICE` and the data card)

- Common Voice (CC0) — no attribution required, cited anyway.
- MLS (CC-BY 4.0) and LibriSpeech (CC-BY 4.0), FLEURS (CC-BY 4.0) — attribution and license links required.
- Whisper weights and `mel_filters.npz` (MIT) — copyright and permission notice must be included.
- LLM-written sentences — check the provider's terms before publishing them as a dataset (open question).
- Speakers' recordings — per-speaker consent (§7).
