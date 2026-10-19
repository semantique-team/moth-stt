# 05 — Evaluation

One scorer (`stt/eval.py`), one normalizer (`stt/text.py`), one manifest format. Baselines and our model are
different *producers of hypothesis files*; they never touch the scoring code.

## 1. Text normalization — the one rule

`normalize(text, lang_hint="it") -> str` is the only place that decides what counts as the same text. It is used for
tokenizer training, training targets, reference texts, and every system's hypotheses.

1. Unicode NFKC; lowercase.
2. Typographic apostrophes (`’ ‘ ´`) → `'`; remove whitespace after an apostrophe (`l' ho` → `l'ho`).
3. Digits → words via `num2words` in `lang_hint` (default Italian). Training drops utterances containing digits;
   test scripts spell numbers as words; this rule exists so *hypotheses* from systems that emit digits are comparable.
   (Limitation: an English-context number in a baseline's output is spelled in Italian. Rare; noted in reports.)
4. Hyphens, dashes and slashes → space.
5. Keep `a–z`, `à è é ì í ò ó ù ú`, `'` and space; fold other Latin diacritics to the ASCII base letter (`ü`→`u`, `ç`→`c`,
   `ñ`→`n`); drop every other character (punctuation, symbols, emoji).
6. Collapse runs of whitespace; strip.

Words are the whitespace-separated tokens of the normalized text (so `l'ho` is one word for every system).
Properties tested: idempotent; no digits or punctuation in the output; fixture table of tricky strings.

## 2. Manifests and hypothesis files

- **Reference manifest** (JSONL): the format in `03-data.md` §4. `category` and `en_spans` are present on the mixed sets.
- **Hypothesis file** (JSONL): `{"id": ..., "hyp": raw system output, "latency_s": optional}`.
- `eval.py --manifest M --hyp H [--breakdown category,speaker] [--bootstrap 1000]` writes one `metrics.json`
  (below) and prints a table. Baseline scripts in `baselines/` implement `transcribe(wav) -> str` and write the
  hypothesis file; our model's `decode` does the same.

## 3. Metrics

| Metric | Definition | Set |
|---|---|---|
| **WER** | Word-level Levenshtein (substitutions + deletions + insertions) / reference words, corpus-level (sum over utterances). | every set |
| **Per-category WER** | WER restricted to utterances of each category (`03` §6). | mixed test |
| **Per-speaker-group WER** | S1+S2 (seen speakers from H3 onward, always unseen text) vs held-out S3. | mixed test |
| **ETA** (English-term accuracy) | Of reference words inside `en_spans`, the fraction aligned as an *exact match* in the Levenshtein alignment. Errors are listed per term. | mixed test |
| **Insertion rate** | Hypothesis words inserted per 100 reference words (looping/hallucination indicator). | mixed + FLEURS |
| **Silence/noise words** | Words emitted over the 1-hour noise set, counted as total words and as the number of non-empty 10 s clips; **music clips reported separately**. | noise set |
| **RTFx (browser)** | Audio seconds / wall seconds for 5, 10 and 20 s clips, median of 10 runs after 3 warm-ups, Chrome and Safari on the author's M5, 1 and 4 threads. | demo page |
| **Download size** | Sum of bytes of every file the runtime fetches (model, vocab, config, mel filters). Demo page assets excluded. | export |

"Mixed error rate" in the character-level sense is unnecessary: both languages are Latin script, so word-level WER already covers both.

`metrics.json` carries: the manifest hash, the hypothesis hash, the normalizer version, all metrics, bootstrap CIs, per-term
errors, and (for our runs) the git SHA and config path.

## 4. Statistics

- **Paired bootstrap over utterances** (1,000 resamples, seeded): 95% CI of each WER and of the WER **difference** between two
  systems on the same set.
- A gap is called a **tie** when its CI includes 0. With ≈ 600 utterances a difference below roughly 2–3 absolute WER
  points is expected to be noise — report the interval instead of the headline difference.
- Dev sets are used for selection only; the test sets are scored **once per candidate** and each scoring is logged.

## 5. Bars and gates (all relative to H0 measurements)

| Tier / gate | Condition |
|---|---|
| **Ship v0.1** | ≤ 20 MB download. Mixed-set WER **and** ETA better than Whisper-tiny's *best* H0 variant (auto or forced Italian): the 95% paired-bootstrap CI of the WER difference (ours − Whisper-tiny) lies entirely below 0, and the ETA difference entirely above 0. 0 words on non-music noise/silence. Browser RTFx ≥ 10 on the author's M5. int8 within the noise interval of fp32 on dev and the mixed test. |
| **Stretch** | Mixed-set WER within the noise interval of Whistle. |
| **H1 gate** | Head learns; truncated not worse than padded beyond the noise interval (`04` §5). |
| **H2 gate** | Beats Whisper-tiny on Italian FLEURS and Common Voice dev; else escalation list in `04`. |
| **H3 gate** | Keep a data source only if it improves mixed WER or ETA beyond the noise interval without hurting FLEURS IT/EN. |
| **Export gate** | PyTorch fp32 vs ONNX fp32 logits within 1e-3 max-abs; int8 greedy transcripts within the noise interval; browser vs Python transcripts equal on the 20-clip e2e set (`06`). |

## 6. Reference points from other people's reports (not our measurements)

For orientation only; **H0 replaces them**:

- Fine-tuned `whisper-tiny-it`: 27.1% WER on Common Voice 25 (third-party model card).
- Parakeet TDT 0.6B v3 Italian: FLEURS 2.96, MLS 10.12, Common Voice 3.65 (average ≈ 5.6%) — independent benchmark.
- A third-party card (Duyu-small) lists Cactus Whistle at 14.37% on FLEURS en_us with the language forced to English.
- Whistle's own card reports a 7-language average ahead of Whisper base on FLEURS and behind on MLS; its Italian figure is only
  in a chart image and was not read.

## 7. H0 baselines — exact configurations

Run on the Mac, CPU (Parakeet via an ONNX/sherpa-onnx int8 export), no training. Each row produces a hypothesis file per set.

| System | Variants | Notes |
|---|---|---|
| Whisper tiny (39 M) | language auto; language forced `it` | greedy and beam 5; record whether the 30 s padding is used |
| Whisper base (74 M) | auto; forced `it` | same |
| Cactus Whistle (16.9 MB) | auto; forced `it`; ± keyword list | via the `needle` CLI or `cactus-needle` |
| Parakeet TDT 0.6B v3 | ± hotwords | offline accuracy ceiling; ≈ 640 MB int8 |

**Hotword fairness:** a keyword/hotword list that includes the test set's English terms is an **oracle** and is labelled as
such. The fair variant uses terms from the *R script and the English-terms list* only. Our own hotword results follow the same
two-line convention.

Also record per system: size on disk, RTFx on the Mac CPU, and any hallucination on the noise set.

## 8. Reporting

Every experiment appends an `EXPERIMENTS.md` entry (`07` §4) with the metrics table below filled in, linked to its `runs/<id>/metrics.json`:

| System | Mixed WER [CI] | single_word | phrase | brand | acronym | italianized | switch | ETA | FLEURS-IT | FLEURS-EN | CV-IT | noise words | RTFx | size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Systems whose output on the noise set is non-empty list the clip counts in the entry.
