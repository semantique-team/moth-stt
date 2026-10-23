# Recordings — protocol and sentence scripts

> External speakers: see [`EXTERNAL-SPEAKERS.md`](EXTERNAL-SPEAKERS.md) for the Mac recorder,
> the Windows/phone path, speaker roles and the privacy rules.

This folder holds the recording protocol, the script tooling and (soon) the data card. Audio files
never enter git; they live on Hugging Face once the dataset is published (see `docs/03-data.md` §7).

## Sentence scripts

The mixed IT+EN sentences that speakers read are imported into JSONL files under
`recordings/scripts/`:

| File | Content | Size target |
|---|---|---|
| `T.jsonl` | Test script, split across S1–S4 | 800 (200 per speaker) |
| `A.jsonl` | Anchor: 60 T sentences read by every speaker (separate audio folder) | 60 |
| `R-S1.jsonl` / `R-S2.jsonl` | Training script, one per speaker (S3/S4 never) | 600–1000 each |
| `pool.jsonl` | Extra sentences for tokenizer text and later synthesis | 3,000–5,000 |

`T` and `R` are disjoint; duplicates (exact or near, word-3-gram Jaccard > 0.8) are refused
across every file.

### Candidate format

The texts are generated **externally** (any LLM, or by hand). One JSON object per line:

```json
{"text": "ho spostato il meeting a domani", "category": "single_word", "en_spans": [[3, 4]], "generator": "claude-sonnet-4-6", "date": "2026-10-19"}
```

Rules enforced by the tool (`generate_scripts.py`):

- `text` is already **normalized**: lowercase, no punctuation or digits, apostrophes only where
  needed (`l'ho`, `un'idea`); English words in standard spelling (`meeting`, `deploy`), Italianized
  forms as Italians write them (`deployato`, `forwardare`).
- `category` is one of `single_word`, `phrase`, `brand`, `acronym`, `italianized`, `switch`,
  `control_it`, `control_en` (`docs/03-data.md` §6).
- `en_spans` are `[start, end)` index pairs into `text.split()`, covering the **English-origin**
  words (including Italianized forms): empty for `control_it`, exactly one one-word span for
  `single_word`, at least two words for `phrase` and `switch`, at least one for the others.

### The generation prompt (paste into your LLM)

```text
Genera frasi italiane con parole inglesi mescolate (code-switching IT+EN), nell'italiano
parlato quotidiano di lavoro, tecnologia e vita quotidiana.

Regole rigide:
- minuscolo, nessuna punteggiatura, nessuna cifra, apostrofi solo per elisione (l'ho, un'idea);
- parole inglesi in grafia inglese standard (meeting, deploy, update); forme italianizzate
  scritte come le scrivono gli italiani (deployato, forwardare, taggare);
- niente nomi di persona, niente contenuti sensibili o politici;
- per ogni frase indica en_spans come indici [inizio, fine) sulle parole separate da spazio,
  riferiti alle parole di origine inglese.

Categorie e quote (per il set T, 800 frasi): single_word 200 (esattamente una parola inglese),
phrase 133 (almeno 2 parole inglesi consecutive), brand 133 (app/azienda/prodotto),
acronym 80 (sigla stile KPI), italianized 107 (forma italianizzata), switch 80 (cambio di lingua
a livello di frase, almeno 2 parole inglesi), control_it 40 (italiano puro, en_spans []),
control_en 27 (inglese puro).

Domini: lavoro/tech (meeting, deploy, ticket, calendario), vita quotidiana (app, streaming,
shopping), brand e prodotti, sigle. Varia le strutture: non ripetere lo stesso stampo.

Output: una riga JSON per frase, senza altro testo attorno.
Schema: {"text": "...", "category": "...", "en_spans": [[i, j]], "generator": "<modello>", "date": "YYYY-MM-DD"}
```

For `R` scripts ask for 600–1000 sentences per speaker (same format, any category mix); for the
`pool` ask for 3,000–5,000 sentences (categories useful but not quota-bound).

### Workflow

```bash
# import a batch of generated candidates (append + dedupe + validate + quotas)
uv run recordings/generate_scripts.py import --kind T --in candidates.jsonl
uv run recordings/generate_scripts.py import --kind R --speaker S1 --in candidates.jsonl
uv run recordings/generate_scripts.py import --kind pool --in candidates.jsonl

# quotas and counts
uv run recordings/generate_scripts.py stats

# list sentences still to review
uv run recordings/generate_scripts.py review --kind T --limit 50

# after the human review of T (docs/03 §6: every T sentence is reviewed by hand)
uv run recordings/generate_scripts.py mark-reviewed --kind T --reviewer tancredi
```

Only fully-imported `T.jsonl` (with `reviewed: true` rows) may be recorded.

The anchor (`A.jsonl`) is derived from 60 reviewed T sentences and read by **every** speaker; its audio
lives outside the T folders:

```bash
uv run recordings/record.py --speaker S1 --kind A     # files under recordings/audio/anchor/S1/
uv run recordings/build_manifest.py build --kind A --speaker S1   # -> recordings/manifests/A-S1.jsonl
```

## How to record

Prerequisites: `brew install ffmpeg`; macOS asks for microphone permission the first time a
terminal records (if denied: System Settings → Privacy & Security → Microphone → allow the
terminal app).

**Consent first.** Print `consent-template.md`, have every speaker read and sign it (their
release choice goes in `data-card.md`).

**Laptop path (S1, S2) — guided recorder:**

```bash
uv run recordings/record.py --speaker S1 --kind T
```

It shows one sentence at a time (`[12/200] T-0012  (single_word)`), Enter starts the recording,
Enter stops it; after each take `[Enter]` = next, `p` = play back, `r` = redo the same sentence;
`s` = skip, `q` = quit. T files land as `recordings/audio/S1/S1_0001.wav`; R files under
`recordings/audio/R-S1/S1/`, A under `recordings/audio/anchor/S1/`. Retakes and skips are
appended to `session.log.jsonl` next to the audio. **Rerunning the same command resumes
automatically**: existing files are skipped, so `--start N` is only an override. Useful flags:
`--start N` (force an index), `--dry-run` (list only what is missing), `--overwrite`, `--device` (check
`ffmpeg -f avfoundation -list_devices true -i ""` for the microphone index).

**Phone path (S2, S3) — one memo per sentence:** read from the cue sheet, record one sentence per
file, export, then convert and rename:

```bash
uv run recordings/build_manifest.py cue --kind T --speaker S3 --out recordings/cues/T-S3.txt
# then, one by one (or with a small loop):
ffmpeg -i memo.m4a -ac 1 -ar 48000 recordings/audio/S3/S3_0001.wav
```

The builder also accepts `.m4a` and `.flac` files that follow the `S3_0001.m4a` naming.

**Sessions:** ~50–60 sentences per sitting (≈ 10–15 minutes of speech); natural conversational
speed; English words pronounced the way you would say them in conversation; do not re-read a word
that came out Italianized; retakes only for flubs (the recorder logs them). The first file of a
speaker is a good moment to check level: aim for peaks around −12 to −6 dBFS, no clipping.

**Check and build the manifest:**

```bash
uv run recordings/build_manifest.py build --kind T
```

It validates every file (naming, 1–30 s duration, silence, clipping ≤ 1%), writes
`recordings/manifests/T.jsonl` and prints a per-speaker report (`2/3 clips, 12.5s total`, errors,
warnings). Fix the flagged files and rerun. For the training script:
`uv run recordings/build_manifest.py build --kind R --speaker S1`.

## Recording protocol (from `docs/03-data.md` §7)

- **Speakers:** S1 (the author), S2 (a friend), S3 and S4 (**held out** from every training use;
  S3/S4 read only the T script).
- **Order:** T first — it unblocks H0 — then R for S1/S2. Optional anchor A: 60 T sentences read
  by all three speakers.
- **Conditions:** quiet room; at least one speaker through the browser-demo path on a laptop
  microphone, one through a phone; record in the conditions the model will see.
- **Format:** WAV 16-bit mono (native rate is fine, the pipeline resamples); one sentence per
  file, named `S1_0001.wav`; light silence trim only.
- **Delivery:** natural conversational speed; English words pronounced the way you would say them
  in conversation; do not re-read a word that came out Italianized; retakes only for flubs and
  the retake count is logged.
- **Budget:** ≈ 6 s per sentence → about 2 h of sessions per 600 sentences including retakes.

## Consent and licensing

Before any recording is used beyond the private experiments: each speaker consents **in writing**
to the intended release (CC-BY 4.0 proposed for the dataset; private otherwise). The choice and
the per-speaker consent are recorded in the data card. The consent template and the data card
arrive with the manifest builder task.
