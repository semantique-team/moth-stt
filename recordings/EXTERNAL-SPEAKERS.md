# Registrazioni Moth — guida per S2, S3, S4

Questa guida è per gli speaker esterni e per il loro agente AI. Contiene i comandi, i vincoli e
le cose da **non** fare. La macchina principale del progetto resta quella di S1/Tancredi: qui
spieghiamo solo come registrare e come restituire l’audio.

## 1. Ruoli e set

| Speaker | T (test) | Anchor A | R (training) |
|---|---|---|---|
| S1 | 200, già fatto | 60, già fatto | R-S1, in corso |
| **S2** | **200** | **60** | **R-S2** (600–1000, quando lo script esiste) |
| **S3** | **200** | **60** | **mai** (held-out) |
| **S4** | **200** | **60** | **mai** (held-out) |

- **T** è il test set misto: ogni frase è stata revisionata a mano. S4 può registrare **solo dopo**
  che le sue 200 frasi sono state revisionate.
- **A** è l’anchor: 60 frasi di T lette da tutti gli speaker.
- **R** è il training gold per H3. Solo S1 e S2. S3/S4 non registrano mai R.
- T e R sono script diversi: non confondere le cartelle (vedi sotto).

## 2. Prima di registrare

1. Leggere e firmare `recordings/consent-template.md`. La scelta di release va in
   `recordings/data-card.md`. Senza consenso non si registra.
2. Non caricare audio su GitHub, issue, PR o servizi pubblici. L’audio è dato personale: va
   trasferito privatamente (Drive/Dropbox/WeTransfer) all’organizzatore.
3. Su Mac servono: `git`, Python ≥3.12, `uv`, `ffmpeg`.
4. Non servono dati pesanti, checkpoint, tokenizer o shard: la repo basta.

```bash
git clone git@github.com:semantique-team/moth-stt.git   # oppure HTTPS, se non hai chiavi SSH
cd moth-stt
git checkout main          # oppure il commit che ti passa l’organizzatore
uv sync                    # ~2.5 GB: installa anche torch, che per registrare non serve
brew install ffmpeg         # solo macOS
```

Se vuoi evitare i 2.5 GB di torch, il recorder gira anche senza l’ambiente del progetto
(servono solo `numpy` e `soundfile`; verificato):

```bash
PYTHONPATH=. uv run --no-project --with numpy --with soundfile \
    python recordings/record.py --speaker S2 --kind T --dry-run
```

Non fare `git pull` a metà sessione e non modificare script o manifest.

## 3. Mac: recorder guidato

Il recorder `recordings/record.py` usa **ffmpeg/AVFoundation**, quindi funziona **solo su macOS**.
Chiede una frase alla volta e salva un WAV per frase.

### T — S2 e S3 (S4 solo dopo review)

```bash
uv run recordings/record.py --speaker S2 --kind T
uv run recordings/record.py --speaker S3 --kind T
# S4: attendere la review delle T-0601…T-0800
uv run recordings/record.py --speaker S4 --kind T
```

Comandi durante la sessione:

- **Enter**: avvia la registrazione; **Enter** di nuovo: ferma e salva.
- **p**: riascolta; **r**: rifai; **s**: salta; **q**: esci.
- Al primo utilizzo macOS chiede il permesso microfono al terminale. Se lo neghi:
  Impostazioni di Sistema → Privacy e sicurezza → Microfono → abilita il terminale.
- Durata target: 1–30 s, parlato naturale, picchi circa −12/−6 dBFS, niente clipping.
- Meglio sessioni da **50–60 frasi** (~10–15 min di parlato), non 200 di fila.

### Anchor A — S2, S3, S4

Dopo il T:

```bash
uv run recordings/record.py --speaker S2 --kind A
uv run recordings/record.py --speaker S3 --kind A
uv run recordings/record.py --speaker S4 --kind A
```

### R-S2 — solo S2

Lo script R-S2 deve arrivare a **600 frasi** (2026-10-24: 344, in completamento) e va **rivisto
dall’organizzatore** prima di costruire il manifest. Quando l’organizzatore conferma che è pronto:

```bash
uv run recordings/record.py --speaker S2 --kind R
```

Il builder applica il controllo `reviewed` **solo a T e A**, non a R: per R la revisione resta una
scelta di qualità dell’organizzatore, non un blocco tecnico. Per T e A, invece, una frase non
revisionata ferma il manifest.

### Percorsi audio

| Set | Percorso |
|---|---|
| T | `recordings/audio/S2/S2_0001.wav` |
| A | `recordings/audio/anchor/S2/S2_0001.wav` |
| R-S2 | `recordings/audio/R-S2/S2/S2_0001.wav` |

T e R non condividono più la stessa cartella: è un fix importante. Non spostare o rinominare i
file: il manifest si costruisce **per posizione** (`S2_0001.wav` = prima frase dello script).

## 4. Riprendere una sessione

**Non serve `--start`.** Rilanciando lo stesso comando, il recorder:

1. legge lo script;
2. guarda quali file `S2_0001.wav`, `S2_0002.wav`, … esistono già;
3. salta quelli e riparte dalla prima frase mancante.

```bash
# basta rilanciare lo stesso comando
uv run recordings/record.py --speaker S2 --kind T
```

- `--dry-run` mostra solo cosa manca, senza registrare.
- `--start N` è un override: forza l’inizio dalla frase N (non serve per riprendere).
- `--overwrite` rifà anche i file esistenti: usarlo solo se l’organizzatore lo chiede.
- `--audio-dir` esiste per casi speciali; normalmente non va usato.

Il resume è esatto perché T e R hanno cartelle separate. In precedenza R-S1 condivideva
`recordings/audio/S1` con T: è stato corretto e le registrazioni R già fatte sono state spostate in
`recordings/audio/R-S1/S1/`.

## 5. Costruire il manifest e restituire l’audio

Non costruire un manifest parziale e non committarlo. A fine set:

```bash
# T di tutti gli speaker presenti
uv run recordings/build_manifest.py build --kind T

# Anchor di uno speaker
uv run recordings/build_manifest.py build --kind A --speaker S2

# R-S2
uv run recordings/build_manifest.py build --kind R --speaker S2
```

Il builder controlla nome, durata 1–30 s, silenzio, clipping ≤1% e frasi T revisionate. Se ci sono
errori, correggi i file e rilancia.

Poi zippa **solo l’audio** e invialo privatamente all’organizzatore:

```text
recordings/audio/S2/              # T
recordings/audio/anchor/S2/       # A
recordings/audio/R-S2/S2/         # R-S2
```

Mantieni la struttura di cartelle. Non fare `git add` dell’audio: `recordings/audio/` è
gitignored.

## 6. Windows/Linux o telefono

`record.py` **non funziona** su Windows/Linux: usa `avfoundation`, che è solo macOS. Non provare a
forzarlo. Usa il **phone/desktop path**:

1. L’organizzatore genera un cue sheet:

```bash
uv run recordings/build_manifest.py cue --kind T --speaker S3 --out recordings/cues/T-S3.txt
```

2. Lo speaker registra **un file per frase**, seguendo l’ordine del cue sheet, con il registratore
   del telefono o del PC.
3. Nomi obbligatori, uno per frase:
   `S3_0001.m4a`, `S3_0002.m4a`, … (oppure `.wav`/`.flac`).
4. Durata 1–30 s, parlato naturale, niente clipping.
5. Zippa i file mantenendo la struttura `recordings/audio/S3/` e invia privatamente.

Per A e R il cue si genera con `--kind A --speaker S2` o `--kind R --speaker S2`; i percorsi finali
sono quelli della tabella sopra.

## 7. Cose da non fare

- Non registrare T per S4 prima della review delle sue 200 frasi.
- Non registrare R per S3/S4: sono held-out.
- Non usare `--overwrite` senza motivo.
- Non rinominare i file o cambiarne l’ordine.
- Non committare audio, candidati, cue o dati personali.
- Non usare `--start` per riprendere: basta rilanciare.
- Non fare tuning o training su T; T è test.

## 8. Istruzioni per l’agente AI dello speaker

Se sei un agente che assiste S2/S3/S4:

1. Identifica lo speaker e il set consentito (tabella §1). In caso di dubbio, chiedi.
2. Su macOS puoi usare `record.py`; su Windows/Linux **non** usare `record.py`: genera/usa il cue
   sheet e fai registrare un file per frase.
3. Prima di registrare esegui `--dry-run`; poi rilancia il comando senza `--start` per riprendere.
4. Non usare `--overwrite` e non modificare script, manifest o nomi dei file.
5. Non marcare T come reviewed: la review è umana. S4 aspetta.
6. Dopo la sessione esegui il `build_manifest` del set e riporta gli errori. Non usare
   `--allow-unreviewed` con T o A senza autorizzazione scritta dell’organizzatore: salta la
   revisione umana delle frasi.
7. Zippa l’audio con la struttura corretta e caricalo su un servizio privato concordato; non
   usare git.
8. Verifica che `git status` non mostri audio o file personali.

## 9. Checklist finale

- [ ] Consenso firmato.
- [ ] Script del set giusto (T/A/R) e speaker giusto.
- [ ] `--dry-run` controllato.
- [ ] Tutte le frasi registrate (o consapevolmente saltate).
- [ ] `build_manifest` senza errori.
- [ ] Audio zippato con i percorsi corretti.
- [ ] Nessun audio in git.
