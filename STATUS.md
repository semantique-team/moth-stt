# STATUS — snapshot per riprendere i lavori

> Aggiornato: **2026-10-21** — ultimo commit su `origin/main`: vedi `git log -1`. Suite: **206 test verdi**, `ruff` pulito.
> Questo file dice **dove siamo**; il *come* e il *perché* stanno nei design doc (`docs/01`…`09`) e nel
> piano (`docs/plans/2026-10-19-implementation-plan.md`, che ha una nota di sync in testa: i suoi checkbox
> non sono mantenuti, questo file è la fonte live).

## Come ripartire in una nuova conversazione

1. `cd /Users/tancredi/Desktop/semantique/moth-stt`
2. Al nuovo agente incolla:

   > Leggi `AGENTS.md`, `STATUS.md`, `docs/README.md` e `docs/plans/2026-10-19-implementation-plan.md`,
   > poi `git log --oneline -10` e `uv run pytest -q`, e dimmi da dove ripartire.

3. Regola del repo (`AGENTS.md`): **commit e push solo su richiesta esplicita**.

## Fatto (committato e testato)

| Area | Cosa | Dove |
|---|---|---|
| M0 | normalize, scorer (WER/ETA/bootstrap), manifest schema, E-000 | `stt/text.py`, `stt/eval.py`, `stt/manifest.py`, `EXPERIMENTS.md` |
| M1 (1.1–1.3) | script generator (quote/dedupe/review), recorder guidato + manifest builder + consenso, runner baseline (Whisper/Whistle/Parakeet) | `recordings/`, `baselines/` |
| Dati (2.1–2.4, 2.6) | registry+download+reader, filtri, BPE tokenizer (codice), log-mel con parità Whisper (+ asset 64 KB), dataset+augmentation | `stt/data/`, `stt/features.py`, `stt/tokenizer.py`, `stt/assets/`, `data/en_terms.txt` (580 termini) |
| **2.3 tokenizer** | **BPE IT+EN addestrato**: 1024 pezzi, corpus 100.429 righe (MLS IT 169,9 h + LS EN 100 h pesati 70/30 per parole, pool, 580 termini al 5%), `tokens/word 1.945`, `tokens/s 4.70`, termini ≤3 pezzi 0.488; CLI `stt/data/build_tokenizer.py`, stats nel build report, 6 test | `data/tokenizer/` (gitignored), `data/manifests/build_report.json` (chiave `tokenizer`) |
| **1.1 pool** | **pool.jsonl: 3.137 frasi** (8 categorie, 0 invalide, 58 doppioni scartati), generator `deepseek-v4.1-flash` | `recordings/scripts/pool.jsonl` |
| **T-S1** | **200 take di S1** (481 s, 12 retake, 48 kHz mono, 1,3–5,6 s) + manifest parziale (200/800) | `recordings/audio/S1/` (fuori git), `recordings/manifests/T.jsonl` |
| **4.2 H1** | **probe completato** (E-004, poi corretto): encoder congelato + testa lineare su MLS IT. Padded30 96 s 0.7079 vs truncated 320 s 0.7577; il controllo a parità di schedule (C2: truncated 96 s, cosine 3 epoche, fermato a 1) fa **0.7375** [0.7319, 0.7431] contro ~0.7406 del padded → **padding e troncamento sono equivalenti**; i 3,2 punti iniziali erano lo schedule (cosine 1 vs 3 epoche). **Decisione: H2 usa truncated** | `stt/train.py`, `runs/E-004/`, `EXPERIMENTS.md` |
| **T 800 + A** | T portato a **800 frasi (200 per S1–S4)** con quote per categoria; **anchor A** di 60 frasi; recorder/manifest supportano `S4` e `--kind A` | `recordings/scripts/T.jsonl`, `recordings/scripts/A.jsonl`, `recordings/`, docs |
| **2.5 mel cache** | **cache reale costruita**: **12 set, 509.8 h, 27 GB** (MLS 169.9, CV IT 167.3, LS EN 100, dev/test 72.6), **0 errori**, spot check bit-exact su ogni set → `data/shards/`; report in E-002 + addendum CV | `stt/data/build_mel.py`, `EXPERIMENTS.md` (E-002) |
| **CV IT 27** | **scaricato e buildato** (2026-10-21): 256.7 h train disponibili → campione 170 h → **167.8 h / 113.118 clip** (593 overlap scartati contro dev/test/T), dev 24.86 h, test 25.81 h; split speaker-disjoint verificati (0 parlanti in comune); id clip univoci | `data/manifests/cv_it_*.jsonl`, `build_report.json`, `runs/E-005/` |
| Modello (3.1–3.3) | encoder Whisper-tiny + testa CTC (8.027.009 param), conversione pesi + parità, decode greedy | `stt/model.py`, `stt/convert_whisper.py`, `stt/decode.py` |
| Conversione (3.2) | **verificata sui pesi reali** e materializzata: `models/whisper_tiny_encoder.pt` (30,6 MB; parità 5,2e-4 vs gate 1e-3 contro eager con posizioni del checkpoint; errore relativo 2,8e-5) | `models/` (gitignored), docstring di `stt/convert_whisper.py`, `docs/02` §3 |
| Export + web (3.4–3.5) | ONNX opset 17 (sinusoidi in-graph) + int8: **8.32 MB totali**; smoke browser Playwright; **RTFx 65–74×** (4 thread) / 33–38× (1 thread) con pesi random; E-003 | `stt/export.py`, `web/`, `runs/E-003/` |
| **4.1 train.py** | **loop di training completo**: resume a metà epoca, checkpoint last-3+best, log JSONL, guardie non-finite, dev WER, parity guard MPS↔CPU (**misurata Δ=0.0** su modello e dati reali), input truncated/padded30, augment none/light/full; 8 test; config `runs/E-004/config.yaml` | `stt/train.py`, `tests/test_train.py`, `runs/E-004/` |
| Script | **T.jsonl** (600, riviste) e **R-S1.jsonl** (600) | `recordings/scripts/` |

## In corso / fuori dal repo

- **Registrazioni T**: **S1 completo, 200/200** (manifest `recordings/manifests/T.jsonl`, 200 righe, 481 s,
  12 retake). Il set è ora **800 frasi, 200 per S1–S4**; mancano S2, S3 e S4 (leggono solo T). Comandi:
  `uv run recordings/record.py --speaker S2 --kind T` (idem S3, S4).
- **Anchor A**: 60 frasi di T lette da tutti; 0/240 take. Dopo il T di ogni speaker:
  `uv run recordings/record.py --speaker S2 --kind A` (audio in `recordings/audio/anchor/`).
- **Review**: le 200 frasi nuove (T-0601…T-0800) sono `reviewed: false`; vanno riviste prima che S4 registri
  (`uv run recordings/generate_scripts.py review --kind T`).
- **Da generare/importare (tu)**: R-S2 (600–1000) con `recordings/generate_scripts.py` (il pool è fatto).
- **H2 (E-005) in corso**: `runs/E-005/` — encoder sbloccato, init encoder + testa H1, 438 h (CV IT 167.8 +
  MLS 169.9 + LS EN 100), **truncated 192 s / 32 clip**, 8 epoche (~72.6k step), selezione macro WER IT+EN.
  Misurato: 430–520 audio-s/s, 4,2–8,4 GB con `empty_cache_every: 50`; stima **~8–9 h**. Log:
  `runs/E-005/train.log.jsonl`.
- **Dati**: CV IT 27 scaricato ed estratto in `data/raw/cv_it_27/` (11 GB di clip + archivio 9.8 GB, ancora su
  disco); manifest **12 set** in `data/manifests/` + `build_report.json` (E-002 + addendum); cache mel 27 GB.
  Audio raw MLS/LibriSpeech **cancellato** dopo spot check (ricostruibile con `data/raw/fetch.sh`); resta
  **FLEURS (1.7 GB)** per le baseline di 1.4. Nessun dataset mancante.

## Da fare (prossimi task)

| Task | Cosa | Blocco |
|---|---|---|
| 1.4 | E-001: restano base, beam 5, Whistle, Parakeet, mixed T, noise; **CV-IT ora disponibile** (14.643 clip test) | S2/S3/S4 + anchor; CLI Whistle/Parakeet |
| M6 | H3a/H3b dopo H2: continue 1–2 epoche, LR 5e-5/3e-4, miscela IT/EN/gold; richiede R-S1 (fatto) e R-S2 | H2 + registrazioni R-S2 |
| M7–M8 | demo e release; benchmark browser con **pesi veri** | dopo H3 |

## Backlog ottimizzazioni (`docs/10`, living review)

Il review **non cambia il piano** e non tocca H2: un item diventa un esperimento solo quando gira, con entry
E-0NN (protocollo: 200 step, stessa config, una variabile). Snapshot del review: H2 a 42,6k step; i numeri
live sono nei log.

- **P0 — post-H2, worktree fresco da `main`:** fix `dataset.epoch` (oggi l'augmentation è identica a ogni
  epoca), cache dei memmap, copie ridotte, prefetch 1–2 batch, eval con `inference_mode`/argmax su device.
- **P1 — post-H2, pesi veri:** checkpoint averaging/EMA, SDPA + time-major + QKV fuso (con test di parità),
  export int8 reale (WER + RTFx), beam/hotword.
- **P2 — H3:** InterCTC (prima), CR-CTC short fine-tune, noise augmentation, frame-aware batching, LM
  piccolo solo se l'ETA lo richiede.
- **P3 — solo se il gate H2 fallisce:** conv head → +2 blocchi → 25 Hz+2 blocchi → Whisper-base.
- **Vincolo:** niente merge di codice in `main` durante H2, soprattutto il fix `dataset.epoch`
  (cambierebbe l'augmentation al resume delle epoche 5–8).

## Serve da te

- Registrazioni T per **S2/S3/S4** e anchor per tutti (quando vuoi); R-S2 per H3.
- Revisione a mano delle 200 frasi nuove di T prima che registri S4.
- `en_terms.txt` è a 580 voci (target design 1–2k): estendibile quando serve.

## Note di contesto utili

- **H1 (E-004, corretto)**: la testa lineare su encoder congelato arriva a 0.7079 su MLS dev e si appiattisce.
  Il primo confronto dava 3,2 punti al padding, ma il controllo con lo **stesso schedule** (cosine 3 epoche,
  fermato a 1) dà 0.7375 vs ~0.7406: **equivalente**. L'encoder maschera il padding e usa sinusoidi generate,
  quindi i due input calcolano la stessa funzione per clip. H2 usa **truncated**: 4× meno memoria, ~2,5× meno
  calcolo, e i clip corti di CV riempiono i batch. Vedi l'addendum in `EXPERIMENTS.md`.
- **CV IT 27**: 256,7 h train disponibili, campione deterministico 170 h → 167,8 h dopo i filtri; dev/test
  ufficiali 24,86/25,81 h; split **speaker-disjoint** (0 client_id condivisi); 593 righe train scartate per
  overlap con dev/test/T. Gli id sono `cv-<stem del clip>` (il client_id non è univoco).
- **MPS**: con batch a lunghezza variabile la cache dell'allocator cresce fino a 20 GB e va in swap;
  `empty_cache_every: 50` la tiene a **4,2–8,4 GB** con **430–520 audio-s/s** (smoke E-005).
- **Tokenizer**: 1024 pezzi, 1.945 token/parola e 4.70 token/s sulle trascrizioni (contro i 50 frame/s
  dell'encoder); i termini inglesi entro 3 pezzi sono il 48.8% e alzare l'up-weight (0.05 → 0.30) non li
  migliora (misurato): il limite è V=1024, non la quota. Pezzi, id ed encoding sono stabili fra run; i byte
  del `.model` variano per rumore float, quindi il build report registra lo sha256 dell'artefatto costruito.
  `data/tokenizer/` è gitignored: da rilanciare il CLI (~3 s) dove servono i file.
- **Parità MPS↔CPU del CTC loss: Δ = 0.0** (identica) su modello e dati reali — il dubbio di `docs/04` §1 è risolto.
  Smoke di 3 step su MLS dev (MPS, bf16, batch 320 s): ~330 audio-s/s, ~6.5 GB peak (non è un benchmark).
- **Posizioni**: la tabella `embed_positions` del checkpoint HF differisce dalla formula delle sinusoidi di 3,03e-4
  (artefatto di precisione del training originale) e HF stesso si muove di 2,2e-3 tra i suoi kernel sdpa/eager:
  il gate di conversione è 1e-3 con posizioni del checkpoint sostituite e riferimento eager (misurato 5,2e-4).
  Noi **rigeneriamo** la tabella (design `docs/02` §3): l'init è una perturbazione minima di whisper-tiny.
- Parametri totali **8.027.009** (V_ctc = V_sp+1); sinusoidi in **float32** come Whisper.
- Il test set T **non** si usa per training né per tuning; i sorgenti eval-only (FLEURS) sono rifiutati per `train`.
- `build_mel.py` consuma i risultati a lotti (bounded memory): senza il fix il processo padre avrebbe tenuto
  ~10 GB di float16 per `mls_it_train` su 16 GB di RAM.
- Audio raw: MLS IT + LibriSpeech cancellati dopo secondo spot check (72.524 file, ~22 GB liberati); FLEURS
  (1.7 GB) resta finché le baseline di 1.4 non sono chiuse, poi si può cancellare anche quello.
- Test int8 dell'export: pesi seeded e bound per piattaforma — i kernel int8 x86_64 di onnxruntime derivano
  ~40× di più che su ARM64 (misurato, E-003 addendum); il gate int8 vero (Task 7.1) va misurato sulla
  piattaforma target.
- Macchina: M5 16 GB; `torch 2.14.1` pinnato; `ffmpeg` installato; `uv` gestisce l'ambiente.
