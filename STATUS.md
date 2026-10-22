# STATUS — snapshot per riprendere i lavori

> Aggiornato: **2026-10-22** — ultimo commit su `origin/main`: vedi `git log -1`. Suite: **221 test verdi**, `ruff` pulito.
>
> **Decisione (2026-10-22):** per ora si resta sul **modello piccolo** (8 → 11,6 M, ≤ ~12 MB int8) e lo si
> ottimizza al massimo (P0 velocità, +2 blocchi, beam/LM, InterCTC/CR-CTC). Un modello grande (Whisper-base
> o superiore) è rimandato a una fase successiva.
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
| **5.1–5.3 H2** | **E-005 completo**: encoder sbloccato, 437,7 h, **72.592 step**, truncated 192 s/32. Dev best macro **33,77%** (IT 38,08 / EN 29,46). Held-out: **FLEURS-IT 37,41%** vs Whisper-tiny 32,09% (**+5,32** [4,10, 6,69]), FLEURS-EN 62,44%, **CV dev 37,60% vs Whisper 48,53% (−10,93** [−11,33, −10,51]**)**, CV test 39,48%, **T S1 65,35% ETA 16,4%** → **gate fallito solo su FLEURS-IT** | `runs/E-005/`, `EXPERIMENTS.md` (E-005) |
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
- **H2 (E-005) completato** (2026-10-21, 72.592 step, ~8 h di training): gate **fallito solo su FLEURS-IT**
  (37,41% vs 32,09%); su CV dev batte Whisper-tiny (37,60% vs 48,53%). Artefatti: `runs/E-005/`,
  `EXPERIMENTS.md` (E-005).
- **E-006 (+2 blocchi) completato** (2026-10-22, 36.296 step, ~5,4 h di training): **nuovo best Moth**.
  Dev **31,27%** (IT 35,56 / EN 26,99; E-005 33,77). Held-out: FLEURS-IT **34,11%** (E-005 37,41;
  Whisper-tiny 32,09), FLEURS-EN 61,45%, CV dev **34,45%** (Whisper-tiny 48,53), CV test 36,16%,
  T **59,50%** (ETA 20,0%, E-005 16,4%). Il gate H2 resta **fallito solo su FLEURS-IT (+2,02)**,
  ma il gap è più che dimezzato. Artefatti: `runs/E-006/` (config, log, `eval/best/`), `EXPERIMENTS.md`
  (E-006). Il codice P0 + `head_blocks` e gli artefatti E-005/E-006 sono **committati su `main`**
  (2026-10-22).
- **E-007 (beam + hotwords) completato** (2026-10-22, decode-only, nessun training): E-006 congelato,
  beam 5/top-8. Dev macro **30,84%** (−0,44 vs greedy); FLEURS-IT **33,77%** (gap **+1,68** vs
  Whisper-tiny 32,09), FLEURS-EN 60,83%, CV dev **33,98%** (Whisper-tiny 48,53), CV test 35,68%,
  T 59,07% (ETA 20,9%). Hotwords 580 termini, bonus 1,0: FLEURS/CV neutre, T 58,73% ma **ETA 21,8%**
  (peggio del greedy 20,0%): non bastano. Artefatti `runs/E-007/`, `EXPERIMENTS.md` (E-007).
  Prossimo: **LM piccolo (E-008)**; se non chiude, conv head / 25 Hz+2.
- **Dati**: CV IT 27 scaricato ed estratto in `data/raw/cv_it_27/` (11 GB di clip + archivio 9.8 GB, ancora su
  disco); manifest **12 set** in `data/manifests/` + `build_report.json` (E-002 + addendum); cache mel 27 GB.
  Audio raw MLS/LibriSpeech **cancellato** dopo spot check (ricostruibile con `data/raw/fetch.sh`); resta
  **FLEURS (1.7 GB)** per le baseline di 1.4. Nessun dataset mancante.

## Da fare (prossimi task)

| Task | Cosa | Blocco |
|---|---|---|
| 1.4 | E-001: restano base, beam 5, Whistle, Parakeet, mixed T, noise; **CV-IT ora disponibile** (14.643 clip test) | S2/S3/S4 + anchor; CLI Whistle/Parakeet |
| H2 esc. | **Gate H2 fallito solo su FLEURS-IT (+1,68 dopo E-007 beam)**: FLEURS-IT 37,41→34,11→33,77. Beam aiuta ma non chiude; hotwords non migliorano l'ETA. Prossimo **LM (E-008)**; se non basta conv head / 25 Hz+2; Whisper-base resta l'ultima risorsa (tetto ~20 MB, decisione esplicita) | E-008 |
| M6 | H3a/H3b: continue 1–2 epoche, LR 5e-5/3e-4, miscela IT/EN/gold; H2 fatto, manca R-S2 | registrazioni R-S2 |
| M7–M8 | demo e release; benchmark browser con **pesi veri** | dopo H3 |

## Backlog ottimizzazioni (`docs/10`, living review)

Il review **non cambia il piano** e non tocca H2: un item diventa un esperimento solo quando gira, con entry
E-0NN (protocollo: 200 step, stessa config, una variabile). Snapshot del review: H2 a 42,6k step; i numeri
live sono nei log.

- **P0 — fatto (2026-10-22):** fix `dataset.epoch` (+ regression test), cache memmap (nessun guadagno:
  GPU-bound), eval con `inference_mode`/argmax su device; 214 test verdi.
- **P1 — fatto in parte (E-007):** beam implementato e misurato (utile ~0,3–0,6 punti); hotwords
  implementate ma neutre/non utili sull'ETA. Restano checkpoint averaging/EMA, SDPA + time-major + QKV
  fuso (con test di parità), export int8 reale (WER + RTFx), redesign hotword (prefix/fuzzy/word-level).
- **P2 — H3:** InterCTC (prima), CR-CTC short fine-tune, noise augmentation, frame-aware batching;
  il LM piccolo è passato a E-008 (serve per FLEURS, non solo per l'ETA).
- **P3 — +2 blocchi fatti (E-006); restano:** conv head → 25 Hz+2 blocchi → Whisper-base.
- **Vincolo:** niente merge di codice durante un training attivo; E-006 è chiuso ed E-007 è
  decode-only, quindi il prossimo run (LM) può partire dal working tree attuale.

## Serve da te

- Registrazioni T per **S2/S3/S4** e anchor per tutti (quando vuoi); R-S2 per H3.
- Revisione a mano delle 200 frasi nuove di T prima che registri S4.
- `en_terms.txt` è a 580 voci (target design 1–2k): estendibile quando serve.

## Note di contesto utili

- **P0 (2026-10-22)**: fix `dataset.epoch` (l'augmentation ora varia ogni epoca), cache memmap per shard,
  eval con `inference_mode` + argmax su device; 214 test verdi. La cache memmap **non** ha cambiato il
  throughput (473 vs ~477 audio-s/s): il training è GPU-bound, non loader-bound.
- **Whisper-base zero-shot** (riferimento per la futura variante grande): FLEURS-IT **18,27%**, FLEURS-EN
  **9,88%**, ~20,35 M param → int8 ~21 MB, sopra il tetto dei 20 MB. Whisper-tiny: FLEURS-IT 32,09 / EN
  13,86; CV dev 48,53.
- **Working tree**: P0 + `head_blocks` + artefatti E-005/E-006 sono **committati e pushati** su `main`
  (2026-10-22). Checkpoint, audio e shard mel restano fuori da git (`.gitignore`).
- **H2 (E-005)**: full fine-tune 437,7 h, 72.592 step. Dev best macro **33,77%** (IT 38,08 / EN 29,46);
  held-out **FLEURS-IT 37,41%** (Whisper-tiny 32,09%, differenza appaiata +5,32 [4,10, 6,69]),
  FLEURS-EN 62,44%, **CV dev 37,60% vs Whisper-tiny 48,53% (−10,93 [−11,33, −10,51])**, CV test
  39,48%, **T S1 65,35%** con ETA 16,4%. Il gate fallisce solo su FLEURS-IT; su CV dev il modello
  piccolo batte già Whisper-tiny di ~11 punti. Il checkpoint migliore
  è il 62k (0.3377), non l'ultimo; l'averaging degli ultimi 3 non aiuta. La curva si è appiattita dopo
  ~62k: più epoche da sole difficilmente chiudono i 5 punti. `runs/E-005/eval/best/` ha ipotesi e metriche.
- **`dataset.epoch`**: in H2 ogni epoca ha visto la stessa augmentation (`train.py` non aggiornava
  l'epoca). **Corretto in P0** con regression test; E-006 è il primo run con augmentation variabile.
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
