# STATUS — snapshot per riprendere i lavori

> Aggiornato: **2026-10-24** — ultimo commit su `origin/main`: vedi `git log -1`. Suite: **246 test verdi**, `ruff` pulito.
>
> **Decisione (2026-10-22):** per ora si resta sul **modello piccolo** (8 → 11,6 M, ≤ ~12 MB int8) e lo si
> ottimizza al massimo (P0 velocità, +2 blocchi, beam/LM, InterCTC/CR-CTC). Un modello grande (Whisper-base
> o superiore) è rimandato a una fase successiva.
> Questo file dice **dove siamo**; il *come* e il *perché* stanno nei design doc (`docs/01`…`09`) e nel
> piano (`docs/plans/2026-10-19-implementation-plan.md`, che ha una nota di sync in testa: i suoi checkbox
> non sono mantenuti, questo file è la fonte live).
> **Materiale per il paper:** `docs/11-evidence-and-decisions.md` — decisioni eseguite con l'evidenza,
> risultati negativi, le tre autocorrezioni e i limiti (single seed, T = solo S1, FLEURS pubblica, ecc.),
> più l'indice di riproducibilità di ogni numero.

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
| Script | **T.jsonl** (800 = 200×S1–S4; 600 riviste, S4 200 da rivedere) e **R-S1.jsonl** (600) | `recordings/scripts/` |

## In corso / fuori dal repo

- **Registrazioni T**: **S1 completo, 200/200** (manifest `recordings/manifests/T.jsonl`, 200 righe, 481 s,
  12 retake). Il set è ora **800 frasi, 200 per S1–S4**; mancano S2, S3 e S4 (leggono solo T). Comandi:
  `uv run recordings/record.py --speaker S2 --kind T` (idem S3, S4).
- **R-S1 COMPLETO** (2026-10-29): **635/635 take, 0 errori, 0 warning, 29,3 min di parlato**
  (`recordings/manifests/R-S1.jsonl`, audio in `recordings/audio/R-S1/S1/`). Cache mel:
  `data/shards/R-S1` (0,49 h, spot check bit-exact su 3 clip). Oversampling ×50 →
  `data/shards/R-S1-x50` (24,4 h, indice duplicato, nessuna copia extra dell'audio).
  **E-016 lanciato** (`runs/E-016/`): 572,4 h totali (gold 4,3%), 2 epoche = 24.478 step.
- **Anchor A**: S1 **60/60** completo; mancano S2/S3/S4 (60 ciascuno). Dopo il T di ogni speaker:
  `uv run recordings/record.py --speaker S2 --kind A` (audio in `recordings/audio/anchor/`).
- **Review**: le 200 frasi nuove (T-0601…T-0800, S4) sono `reviewed: false`; vanno riviste prima
  che S4 registri (`uv run recordings/generate_scripts.py review --kind T`).
- **R-S2**: **344 frasi** (batch 1+2: 293 nette, 8 categorie, 18/18 termini rari, 0 doppioni); ne mancano ~256, da **generare
  nuovi** — il pool non si può riusare, `docs/03` §6 impone il dedupe su T/R/pool. Poi S2 registra
  `uv run recordings/record.py --speaker S2 --kind R`. R-S2 è il blocco residuo di H3a insieme a
  T S2–S4.
- **Guida per speaker esterni**: `recordings/EXTERNAL-SPEAKERS.md` (Mac guided recorder;
  Windows/Linux/phone con cue sheet; consenso, privacy, resume e istruzioni per i loro agenti).
- **H2 (E-005) completato** (2026-10-21, 72.592 step, ~8 h di training): gate **fallito solo su FLEURS-IT**
  (37,41% vs 32,09%); su CV dev batte Whisper-tiny (37,60% vs 48,53%). Artefatti: `runs/E-005/`,
  `EXPERIMENTS.md` (E-005).
- **E-006 (+2 blocchi) completato** (2026-10-22, 36.296 step, ~5,4 h di training): **nuovo best Moth**.
  Dev **31,27%** (IT 35,56 / EN 26,99; E-005 33,77). Held-out: FLEURS-IT **34,11%** (E-005 37,41;
  Whisper-tiny 32,09), FLEURS-EN 61,45%, CV dev **34,45%** (Whisper-tiny 48,53), CV test 36,16%,
  T **59,50%** (ETA 20,0%, E-005 16,4%). Il gate H2 resta **fallito su FLEURS-IT** contro la
  barra greedy (+2,02) e di più contro la barra vera (+6,69, vedi sotto); il gap era più che
  dimezzato rispetto a E-005. Artefatti: `runs/E-006/` (config, log, `eval/best/`), `EXPERIMENTS.md`
  (E-006). Il codice P0 + `head_blocks` e gli artefatti E-005/E-006 sono **committati su `main`**
  (2026-10-22).
- **E-007 (beam + hotwords) completato** (2026-10-22, decode-only, nessun training): E-006 congelato,
  beam 5/top-8. Dev macro **30,84%** (−0,44 vs greedy); FLEURS-IT **33,77%** (gap **+1,68** vs
  Whisper-tiny 32,09), FLEURS-EN 60,83%, CV dev **33,98%** (Whisper-tiny 48,53), CV test 35,68%,
  T 59,07% (ETA 20,9%). Hotwords 580 termini, bonus 1,0: FLEURS/CV neutre, T 58,73% ma **ETA 21,8%**
  (peggio del greedy 20,0%): non bastano. Artefatti `runs/E-007/`, `EXPERIMENTS.md` (E-007).
  Prossimo: **LM piccolo (E-008)**; se non chiude, conv head / 25 Hz+2.
- **E-008 (LM) completato** (2026-10-22, decode-only): BPE trigram **3,53 MB** (195k bigrammi,
  800k trigrammi, backoff 0,4), shallow fusion α=0,2 sul beam E-007. **Contro la barra greedy**
  (Whisper-tiny 32,09%) il gate H2 passava: FLEURS-IT **29,02%**, CV dev **28,39%** vs 48,53%.
  **Correzione 2026-10-23 (E-001 addendum):** il miglior H0 è Whisper-tiny **auto + beam 5** a
  **27,42%** su FLEURS-IT; Moth 29,02 è **+1,60 [+0,32, +2,83]**, quindi il gate **non passa** su
  FLEURS-IT. CV dev resta vinto (−20,13). FLEURS-EN 56,99%, CV test 30,07%, T 55,37%
  (ETA 21,5%). Dev macro **27,05%**. Totale ~15,5 MB. Artefatti `runs/E-008/`, `EXPERIMENTS.md` (E-008). Prossimo:
  export LM/beam browser + RTFx; poi ETA su T e H3.
- **E-009 (export/browser) fatto** (2026-10-23): E-006 int8 **11,91 MB**, fp32 Δ **2,17e-4**,
  int8 top-1 100%; su 100 clip FLEURS-IT WER 30,65% vs 30,92% PyTorch. Browser WASM reale:
  **4T 66,4× / 61,3× / 54,6×** e **1T 29,3× / 28,3× / 24,6×** su 5/10/20 s; load 138 ms.
  Il browser misura ancora il path **greedy**: beam + LM JS non sono ancora portati; totale
  previsto ~15,5 MB. Artefatti `runs/E-009/`.
- **E-010 (browser beam+LM) fatto** (2026-10-23): LM binario CSR **5,09 MB**, parser JS
  (`web/lm.js`), beam JS (`web/decode.js`), `web/ort.js` carica LM e seleziona beam/greedy.
  Release **17,08 MB**. RTFx quality path: **4T 65,2× / 63,3× / 58,1×** e **1T 30,0× / 28,3× /
  25,2×** su 5/10/20 s; decode 4–16 ms. Su **200 clip T**: WER browser **56,66%** vs Python E-008
  55,37% (diff +1,29 [−0,34, +3,02], **tie**); ETA 21,19% vs 21,49%. Differenze per-clip da int8.
  Demo: aggiunto **microfono** (AudioWorklet → 16 kHz, max 30 s), testato con fake device. Artefatti
  `runs/E-010/`. Prossimo: demo e2e + H3.
- **H0 (E-001) quasi completo** (2026-10-23): **Whisper-tiny auto + beam 5 = 27,42% FLEURS-IT**
  e **11,86% FLEURS-EN**; forced `it` + beam 5 è il miglior WER su **T: 53,05% / ETA 37,61%**.
  Moth E-008 su T: 55,37% / ETA 21,49% → **WER tie** (+2,32 [−3,22, +8,02]) ma **ETA −16,12**
  [−25,31, −6,02] → **il gate misto non passa** (serve WER < 0 e ETA > 0). Anche FLEURS-IT non
  passa (29,02 vs 27,42 = +1,60 [0,32, 2,83]); CV dev resta vinto. **Whistle** (16,9 MB) è a
  **11,85% FLEURS-IT** e **29,49% / ETA 50,15% su T**: molto avanti (stretch gate lontano; caveat
  FLEURS pubblico, T è il set pulito); su **CV-IT test** Whistle 20,63 vs Moth 30,07. **Parakeet**
  (639 MB) è il ceiling: **10,66% / ETA 83,88%** su T, **3,65% FLEURS-IT**. Whisper base greedy
  (18,27 IT / 9,88 EN) è solo riferimento.
  Tutti i numeri: addendum E-001. **v0.1 non è spedibile** così: l'ETA è il problema dominante e
  strutturale (BPE V=1024).
- **E-011 noise set fatto** (2026-10-23): **360 clip × 10 s = 60 min** (20 silenzio digitale,
  20 MUSAN noise, 20 MUSAN music), MUSAN **CC-BY 4.0** verificata, seed 0, livelli naturali.
  Non-music (240 clip): Moth greedy **24 parole / 18 clip**, Moth beam+LM **259 / 134** (tutte le
  120 clip di silenzio dicono "non mi": l'LM non ha un prior di blank), Whisper tiny beam5
  **54 / 13**, Whistle **264 / 115**, Parakeet **0 / 0**. Music riportata a parte. **Solo Parakeet
  passa il gate 0 parole**; servono energy gate + noise augmentation (H3). Dettagli: E-011.
  Room tone (20 min) ancora da registrare per arrivare a 1 h non-music.
- **E-014 (H3b splicing) fatto — nuovo best Moth** (2026-10-24): 548 h (438 naturali + **110 h
  spliced**, 21.205 clip costruiti sul mel cache), **22.494 step** da E-006 best in 4 h 16 (uno
  stallo da memory pressure a ~3.040, ripreso da checkpoint). Beam+LM: FLEURS-IT **26,75** (E-008
  29,02; **−2,27** [−2,86, −1,72]), FLEURS-EN **56,04** (−0,95 [−1,57, −0,37]), CV test 30,10
  (tie), **T 57,09 / ETA 20,60** (WER +1,72 tie, ETA −0,90 tie), noise non-music **13** parole
  (era 19). **Contro la barra:** FLEURS-IT vs Whisper-tiny auto+beam5 = **−0,66** [−1,80, +0,47],
  **tie** (E-008 era +1,60 significativo); il gate misto T resta **non passato** (ETA **−17,01**
  [−25,55, −7,01]). **Lo splicing non muove l'ETA** (nessuna categoria migliora; italianized
  7,4 → 3,7): ipotesi H3b respinta. Artefatti `runs/E-014/`, `EXPERIMENTS.md` (E-014).
- **Diagnosi ETA fatto** (2026-10-24): il problema è **copertura lessicale**. T annota 94 termini
  inglesi, **47 non compaiono mai** nel testo di training (CV+MLS+spliced) e **18 non erano in
  nessuno script** (esistevano solo in T): `backlog, cfo, cpu, crm, deadline, deploy, erp, kpi,
  merge, milestone, okr, onboarding, paypal, phishing, url, warranty`… Sulle **41 frasi di T** con
  un termine raro: Parakeet **32/41 (78%)**, Whistle 17/41, Whisper 12/41, **Moth E-014 2/41
  (5%)**. Spiega perché E-012 e E-014 non hanno mosso l'ETA: l'inglese aggiunto era prosa, non
  lessico tecnico. Rimedio: **35 frasi aggiunte a R-S1** e **54 a R-S2**, entrambi gli script ora
  coprono tutti i 18 termini. Protocollo distillazione: `runs/E-015/PLAN.md` (bloccato sui dati,
  non sul compute).
- **Audit script fatto** (2026-10-24): tutte le 4.976 frasi di T/A/R-S1/R-S2 controllate
  (`recordings/scripts/audit-2026-10-24.md`). Esito: **0 non-canoniche, 0 duplicati, 0 violazioni
  di categoria**, R-S1/R-S2 **0 near-duplicate contro T**; copertura **18/18** termini rari.
  Corretti: **15 buchi di annotazione in T** (tutte frasi non ancora registrate: le 200 di S1 erano
  già pulite, quindi gli ETA riportati non cambiano), **5 in R-S1/R-S2**, e **1 near-duplicate**
  rimosso da R-S2. Aperti: **T-0714** e **T-0778** (`single_word` con 2–3 parole inglesi, nel
  blocco S4 non ancora rivisto) e le **190 frasi R non riviste**.
- **E-016 (gold S1) fatto — diagnosi, non nuovo modello** (2026-10-29): R-S1 completo
  (635 take, 29,3 min) ripetuto **×50** (24,4 h) dentro la miscela E-014 (572,4 h), 24.478 step in
  4 h 20 (uno stallo da memoria a ~2.700, ripreso con `empty_cache` 50 → 10). Beam+LM:
  FLEURS-IT 26,98 (tie), FLEURS-EN 56,22 (tie), CV test 30,02 (tie), **T 35,51 / ETA 40,60**
  (WER −21,58 [−25,38, −17,99], ETA +20,00 [+13,69, +25,70]). **Ma il controllo decide:** il
  control_it (italiano puro, nessun inglese) passa da 34,7 a **18,1** e i termini **visti** in R-S1
  guadagnano +27,4 contro +11,9 degli **invisibili**: è soprattutto **adattamento alla voce di S1**,
  non apprendimento dei termini. Contro Whisper su T il WER vince (−17,54) ma l'**ETA è tie**
  (+2,99 [−6,23, +12,29]) → gate misto **non passato**. **E-017 obbligatorio**. Artefatti
  `runs/E-016/`.
- **T di S2 integrato + check gratuito** (2026-10-30): zip ricevuto (260 wav: 200 T + 60 A,
  PCM 16 bit mono 48 kHz, 0 warning), copiato in `recordings/audio/S2` e `anchor/S2`, manifest
  **T a 400 righe** e `A-S2` a 60, cache mel di T ricostruita (400 clip). Sui **200 clip di S2**
  (speaker mai visto in training): E-016 vs E-014 **WER 51,87 → 42,22** (−9,64 [−12,55, −6,93]) e
  **ETA 20,68 → 29,75** (+9,07 [+3,97, +14,61]), entrambi significativi. Quindi circa **metà** del
  guadagno di S1 (−21,6 / +20,0) **trasferisce** a uno speaker nuovo: lo speaker appaiato gonfia
  il guadagno ~2×, non lo inventa. **Da 2026-10-30 tutti i numeri su T sono a 400 clip**; quelli
  precedenti sono sui 200 di S1 e non sono confrontabili direttamente.
- **No-speech guard fatto** (2026-10-24, addendum E-011): `stt/no_speech.py` + `isNoSpeech` in
  `web/mel.js`, soglia **−1.0** calibrata sui shard (silenzio **−1.50** piatto, clip di parlato più
  debole **+0.10**, MUSAN noise +0.64). Ferma **120/120** clip di silenzio in entrambi i path e
  **0/897** clip di parlato (200 T + 697 FLEURS-IT, WER identico a E-006); beam+LM scende da
  **259 → 19** parole non-music, silenzio **240 → 0**. Sul browser il clip silenzioso non arriva
  al modello (`guarded: true`, solo feature). **Il gate noise resta non passato**: 19–24 parole
  su MUSAN noise, che richiedono l'augmentation con target vuoti.
- **E-013 (tokenizer V=4096) completato** (2026-10-23): dev **32,79%** (IT 37,44 / EN 28,14) e
  a 18k meglio di E-006 (34,09), ma sui set veri: FLEURS-IT **36,30 greedy / 36,02 beam** vs E-006
  34,11; FLEURS-EN **60,02 / 59,71** vs 61,45 (meglio); T **59,93 / 59,85**, **ETA 19,7 / 20,0**
  vs E-006 20,0. **Il tokenizer non alza l'ETA** e peggiora IT/T: B da solo non si adotta.
  Artefatti `runs/E-013/`.
- **E-012 (più inglese) completato** (2026-10-23): 200 h EN extra (638 h totali), 2 epoche da
  E-006. Dev **29,50%** (IT 35,78 / EN 23,22; E-006 best 31,27). Sui set veri: FLEURS-IT **29,79**
  beam+LM (E-008 29,02), FLEURS-EN **51,56** (56,99, **+5,4**), **T 60,53 / ETA 21,49** (55,37 /
  21,49), CV test greedy 36,60 (36,16). **A migliora l'inglese ma peggiora T/FLEURS-IT e non muove
  l'ETA**: non si adotta. Crash a 9.350 per tempo-stretch >30 s, corretto `_prepare_batch` (crop
  3000 + test), ripreso da 9.072. Artefatti `runs/E-012/`.
- **Dati**: CV IT 27 estratto in `data/raw/cv_it_27/` (11 GB di clip; archivio cancellato per fare
  spazio); manifest **13 set** in `data/manifests/` + `build_report.json` (E-002 + addendum); cache
  mel **~34 GB** (27 GB + 7 GB LS360). Audio raw MLS/LibriSpeech **cancellato** dopo spot check
  (ricostruibile con `data/raw/fetch.sh`); resta **FLEURS (1.7 GB)** per le baseline di 1.4.

## Da fare (prossimi task)

**Piano operativo attuale:** [`docs/plans/2026-10-24-post-E014-plan.md`](docs/plans/2026-10-24-post-E014-plan.md)
— fasi 0–4 con dipendenze, criteri di accettazione e stime. Ordine: **0** registrazioni (review 89
frasi, poi R-S1), **1** E-016 gold, **2** E-015 distillazione (dopo la decisione sulla fonte audio),
**3** E-017 cross-speaker (S2/S3), **4** re-scope. Trasversali: release con E-014, room tone,
noise augmentation.

| Task | Cosa | Blocco |
|---|---|---|
| 1.4 | E-001: Whisper tiny greedy/beam5 (FLEURS, T), Whistle e Parakeet (FLEURS, T; CV-IT Whistle in corso), base greedy già fatto; CV-IT Whisper non fatto (informativo) | — |
| H2 gate | **E-016: +20 ETA su T ma è adattamento a S1** (control_it 34,7 → 18,1; visti +27,4 vs invisibili +11,9); contro Whisper ETA tie. Serve **E-017** (S3 held out). **Storia:** ETA = copertura lessicale (47/94 termini mai visti, 18 assenti dagli script, Parakeet 78% vs Moth 5%): il prossimo test è **E-016 gold**, poi **E-015** se serve scala. **FLEURS-IT: ora tie** (E-014 26,75 vs Whisper-tiny auto+beam5 27,42; −0,66 [−1,80, +0,47]) — non più dietro in modo significativo, ma il gate chiede una CI tutta sotto 0. **Gate misto T non passato**: E-014 57,09/20,60 vs Whisper it+beam5 53,05/37,61 → WER tie (+4,04 [−1,20, +9,60]), **ETA −17,01** [−25,55, −7,01]. **Noise**: silenzio risolto dal guard (240 → 0), restano **13–24 parole su MUSAN noise**. **Respinti: A (200h EN), B (V=4096), H3b splicing**. Prossimo: distillazione da Whistle/Parakeet, oppure gold H3a, oppure re-scope | decisione |
| M6 | H3b fatto e respinto (E-014); **H3a (gold) pronto da eseguire**: R-S1 76/635, R-S2 344 (servono ~256 frasi nuove), T S2–S4 mancanti | registrazioni R-S1/R-S2/T |
| M7–M8 | demo e release; benchmark browser con **pesi veri** | dopo H3 |

## Backlog ottimizzazioni (`docs/10`, living review)

Il review **non cambia il piano** e non tocca H2: un item diventa un esperimento solo quando gira, con entry
E-0NN (protocollo: 200 step, stessa config, una variabile). Snapshot del review: H2 a 42,6k step; i numeri
live sono nei log.

- **P0 — fatto (2026-10-22):** fix `dataset.epoch` (+ regression test), cache memmap (nessun guadagno:
  GPU-bound), eval con `inference_mode`/argmax su device; 214 test verdi.
- **P1 — fatto (E-007/E-008/E-009/E-010):** beam (+0,3–0,6), LM trigram (4–6), int8 reale
  (11,91 MB), **port JS beam+LM** (RTFx 58,1–65,2× 4T, pacchetto 17,08 MB), T200 e2e tie.
  Restano checkpoint averaging/EMA, SDPA + time-major + QKV fuso, redesign hotword, demo e2e.
- **P2 — H3:** InterCTC (prima), CR-CTC short fine-tune, noise augmentation, frame-aware batching;
  il LM piccolo è passato a E-008 (serve per FLEURS, non solo per l'ETA).
- **P3 — +2 blocchi fatti (E-006); con LM e splicing (E-014) il gate passa su CV-dev e su
  FLEURS-IT è un tie, non ancora un pass.** Capacity ulteriore
  (conv head → 25 Hz+2) non serve per il gate; resta un'opzione se si vuole alzare la qualità
  oltre il tetto attuale.
- **Vincolo:** niente merge di codice durante un training attivo; E-006 è chiuso ed E-007 è
  decode-only, quindi il prossimo run (LM) può partire dal working tree attuale.

## Serve da te

- **Decisione bloccante (2026-10-24, aggiornata)**: il problema dell'ETA non è capacità né
  struttura: è **lessico mai sentito** (47/94 termini assenti dal training, 18 assenti dagli
  script). Piano concordato: **(2) gold prima** — R-S1 è a 76/635 (35 frasi nuove sui termini),
  R-S2 a 344 e va completato con ~256 frasi nuove — poi **E-016** (fine-tune sul gold) misura se
  l'ipotesi originale chiude l'ETA; **(1) distillazione** da Parakeet (78% sui termini rari) è
  pronta come protocollo (`runs/E-015/PLAN.md`) ma **bloccata sulla fonte audio**: gold, audio
  CC0/BY, o TTS (H3c) — serve una decisione di provenienza, non compute; **(3) re-scope** si
  valuta solo dopo i risultati di E-016/E-015. Non registrare R-S1 alla cieca non è più il
  vincolo: ora si sa *quali* frasi servono.
- **S1**: continuare R-S1 da 0023 (524 mancanti) e completare le 600; poi build manifest.
- **Room tone (20 min)**: serve per chiudere il noise set a 1 h non-music. Trova il device con
  `ffmpeg -f avfoundation -list_devices true -i ""`, poi registra con
  `ffmpeg -f avfoundation -i ":<n>" -t 1200 -ar 16000 -ac 1 data/raw/room_tone.wav`
  (ventilatore/tastiera inclusi, stanza silenziosa). Poi:
  `uv run python -m stt.data.build_noise --room-tone data/raw/room_tone.wav --categories room_tone`
  e ricostruisci gli shard `data/shards/noise` con `build_mel.py`.
- **S2/S3/S4**: T + anchor quando disponibili; S4 solo dopo review delle sue 200 T.
- **R-S2**: l'organizzatore genera lo script dal pool; poi S2 registra.
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
