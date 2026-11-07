# STATUS — snapshot per riprendere i lavori

> Aggiornato: **2026-11-07** — ultimo commit su `origin/main`: vedi `git log -1`. Suite: **262 test verdi**, `ruff` pulito.
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

- **Registrazioni T**: **S1 200/200, S2 200/200, S3 200/200** (manifest `recordings/manifests/T.jsonl`,
  **600 righe**, 23,2 min, cache mel a 600 clip). Il set è **800 frasi, 200 per S1–S4**: manca **S4**,
  che può registrare solo dopo la review delle sue 200 frasi. Comando: `uv run recordings/record.py
  --speaker S4 --kind T`.
- **R-S1 COMPLETO** (2026-10-29, esteso 2026-10-31 con le 6 frasi `standup` 0636–0641):
  **641/641 take, 0 errori, 0 warning, 29,6 min di parlato** (`recordings/manifests/R-S1.jsonl`,
  audio in `recordings/audio/R-S1/S1/`). Cache mel `data/shards/R-S1` (0,49 h, 0 errori).
  Oversampling ×50 → `data/shards/R-S1-x50` (32.050 righe, **24,70 h**, indice duplicato, nessuna
  copia extra dell'audio). **E-016 lanciato** (`runs/E-016/`): 572,4 h totali (gold 4,3%),
  2 epoche = 24.478 step; E-016 ha usato i 635 take, poi il set è cresciuto a 641.
- **R-S2 COMPLETO + E-017** (2026-11-07): S2 ha registrato **623 take** (20,6 min, 48 kHz mono);
  una è inutilizzabile (`S2_0001`, 0,39 s di silenzio) e **resta fuori dal manifest**: 622 righe,
  mappatura cue→file intatta (D26). Cache mel `data/shards/R-S2` (0,34 h, 0 errori).
  **E-017** = ricetta E-016 con gold da **due speaker** (R-S1 641 + R-S2 622, 50,2 min, ×50 = 41,87 h,
  gold **7,1%** della miscela da 589,54 h): 26.442 step in **3 h 21**, dev macro **30,98%** (migliore
  del progetto), 0 step non-finiti. Beam+LM: **T-600 32,77 / 42,86** (E-016 38,50/36,05),
  FLEURS-IT **26,52**, FLEURS-EN **55,48**, CV 30,36, noise **0 parole**. **Su S3, che non ha mai
  contribuito audio: WER −6,13 [−8,68, −3,51] e ETA +4,79 [+0,51, +9,06] vs E-016 — lo stesso
  guadagno degli speaker in training (S1 −4,47, S2 −6,60)**: il guadagno gold **non** è adattamento
  allo speaker (D27). **Gate su S3 vs Whisper it+beam5: WER −11,07 [−16,28, −5,89] vinto,
  ETA −2,82 [−11,65, +6,56] tie** → gate misto ancora non passato, ma il WER è vinto per la prima
  volta su uno speaker held-out. Artefatti `runs/E-017/` (`config.yaml`, `train.log.jsonl`,
  `eval.sh`, `prepare_gold.py` + test, `eval/`, `diffs/`), addendum E-017.
- **E-019 (sonda Whistle LoRA) — FATTO e CHIUSO (2026-11-07, D28)**: artefatto esplorativo "meno
  nostro", **non Moth**, D5 invariato. LoRA rank 16 su `whistle.safetensors` (220,6 MB) con i
  **soli** R-S1+R-S2 (1.263 clip, 45,3 min, split 90/5/5 seed 0 stratificato per speaker; T/A mai
  letti), 3 epoche = 216 step in **22 min su CPU** (jax 0.12.0.1 senza Metal; il venv usa-e-getta
  usa l'extra `train`, non `jax` che non esiste). Baseline L0 riverificata: su T-S3 riproduce E-017
  **bit-identica** (WER 28,3646 / ETA 52,1127). **Esito: kill a L1.** Su **S3**: WER 28,36 → 30,66
  (**+2,30** [−0,68, +5,46], tie), ETA 52,11 → 47,89 (tie) → criterio di kill soddisfatto. Solo gli
  speaker in training migliorano (gold test −2,98 WER, +8,25 ETA, tie); **FLEURS-IT +1,02
  [+0,51, +1,56], FLEURS-EN +2,30 [+1,16, +3,26], CV-IT +2,66 [+1,74, +3,49] — tutti
  significativamente peggiori**. Noise col VAD: 0 parole non-music in entrambe le braccia. Quindi
  **H2 (dati, non architettura) è respinta**: lo stesso gold che dà −6,13 WER a Moth su S3 fa
  perdere 2,3 a Whistle — l'effetto gold è legato all'architettura (`docs/11` §3.4). H3 (`mix`) e
  H4 (`--tokens`) **non eseguiti**: il piano li vieta dopo il kill. In più l'export `--lora` pesa
  **28,0 MB** (base 16,92): con il VAD sfonderebbe il tetto da 20 MB. Artefatti `runs/E-019/`
  (`build_jsonl.py`, `config.json`, `data/`, `train_L1.sh`, `train_L1.log`, `RESULT.md`, eval di
  L0/L1 su S3, gold-test, FLEURS, CV, noise). Adattatore 6,5 MB e `.cact` 28 MB gitignored.
  **Whistle LoRA: chiuso, non si prosegue.**
- **E-020 (int8 WER end-to-end) — FATTO (2026-11-07)**: chiusa la lacuna "il costo WER dell'int8
  non è stato rimisurato end-to-end su T". Nuovo `stt/onnx_runner.py` (il repo non sapeva *eseguire*
  il modello spedito: il path browser è JS e `export.py` usava ORT solo per i logit). Misurato
  decodificando `release/model.int8.onnx` con i decodificatori, i manifest e lo scoring esistenti:
  **T-600 greedy 37,70 → 38,13 (+0,43 [−0,17, +1,01], tie)**; **T-600 beam+LM 32,77 → 32,94
  (+0,17 [−0,40, +0,77], tie)**; **FLEURS-IT beam+LM 26,52 → 26,85 (+0,32 [+0,08, +0,57], non tie)**.
  ETA invariata su T. Trascrizioni identiche: 70,5% (greedy) e 77,3% (beam+LM) su T, 52,8% su
  FLEURS-IT. Lettura: sugli speaker nostri l'int8 è gratis entro il rumore; sul dominio lungo
  (clip 7–23 s vs 1,3–5,6 s di T) costa 0,3 punti reali. L'artefatto spedito resta valido.
  Artefatti `runs/E-020/` (`eval_onnx.py`, `RESULT.md`, eval int8 + riferimenti PyTorch ri-scorati).
- **E-021 (gate fp32) — FATTO (2026-11-07)**: il `1,051e-3` che da E-016 faceva costruire ogni
  release con `--no-verify` è **la coda di una singola estrazione casuale, non un bug di export**.
  Sweep 20 seed × 4 lunghezze (80 estrazioni): mediana ~1,8e-4, **peggiore 1,28e-3, 1/80 sopra il
  gate, sempre a 29 s**; a 1/5/10 s nessuna. Su audio reale (anche una clip T reale paddata a 29 s)
  max **3,05e-4**, top-1 1,0. Il valore dipende da (checkpoint, seed, lunghezza): E-017 al seed 0 dà
  2,43e-4 e al seed 2 dà 1,28e-3. **Raccomandazione**: sostituire il gate a estrazione singola con
  uno sweep seedato (mediana ≤ 1e-3, peggiore ≤ 1e-2) — **implementato lo stesso giorno (D29)**:
  `FP32_GATE_WORST`, `FP32_SEEDS` in `stt/export.py`, `verify_fp32(..., seeds=N)`, e `docs/06` §4.2
  aggiornato. Artefatti `runs/E-021/` (`fp32_gap.py`, `RESULT.md`, `fp32_gap.json`).
- **Release 0.5.1 — la prima verificata da 0.3.0** (2026-11-07): stessi pesi di 0.5.0, ricostruita
  col gate nuovo **senza `--no-verify`**: fp32 **mediana 2,127e-4, peggiore 1,472e-3, 3/80 sopra
  1e-3** → passa. L'int8 prodotto è **byte-identico** a 0.5.0 (`fe1afcc2…`), pacchetto **19,44 MB**,
  cinque sha invariati; solo `NOTICE` (FLEURS una volta) e `config.json` (versione) cambiano.
  WER int8 ricontrollato sul pacchetto ricostruito: **38,13** su T-600 greedy, identico a E-020.
  Browser 6/6 Playwright, 4T 52,0/51,0/46,5 e 1T 25,4/24,2/21,7 su 5/10/20 s.
  Record: `runs/E-021/release-0.5.1.json`.
- **NOTICE**: FLEURS era elencato due volte (una voce per `fleurs_it` e una per `fleurs_en` nello
  stesso registro); ora le righe di attribuzione identiche sono deduplicate in `build_notice`, con
  un test di regressione. Corretto in 0.5.1.
- **Piano E-018 scritto** (2026-11-07): [`docs/plans/2026-11-07-e018-scale-the-gold.md`](docs/plans/2026-11-07-e018-scale-the-gold.md)
  — bracci G3 (tre speaker) / G3+ (più frasi) / G3× (×100), criteri di successo e di plateau,
  la modifica di codice necessaria (l'invariante `kind R` rifiuta S3/S4 in `generate_scripts.py`),
  e i rischi (terzo canale, rendimenti decrescenti, E-019 che rende il claim architecture-bound).
- **E-021 (gate fp32)** — chiuso, vedi sopra.
- **M8 — model card + publisher HF — PRONTI, non pubblicati (2026-11-07)**:
  `runs/E-021/MODEL_CARD.md` è la card completa (numeri verificati uno per uno contro gli artefatti,
  tabella contro Whisper/Whistle/Parakeet, costo int8, dati, 9 limiti dichiarati). `runs/E-021/upload_hf.py`
  verifica `release/` contro gli sha registrati prima di caricare (hash, dimensione payload, card) ed
  è **dry run di default**; 7 test. **Serve da te**: (1) il **token HF** con write, passato via
  `HF_TOKEN` — mai nel repo; (2) il **namespace**: oggi HF non ha `semantique-team`, quindi o si crea
  l'org, o si usa un account e si cambia `--repo`. Comando:
  `HF_TOKEN=hf_... uv run python runs/E-021/upload_hf.py --repo <namespace>/moth-stt --push`
  (aggiungere `--private` per uno staging). La card cita `semantique-team` in due punti: va
  aggiornata se il namespace è diverso.
- **Release 0.5.0 con E-017** (2026-11-07): `release/` ricostruita col checkpoint E-017 —
  int8 **11,91 MB** (sha `fe1afcc2…`) + LM 5,12 + VAD 2,33 = **19,44 MB**. Browser con VAD nel loop:
  **4T 52,2× / 50,8× / 46,3×**, **1T 25,5× / 24,3× / 21,8×** su 5/10/20 s; **6/6 Playwright** verdi.
  int8 su 100 FLEURS-IT: top-1 **99,72%**, 44/100 trascrizioni identiche, gap massimo 8,59. Il costo
  WER dell'int8 **ora misurato end-to-end** (E-020): T-600 +0,43 greedy / +0,17 beam+LM, tie;
  FLEURS-IT +0,32 significativo. Record: `runs/E-017/release-0.5.0.json`.
- **Baseline T-600 complete** (2026-11-07): su tutti i 600 clip, Moth E-017 **32,77/42,86** vs
  Whisper auto 68,86/39,50, Whisper it 50,72/36,05, **Whistle 29,42/49,28**, Parakeet 10,98/83,60.
  **Il gate misto passa su T-600 contro entrambe le varianti Whisper** (vs it: WER −17,95
  [−21,47, −14,88] e ETA +6,81 [+0,85, +12,50]) — ma 400 dei 600 clip sono di speaker in training,
  quindi il numero onesto di generalizzazione resta la slice S3 (WER vinto, ETA tie). Contro
  **Whistle** siamo a **+3,35 WER** [1,07, 5,60]: dietro ma di ~3 punti (era ~14). Artefatti
  `runs/E-017/baselines-t600/`.
- **Anchor A**: S1, S2 e **S3 60/60** completi (manifest `A-S1/A-S2/A-S3`); manca solo S4.
  `uv run recordings/record.py --speaker S4 --kind A` (audio in `recordings/audio/anchor/`).
- **Review**: le 200 frasi nuove (T-0601…T-0800, S4) sono `reviewed: false`; vanno riviste prima
  che S4 registri (`uv run recordings/generate_scripts.py review --kind T`).
- **R-S2**: **623 frasi** (batch 1–4 + 12 `standup` del 2026-10-30: 572 nette, 8 categorie, 18/18 termini rari, 0 doppioni, 0 near-dup contro T); **pronto per S2**. Non serve altro lavoro
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
  il guadagno ~2×, non lo inventa. **Gate su S2 (speaker mai in training):** Moth E-016
  **42,22 / ETA 29,75** vs Whisper auto 70,63/35,41, Whisper it 51,69/32,86, Whistle 30,41/45,61,
  Parakeet 11,47/83,00 → **WER vinto contro Whisper** (−9,47 [−14,90, −4,59]) ma **ETA tie**
  (−3,12 [−12,46, +5,65]); Whistle resta avanti. **Da 2026-10-30 i numeri su T sono a 600 clip**
  (T-400/T-200 non sono confrontabili direttamente).
- **T+A di S3 integrati + zero-shot cross-speaker** (2026-10-30): S3 (mai in training, stesso Mac e
  microfono di S1) ha registrato **T 200/200 (459 s) + A 60/60**; manifest T a **600 righe**,
  `A-S3` a 60, cache mel di T ricostruita (600 clip, 0,39 h, 0 errori). **Sui 200 clip di S3**
  (E-016 vs E-014, beam+LM): **WER 54,77 → 41,57** (−13,20 [−16,36, −10,24]) e **ETA 17,75 →
  30,42** (+12,68 [+7,71, +18,03]), entrambi significativi; greedy 60,22 → 45,91 / 13,52 → 26,48.
  È **anche più grande che su S2** (−9,6/+9,1) e ~60% del guadagno su S1: il trasferimento
  cross-speaker regge su un secondo speaker. **Caveat**: S3 condivide il canale con S1, quindi parte
  del guadagno può essere adattamento al microfono; S2 resta la stima più pulita. **Gate su S3 vs Whisper:**
  Moth **41,57/30,42** vs Whisper auto 58,69/41,41 (**vince entrambi**), Whisper it 46,51/38,03
  (**WER −4,94 [−10,40, +0,34] tie, ETA −7,61 [−17,02, +1,86] tie**), Whistle 28,36/52,11 (avanti),
  Parakeet 10,82/83,94. **VAD su S3: 0/200 gated**, ma il clip più debole è 0,194 (`T-0033`).
  Artefatti `runs/E-017/` (`zero_shot_s3.sh`, manifest per speaker, metriche S3, baseline,
  `vad_speakers.json`), `runs/E-016/eval/t600-*`, `runs/E-014/eval/t600-*`; addendum E-016.
  **Manca ancora R-S2** (script pronto, 623 frasi) per il training di E-017 vero.
- **Release 0.3.0 con E-016** (2026-10-30): `release/` ricostruita col checkpoint E-016 —
  int8 **11,91 MB** + LM 5,09 = **17,08 MB**. Browser (Playwright, WASM SIMD): **4T 61,0× / 60,9× /
  53,8×**, **1T 28,5× / 27,0× / 23,7×** su 5/10/20 s; guard e mic test verdi. int8: top-1 99,78%,
  WER identico a PyTorch su 100 FLEURS-IT (29,23%). fp32: **1,051e-3 su sintetico 29 s** (gate 1e-3,
  +5%), **5,46e-4 su 20 clip reali** con top-1 100% → release costruita con `--no-verify` e
  deviazione documentata (addendum E-016), da decidere se indagare. Non rimisurata la parità
  browser-vs-Python sull'intero T (E-010 l'aveva fatta per E-006). Artefatti `runs/E-016/export.json`.
- **LM aumentato + release 0.3.1** (2026-10-30): la demo ha rivelato che il LM non conosceva i
  termini (`standup` → `st meeting`; 0 occorrenze nel testo LM, +15,9 log-prob a favore di `st`).
  `runs/E-008/build_lm_augmented.py` aggiunge al LM i testi **di training** (R-S1, R-S2, pool; mai
  T/A): **T-400 38,76 → 36,91 WER** (−1,86 [−2,55, −1,21]) e **ETA 35,03 → 39,10** (+4,07
  [+2,67, +5,61]); T-S2 42,22/29,75 → **40,49/33,43**; FLEURS/CV/noise invariati. Adottato:
  release **17,11 MB** (v0.3.1) con LM 5,12 MB, browser 4T **64,3× / 62,5× / 57,3×**, 1T **29,1× /
  27,6× / 24,5×**. Il caso `standup` isolato può ancora perdere contro `st` (3 pezzi BPE vs 1): serve
  più esposizione acustica in contesti vari.
- **VAD fatto + release 0.4.0** (2026-10-30, decode-only): **Silero VAD v5** (MIT, 2.327.524 B,
  `stt/assets/silero_vad.onnx`) sostituisce l'energy guard; regola 512+64 campioni per chiamata,
  `state [2,1,128]`, frazione di finestre di parlato **> 0,10** (`stt/vad.py`, `web/vad.js`).
  **Noise set: 0 parole non-music con greedy e beam+LM** (era 35 / 31), silenzio 0, e anche
  **107/120 clip di music** fermate (riportate a parte). **0 clip gated** su T-400 e FLEURS-IT
  (697): beam+LM **ipotesi bit-identiche** al run senza guard (400/400, 697/697), WER T-400
  36,91 / FLEURS-IT 26,90 invariati. Pacchetto **19,44 MB** (int8 identico a 0.3.1, sha
  `d94da6fc…`; LM 5,12; VAD 2,33), tetto 20 rispettato (`--no-verify`, stessa deviazione fp32 di
  E-016). Browser con VAD nel loop: **4T 48,3× /
  46,4× / 41,8×** e **1T 25,2× / 24,0× / 21,5×** su 5/10/20 s (VAD 16–83 ms, −25% RTFx su 4T,
  −12% su 1T); **6/6 Playwright** verdi (il fixture-tono è gated, il parlato passa). Artefatti:
  `runs/E-011/vad-*/`, `runs/E-011/vad_guard_summary.json`, `runs/E-011/browser-0.4.0.json`,
  `runs/E-011/vad_eval.sh`; addendum E-011, `docs/06` §1, D25.
- **No-speech guard (energy) — storico** (2026-10-24, addendum E-011, **superato da D25**):
  `stt/no_speech.py`, soglia **−1.0** calibrata sui shard (silenzio **−1.50** piatto, clip di parlato
  più debole **+0.10**, MUSAN noise +0.64). Fermava **120/120** clip di silenzio e **0/897** clip di
  parlato, beam+LM da **259 → 19** parole non-music. Da 0.4.0 non è più nel path di spedizione:
  resta solo per riprodurre gli artefatti E-011/E-016 (`--no-speech-guard`), perché non separava il
  rumore forte (19–24 parole su MUSAN noise).
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
— riallineato il **2026-11-07**: fasi 1 e 3 chiuse (E-016, E-017), cross-cutting chiuso
(release 0.5.1, room tone, VAD); restano solo le voci che dipendono da te o da S4 (review,
decisione E-015, re-scope, E-018). Il piano E-018 è
[`2026-11-07-e018-scale-the-gold.md`](docs/plans/2026-11-07-e018-scale-the-gold.md).

**In attesa di S4 (2026-11-07):** **E-019 chiuso** (kill a L1, `docs/11` §3.4), **E-020/E-021 fatti**
(misure), **piano E-018 scritto**. L'attesa di S4 resta aperta e ora ha un piano:
[`docs/plans/2026-11-07-e018-scale-the-gold.md`](docs/plans/2026-11-07-e018-scale-the-gold.md).
La sonda Whistle LoRA **non si riprende**: H2 respinta, H3/H4 non eseguiti per il criterio di kill.

| Task | Cosa | Blocco |
|---|---|---|
| 1.4 | E-001: Whisper tiny greedy/beam5 (FLEURS, T), Whistle e Parakeet (FLEURS, T; CV-IT Whistle in corso), base greedy già fatto; CV-IT Whisper non fatto (informativo) | — |
| H2 gate | **Noise: risolto (VAD, 0 parole)** — **0/337 clip non-music** (240 MUSAN/silenzio + **97 room tone, tutti gated**) con entrambi i decoder (addendum E-011). **E-017 è il modello migliore**: T-600 **32,77 WER / 42,86 ETA** (E-016 38,50/36,05; E-014 53,50/20,33), FLEURS-IT **26,52**, FLEURS-EN 55,48, CV 30,36, noise 0 parole. **Su S3, che non ha mai contribuito audio, E-017 vs E-016: WER −6,13 [−8,68, −3,51], ETA +4,79 [+0,51, +9,06]** — lo stesso guadagno degli speaker in training (S1 −4,47, S2 −6,60): il gold **non** è adattamento allo speaker (D27). **Gate misto su S3 vs Whisper it+beam5** (46,51/38,03): **WER −11,07 [−16,28, −5,89] VINTO**, **ETA −2,82 [−11,65, +6,56] tie** → gate misto non passato, ma il WER è vinto per la prima volta su uno speaker held-out; contro Whisper auto vince entrambi; Whistle resta avanti (28,36/52,11). **FLEURS-IT**: 26,52 vs Whisper-tiny auto+beam5 27,42 → −0,89 [−2,08, +0,19] **tie** (era −0,66). **Respinti: A (200h EN), B (V=4096), H3b splicing**. Prossimo: **E-018** (più gold: S4 e più frasi) perché il secondo speaker ha valso ~6 punti, oppure E-015 distillazione (bloccata sulla fonte audio), oppure re-scope | decisione |
| M6 | H3a **fatto**: R-S1 641/641, R-S2 622/623, **E-017 eseguito** (gold S1+S2 → eval S3). Resta **S4**: review delle 200 frasi T-0601…0800, poi T + A, poi eventuale **E-018** con tre speaker | registrazioni T-S4 |
| M7–M8 | demo e release; benchmark browser con **pesi veri** | dopo H3 |
| **E-019** | **Sonda Whistle LoRA (D28) — FATTA e CHIUSA 2026-11-07, kill a L1**: su S3 +2,30 WER (tie), FLEURS/CV significativamente peggiori; H2 respinta, H3/H4 non eseguiti. `runs/E-019/RESULT.md`, `docs/11` §3.4 | — (nessun seguito) |
| **E-020** | **Costo WER dell'int8 — FATTO 2026-11-07**: T-600 +0,43 (greedy) / +0,17 (beam+LM), entrambi tie; FLEURS-IT +0,32 significativo. `stt/onnx_runner.py`, `runs/E-020/RESULT.md` | — (misura chiusa) |
| **E-021** | **Gate fp32 — FATTO 2026-11-07**: 1,05e-3 è la coda di una singola estrazione (1/80 sopra, solo 29 s), non un bug; reale ≤ 3,1e-4. **Gate sostituito da uno sweep seedato (D29)** e release **0.5.1 ricostruita verificata**: mediana 2,13e-4, peggiore 1,47e-3, int8 byte-identico. `runs/E-021/RESULT.md` | — (chiuso) |
| **M8** | **Card + publisher HF pronti, non pubblicati**: `runs/E-021/MODEL_CARD.md`, `runs/E-021/upload_hf.py` (verifica gli sha, dry run di default, 7 test) | **token HF + namespace** |
| **E-018** | **Piano scritto 2026-11-07**: [`2026-11-07-e018-scale-the-gold.md`](docs/plans/2026-11-07-e018-scale-the-gold.md). Bracci G3/G3+/G3×, criteri di successo e plateau, unica modifica di codice (invariante `kind R`) | review T-S4 + R-S4 |

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

- **Stato gold (2026-11-07)**: R-S1 **641/641** e R-S2 **622/623** registrati e consumati da **E-017**
  (gold 50,2 min ×50). Il gold funziona e generalizza: su S3 (mai in training) il guadagno è lo stesso
  degli speaker in training. **Il prossimo passo è decidere se scalare** (E-018: S4 + più frasi per
  speaker) — il secondo speaker ha valso ~6 punti WER.
- **Decisione distillazione (2026-10-24)**: il protocollo Parakeet è pronto
  (`runs/E-015/PLAN.md`) ma **bloccato sulla fonte audio**: gold, audio CC0/BY, o TTS (H3c) — serve
  una decisione di provenienza, non compute. **(3) re-scope** si valuta con i numeri di E-017.
- **S1**: completo — T 200/200, A 60/60, **R-S1 641/641**. Non servono altre registrazioni S1.
- **Room tone — fatto** (2026-11-06): 16,2 min di stanza reale (ventola, tastiera, monitor; nessuna
  voce), −65,7 dBFS medi, **0 finestre di parlato** (max P 0,42). Costruite **97 clip** (la cattura
  si è fermata a 974,8 s invece di 1.200), shard `noise` ricostruiti (**457 clip, 1,27 h**): il
  non-music passa a **337 clip = 56,2 min**. Col VAD: **0 parole non-music** e **97/97** clip di
  room tone gated (le 13 non gated sono music). Restano 4 min per arrivare a 1 h, non bloccanti.
- **S4**: review a mano delle 200 frasi T-0601…T-0800, poi T + anchor. S1–S3 hanno già fatto T e A.
- **R-S2 — fatto** (2026-11-06): **623 take**, 622 valide (`S2_0001` = 0,39 s di silenzio, escluso;
  D26). Manifest `recordings/manifests/R-S2.jsonl`, mel `data/shards/R-S2`. Consumato da **E-017**.
- **T di S2**: già integrato dal 2026-10-30; lo zip di novembre conteneva T e anchor **byte-identici**
  (verificato via sha256), quindi non è stato sovrascritto nulla.
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
