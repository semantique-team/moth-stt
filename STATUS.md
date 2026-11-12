# STATUS — snapshot per riprendere i lavori

> **Handoff di sessione (2026-11-11, HEAD `9bd552c`):** tutto il contesto è compattato in
> [`docs/15-session-handoff-2026-11-11.md`](docs/15-session-handoff-2026-11-11.md): PR #3, E-026,
> E-027, dati S2, E-028 fp32/INT8, sweep di quantizzazione, decisioni D36-D37. **0.8.0 è
> pubblicato dal 2026-11-12** su Hugging Face e Space, con il decoder E-030 e un'eccezione
> documentata sul gate raro (28,4% vs 30%). **E-029 QAT è negativo**; **E-030** (decoder) porta
> E-028 a 5 gate su 6. [Piano](runs/E-029/PLAN.md), [QAT](runs/E-029/RESULT.md),
> [decoder](runs/E-030/RESULT.md), [record 0.8.0](runs/E-028/release-0.8.0.json).

> **Acceptance/preflight locale (2026-11-10, mergiato con PR #3):** Chromium/WASM reale passa
> quattro configurazioni (isolated auto/1/4 thread, nonisolated auto); VAD, guardie,
> concorrenza e uso effettivo del contextual scorer verificati. Nessun claim WER/ETA
> naturale e nessun nuovo modello. [Report](runs/browser-acceptance-2026-11-10/RESULT.md).
> `stt.preflight` inventaria dipendenze, file mancanti, hash e disco senza training:
> [guida altro PC](docs/guides/other-pc-preflight.md). Le registrazioni dell'altro PC
> vanno associate a manifest e storia di esposizione; non sono automaticamente test indipendente.
>
> **Audit follow-up (2026-11-10, mergiato con PR #3):** in main sono entrati safe tempo,
> InterCTC opzionale solo training, preflight/resume con hash, selezione separata dei
> finalisti su mixed-dev, diagnostica CTC, bias contestuale opt-in Python/JS e strumenti LM
> da corpus congelato. Nessun nuovo modello addestrato o peso pubblicato. La compattazione
> lossless del LM attuale risparmia **0 byte**. Uso, limiti e lavoro sull'altro PC:
> [guida audit](docs/guides/audit-improvements-no-training.md); findings e budget:
> [audit 14](docs/14-architecture-audit-2026-11-10.md).
>
> **Diagnostica E-026 (2026-11-11, no training):** sui 47 rari di T il path di riferimento è
> raggiungibile nel top-k in **41/47** casi (6/47 completamente potato: `startup`, `prototype`,
> `chromecast`, `charger`, `exam`, `mondays`). L'LM è la leva principale (ETA **17,3% greedy →
> 23,5% beam+LM**), non il top-k. L'ETA allineata sui 47 passa da **19/81 = 23,5%** a
> **25/81 = 30,9%** con E-015, ma il T-600 completo scende da **54,17% a 53,40%**: il prossimo
> braccio usa ETA allineata primaria e presence come diagnostica. [Report](runs/E-026/RESULT.md).
>
> **Bias contestuale E-027 (2026-11-11, no training):** il glossario prodotto pre-registrato
> (24 termini work/meeting) **non muove i 47 rari** (ETA 22,22% → 22,22%, presence 8/47), T-600
> ETA +0,10 (CI include zero), nessuna inserzione falsa sugli italiani; il controllo con glossario
> irrilevante cambia leggermente (non specifico). **Respinto** dal gate ≥ +1,0 ETA. Il meccanismo è
> sicuro ma non è la leva mancante; il prossimo passo resta S2 + fine-tune gentile e un mixed-dev
> vero. [Report](runs/E-027/RESULT.md).
>
> **E-028 (E-015 v2) finito (2026-11-11):** S2 + head 1e-4 / encoder 1e-5, 13.283 step.
> **Passa i gate pre-registrati sul fp32:** rare presence **17,0% → 27,7%** (+10,6 [+2,1, +21,3]),
> ETA allineata rara **23,5% → 30,9%** (+7,4 [0,0, +15,4]), naturali **439 → 438**, T-600
> **25,20 → 24,66** WER / **54,17 → 54,65** ETA, T-S3 **26,92 → 25,55**. Caveat: dev **31,01 vs
> 30,81** (+0,20), S1 WER +0,43, FLEURS-IT **+0,33** significativo (dentro il margine 0,5), 16 gold
> **20/27 → 18/27**. **INT8** (l'artefatto che si scarica): pacchetto 19.440.760 byte, browser 4/4,
> ma T-600 **25,49 / 53,50** contro 25,14 / 54,65 di 0.7.0 INT8 (tie), T-S3 25,89 / 49,01, rari
> 8/47 → 12/47, naturali **441 → 428**. Lo sweep non recupera: default migliore dentro 20 MB,
> head-fp32 20,62 MB per +0,08 dev, statica QDQ rotta. **0.8.0 è stato poi pubblicato il
> 2026-11-12 con l'arm E-030** (vedi sotto). [Risultato](runs/E-028/RESULT.md),
> [sweep](runs/E-028/QUANTIZATION.md).
>
> **E-029 QAT (2026-11-11, probe):** STE custom su MPS (pesi per-canale int8 + attivazioni
> uint8), 4 bracci da 200–500 step da E-028. Il migliore (500 step, encoder congelato) chiude
> il gap INT8–fp32 su dev (**31,13 vs 31,08**; E-028 INT8 31,14), ma sul prodotto fallisce:
> T-600 **25,34 / 53,21** (gate ≤25,14 / ≥54,65), rari **10/47**, ETA rara 27,2%, naturali
> **428/817**. Anche il controllo fp32 è sotto E-028 (24,97 / 54,07, naturali 434). **Braccio
> completo non finanziato.** [Risultato](runs/E-029/RESULT.md).
>
> **E-030 decoder (2026-11-11, stesso artefatto INT8):** ri-calibrazione del decoder
> (beam 12 / top-k 10 / LM 0,55 / bonus 2,7), selezionata su dev. E-028 INT8 passa a
> T-600 **24,71 / 54,17**, T-S3 **25,21 / 50,14**, naturali **436/817**, rari **11/47** con
> ETA rara **28,4% (23/81)** contro 22,2% di 0.7.0 (+6,2 [+1,1, +11,8]). **5 gate su 6**:
> manca il gate primario raro per **2 occorrenze**; head-fp32 non recupera (23/81) ed è
> fuori budget. **Pubblicato il 2026-11-12 come 0.8.0** con l'eccezione documentata; 0.7.0
> resta lo storico. [Risultato](runs/E-030/RESULT.md).

## Stato Corrente Verificato (2026-11-10)

Questa sezione sostituisce le priorita' e le affermazioni di stato dello snapshot storico sotto.
Audit su `main` **`cd1ce1b`**; la PR #3 (audit + tooling) e' stata **mergiata l'11-11-2026**.
L'audit in se' non ha modificato pesi, checkpoint o audio e non ha eseguito training.
La verifica di questa macchina e' riportata sotto la tabella.

| Area | Stato verificato / fonte |
|---|---|
| Modello pubblicato | **0.8.0 / E-028 + decoder E-030**, 11.575.169 parametri acustici; HF e Space verificati live il 2026-11-12 (model sha `7bbfad7e…`, config 0,55/2,7) |
| Pacchetto | **19.440.761 byte**, sette file; il payload runtime è 19.436.178 byte, config+NOTICE 4.583. ORT/WASM e dipendenze esclusi |
| Qualita' fp32 di riferimento | E-028 fp32: T-600 **24,66 / 54,65**; S3 **25,55 / 49,01**; FLEURS-IT **20,47**, EN **49,38**; CV-IT dev **19,80** |
| Qualita' int8 pubblicata | T-600 **24,71 / 54,17**; S3 **25,21 / 50,14**; rari **11/47**, ETA rara **28,4%**; naturali **436**. Eccezione documentata: gate raro 30% mancato di 2 occorrenze |
| Decoder adottato | beam 12 / top-k 10 / LM weight **0,55** / token bonus **2,7**, trigram aumentato, Silero VAD |
| Ultimo training accettato | **E-028 (0.8.0)**: S2 + ricetta gentile + decoder E-030; E-024/E-025 restano l'evidenza rumore |
| Ultimo training respinto | E-015 primo addendum gold: rare-term 8/47 -> 11/47, CI include zero; S3 invariato, FLEURS peggiora. Distillazione Parakeet originaria mai eseguita |
| Ultimo training candidato | nessuno; E-028 è pubblicato come 0.8.0 con eccezione rara documentata |
| Non eseguito | E-018 gold scaling, E-015 seconda voce/ricetta gentile, nuovi esperimenti LM; E-026 (diagnostica) e E-027 (bias) sono eseguiti, E-027 negativo. Previsioni non sono risultati |
| Runtime su main | PR #2 gia' integrata: loader release, LM binario, budget 20/25 MB, dipendenze separate. Pesi invariati; non prova che il codice della demo pubblicata sia aggiornato |
| Audit/tooling su main | PR #3 mergiata 2026-11-11: audit 20/25 MB, tempo sicuro, InterCTC opt-in, preflight/resume, finalist eval, diagnostica, bias contestuale, LM builder/compaction. Nessun peso nuovo; il LM attuale risparmia 0 byte |
| Diagnostica rari | E-026 (2026-11-11, no training): 41/47 riferimenti nel top-k, 6/47 potati; ETA allineata 47 rari 23,5% → 30,9% con E-015, ma T-600 54,17% → 53,40% |
| Bias contestuale | E-027 (2026-11-11, no training): glossario 24 termini, 47 rari invariati (22,22%), T-600 ETA +0,10 tie; **respinto** |
| Dati S2 | 2026-11-11: addendum 50/50 valido, R-S2 672 righe; E-028 (E-015 v2) eseguito |
| INT8 E-028 (storico) | 2026-11-11: vecchio decoder, T-600 25,49 / 53,50; superato dall'arm E-030 |
| QAT E-029 | 2026-11-11 probe: 4 bracci 200–500 step; migliore frozen-500 dev **31,13** (E-028 INT8 31,14), T-600 **25,34 / 53,21**, T-S3 **25,55 / 49,30**, rari **10/47**, naturali **428/817**; **gate falliti, braccio completo non finanziato** |
| E-030 decoder | 2026-11-11: stesso E-028 INT8, decoder 12/10/0,55/2,7; T-600 **24,71 / 54,17**, T-S3 **25,21 / 50,14**, naturali **436**, rari **11/47** / ETA **28,4%**; **5/6 gate**, manca il raro di 2 occorrenze; **pubblicato come 0.8.0 il 2026-11-12** |
| Prossimo passo | Monitorare 0.8.0; raccogliere un **mixed-dev naturale** e/o **dati rari mirati** per chiudere il gate raro e la selezione prodotto |
| Dati mancanti qui | Release, checkpoint, tokenizer originale, raw audio e mel shard. La repo contiene log/metriche/ipotesi, non un ambiente capace di ripetere subito il training |
| Speaker | T registrato 600/800, S1/S2 in training, S3 senza audio in training ma test ripetutamente osservato e stesso microfono di S1; S4 non registrato. Review S4 completata |

**Percorsi autorevoli per lavorare e scrivere il paper:**

- [Workflow operativi verificati](docs/guides/repository-workflows.md): installazione, dati,
  registrazioni, nuovi esperimenti, scoring, export e pubblicazione.
- [Registro E-000-E-025 e audit extra](docs/12-experiment-ledger.md): protocolli, numeri,
  benchmark/split, accettazioni, scarti, confondenti e link agli artefatti.
- [Modello corrente e paper readiness](docs/13-current-model-and-paper-readiness.md): architettura,
  lineage, tabelle fp32/int8/baseline, statistica, riproducibilita' e lacune.
- [Decisioni corrette](docs/11-evidence-and-decisions.md): ragionamento e risultati negativi.
- [Audit architetturale e budget 20/25 MB](docs/14-architecture-audit-2026-11-10.md):
  parametri, asset, rischi osservati e sequenza raccomandata; non certifica guadagni.

**Prima del paper:** riconciliare il best E-024 (log: step 26.000; release: 26.442),
le differenze S3 copiate nelle card e i riferimenti esatti delle baseline. Una ricetta Whistle LoRA
fallita non dimostra causalita' architetturale. Il recupero VAD dell'80% non e' verificato dagli
artefatti conservati. I CI sono nominali per utterance, non su seed/speaker/stanza; "tie" non
significa equivalenza. Servono test misti nuovi congelati, piu' speaker/canali e seed multipli.
Il dossier elenca i deliverable e separa incongruenze correggibili da misure da rifare.

**Verifica locale dell'audit (Windows, 2026-11-10):** `ruff check .` pulito, test unitari JS
**7/7** passati. `pytest -q` si ferma in collection su `test_train.py` per `resource` assente;
`pytest -q --ignore=tests/test_train.py`: **450 passati, 3 skipped, 5 falliti**. I fallimenti
sono nei test baseline (separatori path e quoting subprocess con spazi) e LibriSpeech
(separatori path). Non sono stati nascosti con patch o dichiarati verdi. Nessun training,
export, inferenza con pesi, browser end-to-end o controllo live HF eseguito.

**Verifica locale della merge (macOS, 2026-11-11):** `ruff check .` pulito, `git diff --check`
pulito; `pytest -q` **881 passati, 1 skipped**; unit test Node **17/17** con
`--experimental-vm-modules`; Playwright **11/11**. L'acceptance WASM reale (4/4) e' quella
riportata dal report della PR su Windows e non e' stata rieseguita qui: richiede la release
esterna e un host con Chromium. Nessun training, export, inferenza con pesi o controllo live HF.

## Snapshot Storico (Non Operativo)

Il testo seguente conserva il contesto delle sessioni precedenti, **incluse affermazioni superate
o contestate dall'audit**. Numeri di test, path Mac, blocchi "in corso", richieste HF, stime,
claim sui gate e prossimi passi qui sotto non descrivono automaticamente lo stato corrente.
Per comandi e conclusioni usare le fonti sopra, non questo archivio.

> **Runtime follow-up (2026-11-10, non pubblicato):** implementati caricamento release Python,
> lettura LM binario, guardie/parita' decoder e budget, dipendenze runtime separate e test CI
> leggeri. Pesi 0.7.0 invariati; nessun guadagno qualitativo dichiarato. Piano/stato:
> [`docs/plans/2026-11-10-lightweight-it-en-runtime-plan.md`](docs/plans/2026-11-10-lightweight-it-en-runtime-plan.md).
> Acceptance su x86_64 e campione riusato: [`runs/runtime-2026-11-10/RESULT.md`](runs/runtime-2026-11-10/RESULT.md).
> Mixed-dev naturale e nuovi esperimenti LM/glossario restano bloccati sui dati mancanti.

> Aggiornato: **2026-11-09** — ultimo commit su `origin/main`: vedi `git log -1`. Suite: **300 test
> verdi** + 11 Playwright, `ruff` pulito.
>
> **Decisione (2026-11-09, D32) — release 0.7.0:** il modello ora **sopravvive al rumore**. Metà di
> CV-IT è stata riresa con rumore a 10–30 dB, miscela identica (589,5 h), nessun'altra modifica.
> Sulla curva SNR controllata: **−6,28 a 15 dB, −11,54 a 10 dB, −9,68 a 5 dB** (media −6,18 su
> 10–30 dB), con il pulito **non peggiore** (T-600 −0,72, S3 +0,26, CV-IT test **−0,52**
> significativo). Prima: a 15 dB il modello faceva 38,4%, cioè **peggio di Whisper su audio pulito**.
> Pesi nuovi (`7a075da4…`), architettura e LM invariati, pacchetto 19,44 MB. Pubblicato su HF:
> modello e demo sono alla 0.7.0.
>
> **Decisione (2026-11-09, D34):** l'**ETA resta un gate**: il caso d'uso richiede i termini inglesi
> rari usati in italiano. E-015 riparte come **gold addendum** (100 frasi, 44 termini mai visti),
> non Parakeet. Misura su 0.7.0: **17,0%** sui 44 termini mai visti, **74,1%** sui 18 già coperti;
> il tetto diretto è **+3,7 ETA**, non +5. Le 50 S1 sono registrate e validate; le 50 S2 aspettano la seconda voce.
>
> **Decisione (2026-11-09, D35):** il **primo braccio E-015 è killato**. Rare-term **17,0% →
> 23,4%** (8/47 → 11/47, **+6,4 [−4,3, +17,0]**, non significativo), guadagno solo su S1/S2 e
> **S3 invariato**; termini naturali 439 → 423; FLEURS-IT **+0,72** e FLEURS-EN **+1,31**
> significativamente peggiori; dev strappato da 30,81 a 33,64 e mai recuperato (best = step 20,
> cioè il baseline). **0.7.0 resta il modello pubblicato.** I dati sono riutilizzabili: il
> prossimo braccio cambia **due** variabili, S2 come seconda voce e un fine-tune più gentile.
>
> **PUBBLICATO (2026-11-09):** il modello è **online** — <https://huggingface.co/semantique-team/moth-stt>
> (pubblico, artefatti ri-scaricati e verificati sha256 contro `runs/E-024/release-0.7.0.json`) e la
> **demo gira nel browser** — <https://semantique-team-moth-demo.static.hf.space> (Space statico:
> l'audio non lascia il tab; 19× realtime a 1 thread perché gli Space statici non hanno COOP/COEP,
> quindi niente SharedArrayBuffer). Ripubblicabili con `runs/E-021/upload_hf.py` e
> `runs/E-022/publish_space.py`.
>
> **Chiuso (2026-11-09, D33):** **E-025** ha verificato il caveat di E-024 su una **stanza registrata
> dopo il training**, mai vista dal modello: il guadagno **non è un artefatto del pool condiviso**, è
> **più grande** (15 dB −11,21, 10 dB −18,34, 5 dB −14,60; media −10,06 su 10–30 dB contro −6,18
> sullo stesso pool). La stanza è più dura a parità di SNR, quindi il valore assoluto è peggiore
> (36,5 a 15 dB contro 27,3): i due numeri vanno citati insieme. Nessun peso nuovo, card aggiornata
> in place.
>
> **Decisione (2026-11-09, D30):** il **decoder** era mal calibrato, non il modello. `beam_search`
> pagava il costo LM per token emesso senza contrappeso, quindi cancellava ~12% delle parole. Con
> `token_bonus` 2.4 + `lm_weight` 0.5 (beam 12/top-k 10) il WER scende di **6–9 punti su ogni set**
> **senza training** (T-600 32,77 → 25,92; S3 held-out 35,43 → 26,66; FLEURS-IT 26,52 → 20,46) e
> **FLEURS-IT batte finalmente Whisper** — la barra che bloccava il progetto da E-008. Release
> **0.6.0**, pesi invariati, int8 e LM byte-identici a 0.5.1, pacchetto 19,44 MB.
> Browser↔Python verificato in parità (tie, 77% trascrizioni identiche). Vedi `runs/E-022/`.
>
> **Nota (2026-11-09):** la parità browser↔Python è **verificata, non aperta** — al decode E-022 è
> un **tie** (−0,09 [−0,012, +0,010], 154/200 trascrizioni identiche, il 77% contro il 70% del
> decode 0.5.1). Il "+3,27 WER" del primo passaggio era un artefatto del packaging: il packager
> puntava all'LM **non aumentato** per default; ora punta a `lm_bpe3_aug`.
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
| **E-024 rumore** | **Release 0.7.0 (2026-11-09, D32)**: metà di CV-IT riresa con rumore (MUSAN + room tone, SNR 10–30 dB dall'hash dell'id). Miscela invariata a 589,5 h. Curva SNR appaiata: **15 dB −6,28** [−10,58, −2,38], **10 dB −11,54** [−16,18, −7,43], **5 dB −9,68**; 30 e 20 dB tie. Pulito: T-600 25,20 / 54,17, S3 26,92 / 48,45 (tie), FLEURS-IT 20,14, FLEURS-EN 49,14, CV-IT test 24,37 (**−0,52 significativo**). Build dei dati **1,1 min** (si calcola solo la metà rumorosa) | `runs/E-024/`, `data/manifests/cv_it_train_{clean,noisy}50.jsonl` |
| **E-025 hold-out rumore** | **Chiuso (2026-11-09, D33)**: stanza registrata **dopo** il training (38 s, broadband, VAD 0.0), 100 clip S3 appaiati. 0.6.0→0.7.0: 30 dB −3,90, 20 dB −6,79, **15 dB −11,21**, **10 dB −18,34**, **5 dB −14,60**; media −10,06 contro −6,18 sullo stesso pool. Clean riprodotto esattamente (22,24/23,94). La stanza è più dura: assoluto 36,5 a 15 dB | `runs/E-025/` |
| **Pubblicazione + demo** | **Modello su HF** (`semantique-team/moth-stt`, pubblico) e **demo nel browser** (`semantique-team-moth-demo.static.hf.space`): Space **statico**, la card verifica gli sha256 nel browser prima di eseguire il modello, l'audio non esce dal tab. ~6 s al primo caricamento (19 MB), 19× realtime a 1 thread. Lo Space si ricostruisce con un comando (`publish_space.py`) | `runs/E-021/upload_hf.py`, `runs/E-022/publish_space.py`, `runs/E-022/space/` |
| **Segmentazione** | **Registrazioni multi-frase**: 5 frasi concatenate fanno **36,6% WER** contro **26,6%** decodificate separate (−9,37 [−12,22, −3,51]); il VAD già spedito recupera l'**80%** (→ 28,5). Si segmenta **solo** se trova ≥2 sequenze (tagliare una singola costa +1,6). La durata non c'entra: la curva su FLEURS non è monotona | `stt/vad.py`, `web/vad.js`, `runs/audit-2026-11-09/` |
| **E-022** | **Ricalibrazione del decoder (2026-11-09, D30)**: `token_bonus` (γ=2,4) + `lm_weight` 0,5 + beam 12/top-k 10. T-600 **−6,85** [−7,94, −5,72], **S3 held-out −8,77** [−11,13, −6,50], FLEURS-IT −6,06, FLEURS-EN −5,85, CV-IT dev −5,01 — tutti i CI escludono 0. Greedy invariato, `token_bonus=0` no-op bit-esatto, int8 byte-identico, **nessun training**. Release **0.6.0** (19,44 MB), browser 1T 20,3× a 20 s, parità browser↔Python tie. **FLEURS-IT vs Whisper auto+beam5: −6,95 [−8,02, −5,87], vinto** | `stt/decode.py`, `web/decode.js`, `runs/E-022/` |
| **E-023** | **Re-tune LM — FATTO (2026-11-09), niente adottato (D31)**: su dev completo (3.939 clip, 10× il set di E-022) l'ottimo è **esattamente** la cella di E-022 (w 0,5 / γ 2,4, macro **22,98**); backoff 0,4 resta il migliore. L'LM **non potato** (pruning ≥1) vale **−0,65 punti dev** e vince i gate (T-600 −0,77, FLEURS-IT −0,72, CV-IT −0,65) **ma pesa 12,31 MB** → pacchetto **26,63 MB** contro il tetto di 20, ed è un **tie sullo speaker held-out** (+0,43). Quindi: il tetto costa WER italiano, **non** il claim code-switching | `runs/E-023/` |
| **Browser↔Python** | **Parità verificata (tie)**: al decode E-022, 200 clip S1 su int8 → browser 23,73 vs Python 23,82, **−0,09 [−0,012, +0,010]**, **154/200 identiche (77%)** contro 140/200 (70%) del decode 0.5.1 (+1,29 [0,0009, 0,0259], il numero di E-010). Il beam più largo **migliora** la parità. Il primo "+3,27" era un artefatto del packaging (LM non aumentato per default) | `runs/E-022/parity-wer.json` |
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
- **M8 — model card + publisher HF — PUBBLICATI (2026-11-09, aggiornato alla 0.7.0)**:
  `runs/E-024/MODEL_CARD.md` è la card della **0.7.0** (numeri presi dagli artefatti e verificati uno
  per uno, sezione sulla curva del rumore col suo caveat, costo int8 rimisurato, **11** limiti
  dichiarati), con il logo. Le card 0.5.1 e 0.6.0 restano in `runs/E-021/` e `runs/E-022/`. Modello e
  Space sono **online** con gli sha verificati dopo il caricamento.
  `runs/E-021/upload_hf.py` verifica `release/` `runs/E-021/upload_hf.py` verifica `release/`
  contro gli sha registrati **e che la card dichiari la stessa versione della release** (guard
  aggiunto il 2026-11-09: senza, si poteva pubblicare una card vecchia con artefatti nuovi), è
  **dry run di default**, 8 test. **Serve da te**: (1) il **token HF** con write, passato via
  `HF_TOKEN` — mai nel repo; (2) il **namespace**: oggi HF non ha `semantique-team`, quindi o si crea
  l'org, o si usa un account e si cambia `--repo`. Comando:
  `HF_TOKEN=hf_... uv run python runs/E-021/upload_hf.py --repo <namespace>/moth-stt --push`
  (aggiungere `--private` per uno staging). La card cita `semantique-team` in due punti: va
  aggiornata se il namespace è diverso.
- **Review delle frasi — FATTA (2026-11-07)**: lette tutte le **1.464** righe (R-S1 641, R-S2 623,
  S4 200), marcate `reviewed: true`. Report:
  [`recordings/scripts/audit-2026-11-07.md`](recordings/scripts/audit-2026-11-07.md). Struttura:
  **0 violazioni di categoria, 0 duplicati, 0 near-duplicate** contro T (Jaccard 3-gram ≥ 0,8).
  **T (S4, non registrata)**: 24 righe corrette — articoli (`lo status`, `lo sprint`, `lo ssd`,
  `lo sdk`, `lo sso`), accordi (`una demo call`, `la release note`, `la timeline`), frasi rotte
  (`una new version` / `un better support` riscritte), un apostrofo perso, e **8 prestiti inglesi
  che nessun italiano usa** (`printer`, `contract`, `invoice`, `delivery`, `parcel`, `shipment`,
  `courier`, `accountant`) sostituiti con gli equivalenti italiani. `T-0714`/`T-0778` portate a
  `switch` con **tutti** i termini annotati (`bug`, `login`, `team` prima invisibili all'ETA).
  **13 righe R** hanno errori di articolo che restano perché **l'audio li contiene**: lo speaker ha
  letto il testo verbatim (verificato con Whistle su S1_0077 → "Il studio di registrazione è in
  centro"), quindi riscrivere il testo desincronizzerebbe audio e trascrizione. Documentati, non
  toccati. **0 righe registrate modificate**, nessun manifest toccato: verificato contro `HEAD`
  (22 testi + 3 blocchi span/categoria, tutti non registrati). Le sigle (`un SMS`, `il XML`) sono
  **corrette**: si leggono lettera per lettera.
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
- **Review**: **completata il 2026-11-07** — tutte le 1.464 righe lette e marcate. Le 24 correzioni
  del blocco S4 sono in `recordings/scripts/audit-2026-11-07.md`; nessuna riga registrata è stata
  toccata. S4 può registrare.
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
  rimosso da R-S2. **Aperti risolti il 2026-11-07**: `T-0714` e `T-0778` sono ora `switch` con tutti
  i termini annotati, e le frasi R sono state lette tutte (`audit-2026-11-07.md`).
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

**Aggiornato 2026-11-09 dopo D35.** L'**ETA resta un gate**, ma il primo braccio E-015 è fallito:
**0.7.0 resta** e le 50 S1 non bastano. La priorità è **E-015 v2** — S2 come seconda voce (le 50
frasi sono già importate) e un fine-tune più gentile — poi il memo che chiude il re-scope e
riallinea il piano operativo, che resta l'unico documento stantio.

### Prossimi passi, in ordine di valore

| # | Cosa | Cosa vuol dire | Tempo | Training | Guadagno atteso | Stato |
|---|---|---|---|---|---|---|
| 1 | **E-015 v2 (dopo il kill del primo braccio)** | S2 come seconda voce (50 frasi già importate) + fine-tune gentile (head 5e-5 / encoder 1e-5, o encoder congelato). Il primo braccio: **17,0 → 23,4** non significativo, **S3 invariato**, FLEURS-IT/EN peggiorano, dev 30,81 → 33,64 | 7 min registrazione S2 + ~2 h fine-tune + 15 min eval | **Sì** | **+2,8–3,7** se il transfer cross-voice funziona; il primo braccio non l'ha dimostrato | registrare S2 (`--start 624`), poi run gentile |
| 2 | **Chiudere re-scope (D34) + piano** | Documentare "ETA resta un gate" e riallineare `2026-10-24-post-E014` (distillazione fase 2, re-scope fase 4) | ~45 min | No | 0; evita di ripartire da premesse false | dopo la registrazione |
| 3 | **InterCTC** | CTC ausiliario su un blocco intermedio: regolarizza l'allenamento e **non aggiunge nulla al modello esportato**. Sonda: **+4,4% di tempo per step** | ~4 h training + ~1 h codice/eval | **Sì** | **+0,2–0,7 WER** (stima da letteratura; con 4 blocchi è probabile il lato basso) | pronto, nessun blocco |
| 4 | **Rumore su MLS/LS** | E-024 ha augmentato solo CV-IT (84 h); MLS+LS sono altre 270 h di parlato letto ancora pulite | 22 GB + ~4 h | **Sì** | **+1–2 a 10–15 dB, ~0 sul pulito** (stima non misurata) | **bloccata dal disco**: 19 GB liberi |
| 5 | **LM 4-gram** | Sostituire il trigramma. Il tetto di 20 MB ha **0,56 MB** di margine, quindi serve un formato compresso nuovo (`web/lm.js`) | ~1 giorno | No | **+0,5–1,0 WER** *se* entra nel tetto (stima); rischio alto | dopo il re-scope |
| 6 | **Word timestamps** | Viterbi forzato sulla matrice CTC: tempi di parola gratis dal modello attuale | ~mezza giornata | No | **0 WER**; è una feature (sottotitoli, evidenziazione) | solo se serve UX |
| 7 | **Frame rate 25 Hz** | Pooling dei frame dopo l'encoder: metà frame, più profondità a parità di compute (`docs/10`: "25 Hz + 2 blocchi") | ~4 h + modifiche runtime | **Sì** | **20–30% più veloce**, WER ±0,5–1,5 (stima) | cambio architetturale, non ora |
| 8 | **`en_terms.txt` 580 → 1–2k** | Espandere la lista di termini del LM | ~1 h + rebuild LM | No | **+1–2 ETA** sui termini plausibili; **0 sui 44 mai sentiti** (diagnosi E-015) | basso valore senza audio nuovo |
| 9 | **Secondo seed di E-024** | Rifare il fine-tune del rumore con un seed diverso | ~3,5 h | **Sì** | **0 guadagno**; dice se il **−0,52 CV-IT** sul pulito è reale o rumore di run | rigore, non prodotto |
| 10 | **E-018 (gold)** | Terzo speaker per la domanda "il gold scala?" | registrazioni + review | **Sì** | ~2–3 WER da un terzo speaker (stima), ma il rendimento marginale cala (−5,73 → −4,07) | **morto senza S4** |

**Misurato vs stimato**: l'unico costo misurato è il **+4,4% per step** di InterCTC; per E-015 il
primo braccio ha **misurato** 17,0 → 23,4 (non significativo) e il tetto **+2,8–3,7** resta una
stima del secondo. Due previsioni consecutive (E-024, E-025) sono state troppo prudenti, quindi
tratta gli altri numeri come ordini di grandezza.

**Non adesso, e perché**: rumore MLS/LS (disco), LM 4-gram (tetto di 20 MB), 25 Hz (cambio
architetturale), `en_terms` (non risolve i termini mai sentiti), word timestamps (feature, non
accuratezza). Il **divario su FLEURS-IT** (Whistle +8,29) è *architecture-bound* — CTC contro
decoder — e l'unica strada è la distillazione (E-015), non un knob. Una **seconda stanza** (café)
rafforzerebbe E-025 ma non migliora il modello: ~30 min, solo evidenza.

### Fatto di recente, per contesto

- **0.7.0 pubblicato** (modello + demo), artefatti verificati sha256 dopo il caricamento
- **E-015 primo braccio: kill** — 50 frasi S1, x20, 1 epoca: **17,0 → 23,4** non significativo,
  **S3 invariato**, FLEURS-IT/EN peggiorano; dev 30,81 → 33,64. **0.7.0 resta**; dati e readout
  riutilizzabili
- **E-025** hold-out su una stanza mai vista: il guadagno del rumore **trasferisce ed è più grande**
  (media −10,06 su 10–30 dB), caveat del pool condiviso **chiuso**; nessun peso nuovo
- **E-022** decoder ricalibrato (−6,9 WER, zero training), **E-023** re-tune LM (niente da adottare),
  **E-024** noise augmentation (−6 a −11 con rumore, pulito tie)
- **Segmentazione** delle registrazioni multi-frase (il VAD recupera l'80% del divario)
- **Guida alla pubblicazione HF** per colleghi: [`docs/guides/publishing-to-huggingface.md`](docs/guides/publishing-to-huggingface.md)

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
- **S4**: review **fatta** (2026-11-07, `audit-2026-11-07.md`); ora S4 può registrare T + anchor.
  S1–S3 hanno già fatto T e A.
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
