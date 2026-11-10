# Piano: Runtime Leggero E Code-Switching IT/EN

Data: 2026-11-10. Base esaminata: `a64f2a1f06843fc34d4ad1e1d948de67476263f1`.
Stato: infrastruttura implementata sul branch `feat/lightweight-release-runtime`;
esperimenti di qualita' e raccolta dati non completati. Vedi Implementation Status.

## Obiettivo E Perimetro

STT locale per italiano con parole/frasi inglesi inserite nel discorso, su CPU e browser/WASM.
Preservare la release 0.7.0 come controllo. Preferire cambi piccoli, senza nuovi servizi,
autoregressione, LLM correttori o dipendenze di training nel runtime.

Budget proposto dalla richiesta: asset del modello <=20 MB preferiti, <=25 MB massimo opzionale.
Qui MB significa 1.000.000 byte. Questa interpretazione va confermata: il requisito storico
parla anche di download totale, ma i 19,44 MB pubblicati NON includono ORT Web/WASM.
Misurare separatamente asset, applicazione browser completa e trasferimento HTTP a cache fredda.
Non dichiarare il browser completo sotto 25 MB senza una misura che lo dimostri.

Questa fase modifica solo documentazione. Nessun training, rebuild LM, modifica dei pesi,
download massivo di dati, pubblicazione o commit. I task seguenti richiedono approvazione.
Il precedente vincolo di evitare fine-tuning resta valido: le opzioni di training sono solo
un'appendice condizionale e non un passo implicito del piano.

## Evidenza Che Cambia Le Priorita'

| Fatto verificato nel repository | Conseguenza |
|---|---|
| 11,6 M parametri, encoder Whisper-tiny + 2 blocchi aggiuntivi + testa CTC | Le descrizioni iniziali da 8 M sono storiche, non il baseline attuale |
| VAD e segmentazione gia' in `stt/vad.py`, `web/vad.js`, `web/ort.js` | Integrare e proteggere il percorso esistente, non reinventarlo |
| E-022 ha gia' corretto il bias di lunghezza del decoder; E-023 ne conferma i parametri | Non ripetere una ricerca generica LM weight/token bonus come presunto easy win |
| `OnnxMoth.transcribe` non passa `token_bonus`; Python legge pickle, non il `.bin` spedito | Prima rendere eseguibile correttamente il pacchetto pubblicato |
| Python e JS hanno semantiche top-k diverse | La parita' statistica misurata non significa algoritmi identici |
| E-023: LM completo 26,63 MB totali, S3 WER +0,43 punti, tie | Il modello grande non entra neppure nei 25 MB e non dimostra il beneficio prioritario |
| E-015 primo addendum killato, S3 invariato e regressioni IT/EN | Non proporre di ripetere lo stesso fine-tuning aggressivo |
| Audio, checkpoint, tokenizer binario e shard non sono in git | La clone non permette da sola di riprodurre tutti i benchmark |

Fonti: `STATUS.md` (D30-D35 e prossimi task), `runs/E-023/RESULT.md`,
`runs/E-024/RESULT.md`, `runs/E-024/release-0.7.0.json`, `runs/E-015/RESULT.md`,
`stt/onnx_runner.py`, `stt/decode.py`, `stt/lm.py`, `web/decode.js`.
Le vecchie checklist non prevalgono sui risultati eseguiti e sullo stato live.

## Budget Reale Degli Asset

Valori dal record della release, non ricostruiti in questa sessione:

| Componente | Byte |
|---|---:|
| ONNX acustico INT8 | 11.910.677 |
| LM trigramma binario | 5.116.464 |
| Silero VAD | 2.327.524 |
| Vocabolario | 17.193 |
| Filtri mel | 64.320 |
| Config + NOTICE | 4.582 |
| Totale sette file | 19.440.760 |

Margine: 559.240 byte a 20 MB, 5.559.240 byte a 25 MB.
Con gli altri asset invariati, LM massimo teorico 5.675.704 / 10.675.704 byte.
Target ingegneristici con un piccolo margine: LM <=5,5 MB oppure <=10,5 MB.
Il limite deve contare anche eventuali nuovi asset obbligatori. README, loghi, codice, ORT,
RAM di picco e dipendenze installate vanno rendicontati separatamente, non nascosti.

## P0: Riprodurre Il Prodotto Spedito

### A. Contratto Unico Del Runtime

- [ ] Passare `token_bonus` nelle API di `stt/onnx_runner.py`, con test contro `beam_search` diretto.
- [ ] Aggiungere in `stt/lm.py` un lettore/scorer del formato `MOTHLMB1` gia' spedito.
- [ ] Usare NumPy/stdlib esistenti, mantenendo CSR e probabilita' quantizzate: non inventare conteggi per ricostruire il pickle.
- [ ] Validare header, dimensioni, offset CSR, indici, valori finiti, troncamenti e versione; verificare anche il lettore JS.
- [ ] Offrire un percorso Python da directory release e PCM mono 16 kHz, caricando sessioni/LM una volta sola.
- [ ] Riutilizzare config, filtri, vocab, hash, VAD e segmentazione della stessa release, senza mescolare asset installati e scaricati.
- [ ] Usare `stt.text.normalize` per training/scoring: eliminare normalizzazioni ad hoc dai futuri benchmark.

File: `stt/onnx_runner.py`, `stt/lm.py`, `stt/features.py`, `stt/vad.py`,
`tests/test_onnx_runner.py`, `tests/test_lm.py`, nuovi test di integrazione minimi.

Accettazione: inferenza Python dal solo pacchetto HF senza pickle/checkpoint/tokenizer di training;
beam 12, top-k 10, LM weight 0.5, token bonus 2.4 dal config; nessuna modifica dei pesi o
asset binari richiesta. Una API configurata dalla release evita di alterare silenziosamente
gli esperimenti storici; non aggiungere alias o compatibilita' non necessari.

### B. Parita', Durata E Concorrenza

- [ ] Definire top-k come numero di candidati nonblank, blank separato, con tie-break per ID; verificare e misurare il cambiamento prima di adottarlo.
- [ ] Testare prima decoder su logit identici, poi stesso LM binario, poi mel da PCM identico, infine pipeline reale.
- [ ] Verificare BOS/EOS, shift CTC->LM, ripetizioni, bonus, apostrofi e word boundary.
- [ ] Rifiutare chiaramente input non finiti, forma/sample rate errati e segmenti >30 s: la segmentazione attuale non garantisce il limite.
- [ ] Serializzare le trascrizioni browser: VAD usa buffer condivisi e le chiamate asincrone sovrapposte possono interferire.
- [ ] Non introdurre chunking arbitrario per file lunghi finche' il caso d'uso non lo richiede; mantenere la regola di segmentare solo con >=2 sequenze rilevate.

File: `stt/decode.py`, `web/decode.js`, `web/ort.js`, `web/vad.js`, UI pubblicata,
`tests/test_decode.py`, `web/tests/`.

Accettazione: equivalenza del decoder su fixture comuni; tolleranze mel esplicite; parita'
WER/ETA verificata separatamente dall'identita' delle trascrizioni; test >30 s e concorrenza.
La parita' corrente e' gia' un tie, non un problema di qualita' dimostrato da correggere alla cieca.

## P1: Misure E Release Riproducibili

- [ ] Recuperare autorizzazioni e percorsi degli audio T/A/R, checkpoint E-024 e corpus LM congelato; inventariare cio' che manca, senza copiarlo in git.
- [ ] Eseguire il pacchetto INT8 originale su questa macchina x86_64 e confrontare con gli artefatti, usando scorer e normalizzatore ufficiali.
- [ ] Registrare un piccolo mixed-dev naturale nuovo, con speaker/consensi/trascrizioni e `en_spans`, distinto da T/A e dai dati gia' usati per training.
- [ ] Separare sviluppo, scelta del candidato e test finale; non usare S3/T per scegliere lessico, pruning o parametri.
- [ ] Rendere espliciti i ruoli speaker nello scorer prima di usare S4 in training: il gruppo fisso S3+S4 non sarebbe piu' held-out.
- [ ] Testare Python/JS in CI con fixture piccole e riproducibili; riservare audio non versionati e RTF hardware ai gate dedicati.
- [ ] Separare dipendenze runtime da training/export in `pyproject.toml`, dopo aver verificato il grafo degli import e gli asset nel wheel.
- [ ] Unificare finalizzazione release: provenienza checkpoint, parametri decoder, hash, lettura LM, byte totali, budget e versione card.
- [ ] Correggere esempi di build che associano E-017 alla 0.7.0, default packager obsoleti e documentazione del download.

File: `stt/eval.py`, `stt/text.py`, `recordings/`, `pyproject.toml`,
`.github/workflows/ci.yml`, `web/tests/`, `stt/export.py`, packager/publisher in `runs/`, README.
Decidere durante l'implementazione se estrarre un piccolo finalizzatore comune; non costruire
un nuovo framework di esperimenti o riscrivere gli script storici.

Il baseline S3 non e' mai stato addestrato, ma e' gia' stato ispezionato ripetutamente nel
progetto. Non descriverlo come test completamente intatto. Se S4 entra nel training,
non presentarlo poi come conferma su speaker sconosciuto. Cercare una conferma finale su
nuove voci; non riusare retroattivamente dati R gia' consumati come dev indipendente.

## P2: Qualita' Senza Fine-Tuning Acustico

Questi sono esperimenti separati, non prerequisiti per chiudere P0. Ogni esperimento
approvato riceve un nuovo ID libero, entry in `EXPERIMENTS.md`, `runs/<id>/PLAN.md`,
config e criteri congelati prima dell'esecuzione. Non assegnare retroattivamente ID ai
tentativi esterni della conversazione, che hanno normalizzazione e test diversi.

### A. Contextual Biasing Piccolo, Non Una Lista Globale

E-007 ha gia' provato 580 hotword sul vecchio checkpoint/decoder: beneficio non dimostrato
sul parlante held-out. Non riproporre lo stesso bonus come feature nuova.

- [ ] Usare il supporto prefix/completion gia' presente in Python per un piccolo glossario di dominio fissato prima del test.
- [ ] Selezionare su mixed-dev naturale, con controllo di inserzioni e italiano senza inglesismi.
- [ ] Confrontare con un glossario irrilevante: serve a scoprire inserzioni indotte dal contesto.
- [ ] Solo se utile, portare la stessa regola bounded e word-aware in JS; nessuna nuova rete.

Glossario da requisiti prodotto/testi training-side, mai dagli errori di T/S3.
Misurare byte effettivi e latenza; non promettere recupero di termini acusticamente sconosciuti.

### B. LM Intermedio Entro Budget

E-023: (2,2) spedito 5,12 MB; (1,1) 12,31 MB, pacchetto 26,63 MB; (2,3)/(3,3)
piu' piccoli ma peggiori su dev. I risultati sono sul checkpoint precedente, non una
previsione per E-024. Prima congelare il corpus esatto: gli script R sono cambiati.

- [ ] Confrontare (2,2) controllo e soglie asimmetriche (1,2)/(2,1), gia' supportate dal builder.
- [ ] Scartare prima del benchmark completo i binari oltre budget; non usare la dimensione pickle come misura browser.
- [ ] Se necessario, studiare retention budgeted di count-one n-gram con statistiche training-side, non test-informed.
- [ ] Selezionare su full dev IT/EN e mixed-dev, poi un solo gate del candidato congelato.
- [ ] Pubblicare un tier 20-25 MB solo con beneficio misurato sul task prioritario, non perche' esiste margine.

Taglia/WER delle soglie asimmetriche sconosciute. Non si presume che la qualita' cresca
monotonicamente coi byte. Il 4-gramma richiederebbe un formato e due reader nuovi:
rimandarlo finche' un trigramma intermedio non giustifica questa complessita'.

## Gate Proposti Prima Di Qualunque Adozione

Confronto sempre contro lo stesso artefatto 0.7.0 INT8, non tra un risultato PyTorch e uno ONNX.
Nessun candidato e' un modello migliorato per il solo fatto che il file e' valido.

| Area | Criterio proposto da congelare nel singolo esperimento |
|---|---|
| Task misto | WER e ETA canonica, per categoria e speaker; per dichiarare un miglioramento confermato sul task, CI WER sotto zero e CI ETA sopra zero su un test indipendente adeguato |
| Non regressione | Su IT/EN, margine esplorativo massimo +0,5 punti WER, da giustificare nel protocollo; un CI inconcludente non prova equivalenza |
| Termini rari | ETA allineata separata da term-presence diagnostico; non scambiarli e non derivare la lista dal test |
| Non parlato | VAD attivo, 0 parole su set non-music di riferimento; musica e fonti mai viste riportate a parte |
| Taglia | Tutti gli asset obbligatori <=20.000.000 byte; tier esteso mai >25.000.000; conteggio separato del browser completo |
| Prestazioni | RTFx >=10 sul dispositivo di riferimento storico; misurare anche x86_64 e browser non isolato, load e RAM; non confrontare direttamente hardware diversi |
| Riproducibilita' | SHA pesi/LM/manifest/config, revisione codice, normalizzatore, seed e parametri di decoding |
| Robustezza | Segmenti validi <=30 s, errori comprensibili, nessun bypass dei gate export o publisher |

Usare bootstrap appaiato e dichiarare se per clip o cluster di speaker. Con una sola voce
held-out, nessun CI per clip dimostra generalizzazione a tutta la popolazione.
Se il campione non ha potenza sufficiente, risultato inconcludente e baseline mantenuto.

## Solo Con Autorizzazione Separata Al Training

Non eseguire questi passi nel perimetro corrente:

- E-015 v2: seconda voce S2 e aggiornamento gentile, seguendo il kill del primo braccio; nessun nuovo dato S3 e gate rari/IT/EN congelati. I 44 termini derivano da analisi del test: confermare su termini/voci indipendenti prima di un claim generale.
- InterCTC: testa ausiliaria solo training, rimossa nell'export; costo +4,4% per step misurato, guadagno qualitativo non misurato.
- E-018: bloccato su registrazioni S4, senza trasformare il suo futuro T in un test held-out speaker.

Non riaprire Whistle LoRA killato, non aggiungere strati senza addestramento e non
promettere che modifiche manuali dei pesi aggiungano capacita' acustiche nuove.

## Ordine Di Consegna E Decisioni Aperte

1. Confermare che 20/25 MB si riferiscano agli asset e scegliere browser, Python o entrambi come target primario.
2. Implementare P0-A (bonus + LM binario + percorso release), con test; nessuna modifica dei pesi.
3. Implementare guardie/parita' P0-B e il minimo P1 necessario per misure affidabili.
4. Recuperare audio mancanti e creare mixed-dev; scegliere UN esperimento P2, non una sweep sul test.
5. Selezionare su dev e verificare una sola volta su test; tenere l'originale se il risultato non e' convincente.

Comandi di verifica previsti: `uv run ruff check .`, `uv run pytest -q`, suite Playwright
e benchmark release su asset reali quando disponibili. Non eseguiti per questo piano:
nessuna dipendenza installata, nessun claim che i test passino su Windows.

I miglioramenti infrastrutturali garantiscono soprattutto uso corretto e riproducibilita'.
I guadagni di STT rimangono ipotesi da misurare, non promesse legate alla roadmap.

## Implementation Status

Aggiornamento dopo l'autorizzazione a implementare il piano. I checkbox precedenti
descrivono il piano originale, non un'affermazione che ogni suo punto sia completato.

| Area | Stato effettivo | Evidenza / limite |
|---|---|---|
| Wrapper e token bonus | Completato | Pass-through e test contro decoder diretto |
| LM binario Python/JS | Completato | CSR validato, BOS/EOS/backoff, input malformati e score parity |
| Runtime release Python | Completato | Hash/budget/config, asset release, VAD singolo, sessioni persistenti, CLI |
| Guardie e decoder | Completato | Top-k nonblank deterministico, cache locale LM, durata/PCM/concorrenza browser |
| Pacchetto leggero | Completato | Dipendenze runtime separate, lock aggiornato, wheel smoke CI |
| CI | Completato per test leggeri | Node unit e parita' via pytest; browser/WASM reale resta un gate separato |
| Budget | Completato per asset/raw browser | 20 MB default /25 MB massimo; trasferimento HTTP e RAM browser non misurati |
| Freeze corpus | Implementato e testato | Input/esclusioni espliciti, normalizzazione canonica, hash, reject overlap |
| Acceptance x86_64 | Eseguito sul campione disponibile | `runs/runtime-2026-11-10/`; audio riusati, non nuovo test indipendente |
| Mixed-dev naturale / nuovo test | Bloccato | Servono persone, consensi, registrazioni e trascrizioni reali |
| Contextual biasing | Protocollo pronto, non eseguito | Mancano mixed-dev indipendente e glossary di prodotto approvato |
| LM (1,2)/(2,1) | Protocollo pronto, non eseguito | Mancano tokenizer e corpus congelato corrispondente al LM pubblicato |
| Ruoli speaker scorer | Differito | Nessun S4 aggiunto al training; cambiare prima di E-018 |
| Training condizionale | Non eseguito | Fuori dal perimetro corrente |

Nessun nuovo peso acustico e nessuna nuova release pubblicata. L'allineamento top-k
Python e' una variazione del percorso di decoding: i vecchi benchmark restano storici,
non vanno attribuiti automaticamente al nuovo runtime.

Verifiche locali: Ruff completo verde; test runtime/scoring/data e Node verdi. Suite
Windows estesa: cinque fallimenti preesistenti (path POSIX e quoting shell nelle baseline),
training non collezionabile per import Unix `resource`. La CI Linux resta necessaria
per il gate completo. Nessuna modifica di componenti training estranei per nascondere i limiti.
