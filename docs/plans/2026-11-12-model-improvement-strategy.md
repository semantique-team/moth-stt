# Strategia Di Miglioramento Entro I Constraint

Data: **2026-11-12**. Baseline prodotto: **Moth 0.8.1**, acustica E-028,
decoder E-030 e LM E-031. Questo documento raccoglie proposte e criteri di scelta:
**non registra esperimenti eseguiti, non riserva nuovi ID e non autorizza training**.

Fonti: [stato corrente](../../STATUS.md), [audit e correzioni](../18-audit-fixes-2026-11-12.md),
[protocollo mixed-dev](../guides/mixed-dev-and-quality-experiments.md),
[audit architetturale](../14-architecture-audit-2026-11-10.md) e risultati
[E-028](../../runs/E-028/RESULT.md), [E-029](../../runs/E-029/RESULT.md),
[E-031](../../runs/E-031/RESULT.md), [E-032](../../runs/E-032/RESULT.md).

Il codice e le ricette gia' pronti (InterCTC opt-in, preflight/resume, finalist eval,
diagnostica, builder LM) sono descritti in
[miglioramenti dell'audit](../guides/audit-improvements-no-training.md): quel documento dice
cosa e' implementato e come si lancia, questo decide priorita' e criteri. Non duplicare la
ricetta InterCTC qui.

Le raccomandazioni qui sotto usano 0.8.1; i riferimenti a 0.7.0 nelle guide precedenti
descrivono il loro contesto storico, non la baseline di nuovi esperimenti.

Per la sola **selezione**, questo piano sostituisce il wording "bloccato finche' non c'e' uno
speaker nuovo" di [`STATUS.md`](../../STATUS.md): un dev su speaker noti (S1/S2, sessioni e
canali nuovi) e' possibile subito. L'**holdout indipendente** con voce nuova resta bloccato e
non va confuso con quel dev.

## Obiettivo E Constraint

Migliorare italiano con inglesismi pronunciati da italiani, non trasformare Moth in un
ASR bilingue generalista. Mantenere record-then-transcribe locale, runtime CPU/browser,
tokenizer e decoder compatibili, licenze e consenso verificabili.

- Budget preferito: **20.000.000 byte di asset obbligatori**. La 0.8.1 occupa
  **19.706.289 byte**, con **293.711 byte** di margine.
- Il tier fino a 25 MB richiede approvazione esplicita; non assumerlo per queste proposte.
- Asset del modello, download ORT/WASM, RAM e latenza sono misure distinte.
- Preferire interventi training-only o sui dati che non aumentano il pacchetto finale.
- Compute e memoria restano limitati; il riferimento storico e' M5/MPS, 16 GB.
  Misurare fattibilita' sulla macchina effettiva, senza assumere prestazioni Windows/Mac uguali.
- Nessun training su T/A, S3 o evaluation. T/S3 sono evidenza riutilizzata, non final test intatti.
- Ogni esperimento effettivo richiede PLAN, config, budget e gate congelati, voce in
  `EXPERIMENTS.md` e artefatti in `runs/<id>/`. Nessuna dipendenza nuova implicita.

La distinzione fondamentale e': **mixed-dev misura e seleziona; TTS puo' aggiungere
training audio. Nessuno dei due da solo dimostra generalizzazione su nuovi speaker reali.**

## Mixed-Dev Reale

### Perche' Serve

Il dev generale MLS-IT + LibriSpeech-EN protegge capacita' di lettura monolingue, ma non
misura direttamente inglesismi intra-frase, pronunce italianizzate e parlato spontaneo.
E-032 migliora il raro mentre peggiora altri endpoint: serve un selettore piu' pertinente
prima del readout su T.

Mixed-dev non migliora automaticamente i pesi. Riduce il rischio di scegliere una ricetta
che migliora il test gia' osservato ma non il prodotto. Diventa sviluppo appena viene usato
per selezione/debugging; non chiamarlo holdout.

### Partire Senza Nuovi Speaker

Nuove registrazioni spontanee di S1/S2 possono essere un primo **dev su speaker conosciuti**:

- Nuove sessioni e, dove possibile, telefono/cuffie/stanza diversi.
- Non leggere T/A/R, non riscrivere errori osservati e non imporre i 44 termini scoperti in T.
- Tenere tutto fuori da training acustico, tokenizer e LM.
- Congelare consenso, audio, manifest, annotazioni, dispositivo/sessione e ruoli prima della selezione.
- Non usare S3 per altro training; preservare il confine esistente.
- Dichiarare che cambi di sessione/canale non equivalgono a nuovi speaker.

Questo dev e' piu' pertinente del solo audiobook dev, ma non certifica trasferimento a
voci nuove. Non dividere clip di uno speaker gia' consumato e chiamare una parte unseen-speaker.

### Raccolta A Basso Attrito

Chiedere **5-10 minuti di note vocali per persona**, invece di 200 frasi lette.
Esempi di elicitation: raccontare una giornata lavorativa, spiegare un problema tecnico,
descrivere un progetto/app/acquisto o dettare un messaggio che si userebbe davvero.
Sono spunti, non quote di lessico da imporre. Consentire distribuzione naturale e categorie sparse.

Trascrivere e annotare senza vedere prima le ipotesi ASR; conservare disfluenze e annotare
esclusioni/retake. Evitare dati personali o contenuti di terzi. Usare il normalizzatore
canonico e span inglesi word-based; verificare consenso e permessi di pubblicazione separatamente.
L'importer non sostituisce review semantica, inventario delle esclusioni o audit dei diritti.

Anche **una nuova persona con dispositivo diverso** e' utile come readout circoscritto,
pur non bastando per un claim generale. Riservare la prima voce nuova disponibile a un
readout del candidato gia' congelato, non inserirla subito nel training o nella selezione.

Un futuro holdout deve essere distinto dal dev, con accesso e ruoli congelati. Una sola
voce non permette inferenze robuste sulla popolazione dei parlanti, anche con molti clip.

## TTS

| Uso | Raccomandazione |
|---|---|
| Training mirato | Promettente: prima un pilot controllato |
| Diagnostica/stress test sintetico | Utile, etichettato come sintetico |
| Mixed-dev naturale o test nuovi speaker | Non sostituisce il reale |

Il TTS e' offline e **non entra negli asset spediti di Moth**. Costo e vincoli si spostano
su generazione, storage, verifica e training. Stessa architettura finale non significa
automaticamente stessa accuratezza o fattibilita' del processo.

### Rischi Specifici

Supportare IT ed EN separatamente non garantisce una buona transizione italiano-inglesismo-italiano.
Un generatore puo' cambiare accento, ritmo o voce sul termine inglese, creare pause artificiali,
pronunciare acronimi male o leggere un termine da madrelingua inglese invece che da italiano.
Si rischia quindi di apprendere il fenomeno sbagliato, come con inglese generico o splicing
tra utterance intere.

Dieci voci dello stesso generatore non equivalgono a dieci speaker reali: condividono
artefatti e meccanismo di sintesi. Rumore, pitch o tempo non trasformano una voce in
evidenza indipendente di diversita' speaker.

La trascrizione desiderata non garantisce che il TTS l'abbia pronunciata correttamente.
Ascoltare e controllare soprattutto termini, forme italianizzate, sigle e transizioni.

### Pilot Proposto

1. Scrivere 30-50 frasi nuove da requisiti di dominio/training-side, senza reference di evaluation.
2. Qualificare pochi generatori con ascolto: naturalezza intra-frase, pronuncia, continuita' di voce.
3. Scartare chi legge bene le lingue separate ma male la commistione.
4. Generare **300-1.000 clip** con contesti/voci variati, non partire da 100 ore.
5. Usare quota sintetica bassa, replay reale e budget di step limitato. Congelare la proporzione
   nel PLAN dopo aver misurato durate/costi; non assumere un rapporto ottimale gia' noto.
6. Confrontare con una continuazione identica **senza TTS**: stesso checkpoint iniziale,
   budget aggiornamenti, schedule e selezione, specificando come si bilanciano le esposizioni.
7. Selezionare su dev reale e proteggere italiano generale, termini comuni, EN e nonspeech.
8. Adottare solo dopo un readout reale; migliorare sul TTS che ha generato training non basta.

Quantita' e budget sono proposte di pilot, non soglie di efficacia dimostrate. Separare
eventuali diagnostiche sintetiche dal dev reale. Uno split per generatore/voce sintetica
riduce riuso diretto, ma non diventa una validazione naturale.

### Scelta Del Generatore

Verificare licenza dei pesi, uso degli output per training, diritti delle voci, costi e
fattibilita' sul proprio hardware. Preferire codice/pesi accessibili e provenance archiviabile,
senza assumere che un servizio gratuito o pubblicamente accessibile conceda ogni diritto.
Non clonare voci di terzi senza consenso. Nessun TTS specifico e' stato confrontato qui:
la selezione deve seguire ascolti e controlli di licenza, non solo una model card multilingue.

## Altre Strade

### InterCTC

Prima prova tecnica ragionevole: loss CTC ausiliaria su rappresentazione intermedia, testa
training-only rimossa all'export. Il supporto e' gia' presente; non cambia tokenizer, LM o
runtime e non richiede asset finali aggiuntivi se esportato correttamente.

Confrontare ricetta identica con/senza loss, budget breve e predefinito. Includere il
checkpoint iniziale tra i candidati: non obbligare la selezione a produrre un modello nuovo.
Misurare memoria e tempo. La sonda storica circa **+4,4% per step**
([E-024 PLAN](../../runs/E-024/PLAN.md), `probe_cost.py`) non e' un risultato WER ne' una
previsione del costo sul run reale. Non abbiamo ancora prova di beneficio su Moth.

### Fine-Tune Che Protegge Il Generale

E-015/E-032 suggeriscono che aggiungere esempi mirati puo' migliorare il target e deteriorare
il resto. Prima di aggiungere capacita', considerare un confronto conservativo:

- Encoder inizialmente congelato, aggiornamento blocchi finali/testa.
- Replay reale mantenuto, senza oversampling aggressivo di pochi clip.
- Step limitati, baseline iniziale eleggibile e stop su dev pertinente.
- Eventuale loss di distillazione/consistenza verso **Moth precedente** sul replay.

Il teacher Moth su stessa architettura/audio rende il confronto dei logit piu' semplice
di una KD diretta TDT/CTC da Parakeet. Il teacher non va spedito; costo training va misurato.
Se le viste hanno tempi diversi non applicare KL framewise senza allineamento corretto.
E' un'ipotesi anti-forgetting, non una garanzia: anche la testa puo' perdere capacita' con
encoder congelato. Provare una modifica alla volta, non un unico braccio con tutte le opzioni.

### Audio Reale Di Dominio Con Diritti Chiari

Cercare tutorial, presentazioni, spiegazioni tecniche o podcast con inserti inglesi,
prima di sintetizzare su larga scala. **Disponibile pubblicamente non significa autorizzato
per training**: servono licenza/permesso, provenance e gestione dei dati personali.

Un teacher puo' aiutare la trascrizione, ma controllare manualmente proprio i termini inglesi.
Accettare solo accordo tra due ASR rischia di selezionare i casi facili. Congelare criteri di
campionamento/filtraggio e verificare overlap speaker/audio/test. Poco reale pertinente e
verificato puo' essere piu' utile di molto sintetico; il vantaggio va misurato.

### LM E Ricerca

E-031 dimostra un miglioramento del sistema senza nuovi pesi, non che ogni altro LM aiutera'.
Possibili domande: pruning budgeted su corpus training-side, equilibrio termini comuni/tecnici,
termine corretto tra candidati e sopravvivenza prefix-beam, ricerca piu' ampia se RAM/latenza reggono.

Non dedurre recuperabilita' del singolo termine dal solo supporto dell'intera reference.
Non ripartire da una grande griglia su T. E-027 non giustifica aspettarsi molto da un altro
glossario generico. Con appena 293.711 B liberi, un LM piu' grande richiede spazio recuperato
altrove; loss training-only e dati sono piu' comodi entro il tetto stretto.

**Gia' misurato (E-034, 2026-11-12, solo dev).** Tenere i trigrammi visti una volta vale
**−0,598 pp** di macro dev con (2,1) e **−0,644 pp** con (1,1), entrambi con CI che esclude
zero, ma i binari pesano 12,03/12,31 MB: pacchetti 26,36/26,64 MB, fuori sia dai 20 MB sia dal
tier 25 MB. Il peso dei termini ×5 e' un tie, quindi ×3 e' saturo. Il seguito coerente e' una
**selezione budgeted** (~65k dei 1.477.990 trigrammi extra nei 293.711 B liberi), da
pre-registrare contro A3; non eseguita. Dettagli:
[E-034](../../runs/E-034/RESULT.md).

## Strade Da Rimandare

| Strada | Motivo |
|---|---|
| Altro addendum S1 fortemente ripetuto | E-032 mostra regressioni e trasferimento insufficiente |
| Piu' inglese generico | Non rappresenta direttamente inglesismi pronunciati dagli italiani |
| Nuovo tokenizer | Richiede testa/LM compatibili; intervento ampio e precedente ricetta negativa |
| QAT lungo | E-029 non ha prodotto un candidato adottabile |
| Altri blocchi acustici | Margine asset quasi esaurito; costo export reale da misurare |
| Piccolo LM neurale | Training, stato beam, export/runtime e calibrazione nuovi |
| CR-CTC | Piu' costo e complessita' di InterCTC; allineamento delle viste da gestire |

Risultati negativi respingono le ricette provate, non tutte le possibili formulazioni.
Le opzioni rimandate possono tornare utili se diagnostiche e nuovi dati le giustificano.

## Sequenza Raccomandata

1. Congelare un piccolo mixed-dev spontaneo reale, anche S1/S2, con limiti speaker/canale espliciti.
2. Provare InterCTC con/senza, mantenendo dati e ricetta controllati
   (pre-registrato in [`runs/E-033/PLAN.md`](../../runs/E-033/PLAN.md), non eseguito).
3. Qualificare TTS con ascolto/licenze e fare un solo pilot contro controllo senza TTS.
4. Riservare una nuova voce disponibile al readout di un candidato gia' congelato.
5. Solo dopo diagnosticare eventuale anti-forgetting, audio reale aggiuntivo o interventi LM/ricerca.

Con una sola azione disponibile: **raccogliere il piccolo dev reale**. Non aggiunge
direttamente accuratezza, ma evita di investire su guadagni visibili solo nell'evaluation consumata.
La mancanza di nuovi speaker non deve bloccare tutte le prove, ma limita le conclusioni.

## Criteri Prima Dell'Esecuzione

Congelare un endpoint primario coerente con il prodotto, piu' controlli di regressione;
non copiare automaticamente soglie storiche sui nuovi denominatori.

- WER, ETA allineata e target-only ETA distinti; riportare conteggi e distribuzione del lessico.
- Dev generale + mixed-dev reale per selezione, baseline iniziale eleggibile.
- Non-inferiority margin deciso prima dei risultati; un tie non dimostra equivalenza.
- Valutare il candidato **INT8 con LM binario e decoder 0.8.1**, non solo fp32/greedy.
- Nonspeech music/non-music separati e percorso VAD prodotto, false inserzioni e termini comuni.
- Asset completi <=20 MB, RAM/load/RTFx su hardware nominato; nessun confronto tra host spacciato per appaiato.
- Ipotesi, reference, config, code/data/model hash e tutti i candidati scartati conservati.
- CI per utterance su poche voci non misurano popolazioni speaker/stanza o variabilita' training.
  Seed multipli e piu' speaker/sessioni restano necessari per conclusioni piu' forti.

## Limiti Di Questa Proposta

Nessun TTS qualificato, costo misurato, dato raccolto o training eseguito per questo documento.
Nessun guadagno WER/ETA previsto e' presentato come risultato. Il progetto resta vincolato a
nuovo parlato reale per stabilire generalizzazione; le prove sintetiche non colmano quel vuoto.
