# Audit Strutturale: Moth STT Entro 20-25 MB

Data: 2026-11-10.
Revisione del codice esaminato: `7acbf3f` sul branch `feat/lightweight-release-runtime`.
La PR #2 e' stata successivamente mergiata in `main` con merge commit `cd1ce1b`;
CI Linux e Node entrambe verdi al merge. L'audit descrive il codice esaminato sopra.
Baseline acustico: release pubblicata 0.7.0, checkpoint E-024.
Stato: audit e proposte, non implementazione o autorizzazione di nuovi esperimenti.

## 1. Sintesi Esecutiva

La struttura attuale e' coerente con un prodotto STT piccolo, locale e specializzato in
italiano con parole e brevi frasi inglesi inserite nel discorso. Non emerge una ragione
misurata per sostituirla immediatamente con un modello piu' grande.

La raccomandazione e' mantenere 0.7.0 come controllo e capire dove si perdono gli inglesismi:
evidenza acustica insufficiente, prior linguistico, potatura dei candidati o metriche non
allineate al task. Queste cause richiedono interventi diversi e non sono intercambiabili.

I rilievi principali sono:

1. Il checkpoint viene selezionato con WER greedy su dev italiani e inglesi separati,
   non con il sistema beam+LM e il task misto che si vuole distribuire.
2. Il tempo-stretch puo' produrre audio oltre 30 secondi che viene troncato mantenendo
   il target testuale completo: un rischio concreto di disallineamento nel training.
3. La valutazione naturale e indipendente del code-switching e' ancora insufficiente;
   S3 non e' mai stato addestrato, ma e' gia' stato esaminato ripetutamente.
4. Il readout dei termini rari non coincide con ETA canonica e l'etichetta "mai visto"
   non ricostruisce tutte le esposizioni nei diversi componenti.
5. I 20 MB lasciano circa 0,56 MB di margine. Prima di aggiungere capacita' conviene
   provare interventi size-neutral e investigare la rappresentazione del LM.
6. Non e' dimostrato che CTC sia il limite insuperabile del progetto: diversi confronti
   storici cambiano contemporaneamente pretraining, adattamento, decoder e dati.

Le modifiche infrastrutturali della PR #2 sono ora in `main`, ma non sono automaticamente
parte degli artefatti della release pubblicata e non dimostrano un nuovo guadagno STT.

## 2. Perimetro E Metodo

L'audit include lettura di architettura, training, tokenizer, LM, decoder, export e
risultati storici; comprende inoltre la verifica del numero di parametri tramite
istanza del modello. Non sono stati eseguiti nuovi training, export, sweep o benchmark
di accuratezza per questo documento. I numeri di qualita' provengono dagli artefatti citati.

Classificazione delle affermazioni:

| Etichetta | Significato |
|---|---|
| Osservato | Direttamente presente nel codice o misurato in un esperimento citato |
| Derivato | Calcolo aritmetico da dimensioni o byte registrati, non una misura prestazionale |
| Ipotesi | Meccanismo o beneficio plausibile, da sottoporre a esperimento |
| Non stabilito | L'evidenza disponibile non permette la conclusione |

Vincoli adottati: CPU e browser/WASM, audio locale, output normalizzato, preferenza per
asset <=20.000.000 byte e tier opzionale mai oltre 25.000.000 byte. MB e' decimale.
Il budget riguarda gli asset, non automaticamente ORT/WASM, applicazione completa,
dipendenze Python, trasferimento HTTP compresso o RAM di picco.

Il vincolo precedente "nessun fine-tuning" non e' revocato da un audit. Tutte le opzioni
di training nelle sezioni successive richiedono un'autorizzazione separata.

## 3. Architettura Effettiva

```text
PCM mono, 16 kHz
  -> log-mel Whisper-compatible [80, T], 100 Hz
  -> Conv1d 80 -> 384, kernel 3, stride 1 + GELU
  -> Conv1d 384 -> 384, kernel 3, stride 2 + GELU
  -> posizioni sinusoidali, sequenza a 50 Hz
  -> 4 blocchi Transformer di origine Whisper-tiny
  -> LayerNorm
  -> 2 blocchi Transformer aggiuntivi
  -> Linear 384 -> 1025
  -> log-softmax FP32
  -> CTC greedy oppure prefix beam search + LM trigramma
```

Ogni blocco ha width 384, 6 attention head e FFN 1536. L'attenzione e' globale e
bidirezionale. I due blocchi aggiuntivi sono Transformer completi, non una piccola
testa lineare. La LayerNorm finale dell'encoder precede i due blocchi aggiuntivi;
non esiste una seconda normalizzazione finale dopo di essi.

Il modello non ha decoder autoregressivo, token lingua, branch language-ID,
convoluzioni temporali interne ai blocchi o loss intermedie nella ricetta produttiva.
Non e' streaming nativo. Il batch di inferenza esportato e' uno e usa la durata reale;
in training e' presente una maschera sulle chiavi per il padding.

Fonti: [`stt/model.py`, righe 27-37, 58-93, 96-164](../stt/model.py#L27),
[`stt/export.py`](../stt/export.py), [`runs/E-024/config.yaml`](../runs/E-024/config.yaml).

### 3.1 Parametri

| Componente | Parametri |
|---|---:|
| Conv1 | 92.544 |
| Conv2 | 442.752 |
| Quattro blocchi encoder | 7.096.320 |
| LayerNorm finale encoder | 768 |
| Encoder totale | 7.632.384 |
| Due blocchi aggiuntivi | 3.548.160 |
| Classificatore CTC | 394.625 |
| **Totale** | **11.575.169** |

Un blocco contiene 1.774.080 parametri: attention 590.976, FFN 1.181.568,
normalizzazioni 1.536. Circa il 92% del totale e' nei sei blocchi; il classificatore
e' circa il 3,4%. Ridurre solo il classificatore non libera molto budget.

Il modello iniziale a quattro blocchi aveva 8.027.009 parametri: le descrizioni da
"8 M" sono storiche. Le posizioni sinusoidali non sono parametri addestrabili e
vengono generate nel grafo ONNX, evitando una tabella persistente ingombrante.

Verifica riproducibile, senza caricare checkpoint o avviare training:

```python
from stt.model import MothSTT

model = MothSTT(head_blocks=2)
print(sum(parameter.numel() for parameter in model.parameters()))  # 11575169
```

### 3.2 Distribuzione Del Costo

Non esiste un unico collo di bottiglia per tutte le piattaforme. Nel browser M5
single-thread il tempo acustico domina; nel recente acceptance Python il decoder e
il calcolo mel pesano piu' dell'inferenza acustica. Una riduzione dei FLOP della rete
non si traduce quindi automaticamente nella stessa riduzione end-to-end.

Fonti: [`release-0.7.0.json`](../runs/E-024/release-0.7.0.json),
[`runtime acceptance`](../runs/runtime-2026-11-10/RESULT.md).

## 4. Budget Dimensionale

Byte registrati per i sette file della release 0.7.0:

| Asset | Byte |
|---|---:|
| `model.int8.onnx` | 11.910.677 |
| `lm_bpe3.bin` | 5.116.464 |
| `silero_vad.onnx` | 2.327.524 |
| `vocab.json` | 17.193 |
| `mel_filters.bin` | 64.320 |
| `config.json` | 1.322 |
| `NOTICE` | 3.260 |
| **Totale** | **19.440.760** |

| Budget asset | Margine | LM massimo sostitutivo, altri file invariati |
|---|---:|---:|
| 20.000.000 | 559.240 | 5.675.704 |
| 25.000.000 | 5.559.240 | 10.675.704 |

Questi massimi sono aritmetici: nuovi metadati o asset consumano parte del margine.
Il record storico `package_bytes` conta cinque payload ed esclude config e NOTICE;
non va confuso con il totale conservativo dei sette file.

Il LM non potato di E-023 occupa circa 12,31 MB nel formato browser e porta il pacchetto
a circa 26,63 MB. Non entra neppure nel tier da 25 MB. La dimensione pickle compressa
non e' un sostituto della misura del binario usato dal runtime.

Fonti: [`runs/E-024/release-0.7.0.json`](../runs/E-024/release-0.7.0.json),
[`runs/E-023/RESULT.md`](../runs/E-023/RESULT.md).

## 5. Rilievi Prioritari

### A1. Selezione Del Checkpoint Non Allineata Al Prodotto

Priorita': alta. Stato: osservato.

`_evaluate_all` seleziona sulla media delle WER dev; `_evaluate` decodifica con argmax
e greedy. Il dev storico usa italiano/inglese separati, mentre il prodotto richiede
inglesismi nel parlato italiano e impiega beam+LM. Un miglioramento sulla metrica di
selezione non garantisce un miglioramento del prodotto.

E-012 e' un esempio: piu' LibriSpeech migliora l'inglese puro ma peggiora il misto.
Questo non significa che il greedy sia inutile: e' un controllo acustico economico,
ma non dovrebbe essere l'unico criterio per il candidato finale.

Proposta: mantenere monitoraggio greedy IT/EN frequente e valutare pochi checkpoint
finalisti col decoder distribuito su mixed-dev naturale. Congelare prima la regola di
scelta, includendo WER, ETA e inserzioni; non usare T/S3 per scegliere il checkpoint.
Non serve cambiare la funzione di loss solo per cambiare la selezione.

Fonte: [`stt/train.py`, righe 659-719](../stt/train.py#L659).

### A2. Troncamento Dell'Audio Con Target Completo

Priorita': alta prima di un nuovo training. Stato: rischio osservato nel codice,
frequenza e impatto non misurati.

`_prepare_batch` taglia mel oltre 3000 frame e limita le lunghezze; `_encode_targets`
continua a usare tutto `row["norm"]`. Un clip allungato dal tempo-stretch puo' perdere
audio finale, ma mantenere le parole corrispondenti nel target CTC.

Proposta: contare gli overflow su una passata dei dati, poi limitare o ricampionare
la trasformazione affinche' conservi l'intera utterance. In alternativa usare la vista
non allungata per quei clip. Non tagliare il testo proporzionalmente alla durata:
senza allineamento temporale sarebbe un'altra etichetta inventata.

Test necessari: clip al limite, stretch oltre limite, contenuto finale conservato e
target invariato solo quando anche l'audio completo e' conservato. Dimensione finale: zero.

Fonte: [`stt/train.py`, righe 534-560](../stt/train.py#L534).

### A3. Evidenza Sul Task Naturale Ancora Limitata

Priorita': alta. Stato: osservato.

T contiene frasi lette; S1/S2 hanno contribuito al training. S3 non e' mai entrato nel
training acustico, ma i suoi risultati sono stati ispezionati piu' volte e condivide il
microfono/canale con S1. E-015 ha scelto termini dopo un'analisi di T. Non equivale a
training sull'audio S3, ma limita l'indipendenza del test.

Le nostre concatenazioni FLEURS e l'acceptance runtime sono controlli tecnici su audio
riusato, non misure indipendenti di inglesismi naturali. Nel report runtime ETA e' null.

Proposta: mixed-dev nuovo per scelta delle modifiche, futuro holdout distinto per frasi,
sessioni e possibilmente voci/dispositivi. Conservare T come riferimento storico e
regression set, dichiarando il riuso. Non promuovere R gia' addestrato a dev indipendente.

Fonti: [`docs/11`](11-evidence-and-decisions.md),
[`protocollo mixed-dev`](guides/mixed-dev-and-quality-experiments.md),
[`runtime acceptance`](../runs/runtime-2026-11-10/RESULT.md).

### A4. ETA E Readout Dei Termini Non Sono Intercambiabili

Priorita': alta per diagnosi e claim. Stato: osservato.

Lo scorer canonico conta i match allineati delle parole inglesi di riferimento.
E-015 converte invece l'ipotesi in un set di parole, premia una presenza ovunque e
nel readout per utterance usa `any` sui termini rari. Le ipotesi mancanti vengono saltate.
Il risultato 17,0% -> 23,4% e' quindi una diagnostica di presenza, non direttamente ETA.

L'inventario NATURAL non comprende esplicitamente tutte le fonti possibili: standalone
LibriSpeech, corpus tokenizer, testo pool e LM vanno ricostruiti separatamente. Una parola
puo' essere nuova per gli audio italiani ma nota al LM o al tokenizer.
La costante `TERMS18` contiene inoltre 16 stringhe: nome, lista e dichiarazioni di copertura
vanno riconciliati prima di riutilizzare quel controllo.

Proposta: inventari storici distinti e hashati per training acustico, tokenizer, LM e
glossario. Metrica primaria ETA allineata; term-presence solo diagnostica chiaramente
etichettata. Trattare ipotesi mancanti come errore di protocollo o fallimento, non ometterle.

Fonti: [`runs/E-015/eval_terms.py`, righe 26-99](../runs/E-015/eval_terms.py#L26),
[`stt/eval.py`](../stt/eval.py).

### A5. Limite CTC Non Dimostrato Causalmente

Priorita': media, importante per la scelta degli investimenti. Stato: non stabilito.

La performance residua e' compatibile con limiti di capacita', ma non li isola.
Whistle LoRA e Moth full fine-tuning differiscono in architettura, pretraining,
parametri aggiornati e ricetta; un confronto del genere non dimostra che CTC sia la
causa del divario. E-022 mostra quanto un difetto nel decoder possa apparire come limite
del modello: circa 6-9 punti WER recuperati senza training.

Non concludere neppure l'opposto: non e' dimostrato che basti il decoder. La domanda
va affrontata con diagnostiche dei candidati e ablation controllate.

Fonti: [`runs/E-022/RESULT.md`](../runs/E-022/RESULT.md),
[`docs/11`](11-evidence-and-decisions.md), [`STATUS.md`](../STATUS.md).

## 6. Tokenizer, Prior E Ricerca

### 6.1 Il BPE Piccolo Non E' Un Dizionario Chiuso

Il tokenizer ha 1024 pezzi SentencePiece, condivisi IT/EN; il blank porta gli output
CTC a 1025. Il corpus storico punta a circa 70/30 IT/EN per parole e include testo
misto e termini. Le statistiche riportate sono 1,945 token/parola, 4,70 token/s e
48,8% dei termini entro tre pezzi. Non sono nuove misure di questo audit.

Una parola non presente intera puo' essere composta da sottoparole/caratteri. Una
frammentazione maggiore aumenta la difficolta' di allineamento e ricerca, ma non
significa automaticamente OOV. `character_coverage=1.0` non e' byte fallback:
l'alfabeto effettivo e i casi `<unk>` vanno verificati sul tokenizer binario congelato,
assente nella clone.

Cambiare `vocab.json` o l'ordine dei pezzi senza riaddestrare la testa cambia il
significato degli output. Un tokenizer diverso richiede testa e LM compatibili.

Fonti: [`stt/tokenizer.py`](../stt/tokenizer.py),
[`stt/data/build_tokenizer.py`](../stt/data/build_tokenizer.py).

### 6.2 Top-K E Beam Eliminano Possibilita' Prima Del Bonus

Il decoder seleziona candidati acustici top-k, poi applica LM/hotword e mantiene un
beam finito. Un token escluso non viene recuperato aumentando il bonus; una sequenza
eliminata non puo' essere corretta dal solo rescoring finale.

Prima di intervenire misurerei, su errori annotati di mixed-dev:

- Top-k/oracle coverage dei token necessari e perdita del prefisso nel beam.
- Score acustico CTC del riferimento contro la trascrizione scelta, con forced alignment.
- Presenza della parola corretta fra i candidati N-best, se si aggiunge questa diagnostica.
- Trascrizione greedy, beam senza LM e beam con LM sugli stessi logit.
- Sostituzioni fonetiche italianizzate, omissioni, fusioni di parole e false inserzioni.

La probabilita' del percorso corretto o l'oracle usa il riferimento per diagnosticare:
non e' un decoder utilizzabile in produzione e non e' una misura di accuratezza deployabile.
Se si modificano beam/top-k, scegliere su dev e contabilizzare tempo/RAM oltre alla WER.

Fonte: [`stt/decode.py`](../stt/decode.py).

### 6.3 Contextual Biasing: Utile Solo Se Controllato

E-007 ha gia' provato 580 hotword, completion bonus 1 e nessun prefix bonus, su un
checkpoint/decoder precedente. Il risultato non dimostra un forte vantaggio held-out,
ma non esclude un piccolo glossario contestuale sulla release attuale.

Il prefix bonus corrente puo' accumulare ricompense senza restituzione quando una
parola viene abbandonata, e il completamento non conferma necessariamente la fine
della parola. Termini lunghi frammentati e prefissi condivisi possono essere favoriti
in modo sproporzionato. Il supporto Python non equivale a un'integrazione browser gia' pronta.

Proposta: glossario piccolo da requisiti prodotto, reward limitato e word-aware;
controlli con glossario irrilevante, italiano puro, non parlato e termini assenti.
Non ricavare il glossario dai termini sbagliati del test. Non presumere che il bias
possa insegnare una corrispondenza acustica completamente mancante.

Fonti: [`stt/hotwords.py`](../stt/hotwords.py), [`stt/decode.py`](../stt/decode.py),
[`EXPERIMENTS.md`, E-007](../EXPERIMENTS.md).

### 6.4 Normalizzazione Del Parlato Tecnico

Numeri in testo misto vengono espansi con fallback italiano; casing e molti simboli
spariscono, trattini/slash diventano spazi. Pronuncia letterale delle sigle e grafia
acronimo non coincidono necessariamente. Il prodotto promette testo normalizzato,
non ortografia tecnica completa.

Conservare il normalizzatore canonico per i confronti e annotare i casi limite;
eventuali metriche alternative su sigle devono essere secondarie e dichiarate, non
introdotte dopo aver visto gli errori per aumentare l'ETA.
Fonte: [`stt/text.py`](../stt/text.py).

## 7. Cosa Insegnano Gli Esperimenti

| Esperimento | Evidenza storica | Conclusione prudente |
|---|---|---|
| E-006, due blocchi aggiunti | FLEURS-IT 37,41 -> 34,11 WER; T 65,35 -> 59,50 | Profondita' piu' continuazione aiutano; non e' una pura ablation di capacita' |
| E-008, testo misto nel LM | T-400 WER -1,86 e ETA +4,07 punti nell'addendum | Il prior di dominio era parte del problema |
| E-012, +200 h inglese | FLEURS-EN migliora, T peggiora, ETA invariata | Inglese generico non equivale a inglesismi italiani |
| E-013, BPE 4096 | Termini entro tre pezzi 48,8 -> 74%; ETA invariata | Tokenizer piu' grande da solo non risolve il task; testa nuova, training breve e LM incompatibile limitano la generalizzazione del risultato |
| E-014, +110 h splicing | Migliora FLEURS, ETA T 21,49 -> 20,60, inconclusiva | Cambio lingua fra utterance non riproduce il prestito breve naturale |
| E-015, primo gold addendum | Presenza rari 8/47 -> 11/47, CI include zero; S3 invariato; IT/EN peggiorano | Ricetta non adottabile, non prova che i termini non si possano imparare |
| E-022, calibrazione decoder | Circa 6-9 punti WER senza pesi nuovi | Separare sempre effetto acustico ed effetto del decoding |
| E-023, LM completo | Guadagni italiani, S3 +0,43 WER con CI che include zero; 26,63 MB | Non entra nel budget e non dimostra beneficio held-out sul misto |
| E-024/025, rumore | Guadagni significativi a SNR bassi, trasferiti a una stanza nuova | Robustezza migliorabile senza espandere il modello; non certifica ogni ambiente |

E-012/013/014 precedono la calibrazione E-022: sono evidenza storica, non confronti
contemporanei contro 0.7.0 col decoder attuale. Alcuni testi storici chiamano un aumento
ETA "peggioramento": per questa metrica piu' alto e' meglio, quindi prevalgono valori
e definizione formale sulle frasi contraddittorie.

Fonti: [`EXPERIMENTS.md`](../EXPERIMENTS.md),
[`runs/E-015/RESULT.md`](../runs/E-015/RESULT.md),
[`runs/E-023/RESULT.md`](../runs/E-023/RESULT.md),
[`runs/E-024/RESULT.md`](../runs/E-024/RESULT.md),
[`runs/E-025/RESULT.md`](../runs/E-025/RESULT.md).

## 8. Miglioramenti Senza Training Acustico

### 8.1 Compressione Della Rappresentazione LM

Il formato CSR corrente usa token uint16 e log-score int16: quattro byte per entry,
piu' chiavi/offset dei contesti. Il vocabolario LM di 1026 simboli richiede teoricamente
11 bit per ID, non 16. Il writer conserva anche contesti trigramma con righe diventate
vuote dopo il pruning.

Prima prova: quantificare righe vuote ed eliminarle preservando esattamente il fallback.
Poi valutare packing/delta encoding di token, score e indici, senza sacrificare lookup,
RAM e latenza. Non e' misurato quanto spazio si recuperi sul binario reale.

Compressione lossless preserva i punteggi, ma non migliora da sola la WER. Quantizzazione
piu' grossolana degli score non e' lossless e richiede gate di decoding e accuratezza.
Un formato nuovo implica writer, due reader, versione e test di input malformati.

Fonte: [`stt/lm.py`, `save_browser`](../stt/lm.py).

### 8.2 Pruning Intermedio

Con corpus e tokenizer congelati, provare le soglie asimmetriche (1,2) e (2,1), non
misurate nei quattro bracci E-023. Se serve, retention budgeted dei singleton basata
su statistiche training-side, non sugli errori T/S3.

Il pruning cambia anche la normalizzazione delle righe, quindi non e' solo eliminare
informazione rara lasciando invariato tutto il resto. Nessuna garanzia di monotonicita'
qualita'/dimensione. Target pratici LM <=5,5 MB oppure <=10,5 MB per il tier esteso.

Prerequisito: il freeze utility nella PR rende riproducibili i futuri input, ma non
ricostruisce il corpus esatto che ha generato il LM storico. Il tokenizer binario e
gli input storici devono essere recuperati prima di un confronto controllato.

### 8.3 Piccolo LM Neurale: Ricerca Successiva

Una rete quantizzata potrebbe sostituire la tabella e modellare contesto piu' lungo.
Tre milioni di pesi INT8 sono circa tre MB ideali, prima di scale/export/runtime:
questa e' solo aritmetica, non un progetto dimensionato o un risultato.

Shallow fusion richiede stato per ipotesi; rescoring N-best non recupera candidati
assenti. Servono operatori browser, cache/stati, training LM, calibrazione e test.
Non lo metterei prima di diagnostica, glossario piccolo e compressione del trigramma.
Nessun LM neurale di questo tipo e' stato valutato nel progetto.

## 9. Opzioni Acustiche Con Training Separato

### 9.1 Dati Mirati E Adattamento Prudente

Per il prodotto conta inglese pronunciato da italiani dentro contesti italiani,
con variazioni di voce, canale e pronuncia. Non basta piu' audiobook inglese o
concatenazione di frasi intere. Il primo addendum E-015 ha fallito; aumentare solo
l'oversampling S1 non affronta regressioni o transfer.

La proposta gia' presente nel progetto e' S2 piu' una ricetta gentile. E' un confronto
pratico che cambia due variabili, non una prova causale pulita della sola diversita'
speaker. Conservare replay dei dati generali, stop su dev e controllo delle regressioni.
Il target rare-term derivato dal test va confermato su termini e voci indipendenti.

Pesi esportati: stessa architettura, dimensione sostanzialmente invariata. Costo reale:
raccolta, consenso, annotazione e training. Un teacher puo' produrre pseudo-label per
audio legittimamente riutilizzabile, ma non offre automaticamente vantaggi su gold gia'
annotato; hard text labels e soft-logit KD fra TDT/CTC non sono la stessa operazione.

### 9.2 InterCTC

Loss CTC ausiliaria su una rappresentazione intermedia, testa rimossa all'export.
Una testa separata 384 -> 1025 aggiunge 394.625 parametri solo durante il training;
dimensione e compute del modello esportato restano invariati se la testa e' esclusa.

La sonda MPS ha misurato mediane 0,25939 s baseline e 0,27072 s InterCTC, rapporto
1,0437, circa +4,4% per step. E' una sonda sintetica, non una misura di guadagno WER.
La sonda attacca al termine dei quattro blocchi encoder, prima dei due aggiuntivi;
non dimostra il costo o la qualita' di un attacco dopo il secondo blocco.

Prima prova tecnica consigliata se si autorizza training. Rischi: loss weight e punto
di attacco, conflitto fra obiettivi e assenza di beneficio specifico sui termini inglesi.

Fonti: [`runs/E-024/probe_cost.py`](../runs/E-024/probe_cost.py),
[`probe_cost.json`](../runs/E-024/probe_cost.json), [`docs/10`](10-optimization-review.md).

### 9.3 CR-CTC

Due viste augmentate dello stesso contenuto, CTC su entrambe e loss di consistenza.
Stessi pesi finali e nessun asset aggiuntivo; training approssimativamente piu' costoso
di due passaggi, con possibile aumento importante della memoria attivazioni.

Se le trasformazioni alterano il tempo, la consistenza framewise richiede viste
allineate o una gestione esplicita dell'allineamento. Non applicare KL punto-punto a
sequenze con tempi diversi come se fossero identiche. Il dataset attuale restituisce
una vista, quindi servono modifiche vere al training. Nessun risultato Moth disponibile.

Priorita' dopo InterCTC; non sostituisce esposizione a pronunce/termini mancanti.

### 9.4 Piccola Testa Convoluzionale

Ipotesi di bias temporale locale, non di nuovo contesto globale: l'attention vede gia'
l'intera sequenza. Esempio dimensionato: due Conv1d depthwise-separable width 384,
kernel 15, depthwise e pointwise con bias.

```text
2 * [(384 * 15 + 384) + (384 * 384 + 384)] = 307.968 parametri
```

Nuovo totale teorico 11.883.137 parametri. Pesi INT8 ideali circa +0,31 MB, oltre a
bias/scale/grafo; FP32 circa +1,23 MB, oltre il margine da 20 MB. Il receptive field
convoluzionale e' 29 frame, circa 580 ms. La struttura esatta, normalizzazioni e
attivazioni vanno specificate prima di calcolare il pacchetto definitivo.

Possibile entro 20 MB solo con export/quantizzazione compatibili e verificati.
Nessun guadagno misurato. Aggiornare anche il forward esportato, non solo il modello PyTorch.

### 9.5 Riduzione A 25 Hz

La posizione del pooling cambia radicalmente il beneficio:

| Posizione | Risparmio possibile |
|---|---|
| Dopo tutti i sei blocchi | Meno frame per classificatore e decoder, nessun risparmio Transformer |
| Dopo quattro blocchi, prima degli ultimi due | Riduce il costo dei due blocchi aggiuntivi |
| Prima dei blocchi pretrained | Risparmio maggiore, ma altera molto l'input del backbone |

Nel secondo caso i blocchi lineari passano da sei equivalenti full-rate a cinque;
l'attention quadratica da sei a 4,5. E' un calcolo, non un benchmark. Non equivale
a essere piu' veloce del vecchio modello a quattro blocchi: quei quattro sono ancora
tutti a pieno rate. Non trasferire automaticamente vecchie stime al baseline attuale.

Rischi: frame CTC insufficienti per casi rapidi/ripetuti, pooling con padding, nuove
lunghezze, posizioni, metadata frame rate, export, decoder e calibrazione LM/bonus.
La media 4,7 token/s non garantisce fattibilita' per ogni utterance.
Dato che la velocita' corrente e' gia' adeguata, priorita' bassa salvo un requisito
misurato su dispositivo piu' debole.

## 10. Interventi Non Prioritari

- Ulteriore BPE expansion senza nuova ipotesi e training/LM compatibili: la prova 4096 non ha risolto l'ETA.
- Piu' inglese generico o altro clause splicing come leva isolata: i risultati precedenti non supportano il task specifico.
- Altri Transformer full-rate: circa 1,77 M parametri ciascuno, piu' compute, prima di verificare ipotesi size-neutral.
- Cambio diretto di bias/pesi senza diagnosi: i due tentativi esterni della conversazione non hanno migliorato il misto; non sono benchmark canonici del repo.
- Dequantizzazione come recupero di conoscenza: cambiare il contenitore non ripristina informazione gia' persa.
- Whisper-base come sostituzione semplice: budget del solo encoder non comprende LM e VAD; il pacchetto completo va misurato.
- LLM correttore e servizi cloud come default: violano lo spirito di un runtime leggero locale e possono alterare il contenuto.

## 11. Rischi Di Implementazione Trasversali

Il forward ONNX in `ExportModel` replica esplicitamente il percorso del modello.
Qualunque conv, pooling o normalizzazione deve raggiungere entrambi e i relativi
test di parita'. Modificare solo `MothSTT.forward` non basta.

Il mapping Whisper usa controlli di parita', ma la conversione deve fissare anche la
revisione immutabile della sorgente. Le qualifiche sulla tabella posizionale vanno
conservate: parita' del confronto non significa che ogni scelta sia identica al modello originale.

Le garanzie sul padding non vanno esagerate: la maschera impedisce attenzione alle chiavi
padded, ma non dimostra invarianti esatte su ogni frame di un modello addestrato dopo
effetti al bordo delle convoluzioni. Nessun grande difetto di qualita' e' dimostrato qui.

Fonti: [`stt/export.py`](../stt/export.py),
[`stt/convert_whisper.py`](../stt/convert_whisper.py),
[`tests/test_model.py`](../tests/test_model.py).

## 12. Sequenza Raccomandata E Gate

| Ordine | Azione | Training acustico | Criterio di avanzamento |
|---|---|---|---|
| 1 | Recuperare tokenizer/corpus e raccogliere mixed-dev naturale | No | Provenienza, esclusioni, consensi, annotazioni e split verificati |
| 2 | Diagnostica acustica/prior/search | No | Errori classificati e ipotesi verificabile, non solo esempi scelti |
| 3 | Glossario piccolo OPPURE compressione/pruning LM | No | Un candidato congelato scelto su dev, entro byte/tempo/RAM |
| 4 | Correggere overflow training e selezione checkpoint | Richiede verifica prima di training | Nessun audio/target disallineato; scelta allineata al prodotto |
| 5 | InterCTC o adattamento mirato prudente | Si', separata autorizzazione | Migliora il task senza regressioni generali oltre il margine concordato |
| 6 | CR-CTC / conv / 25 Hz | Si', separata autorizzazione | Solo se il collo di bottiglia individuato lo giustifica |

Gate da fissare prima di ogni run:

- Controllo esatto 0.7.0 INT8, stessi audio e preprocessing; distinguere runtime storico e nuovo top-k.
- WER e ETA canonica sul misto, per categoria e voce; S/D/I e false inserzioni riportate.
- Dev separato dal test, una lettura finale del candidato; niente mining da S3/T.
- Intervalli appaiati dichiarati per clip o cluster; una voce held-out non rappresenta una popolazione.
- Italiano e inglese puro come regression set; margine di non inferiorita' predefinito e non scelto a posteriori.
- Non parlato con VAD, musica separata, nuove fonti per dichiarare robustezza a rumori non visti.
- Asset completi <=20 MB preferiti, mai >25 MB; browser totale e RAM contati separatamente.
- Tempo end-to-end sul dispositivo target, non solo FLOP o inference acustica.
- Provenienza e hash di checkpoint, LM, tokenizer, manifest, config, normalizzatore e codice.

Se il campione non supporta una conclusione, registrare "inconclusivo" e mantenere
l'originale. Nessuna delle proposte garantisce un guadagno o autorizza a chiamare
"migliorato" un nuovo file solo perche' e' eseguibile.

## 13. Conclusione

Moth e' gia' vicino al limite preferito di dimensione, ma non e' dimostrato che sia
vicino al massimo ottenibile con la stessa architettura. I guadagni storici del decoder
e del gold multi-voce suggeriscono di intervenire su diagnosi, dati e obiettivi prima
di comprare altri parametri.

La strada piu' prudente e' prima separare acoustic evidence, prior e search, poi
scegliere un solo intervento coerente col difetto misurato. Se il training torna
ammesso, InterCTC e audio mirato sono opzioni size-neutral piu' motivate di un aumento
indiscriminato della rete. PR, modello pubblicato e pesi non sono modificati da questo audit.
