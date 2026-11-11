# Miglioramenti Dell'Audit: Codice Pronto, Training Differito

Data: 2026-11-10. Implementazione iniziata su `feat/audit-quality-tooling`,
inclusa nella PR #3 sul branch `docs/lightweight-stt-architecture-audit`.
Riferimento: [audit strutturale](../14-architecture-audit-2026-11-10.md).
Questo lavoro prepara strumenti e ricette, non produce un modello migliore gia' misurato.
Gli asset pubblicati 0.7.0 non sono stati sovrascritti.

## Stato Degli Interventi

| Area | Implementato | Cosa resta da fare sull'altro PC |
|---|---|---|
| Tempo augmentation | Fallback alla durata originale se lo stretch supera 3000 frame; input troppo lunghi rifiutati, mai troncati con target completo | Misurare frequenza del fallback sui dati veri |
| InterCTC | Testa ausiliaria solo training al termine encoder, un solo forward encoder, loss pesata, stato checkpoint separato | Addestrare e confrontare con controllo senza InterCTC |
| Preflight training | `--check-config`, inventario file, hash tokenizer e fingerprint ricetta per resume | Recuperare dati e checkpoint e verificare i percorsi |
| Selezione prodotto | Valutazione separata di finalisti con beam/LM e mixed-dev, ETA e gate regressioni | Registrare mixed-dev indipendente e valutare checkpoint reali |
| Diagnostica | Greedy/beam acustico/beam+LM, scorer canonico; probabilita' CTC riferimento e supporto top-k con tokenizer verificato | Eseguire su errori naturali annotati |
| Bias contestuale | Opt-in Python/JS, reward bounded per parola e refund dei prefissi abbandonati | Scegliere glossario prima del test e misurare false inserzioni |
| Compattazione LM | Rimozione lossless righe trigramma vuote e verifica completa delle righe conservate | Nessun guadagno sul LM attuale: zero righe vuote |
| Builder LM | Corpus congelato con receipt, tokenizer hashato, soglie asimmetriche e output `.bin` | Provare (1,2)/(2,1) su corpus legittimo congelato |

Non implementati intenzionalmente: CR-CTC, convoluzioni aggiuntive, 25 Hz, LM neurale,
distillazione e modifica della definizione ETA storica. Sono esperimenti separati, non
ingredienti da attivare contemporaneamente prima di aver misurato il primo braccio.
Gli inventari storici di esposizione acustica/tokenizer/LM richiedono ancora i dati originali.

## 1. Preparazione Dell'Ambiente

```bash
uv sync --locked --all-extras
uv run --locked --all-extras python -m stt.train --help
uv run --locked --all-extras python -m stt.training_eval --help
```

Trasferire tramite canale appropriato tokenizer binario, checkpoint E-024, manifest e
shard, LM della release, documentazione dei consensi e un mixed-dev nuovo. Audio,
checkpoint e shard non vanno aggiunti a git. Non copiare token o credenziali.

Tutti i comandi usano percorsi relativi alla working directory. Copiare una configurazione
in una NUOVA directory run e aggiornare i percorsi. Non riusare l'output directory di un
training diverso: resume e `init_checkpoint` hanno significati differenti.

## 2. InterCTC E Preflight

Partire da una copia della ricetta E-024, mantenendo dati e schedule sotto controllo.
Per un nuovo esperimento da pesi E-024, impostare `model.init_checkpoint` al checkpoint
originale e NON ripristinare il suo optimizer in una directory esistente.
Aggiungere solo questo blocco per il braccio InterCTC:

```yaml
train:
  interctc:
    weight: 0.25
    tap: encoder_output
```

Il blocco sopra va unito alle impostazioni `train` esistenti, non le sostituisce.
0.25 e' un valore iniziale sperimentale, non un optimum misurato. Il controllo usa
weight 0, default. Le altre impostazioni e i dati devono essere comparabili.

Loss: `(1-weight) * CTC_finale + weight * CTC_ausiliaria`. La proiezione ausiliaria
legge il termine dei quattro blocchi encoder, prima dei due blocchi aggiuntivi.
Non e' un tap dopo il secondo blocco. I calcoli CTC/log-softmax sono FP32.

```bash
# SOLO controlli, nessun modello o forward di training:
uv run --locked --all-extras python -m stt.train --config runs/NUOVO-ID/config.yaml --check-config
```

Il preflight segnala file mancanti e campi errati. Il fingerprint pinna ricetta,
tokenizer, manifest e indici; non calcola hash di tutti i grandi shard mel.
La testa ausiliaria vive in `training_objective`, non nel `model.state_dict()` esportabile.
Il resume rifiuta ricette/tokenizer diversi: usare model-only initialization per un nuovo run.
I vecchi checkpoint senza hash si basano sui file dichiarati ancora disponibili, non
possono ricostruire magicamente l'identita' storica di file successivamente modificati.

Solo quando si vuole realmente avviare training sull'altro PC:

```bash
uv run --locked --all-extras python -m stt.train --config runs/NUOVO-ID/config.yaml --device cuda
```

Usare `--device mps` sul Mac o `cpu` se necessario. Il default storico non e' una
promessa di selezione automatica della GPU CUDA. Nulla di questo documento avvia un run.

## 3. Scelta Dei Checkpoint Finalisti

Il monitoraggio veloce nel training resta greedy IT/EN; `ckpt_best.pt` conserva quel
significato storico. La scelta prodotto avviene SEPARATAMENTE con `stt.training_eval`.
Il comando restituisce ranking/report e non sostituisce silenziosamente `ckpt_best.pt`.
Conservare abbastanza checkpoint per non perdere tutti i candidati prima del confronto.

In una configurazione di valutazione, usare `data.train` come inventario completo
dell'esposizione acustica, inclusa quella delle fasi precedenti. Specificare i nuovi dev:

```yaml
data:
  train:
    - manifest: snapshots/all-training-exposure.jsonl
      shards: data/shards/train
  dev:
    - manifest: snapshots/it-dev.jsonl
      shards: data/shards/it-dev
  dev_en:
    - manifest: snapshots/en-dev.jsonl
      shards: data/shards/en-dev
  dev_mixed:
    - manifest: snapshots/mixed-dev.jsonl
      shards: data/shards/mixed-dev
evaluation:
  selection_metric: mixed_wer
  decoder:
    mode: beam
    lm_path: release/lm_bpe3.bin
    lm_sha256: dcf1d59f075858c7e5f210804b5aabde847fb5d33e8d716e0d534f7c7886003e
    beam_width: 12
    top_k: 10
    lm_weight: 0.5
    token_bonus: 2.4
  wer_margin: 0.005
  min_eta: 0.0
  insertion_margin: 0.0
```

Completare anche `model`, `tokenizer`, `train.input` e batch dalla ricetta. I percorsi
sono esempi, non file gia' presenti. I dev devono dichiarare split `dev`; mixed-dev
richiede `lang: mix` ed `en_spans` reali, eccetto controlli `control_it`.
T/A/R storici e S3 sono rifiutati come dev; gli overlap esatti ID/testo con train sono
rifiutati. Questo non copre near-duplicate o fonti omesse dall'inventario.

```bash
# Inventario, split, tokenizer e LM: nessun modello/forward.
uv run --locked --all-extras python -m stt.training_eval --config finalist.yaml --check-config --out preflight.json

# Inferenza soltanto, nessun optimizer: baseline e poi pochi candidati.
uv run --locked --all-extras python -m stt.training_eval --config finalist.yaml --checkpoint runs/E-024/checkpoints/ckpt_best.pt --out baseline-dev.json
uv run --locked --all-extras python -m stt.training_eval --config finalist.yaml --checkpoint runs/NUOVO-ID/checkpoints/ckpt_0002000.pt --checkpoint runs/NUOVO-ID/checkpoints/ckpt_0004000.pt --baseline-metrics baseline-dev.json --out finalists.json
```

Le soglie WER sono rapporti: 0.005 equivale a 0,5 punti percentuali; insertion margin
e' parole per 100 parole riferimento. Per default un candidato non puo' perdere ETA
rispetto al baseline ne' aumentare le inserzioni nel misto, e deve ridurre la WER
primaria per vincere. Parita' conserva il baseline. Sono gate puntuali di selezione,
non prova statistica di miglioramento. I report includono baseline, hash, batching,
tokenizer e protocollo; protocolli diversi non si confrontano automaticamente.
Il fingerprint include anche gli hash dei mel realmente consumati: cambiare audio con
stessi ID/durate non mantiene valida la comparazione. Per logit forniti esternamente,
la provenienza acustica deve essere attestata esplicitamente dal chiamante.

La valutazione e' FP32 checkpoint-based: il candidato scelto richiede ancora export
INT8, gate dimensione, browser/VAD e un test finale indipendente prima dell'adozione.

## 4. Diagnostica Senza Training

```bash
uv run --locked --all-extras python -m stt.diagnose release --manifest mixed-dev.jsonl --audio-root recordings --out diagnostics-new
# Se disponibile il tokenizer esatto, aggiungere:
# --tokenizer data/tokenizer/moth_bpe.model
```

Il manifest usa lo schema ufficiale e riferimenti corretti, non trascrizioni generate
dal modello. Audio mono 16 kHz, max 30 s, una utterance: VAD e segmentazione sono
disattivati per mantenere l'allineamento diagnostico. Output e ipotesi sono nuovi file.

Tre percorsi: greedy, beam acustico con bonus zero, beam+LM con bonus della release.
Il confronto degli ultimi due varia LM E bonus: non isola causalmente il solo LM.
Con tokenizer verificato pezzo per pezzo si calcola la probabilita' CTC completa del
riferimento e quella conservata dal top-k. Senza tokenizer queste misure sono assenti,
non si ricostruisce un encoder di testo greedy da `vocab.json`.

Le misure col riferimento sono oracle diagnostico, non accuratezza di produzione.
N-best, sopravvivenza nel prefix beam e Viterbi temporale non sono implementati.
Il supporto top-k misura candidati per frame, non garantisce sopravvivenza nel beam.

## 5. Bias Contestuale Opt-In

Python, con encoder che produce CTC ID dal tokenizer congelato:

```python
from stt.hotwords import ContextualBias
from stt.onnx_runner import ReleaseMoth
from stt.tokenizer import encode, load_tokenizer
from stt.diagnose import verify_tokenizer

runtime = ReleaseMoth("release")
tokenizer = load_tokenizer("data/tokenizer/moth_bpe.model")
verify_tokenizer(tokenizer, runtime.model.pieces)
bias = ContextualBias.from_terms(
    ["deploy", "rollback"],
    runtime.model.pieces,
    lambda word: encode(tokenizer, word),
    bonus=1.0,
    prefix_bonus=0.25,
)
text, seconds = runtime.transcribe(audio_16k_mono, contextual=bias)
```

Browser, con le sequenze CTC prodotte dallo STESSO tokenizer (non inventare gli ID):

```javascript
const result = await runner.transcribe(pcm, {
  contextualSequences: verifiedCtcSequences,
  contextualBonus: 1.0,
  contextualPrefixBonus: 0.25,
});
```

Solo parole singole normalizzate; niente frasi multiword. Il reward finale e' al
massimo `bonus` per occorrenza esatta confermata da boundary/EOS; il potenziale di
prefisso viene restituito se abbandonato. Ripetizioni CTC dello stesso frame non
accumulano reward. Il top-k resta una barriera: il bonus non recupera candidati esclusi.
Default disattivato; greedy con bias e' rifiutato, non ignorato.

Bonus e prefisso sono valori iniziali NON tarati. Scegliere glossario prima del test,
includere glossario irrilevante, controlli italiani e non parlato. Misurare latenza:
i microbenchmark sintetici indicano overhead del decoder, non gratuitita'. Nessuna
nuova accuratezza su audio reale e' dimostrata da questi unit test.

## 6. LM: Compattazione E Corpus Congelato

```bash
uv run --locked --no-dev python -m stt.compact_lm --help
uv run --locked --all-extras python -m stt.build_lm --help
```

La compattazione su `lm_bpe3.bin` pubblicato ha verificato 137.976 contesti e
804.969 entry trigramma. Input/output 5.116.464 byte, SHA identico
`dcf1d59f075858c7e5f210804b5aabde847fb5d33e8d716e0d534f7c7886003e`:
**0 byte risparmiati**. Nessun motivo per sostituire la release con la copia.

Per altri LM lo strumento elimina righe trigramma vuote, senza ricostruire conteggi
o riquantizzare score. Usa nuovi output e report, rifiutando sovrascritture.

Per provare in futuro pruning (1,2)/(2,1), prima creare corpus/receipt con
`stt.data.freeze_lm_corpus`, includendo esplicitamente le esclusioni come nella
[guida mixed-dev](mixed-dev-and-quality-experiments.md). Poi:

```bash
uv run --locked --all-extras python -m stt.build_lm --corpus frozen/corpus.jsonl --receipt frozen/receipt.json --tokenizer frozen/tokenizer.model --tokenizer-sha256 HASH_VERIFICATO --min-bigram-count 1 --min-trigram-count 2 --release release --check-inputs
```

`--check-inputs` valida/tokenizza soltanto, non costruisce il LM. Per la build futura
togliere quel flag e aggiungere `--out candidates/lm-1-2.bin --report candidates/lm-1-2.json`.
Usare nomi nuovi per (2,1); hash deve essere 64 caratteri esadecimali, non il placeholder.
Il budget considera la sostituzione del LM, ma la certificazione definitiva richiede
ancora packaging/config aggiornati e `stt.release`. Nessun LM reale e' stato ricostruito
per questo intervento.
Il receipt deve includere inventario training/esclusioni completo e coerente. Questi
controlli non autenticano chi ha prodotto il receipt e non provano che tutte le fonti
di test siano state incluse: serve revisione umana della provenienza.

## 7. Validazione E Limiti

Test sintetici coprono conservazione del contenuto, obiettivi CTC e gradienti,
checkpoint ausiliario separato, guardie resume, scoring/selection, log-prob CTC contro
enumerazione, bias Python/JS e roundtrip del LM. I test di correttezza non misurano WER
su nuovi parlanti e non rendono InterCTC una ricetta gia' validata.

Rimangono i fallimenti Windows preesistenti nelle baseline shell/path; `resource` e'
ora opzionale per importare gli strumenti di training su Windows. MPS/CUDA e la
ricetta completa vanno verificati sull'host di training. Nessuna nuova dipendenza.

Non avviare contemporaneamente tutte le modifiche: controllo senza InterCTC e braccio
InterCTC comparabili; contextual bias/LM come esperimenti separati di decoding.
Annotare la modifica al tempo-stretch come differenza rispetto alle ricette storiche.

## 8. Checklist Del Lavoro Differito

Nessuna voce di questa lista e' dichiarata completata dai test sintetici. Per ogni
esperimento creare prima un ID libero, entry in `EXPERIMENTS.md`, `runs/<id>/PLAN.md`,
config congelata, controllo e criteri di stop; poi registrare risultati anche negativi.

- [ ] Recuperare tokenizer/checkpoint/corpus storici e inventariare le esposizioni per componente, senza dedurle da file correnti mutabili.
- [ ] Raccogliere mixed-dev naturale e un holdout futuro distinto, con consensi, trascrizioni e `en_spans`; non usare T/S3 per tuning.
- [ ] Contare gli overflow tempo sui dati veri e confrontare il controllo con la nuova augmentation sicura; le ricette storiche non sono bit-identiche.
- [ ] Eseguire diagnosi su errori naturali, distinguendo top-k da perdita nel beam; N-best, prefix survival e allineamento Viterbi restano da implementare se necessari.
- [ ] Addestrare controllo e InterCTC con stessa esposizione/schedule, selezionare pochi finalisti su dev e verificare poi export INT8 e test indipendente.
- [ ] Misurare il bias contestuale su glossario pertinente e irrilevante, italiano puro, silenzio/rumore, false inserzioni e latenza; non attivarlo per default prima dei gate.
- [ ] Ricostruire LM (2,2) di controllo e bracci (1,2)/(2,1) da uno stesso corpus congelato; misurare byte, WER/ETA, RAM e tempo e rispettare 20/25 MB.
- [ ] Se il pruning non basta, valutare un formato LM piu' compatto: packing/score quantization richiedono un esperimento nuovo; la rimozione righe vuote attuale risparmia zero.
- [ ] Riconciliare la diagnostica storica dei rari con ETA allineata, inventari completi e ipotesi mancanti; non riscrivere i risultati storici come se fossero una nuova metrica.
- [ ] Verificare su CUDA/MPS la ricetta completa, resume, memoria e runtime browser/WASM reale; i test CPU non certificano questi percorsi.
- [ ] Solo dopo risultati motivanti, progettare separatamente CR-CTC con viste temporalmente allineate, piccola testa convoluzionale o pooling 25 Hz con budget/export aggiornati.
- [ ] Considerare LM neurale, distillazione o altro gold multi-voce soltanto con dati/provenienza/compute e protocollo approvati; nessuno di questi e' gia' implementato o validato.

Per dettagli architetturali e motivazioni delle alternative rinviate vedere le sezioni
8-12 dell'[audit](../14-architecture-audit-2026-11-10.md). Questa guida documenta lo stato
operativo; l'audit non e' una promessa di miglioramento e la PR non certifica nuovi pesi.
