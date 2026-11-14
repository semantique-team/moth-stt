# Valutazione Della Repo E Correzioni Senza Training

Data: 2026-11-13. Baseline della review: `main`, commit `0fde3cb`.
Fetch e pull fast-forward di 40 commit; worktree inizialmente pulito.
Questo documento conserva la valutazione e lo scope autorizzato delle correzioni.
Non e' un nuovo esperimento: nessun training, peso, export di produzione, LM build,
pubblicazione o nuovo gate qualitativo. Release **0.8.1 invariata**.

## Valutazione Generale

Buona repo di ricerca applicata: prototipo distribuibile, architettura comprensibile,
disciplina sperimentale e risultati negativi conservati. Non e' ancora una validazione
forte di un riconoscitore code-switching generalista.

| Area | Valutazione |
|---|---|
| Architettura | Piccola e coerente con il vincolo browser/CPU |
| Software | Buona base e molti test; guardie e lifecycle richiedevano correzioni |
| Esperimenti | Configurazioni, ipotesi e decisioni tracciate; scarti motivati |
| Riproducibilita' | Buona per scoring e artefatti; il clone non ricostruisce il training |
| Evidenza | Tre speaker, parlato letto e benchmark ripetutamente osservati |
| Documentazione | Ricca e franca, ma stratificata e con contraddizioni residue |

Punti forti: dipendenze runtime separate da Torch, loader Python con hash e contenimento
dei path, scorer centralizzato, decoder Python/JS appaiati, checkpoint/resume e corpus
congelati, preservazione dei testi registrati e risultati negativi.
Il debito principale e' la coerenza tra codice, percorsi di valutazione e conclusioni,
non l'assenza di un'altra architettura.

## Modello Pubblicato

0.8.1 = acustica E-028 + decoder E-030 + LM E-031.

| Misura conservata | Valore |
|---|---:|
| Parametri acustici | 11.575.169 |
| Asset obbligatori | 19.706.289 B |
| Margine sul tetto 20.000.000 B | 293.711 B |
| T-600 | 24,08% WER / 55,23% ETA |
| T-S3 | 24,96% WER / 51,27% ETA |
| FLEURS-IT filtrato | 20,16% WER |
| FLEURS-EN filtrato | 49,81% WER |

Sono risultati ONNX reference Python conservati, non nuova inferenza o WER browser.
Gli asset non includono ORT/WASM, pagina o dipendenze Python. Circa meta' dei termini
inglesi annotati non e' riconosciuta; l'inglese puro resta debole.
WER S1/S2/S3: 20,98 / 26,32 / 24,96%; ETA: 63,28 / 51,56 / 51,27%.
Speaker, frasi e canali sono confondenti: il pooled non stima nuove voci.

Il freeze e' motivato: E-032, E-033, E-035 ed E-036 non producono candidati adottabili;
E-034 guadagna sul dev ma supera il budget. E-037 misura un costo INT8 piccolo e non
uniforme; il pacchetto fp32 appaiato pesa 54,16 MB. Restare su INT8 e 0.8.1 e' sensato.

## Findings E Correzioni

| Finding alla baseline | Correzione senza training | Limite |
|---|---|---|
| E-039 MinHash usa un moltiplicatore fisso e offset, selezionando lo stesso minimo | Permutazioni affini indipendenti, deterministiche, aritmetica senza overflow; test contro coppie ad alto overlap | Vecchio risultato negativo superato; corpus non rieseguito; LSH non e' esaustivo |
| Provenienza dei 44 termini omette LibriSpeech standalone | Inclusione LS-EN e fonti esplicite; qualificazione del vecchio conteggio | Nuova tassonomia non ricalcolata; niente nuovo numero dichiarato |
| Audio audit confonde file presenti/cross-source con split completi | Scope esplicito: loose files presenti, byte hash, non equivalenza del segnale | Archivi, audio mancanti, root separati e split non certificati |
| E-036 `x` sul silenzio presentata come regressione del prodotto completo | Distinzione cache mel senza VAD vs runtime completo; silenzio = un segnale x120; conteggi music riconciliati dai JSON | Nessuna nuova misura VAD o inferenza |
| Training aggiorna pesi con loss finita ma gradienti non finiti | Skip/log/abort anche sul backward, azzeramento gradienti, niente optimizer/scheduler step | Test CPU sintetici, non certificazione GPU/MPS |
| Export fp32 puo' accettare NaN nei gap | Rifiuto NaN/Inf in tensori, gap e median/worst gate | Nessun export con pesi reali |
| Init/KD non verifica tokenizer; shape mismatch troppo permissivo | Digest salvato: reset head nell'init, rifiuto KD incompatibile; mismatch non-head rifiutati | Checkpoint storici senza digest: warning, identita' non verificabile |
| Dev puo' essere vuoto, test, lingua errata o duplicato tra gruppi | Validazione ruolo/split, lingua, speech/durata, identita' IT/EN; output/WER finiti | Alias audio rinominati non rilevati tramite byte hash |
| Stop durante permesso microfono non annulla l'avvio | Stato di cancellazione; late grant scartato e risorse liberate | getUserMedia non e' abortibile; cleanup dopo risposta |
| Errori UI non gestiti, PCM vecchio o decode fuori ordine | Controlli concorrenti disabilitati, catch/recovery, invalidazione PCM e richieste obsolete | Test pagina in VM, non microphone acceptance reale |
| Browser valida meno del contratto Python | Schema, asset/hash/path/redirect/budget, vocab, feature/VAD/output finiti; lock benchmark completo | Hash = integrita', non autenticita'; smoke LM-free solo con peso LM zero |
| Note multi-file sovrascrivono skeleton o falliscono dopo output parziale | Skeleton aggregato, collisioni preflight, staging e sostituzione atomica skeleton | Pubblicazione di piu' WAV non crash-transactional |
| E-041 puo' confrontare bundle/input/codice diversi | Pin record 0.8.1, decoder e asset; receipt manifest/audio/runtime; duplicate ID rifiutati; porta/server controllati | Harness non eseguito; input storici E-037 non completamente attestati |

Riferimenti principali: `runs/E-039/audit.py`, `audit2.py`, `RESULT.md`;
`stt/train.py`, `stt/export.py`; `web/index.html`, `web/ort.js`;
`recordings/segment_notes.py`; `runs/E-041/browser_wer.mjs`, `score.py`, `PLAN.md`.
Test di regressione nei rispettivi `tests/` Python e `web/tests/` Node.

## Interpretazione Dell'Evidenza

E-035/E-036/E-037 decodificano cache mel senza VAD/segmentazione (`runs/E-022/eval.py`).
`ReleaseMoth` e browser applicano invece la guardia VAD. Le emissioni sul non-parlato
non dimostrano da sole allucinazioni del runtime completo. E-041 confronta quindi
percorsi end-to-end diversi, non soltanto kernel e feature acustiche.
L'acceptance 4/4 e la parita' sugli stessi logit non sostituiscono WER/ETA browser.

T/S3 hanno informato lo sviluppo; A riusa frasi T; gli addenda sono test-informed.
La separazione letterale delle frasi non restituisce indipendenza. CI per utterance
sono nominali, non incertezza tra speaker/stanze/seed; tie non significa equivalenza.
Il risultato raro 38/81 include 34 co-token: target-only 24/47 va riportato separatamente.
Una ricetta Whistle LoRA negativa non prova causalita' architetturale.

La decisione corrente rinuncia a nuove registrazioni: niente dev mixed spontaneo o
voce nuova. Scope difendibile: case study ingegneristico di parlato letto IT/EN,
deployment leggero e risultati negativi, senza claim su parlato spontaneo/nuove voci.

## Lavoro Residuo

1. Rieseguire E-039 corretto con ricevute e review dei near-duplicate; non sostituire numeri a mano.
2. Eseguire E-041 con asset disponibili, risultati per speaker e ipotesi differenti; distinguere VAD dal gap acustico.
3. E-040: conservare log/readout di tutti i seed; replica E-036 respinto, non robustezza multi-seed di 0.8.1.
4. Chiudere provenienza checkpoint/export/tokenizer/baseline e tabelle benchmark unificate.
5. Mantenere il freeze e i limiti; non riaprire tokenizer/LM neurale/blocchi senza nuova evidenza.

Altri rischi della review non corretti in questo task: fattibilita' CTC esatta dopo
augmentation e target vuoti; payload mel non incluso nel fingerprint resume; pipeline
audio-health incompleta; cap oversampling tokenizer; reference canoniche e protocollo
`--vs`; inventario teacher preflight; resampling Nyquist; esclusioni incrementali basate
su filename; padding dispari convoluzionale; ricevute baseline e lifecycle recorder.
Sono follow-up separati, non evidenza che la release pubblicata sia compromessa.

Non si compatta l'archivio STATUS o la storia dei run: si aggiungono correzioni visibili
e si mantiene l'evidenza originale. Nessuna dipendenza aggiunta.

## Verifica

Baseline prima delle correzioni: Python **1029 passed / 3 skipped**, Node **26/26**,
Ruff e `git diff --check` puliti. `uv` non nel PATH: usato `.venv/Scripts/python.exe`.
Verifica finale delle correzioni:

- `.venv/Scripts/python.exe -m pytest -q`: **1170 passed / 3 skipped**, due warning PyTorch backend.
- `.venv/Scripts/ruff.exe check .`: pulito.
- Node 20.20.2, quattro suite con `--experimental-vm-modules --test-concurrency=1`: **38/38**, nessun test saltato.
- `node --check runs/E-041/browser_wer.mjs`: pulito.
- `git diff --check`: pulito; warning LF/CRLF della configurazione locale, non whitespace errors.
- Review indipendente del diff: regressioni smoke LM-free, URL relativi e identita' legacy corrette; nessun finding bloccante residuo individuato.

Node 20 ha inizialmente avuto access violation nel loader VM sperimentale. Retenere i
grafi di moduli fino al teardown riproduce e risolve il problema nella prova A/B:
30 esecuzioni runtime e 20 suite seriali consecutive passate, poi un ultimo run 38/38.
Nessun retry/skip introdotto nei test. La causa interna V8 non e' provata senza crash stack.
Node 22 (versione CI) non verificato: recupero bloccato da catena certificati npm;
verifica TLS non disabilitata.

Non eseguiti: training reale, export pesi, corpus audit completo, E-040/E-041,
Playwright/WASM, inferenza con release, nuova verifica live HF o pubblicazione.
