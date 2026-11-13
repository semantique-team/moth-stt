# Chiusura Del Modello Entro 20 MB

Data: **2026-11-13**. Baseline prodotto: **Moth 0.8.1** (acustica E-028, decoder E-030, LM E-031).
Questo documento registra **cosa manca prima di concludere il modello** e i criteri di chiusura.
**Non registra esperimenti eseguiti e non autorizza training**: gli esperimenti reali mantengono il
loro PLAN, la voce in `EXPERIMENTS.md` e `runs/<id>/`.

Fonti: [piano strategia](2026-11-12-model-improvement-strategy.md), [STATUS](../../STATUS.md),
[dossier paper](../13-current-model-and-paper-readiness.md) §7, [ledger](../12-experiment-ledger.md),
[E-033](../../runs/E-033/RESULT.md), [E-034](../../runs/E-034/RESULT.md).

## Decisione Del Proprietario

- Il modello **resta entro 20.000.000 B** di asset obbligatori. 0.8.1 occupa **19.706.289 B**,
  margine **293.711 B**. Il tier 25 MB non e' autorizzato e non va assunto in nessuna proposta.
- Un modello piu' grande (ordine **≤ 50 MB**) sara' un progetto separato, in **un'altra repo e un
  altro paper**. Non appartiene a questo repository ne' a questo piano.
- Conseguenza diretta: la via "LM piu' grande" e' **chiusa qui** (E-034), e ogni ulteriore guadagno
  deve stare nei 293.711 B oppure in pesi/ricetta **a parita' di asset** (training-only).

## Stato Degli Item Del Piano

| Item | Stato | Evidenza / cosa manca | Costo residuo | Blocco |
|---|---|---|---|---|
| Dev mixed reale su speaker noti (S1/S2) | **non eseguito** | il kit e' pronto (`recordings/segment_notes.py`, import gia' supportato) | ~1 giornata umana + trascrizione; valutazione in minuti | servono le registrazioni |
| Holdout indipendente con voce nuova | **non eseguito** | nessun audio disponibile | dipende dalla persona | speaker nuovo |
| LM budgeted nei 293.711 B | **eseguito, respinto al readout** | E-035: 73.427 trigrammi count-1, binario **5.675.700 B**, pacchetto **19.999.997 B**, dev macro **22,7849% vs 22,8494%** (**−0,0645 pp**, CI [−0,1144, −0,0134]); **readout prodotto: tie su tutte le metriche primarie** (T-600 +0,0287 pp, T-S3 −0,0852, FLEURS-IT +0,0062, EN −0,0437, rari 38/81 in entrambi), `noise_music` +2 parole, nessun guadagno significativo → **non adottato, 0.8.1 resta** | — (chiuso) | tetto prodotto: nessun guadagno misurabile |
| Fine-tune conservativo (replay + distillazione verso il Moth precedente) | **non eseguito** | E-015/E-032 dicono che aggiungere dati mirati rompe il generale; questa e' la variante protettiva | ~2 h per arma, 2-3 armi | nessuno |
| Pilot TTS per il training | **non eseguito** | nessun generatore qualificato ne' licenza verificata | giorni (ascolto + 1 run vs controllo) | qualifica + licenze |
| Audio reale di dominio con diritti | **non eseguito** | nessuna fonte autorizzata | settimane | reperimento + diritti |
| InterCTC | **respinto** | E-033: +0,0249 pp dev, CI [−0,0998, +0,1557], niente esportato | — | chiuso per questa ricetta |
| LM piu' grande | **respinto** | E-034: −0,598/−0,644 pp dev ma 12,03/12,31 MB → 26,36/26,64 MB | — | tetto byte |
| Addendum acustico S1 | **respinto** | E-032: rari su, T-S3/FLEURS/naturali giu', 6 gate falliti | — | riutilizzabile come dati |
| QAT | **respinto** | E-029: nessun candidato pubblicabile | — | — |
| Bias contestuale decode-time | **respinto** | E-027: rari fermi, tie su T-600 | — | — |
| Nuovo tokenizer / CR-CTC / altri blocchi acustici / piccolo LM neurale | **rimandato** | piano strategia, "strade da rimandare" | — | costo, byte, ricette negative |
| E-018 gold scaling | **mai eseguito** ("morto senza S4") | protocollo scritto nel 2026-11-07, nessun run | — | serve un quarto speaker |

## Evidenze Di Chiusura Non Coperte Dal Piano

Dal [dossier §7](../13-current-model-and-paper-readiness.md); senza queste, il paper resta un case
study ingegneristico/negative-results e non un claim di generalizzazione.

| Gap | Cosa serve | Costo |
|---|---|---|
| Multi-seed | ≥3 seed predeclarati sulle armi abbinate, con curve, dispersione e checkpoint conservati | 3 × ~2 h per arma: 1-2 giorni di macchina |
| fp32 vs INT8 appaiati con lo **stesso** LM E-031 | export fp32 + decode sulle stesse clip | ~mezza giornata |
| Browser acoustic WER/ETA | oltre l'acceptance 4/4 e la parita' decoder: misura WER nel browser | ~1 giorno (harness ripetibile) |
| Leakage audit | hash audio per fonte, near-dup testuale su train/dev/test/T/R/pool, provenienza dei termini | ~1 giorno + review |
| Bootstrap per cluster (speaker/stanza) | richiede piu' speaker e stanze | dipende dai dati |
| Annotation validity | linee guida span, secondo annotatore, accordo | serve una persona |
| Noise generalization | stanze/transienti indipendenti, SNR su parlato attivo | nuove registrazioni |
| Upstream contamination | disclosure overlap Whisper/Whistle/Parakeet | dichiarativo |

## Le Due Chiusure

1. **Chiusura tecnica (freeze del modello, fattibile qui):** E-035 ha mostrato che il margine di
   293.711 B vale un piccolo guadagno dev ma **non un guadagno prodotto**: il candidato non viene
   adottato e 0.8.1 resta. E-036 (fine-tune conservativo) decide la domanda training-side.
   fp32/INT8 appaiato, multi-seed e browser WER rendono difendibili i numeri gia' pubblicati. Se
   E-036 fallisce, il modello si **congela a 0.8.1** con la motivazione scritta, non per
   stanchezza.
2. **Chiusura dell'evidenza (paper):** richiede il **dev mixed reale** e, per l'affermazione piu'
   forte, una **voce nuova**. Nessuna quantita' di compute locale sostituisce questi due.

## Ordine Proposto

1. ~~**E-035 — LM budgeted**~~ — **chiuso**: dev −0,0645 pp, ma readout prodotto tie su tutte le
   metriche primarie e nessun guadagno significativo; **non adottato, 0.8.1 resta**.
2. **E-036 — fine-tune conservativo** (encoder congelato/ultimi blocchi, replay reale,
   distillazione verso il Moth precedente): chiude la domanda training-side senza nuovi dati.
3. **Evidenze di chiusura**: fp32/INT8 appaiato, ≥3 seed, browser WER/ETA.
4. **Lavoro umano in parallelo**: note vocali da 5-10 minuti per il dev mixed; la prima voce nuova
   disponibile va riservata al readout di un candidato gia' congelato, non al training.
5. **Solo se serve**: pilot TTS o audio reale di dominio, con licenze verificate prima.

## Criteri Di Chiusura

- [x] E-035 eseguito e deciso — **dev in avanzo, readout prodotto tie: non adottato**
- [ ] E-036 eseguito e deciso
- [ ] fp32/INT8 appaiato con lo stesso LM registrato
- [ ] ≥3 seed predeclarati con dispersione
- [ ] browser WER/ETA misurato, non solo acceptance
- [ ] dev mixed reale congelato prima di guardare gli errori
- [ ] (opzionale) readout su voce nuova
- [ ] limiti dichiarati nella card; modello congelato; nessun claim oltre l'evidenza

## Limiti

Nessun esperimento e' stato eseguito per questo documento. I costi sono stime dell'ordine di
grandezza, non misure. Il freeze non e' una dichiarazione di qualita': e' la dichiarazione che le
ipotesi residue o richiedono dati nuovi o escono dal tetto di 20 MB.
