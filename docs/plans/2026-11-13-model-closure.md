# Chiusura Del Modello Entro 20 MB

Data: **2026-11-13**. Baseline prodotto: **Moth 0.8.1** (acustica E-028, decoder E-030, LM E-031).
Questo documento registra **cosa manca prima di concludere il modello** e i criteri di chiusura.
**Non registra esperimenti eseguiti e non autorizza training**: gli esperimenti reali mantengono il
loro PLAN, la voce in `EXPERIMENTS.md` e `runs/<id>/`.

> **Aggiornamento 2026-11-13:** E-035 e E-036 sono stati eseguiti e sono **entrambi negativi**.
> **0.8.1 e' congelato**: nessun candidato acustico o LM e' stato adottato. Restano le evidenze di
> chiusura (fp32/INT8 **E-037**, per-speaker **E-038**, leakage audit **E-039**, multi-seed
> **E-040 in corso**, browser **E-041 in coda**). **Nessuna nuova registrazione** (decisione
> dell'organizzatore): niente dev mixed spontaneo e niente voce nuova.

> **Correzione della review senza training (2026-11-13):**
> [report 19](../19-repository-review-2026-11-13.md). E-039 ha metodo/tassonomia corretti,
> ma i vecchi conteggi non sono ricalcolati. E-036 misura silenzio senza VAD su un
> segnale ripetuto x120, non il runtime completo. E-040 non ha log/readout conservati
> in questa baseline; "in corso" sopra non ne certifica l'esecuzione. E-041 e' preparato,
> non misurato; la replica E-040 riguarda E-036 respinto, non tutta la lineage 0.8.1.

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
| Dev mixed reale su speaker noti (S1/S2/S3) | **rinunciato per decisione dell'organizzatore (2026-11-13)** | il kit resterebbe pronto (`recordings/segment_notes.py`, import supportato), ma **non si registra**: nessun parlato spontaneo, nessuna nota vocale | — | decisione, non blocco tecnico: l'evidenza di dominio resta quella scripted |
| Holdout indipendente con voce nuova | **non ottenibile e non perseguito (2026-11-13)** | l'organizzatore ha solo **S1, S2 e S3**; S1/S2 sono nel training e S3 e' stato ispezionato piu' volte. **E-038** riporta T-600 per voce (S1 20,98 / S2 26,32 / S3 24,96 WER; dispersione 5,34 pp WER e 12,02 pp ETA) come sostituto descrittivo | — | nessuna quarta voce disponibile: va dichiarato come limite, non riproposto |
| LM budgeted nei 293.711 B | **eseguito, respinto al readout** | E-035: 73.427 trigrammi count-1, binario **5.675.700 B**, pacchetto **19.999.997 B**, dev macro **22,7849% vs 22,8494%** (**−0,0645 pp**, CI [−0,1144, −0,0134]); **readout prodotto: tie su tutte le metriche primarie** (T-600 +0,0287 pp, T-S3 −0,0852, FLEURS-IT +0,0062, EN −0,0437, rari 38/81 in entrambi), `noise_music` +2 parole, nessun guadagno significativo → **non adottato, 0.8.1 resta** | — (chiuso) | tetto prodotto: nessun guadagno misurabile |
| Fine-tune conservativo (replay + distillazione verso il Moth precedente) | **eseguito, respinto** | E-036: `frozen` 3,94M e `last2` 7,49M param mobili, KD 20; dev **30,9859% / 30,9720%** vs 31,0101 (tie); prodotto tie su WER ma rari **39/81** e **41/81** (target 42), **`x` su tutti i 120 clip di silenzio** (0.8.1: 0), `last2` T-S3 ETA **−1,41 pp** [−2,76, −0,28]. **Nessun candidato, 0.8.1 congelato** | — (chiuso) | il silenzio e i rari non sono recuperabili con questa ricetta |
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
| fp32 vs INT8 appaiati con lo **stesso** LM E-031 | **eseguito (E-037)**: dev +0,1574 pp [+0,0539, +0,2664], T-600 +0,4874 (tie), FLEURS-EN +0,4459 [+0,0087, +0,8652], rari 38/81 =, naturali 444→432; pacchetto 54.164.659 vs 19.706.289 B. INT8 resta | — (chiuso) |
| Browser acoustic WER/ETA | oltre l'acceptance 4/4 e la parita' decoder: misura WER nel browser | ~1 giorno (harness ripetibile) |
| Leakage audit | **primo passaggio (E-039, 2026-11-13)**: 2.055 testi condivisi tutti strutturali o repeat di corpus, 0 coppie inattese; near-dup > 0,8 nei nostri dati **0**; 2.209 wav, 0 hash ripetuti, 0 collisioni tra set, 0 file mancanti; id train/dev/test disgiunti; 44 termini "mai visti" = **mai visti da 0.7.0**, 0.8.1 li ha visti via addenda. Restano: re-hash di `data/raw` (13 GB) e review umana dei near-dup esterni | — (primo passaggio) |
| Bootstrap per cluster (speaker/stanza) | **parziale (E-038)**: CI per-utterance per voce, non cluster; con 3 voci la varianza tra speaker non e' stimabile | dipende da speaker/stanza aggiuntivi |
| Annotation validity | linee guida span, secondo annotatore, accordo | serve una persona |
| Noise generalization | stanze/transienti indipendenti, SNR su parlato attivo | nuove registrazioni |
| Upstream contamination | disclosure overlap Whisper/Whistle/Parakeet | dichiarativo |

## Le Due Chiusure

1. **Chiusura tecnica (freeze del modello):** E-035 ha mostrato che il margine di 293.711 B vale
   un piccolo guadagno dev ma **non un guadagno prodotto**. E-036 ha fallito su entrambi i bracci:
   WER tie, rari sotto target, silenzio allucinato. **Il modello si congela a 0.8.1** con la
   motivazione scritta, non per stanchezza. fp32/INT8 e' stato misurato in E-037; multi-seed e
   browser WER restano le evidenze che rendono difendibili i numeri gia' pubblicati.
2. **Chiusura dell'evidenza (paper):** **nessuna nuova registrazione** (decisione 2026-11-13):
   niente dev mixed spontaneo e nessuna voce nuova. L'evidenza resta **scripted** (T su S1/S2/S3,
   anchor A, FLEURS-IT/EN, CV-IT, MLS-IT, LS-EN, rumore) piu' le misure di deployment (E-037
   INT8, E-038 per voce, E-041 browser) e l'audit E-039. Il paper deve dichiarare che **non**
   sostiene generalizzazione a parlato spontaneo ne' a voci non viste.

## Ordine Proposto

1. ~~**E-035 — LM budgeted**~~ — **chiuso**: dev −0,0645 pp, ma readout prodotto tie su tutte le
   metriche primarie e nessun guadagno significativo; **non adottato, 0.8.1 resta**.
2. ~~**E-036 — fine-tune conservativo**~~ — **chiuso negativo**: WER tie, rari 39/81 e 41/81,
   silenzio 0→120 `x`, `last2` T-S3 ETA −1,41 pp; **0.8.1 congelato**.
3. **Evidenze di chiusura**: ~~fp32/INT8 appaiato~~ (E-037) e ~~per-speaker~~ (E-038) fatti; restano
   ≥3 seed, browser WER/ETA e leakage audit.
4. ~~**Lavoro umano in parallelo**: note vocali per il dev mixed; voce nuova riservata al readout~~
   — **rinunciato (2026-11-13)**: nessuna registrazione, il limite va scritto e non riproposto.
5. **Solo se serve**: pilot TTS o audio reale di dominio, con licenze verificate prima.

## Criteri Di Chiusura

- [x] E-035 eseguito e deciso — **dev in avanzo, readout prodotto tie: non adottato**
- [x] E-036 eseguito e deciso — **entrambi i bracci respinti; 0.8.1 congelato**
- [x] fp32/INT8 appaiato con lo stesso LM registrato — **E-037: costo +0,1574 pp dev, INT8 resta**
- [ ] ≥3 seed predeclarati con dispersione
- [ ] browser WER/ETA misurato, non solo acceptance
- [x] dev mixed reale rinunciato per decisione 2026-11-13; assenza dichiarata come limite
- [x] readout su voce nuova non perseguito; assenza dichiarata come limite
- [ ] limiti dichiarati nella card; modello congelato; nessun claim oltre l'evidenza

## Limiti

Questo documento non esegue esperimenti: i risultati E-035/E-036 vivono nei rispettivi `runs/`. I costi sono stime dell'ordine di
grandezza, non misure. Il freeze non e' una dichiarazione di qualita': e' la dichiarazione che le
ipotesi residue o richiedono dati nuovi o escono dal tetto di 20 MB.
