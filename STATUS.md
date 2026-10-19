# STATUS — snapshot per riprendere i lavori

> Aggiornato: **2026-10-19** — ultimo commit su `origin/main`: `a37b33e`. Suite: **173 test verdi**, `ruff` pulito.
> Questo file dice **dove siamo**; il *come* e il *perché* stanno nei design doc (`docs/01`…`09`) e nel
> piano (`docs/plans/2026-10-19-implementation-plan.md`).

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
| Dati (2.1–2.6) | registry+download+reader, filtri, BPE tokenizer, log-mel con parità Whisper (+ asset 64 KB), shard float16, dataset+augmentation | `stt/data/`, `stt/features.py`, `stt/tokenizer.py`, `stt/assets/`, `data/en_terms.txt` (580 termini) |
| Modello (3.1–3.3) | encoder Whisper-tiny + testa CTC (8.027.009 param), conversione pesi + parità offline, decode greedy | `stt/model.py`, `stt/convert_whisper.py`, `stt/decode.py` |
| Script | **T.jsonl** (600, riviste) e **R-S1.jsonl** (600) | `recordings/scripts/` |

## In corso / fuori dal repo

- **Download in background**: `data/raw/fetch.sh` (v2: LibriSpeech@elda, MLS@MetaS3, FLEURS@HF, in parallelo).
  - fine = `ls data/raw/FETCH_OK`; log = `data/raw/fetch.log`
  - se interrotto: `bash data/raw/fetch.sh` (riprende dai `.part`, salta i completati)
- **Registrazioni T**: S1 a **31/200**. Riprendere con:
  `uv run recordings/record.py --speaker S1 --kind T` — mancano S2 e S3 (200 ciascuno, S3 legge solo T).
- **Da generare/importare (tu)**: R-S2 (600–1000) e pool (3–5k) con `recordings/generate_scripts.py`.

## Da fare (prossimi task)

| Task | Cosa | Blocco |
|---|---|---|
| 1.4 | E-001: baseline H0 sui manifest (interim FLEURS) | serve download FLEURS |
| 2.7 | E-002: manifest reali (filtri) + training tokenizer + mel shard | serve download (+ CV IT) |
| 3.4 | `export.py`: ONNX (sinusoidi in-graph) + int8 + `release/` con hash | nessuno |
| 3.5 | smoke web + **H4 early** RTFX (node/npm presenti; installare `onnxruntime-web` + Playwright) | nessuno |
| M4–M8 | H1, H2, H3, demo, release | da piano |

## Serve da te

- **Common Voice IT 27** (CC0): link dal sito Mozilla dopo aver accettato i termini → poi `stt.data.sources.download`.
- Registrazioni T per S2/S3 (quando vuoi) e revisione/marca-letto se aggiungi frasi a T.
- `en_terms.txt` è a 580 voci (target design 1–2k): estendibile quando serve.

## Note di contesto utili

- Fix già applicati ai doc: parametri totali **8.027.009** (V_ctc = V_sp+1); sinusoidi in **float32** come Whisper
  (scoperta durante il test di parità).
- Il test set T **non** si usa per training né per tuning; i sorgenti eval-only (FLEURS) sono rifiutati per `train`.
- Macchina: M5 16 GB; `torch 2.14.1` pinnato; `ffmpeg` installato; `uv` gestisce l'ambiente.
