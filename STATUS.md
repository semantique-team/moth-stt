# STATUS — snapshot per riprendere i lavori

> Aggiornato: **2026-10-19** — ultimo commit su `origin/main`: vedi `git log -1`. Suite: **194 test verdi**, `ruff` pulito.
> Questo file dice **dove siamo**; il *come* e il *perché* stanno nei design doc (`docs/01`…`09`) e nel
> piano (`docs/plans/2026-10-19-implementation-plan.md`, che ha una nota di sync in testa: i suoi checkbox
> non sono mantenuti, questo file è la fonte live).

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
| **2.5 mel cache** | **cache reale costruita**: 9 set, **291.9 h** (mls_it_train 169.9 h, ls_en_train 100.0 h, dev/test 22 h), 16 GB, **0 errori**, spot check bit-exact su ogni set → `data/shards/`; report in E-002 | `stt/data/build_mel.py`, `EXPERIMENTS.md` (E-002) |
| Modello (3.1–3.3) | encoder Whisper-tiny + testa CTC (8.027.009 param), conversione pesi + parità, decode greedy | `stt/model.py`, `stt/convert_whisper.py`, `stt/decode.py` |
| Conversione (3.2) | **verificata sui pesi reali** e materializzata: `models/whisper_tiny_encoder.pt` (30,6 MB; parità 5,2e-4 vs gate 1e-3 contro eager con posizioni del checkpoint; errore relativo 2,8e-5) | `models/` (gitignored), docstring di `stt/convert_whisper.py`, `docs/02` §3 |
| Export + web (3.4–3.5) | ONNX opset 17 (sinusoidi in-graph) + int8: **8.32 MB totali**; smoke browser Playwright; **RTFx 65–74×** (4 thread) / 33–38× (1 thread) con pesi random; E-003 | `stt/export.py`, `web/`, `runs/E-003/` |
| **4.1 train.py** | **loop di training completo**: resume a metà epoca, checkpoint last-3+best, log JSONL, guardie non-finite, dev WER, parity guard MPS↔CPU (**misurata Δ=0.0** su modello e dati reali), input truncated/padded30, augment none/light/full; 8 test; config `runs/E-004/config.yaml` | `stt/train.py`, `tests/test_train.py`, `runs/E-004/` |
| Script | **T.jsonl** (600, riviste) e **R-S1.jsonl** (600) | `recordings/scripts/` |

## In corso / fuori dal repo

- **Registrazioni T**: S1 a **31/200**. Riprendere con:
  `uv run recordings/record.py --speaker S1 --kind T` — mancano S2 e S3 (200 ciascuno, S3 legge solo T).
- **Da generare/importare (tu)**: R-S2 (600–1000) e pool (3–5k) con `recordings/generate_scripts.py`.
- **Dati**: download completati e verificati (`data/raw/FETCH_OK`); manifest in `data/manifests/` (9 set +
  `build_report.json`, E-002). Audio raw MLS/LibriSpeech **cancellato** dopo spot check (ricostruibile con
  `data/raw/fetch.sh`); resta **FLEURS (1.7 GB)** per le baseline di 1.4. Manca **Common Voice IT 27** (≈ 170 h,
  lato utente).

## Da fare (prossimi task)

| Task | Cosa | Blocco |
|---|---|---|
| 2.3 → 2.7 | **training del tokenizer** (codice pronto, test pronti): testo = trascrizioni + pool + `en_terms` up-weighted; poi stats nel build report | **pool (utente)** |
| 4.2 | H1 (`runs/E-004/config.yaml`): due bracci truncated/padded30, dev WER con CI, decisione al gate — **train.py e config sono pronti** | tokenizer |
| 1.4 | E-001: restano base, beam 5, Whistle, Parakeet, CV-IT, mixed T, noise | CLI Whistle/Parakeet; T registrato; CV IT |
| M5–M8 | H2, H3, demo, release; il benchmark browser con **pesi veri** arriva con M7 | da piano |

## Serve da te

- **Pool** (3–5k frasi, `recordings/generate_scripts.py`) → sblocca tokenizer → H1 (4.2). È l'unico vero blocco rimasto.
- Registrazioni T per S2/S3 (quando vuoi); R-S2 per H3.
- **Common Voice IT 27** (CC0): link dal sito Mozilla dopo aver accettato i termini → poi `stt.data.sources.download`.
- `en_terms.txt` è a 580 voci (target design 1–2k): estendibile quando serve.

## Note di contesto utili

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
