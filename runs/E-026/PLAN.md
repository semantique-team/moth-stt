# E-026 — diagnostic: where do the rare terms go?

**Status:** executed 2026-11-11 as an exploratory diagnostic, then formalized here. This is not a
training run and it does not select or adopt a model. The protocol below is retrospective: the
run was done first to orient the next experiment, and the numbers are reported with that caveat.

## Question

On the published 0.7.0 int8 release, for the 47 T utterances that contain at least one of the 44
terms never seen in training:

1. Is the reference reachable acoustically, or is it already removed by the decoder's frame-level
   top-k?
2. How much does the LM add over greedy and over beam without LM?
3. How different are the E-015 *presence* readout and the canonical **aligned ETA** on the same
   rows?

## Data and protocol

- **Rows:** T rows with at least one English term absent from the pre-E-015 acoustic/tokenizer
  text. The gold vocabulary excludes `R-S1-0642` onward and `R-S2-0624` onward (the E-015/E-028
  addenda themselves), so the groups describe what 0.7.0 had actually seen. This is the same
  selection as `runs/E-015/eval_terms.py`.
- **Audio:** each T file is read with `stt.data.build_mel.load_audio_16k` (mono, 16 kHz, FFT
  resampling when needed). Clips are decoded whole; no VAD and no segmentation. This is deliberate:
  the diagnostic measures the acoustic/decoder path, not the production VAD.
- **System:** `release/` 0.7.0, `model.int8.onnx`, release `vocab.json`, `lm_bpe3.bin`.
- **Decoders:** greedy; beam acoustic (no LM, no token bonus); beam+LM at the release settings
  (beam 12, top-k 10, LM weight 0.5, token bonus 2.4).
- **Reference support:** exact CTC forward probability of the reference and the same probability
  under the top-k frame mask; `stt.diagnose` reports full reachability, top-k reachability and the
  retained probability fraction.
- **Metrics:** WER and aligned ETA via `stt.eval.score_rows`; term presence via
  `runs/E-015/eval_terms.py`; insertion rate.

## Gates and claims

- Diagnostic only: **no adoption, no checkpoint selection, no tuning**.
- T has been inspected repeatedly and is not an untouched test. Results are descriptive.
- The 47-row sample is small; no confidence interval is claimed for the new diagnostic splits.
- Forced-reference likelihood is an oracle diagnostic, not deployable accuracy.

## Artifacts

- `reproduce.py` — rebuilds the exact 47-row 16 kHz manifest and audio in a temp directory.
- `rare47-16k.jsonl` — the manifest used (SHA-256 `e8b4b05…c7b3`).
- `report.json`, `hyp/*.jsonl` — diagnostic report and hypotheses.
- `RESULT.md` — findings and the implication for E-015/E-027.
