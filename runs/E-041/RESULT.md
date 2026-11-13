# E-041 — RESULT: the browser path costs about 0.7 pp WER, and it is the third voice that pays

> Executed 2026-11-13. Read-only: the shipped 0.8.1 bundle decoded in headless Chromium via ONNX
> Runtime Web. No training, no tuning, no release change.

## Verdict

Through the deployed path — JS mel front end, ORT-Web WASM kernels, JS beam decoder with the A3 LM —
the shipped model measures **24.7993% WER / 54.3624% ETA** on T-600, against **24.0826% / 55.2253%**
from the Python ONNX runner on the same clips:

- WER **+0.7167 pp**, 95% CI **[0.0000, +1.4723]** → a tie, with the interval touching zero;
- ETA **−0.8629 pp**, 95% CI **[−2.4390, +0.6757]** → a tie;
- **151 of 600 hypotheses differ** from the Python ones.

A tie by CI, but not "no difference": the point estimate is consistently positive on WER and the
per-voice breakdown is not uniform.

| Voice | Browser WER / ETA | Python WER / ETA | Δ WER | Δ ETA |
|---|---|---:|---:|---:|
| S1 | 21.2382 / 63.5821 | 20.9802 / 63.2836 | +0.2580 pp | +0.2985 pp |
| S2 | 26.6725 / 51.5581 | 26.3249 / 51.5581 | +0.3476 pp | 0.0000 pp |
| S3 | **26.4906 / 48.4507** | 24.9574 / 51.2676 | **+1.5331 pp** | **−2.8169 pp** |
| T-600 | 24.7993 / 54.3624 | 24.0826 / 55.2253 | +0.7167 pp | −0.8629 pp |

**S3 pays five to six times the WER cost of S1/S2**, and loses 2.8 pp of English-term accuracy. The
deployed path is not a uniform shift: the voice that never trained is the one that notices.

## Runtime

1,390.2 s of audio decoded in **36.2 s wall** (600 clips), i.e. **RTFx 38.4** overall with a
per-clip median of **42.30** (min 25.51, max 53.44). This is a desk measurement: headless Chromium,
WASM threads, 10 hardware threads, no GPU, **not** a phone, Safari or end-user latency claim.

## Why The Numbers Move

The browser does not run the PyTorch mel pipeline: `web/mel.js` implements its own front end and
`web/ort.js` feeds ORT-Web. The INT8 graph, the A3 LM and the decoder parameters are identical
(`7bbfad7e…`, `3be1177b…`, beam 12 / top-k 10 / 0.55 / 2.7, VAD gate 0.1), which is why the gap is
of order 0.7 pp rather than a different regime — but 151 utterances change text, and the changes
concentrate on the unseen voice.

## What This Does And Does Not Establish

- **Does**: give the first acoustic WER/ETA measured **through the artifact users run**, replacing
  "same logits, so presumably the same quality" with a measurement and a per-voice breakdown.
- **Does**: quantify where the deployment penalty lands (S3, and ETA on English-origin terms), which
  is more actionable than a single pooled delta.
- **Does not**: validate mobile browsers, Safari, live microphone capture, VAD in real recording, or
  user-perceived latency. It does not make T untouched or speaker-independent, and it does not
  change the release: the numbers describe the shipped bundle.
- **Does not** justify tuning the JS front end on T. Any such change would be a new experiment with
  its own pre-registration, and would have to be measured on the same clips afterwards.

## Artifacts And Cost

`runs/E-041/` — `PLAN.md`, `browser_wer.mjs`, `score.py`, `readout/browser-1/browser.hyp.jsonl`,
`browser.timings.jsonl`, `browser.run.json`, `metrics.json`, this report. Cost: 36 s of browser
decoding plus seconds of scoring, no training, no download, 0 USD.

Reproduction: `node runs/E-041/browser_wer.mjs --name browser-2`, then
`uv run python runs/E-041/score.py --name browser-2`.
