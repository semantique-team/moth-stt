# E-028 — INT8 quantization sweep

The E-028 fp32 checkpoint passes the pre-registered gates, but the shipped artifact is INT8.
The full package must stay at or below 20,000,000 bytes; fp32 is 46.37 MB for the graph alone
(~53.9 MB packaged), fp16 is ~30.6 MB packaged, so INT8 is the only viable format.

## Dynamic quantization variants (selected on dev only)

Dev = CV-IT dev (600) + FLEURS-IT dev (334) + FLEURS-EN dev (343). T was not read for selection.
Package = model + the unchanged 7,530,083 bytes of LM/VAD/vocab/mel/config/NOTICE.

| Variant | Dev WER | CV-IT | FLEURS-IT | FLEURS-EN | Model MB | Package MB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 (reference, not deployable) | **30.93** | 19.78 | 20.54 | 51.17 | 46.37 | 53.90 |
| **INT8 default** | **31.14** | 20.15 | 20.69 | 51.30 | 11.91 | **19.44** |
| INT8 head fp32 | 31.06 | 19.92 | 20.61 | 51.35 | 13.09 | 20.62 ✗ |
| INT8 per-tensor | 31.26 | 20.30 | 20.23 | 52.00 | 11.80 | 19.33 |
| INT8 QUInt8 weights | 31.30 | 20.35 | 20.62 | 51.67 | 11.92 | 19.45 |
| INT8 reduce_range | 31.56 | 20.44 | 20.83 | 52.13 | 11.91 | 19.44 |
| Static QDQ (CV-IT calibration) | **91.44 ✗** | 82.94 | 91.33 | 98.82 | 11.99 | 19.52 |

The default dynamic quantization is the best variant that fits 20 MB. `per_tensor`,
`reduce_range` and `QUInt8` are worse. The head-fp32 variant is only 0.08 dev points better
than the default (and still 0.13 worse than fp32) while exceeding the budget by 0.62 MB; to fit
it we would have to prune the LM and the quality gain is too small to justify that coupled
experiment. Static QDQ is broken on this graph (90%+ WER), not a tradeoff.

## Consequence for the release

The E-028 fp32 checkpoint is a strong candidate, but the INT8 artifact is a tradeoff against
0.7.0 INT8:

| Set | 0.7.0 INT8 | E-028 INT8 |
|---|---:|---:|
| T-600 WER / ETA | 25.14 / **54.65** | 25.49 / **53.50** |
| T-S3 WER / ETA | 27.00 / 48.45 | **25.89 / 49.01** |
| Rare presence | 8/47 | **12/47** |
| Natural-term presence | 441/817 | 428/817 |

The rare gain is significant (+8.5 points presence), but the INT8 artifact gives back 13 natural
terms and 1.15 ETA points on T-600 relative to 0.7.0 INT8 (the T-600 differences are ties by CI).
**0.8.0 is not published.** The remaining serious route is quantization-aware fine-tuning
(fake-quantization in the forward pass / QAT), which is a new training experiment; the cheap
post-training variants are exhausted.

## Artifacts

- `quant_sweep.py`, `sweep.json` — dynamic variants; `.onnx` files are gitignored.
- `quant_static.py`, `static.json` — static QDQ attempt, rejected.
- `../release-0.8.0.json` — hashes, package bytes, fp32/int8 metrics, browser acceptance.
