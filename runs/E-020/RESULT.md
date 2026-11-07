# E-020 — the WER cost of the shipped int8 model

**Question.** The release ships `model.int8.onnx`. Until now nobody had measured what the
quantization costs in WER on our own sets: `runs/E-017/release-0.5.0.json` carries only top-1
logit agreement (99.72%) and 44/100 identical transcripts on 100 FLEURS-IT clips, and neither is
a WER. The one WER measurement that existed was E-016's (`identical on 100 clips`), and the int8
gap has since grown with the E-017 weights.

**Method.** Decode the shipped artifact itself in Python. `stt/onnx_runner.py` (new) wraps the
ONNX graph; the CTC decoders, the manifests, the LM and the scoring are the existing ones, so the
only variable is the quantized acoustic model. PyTorch references are the E-017 hypotheses
re-scored on the same manifest with `stt.eval` (the manifest sha256 matches:
`43f895bf…`).

## Result

| Set | Decoder | PyTorch WER | int8 WER | ΔWER | 95% CI | Verdict |
|---|---|---|---|---|---|---|
| **T-600** | greedy | 37.70 | **38.13** | **+0.43** | [−0.17, +1.01] | tie |
| **T-600** | beam+LM | 32.77 | **32.94** | **+0.17** | [−0.40, +0.77] | tie |
| **FLEURS-IT** | beam+LM | 26.52 | **26.85** | **+0.32** | [+0.08, +0.57] | **slightly worse** |

ETA is unaffected on T (greedy −0.10, beam+LM −0.38, both ties).

Transcript identity: **423/600 (70.5%)** on T-600 greedy, **464/600 (77.3%)** on T-600 beam+LM,
**368/697 (52.8%)** on FLEURS-IT.

## Reading

On our own test set the shipped int8 costs **0.17–0.43 WER points and nothing statistically** —
the release is honest. On FLEURS-IT the loss is small but the CI excludes zero (+0.32
[+0.08, +0.57]): on longer, out-of-domain audio the quantization does cost something real. That is
a fair caveat for the model card, not a reason to change the artifact.

The length split explains the direction: T clips are 1.3–5.6 s (mean 239 mel frames), FLEURS clips
are 7–23 s (mean 1458). Per-frame quantization error accumulates over more frames before the
decoder commits to a token, and beam search then amplifies small logit differences into different
hypotheses (only 52.8% of FLEURS transcripts survive unchanged, against 77.3% on T).

## What this closes

`STATUS.md` said *"Il costo WER dell'int8 non è stato rimisurato end-to-end su T (da fare)"*. It is
done, on 600 clips, with the same rule every other number in the repo uses. The int8 release stands.

## Reproduce

```bash
uv run python runs/E-020/eval_onnx.py --model release/model.int8.onnx --vocab release/vocab.json \
    --sets recorded_t --out runs/E-020/int8-greedy
uv run python runs/E-020/eval_onnx.py --model release/model.int8.onnx --vocab release/vocab.json \
    --sets recorded_t --out runs/E-020/int8-beam-lm --decoder beam \
    --lm runs/E-008/lm_bpe3_aug.pkl.gz --lm-weight 0.2
uv run python -m stt.eval --manifest recordings/manifests/T.jsonl \
    --hyp runs/E-020/int8-beam-lm/recorded_t.hyp.jsonl --out ... --vs ...
```

The PyTorch reference metrics are re-scored from the E-017 hypothesis files into
`runs/E-020/torch-ref/`; re-scoring reproduces 37.70 / 32.77 exactly.