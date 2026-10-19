# moth-stt

**Moth** — a very small, open **Italian + English code-switching speech-to-text** model.
Whisper-tiny encoder (pretrained, MIT) + a new CTC head over a shared IT+EN BPE vocabulary:
≈ 8 M parameters, ≈ 8–10 MB at int8, built to run in a browser tab (ONNX Runtime Web / WASM) and on any CPU.

> **Status:** design phase — nothing is implemented yet. The plan lives in [`docs/`](docs/README.md); read `01` … `09` in order.

- Sibling of [ANT-4B](https://github.com/semantique-team/ant-4b) in the same family of small open models.
- Core job: Italian speech with English words mixed in ("ho spostato il *meeting* a domani").
- Code Apache-2.0 ([`LICENSE`](LICENSE)); the encoder derives from [Whisper](https://github.com/openai/whisper) (MIT) and the `NOTICE` will travel with the released weights.
