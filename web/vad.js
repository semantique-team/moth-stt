/**
 * Silero VAD v5 as the no-speech guard (D25, E-011 addendum), browser path.
 *
 * Direct port of `stt/vad.py`: 16 kHz mono PCM, 32 ms windows, **512 new samples + 64 samples
 * of context** per call (the v5 ONNX graph was trained on the concatenated 576-sample window;
 * passing 512 returns ~0.001 even on speech), `state [2, 1, 128]` float32, `sr` int64 scalar.
 * A window is speech when P(speech) > 0.5; the clip is decoded when the fraction of speech
 * windows is > 0.10. Keep the constants in sync with the Python module
 * (`tests/test_vad.py` reads them).
 */

import * as ort from "./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs";

export const SAMPLE_RATE = 16000;
export const WINDOW = 512;
export const CONTEXT = 64;
export const WINDOW_THRESHOLD = 0.5;
export const GATE_THRESHOLD = 0.10;

const STATE_SIZE = 2 * 1 * 128;

/**
 * Create the VAD session and return `{speechFraction, hasSpeech}` over 16 kHz mono PCM.
 * `gate` is the fraction threshold; speech iff `fraction > gate`.
 */
export async function createVad(modelBuffer, { gate = GATE_THRESHOLD } = {}) {
  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const sampleRate = new ort.Tensor(
    "int64",
    BigInt64Array.from([BigInt(SAMPLE_RATE)]),
    [],
  );
  const context = new Float32Array(CONTEXT);
  const input = new Float32Array(WINDOW + CONTEXT);

  async function speechFraction(pcm) {
    input.fill(0);
    context.fill(0);
    let state = new ort.Tensor("float32", new Float32Array(STATE_SIZE), [2, 1, 128]);
    let speech = 0;
    let windows = 0;
    for (let start = 0; start < pcm.length; start += WINDOW) {
      input.set(context, 0);
      const end = Math.min(start + WINDOW, pcm.length);
      input.set(pcm.subarray(start, end), CONTEXT); // a short tail stays zero-padded
      input.fill(0, CONTEXT + (end - start), WINDOW + CONTEXT);
      const outputs = await session.run({
        input: new ort.Tensor("float32", input, [1, WINDOW + CONTEXT]),
        state,
        sr: sampleRate,
      });
      state = outputs.stateN;
      if (outputs.output.data[0] > WINDOW_THRESHOLD) speech += 1;
      windows += 1;
      context.set(input.subarray(WINDOW, WINDOW + CONTEXT));
    }
    return windows === 0 ? 0 : speech / windows;
  }

  async function hasSpeech(pcm) {
    return (await speechFraction(pcm)) > gate;
  }

  return { session, speechFraction, hasSpeech, gate };
}
