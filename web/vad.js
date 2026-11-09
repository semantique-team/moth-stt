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

/** Silence needed to cut between two sentences when splitting a long recording. */
export const SEGMENT_MIN_SILENCE_S = 0.25;
/** Context kept on each side of a cut: cutting flush removes the leading silence the model expects. */
export const SEGMENT_MARGIN_S = 0.2;
/** Runs shorter than this are dropped rather than decoded on their own. */
export const SEGMENT_MIN_SEGMENT_S = 0.4;

const STATE_SIZE = 2 * 1 * 128;

/**
 * Create the VAD session and return `{speechFraction, hasSpeech, segments}` over 16 kHz mono PCM.
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

  /** Per-window P(speech), in order. Resets state: each call is a fresh clip. */
  async function probabilities(pcm) {
    input.fill(0);
    context.fill(0);
    let state = new ort.Tensor("float32", new Float32Array(STATE_SIZE), [2, 1, 128]);
    const values = [];
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
      values.push(outputs.output.data[0]);
      context.set(input.subarray(WINDOW, WINDOW + CONTEXT));
    }
    return values;
  }

  function fractionFrom(values) {
    if (!values.length) return 0;
    let speech = 0;
    for (const value of values) if (value > WINDOW_THRESHOLD) speech += 1;
    return speech / values.length;
  }

  async function speechFraction(pcm) {
    return fractionFrom(await probabilities(pcm));
  }

  async function hasSpeech(pcm) {
    return (await speechFraction(pcm)) > gate;
  }

  /**
   * Sample offsets `[{start, end}]` of the speech runs, for splitting a long recording.
   *
   * The model is trained on sentence-sized clips and reads several sentences as one unpunctuated
   * stream: five sentences concatenated score 36.6% WER against 26.6% decoded separately, and
   * cutting at the pauses recovers 80% of the difference (runs/audit-2026-11-09/).
   *
   * **Only split when two or more runs come back.** Cutting flush against the first voiced window
   * removes the leading silence the model expects; with a single run the caller must decode the
   * whole clip instead. Mirror of `stt/vad.py::segment_bounds`; keep in sync.
   */
  function segmentsFrom(
    sampleCount,
    values,
    { margin = SEGMENT_MARGIN_S, minSilence = SEGMENT_MIN_SILENCE_S, minSegment = SEGMENT_MIN_SEGMENT_S } = {},
  ) {
    const secondsPerWindow = WINDOW / SAMPLE_RATE;
    const runs = [];
    let index = 0;
    while (index < values.length) {
      if (!(values[index] > WINDOW_THRESHOLD)) {
        index += 1;
        continue;
      }
      const start = index;
      while (index < values.length && values[index] > WINDOW_THRESHOLD) index += 1;
      runs.push([start, index]);
    }
    if (!runs.length) return [];

    const merged = [runs[0]];
    for (const [start, end] of runs.slice(1)) {
      if ((start - merged[merged.length - 1][1]) * secondsPerWindow < minSilence) {
        merged[merged.length - 1][1] = end;
      } else {
        merged.push([start, end]);
      }
    }

    const marginSamples = Math.round(margin * SAMPLE_RATE);
    const bounds = [];
    for (const [start, end] of merged) {
      if ((end - start) * secondsPerWindow < minSegment) continue;
      bounds.push({
        start: Math.max(0, start * WINDOW - marginSamples),
        end: Math.min(sampleCount, end * WINDOW + marginSamples),
      });
    }
    return bounds;
  }

  async function segments(pcm, options = {}) {
    return segmentsFrom(pcm.length, await probabilities(pcm), options);
  }

  return {
    session,
    probabilities,
    fraction: fractionFrom,
    speechFraction,
    hasSpeech,
    segments,
    segmentsFrom,
    gate,
  };
}
