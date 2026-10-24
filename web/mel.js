/**
 * JS port of the Whisper log-mel front-end (docs/02 §2, docs/06 §2).
 *
 * Direct 400-point real DFT through precomputed 201×400 cos/sin matrices — deliberately **not**
 * a zero-padded 512-point FFT: that would move the frequency grid. Constants are bit-exact with
 * `stt/features.py`: periodic Hann, center reflect pad of 200, last STFT frame dropped
 * (`frames = N // 160`), power spectrum, Whisper filterbank, `log10(max(mel, 1e-10))`, per-clip
 * `max(x, x.max() − 8)`, `(x + 4) / 4`.
 */

export const SAMPLE_RATE = 16000;
export const N_FFT = 400;
export const HOP_LENGTH = 160;
export const N_MELS = 80;
export const FLOOR_RANGE = 8.0;

const PAD = N_FFT / 2;
const BINS = N_FFT / 2 + 1;

const HANN = new Float32Array(N_FFT);
for (let n = 0; n < N_FFT; n += 1) {
  HANN[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
}

let tables = null;

/** Lazy `hann * cos` / `hann * sin` tables, `[BINS][N_FFT]`, built once per page. */
function dftTables() {
  if (tables) return tables;
  const cos = new Float32Array(BINS * N_FFT);
  const sin = new Float32Array(BINS * N_FFT);
  for (let k = 0; k < BINS; k += 1) {
    const offset = k * N_FFT;
    for (let n = 0; n < N_FFT; n += 1) {
      const angle = (2 * Math.PI * k * n) / N_FFT;
      cos[offset + n] = HANN[n] * Math.cos(angle);
      sin[offset + n] = HANN[n] * Math.sin(angle);
    }
  }
  tables = { cos, sin };
  return tables;
}

function reflectPad(pcm) {
  const length = pcm.length;
  const padded = new Float32Array(length + 2 * PAD);
  for (let i = 0; i < PAD; i += 1) {
    padded[i] = pcm[PAD - i];
    padded[PAD + length + i] = pcm[length - 2 - i];
  }
  padded.set(pcm, PAD);
  return padded;
}

/**
 * Log-mel for 16 kHz mono PCM: `{data: Float32Array, frames, bins}` with `data` row-major
 * `[80, frames]`, the exact layout of the ONNX input `[1, 80, frames]`.
 */
export function computeMel(pcm, filters) {
  if (pcm.length < N_FFT + HOP_LENGTH) {
    throw new Error(`clip too short: ${pcm.length} samples < ${N_FFT + HOP_LENGTH}`);
  }
  const { cos, sin } = dftTables();
  const frames = Math.floor(pcm.length / HOP_LENGTH);
  const padded = reflectPad(pcm);
  const power = new Float32Array(BINS);
  const frame = new Float32Array(N_FFT);
  const log = new Float32Array(N_MELS * frames);
  let maxLog = -Infinity;

  for (let t = 0; t < frames; t += 1) {
    const start = t * HOP_LENGTH;
    for (let n = 0; n < N_FFT; n += 1) frame[n] = padded[start + n];
    for (let k = 0; k < BINS; k += 1) {
      let real = 0;
      let imaginary = 0;
      const offset = k * N_FFT;
      for (let n = 0; n < N_FFT; n += 1) {
        const sample = frame[n];
        real += cos[offset + n] * sample;
        imaginary += sin[offset + n] * sample;
      }
      power[k] = real * real + imaginary * imaginary;
    }
    for (let m = 0; m < N_MELS; m += 1) {
      let sum = 0;
      const offset = m * BINS;
      for (let k = 0; k < BINS; k += 1) sum += filters[offset + k] * power[k];
      const value = Math.log10(Math.max(sum, 1e-10));
      log[m * frames + t] = value;
      if (value > maxLog) maxLog = value;
    }
  }

  const floor = maxLog - FLOOR_RANGE;
  const data = new Float32Array(log.length);
  for (let i = 0; i < log.length; i += 1) {
    data[i] = (Math.max(log[i], floor) + 4) / 4;
  }
  return { data, frames, bins: N_MELS };
}

/**
 * Loudest normalized log-mel bin below which a clip carries no speech-like energy.
 * Mirrors `SILENCE_MAX` in `stt/no_speech.py`; see that module for the calibration.
 */
export const SILENCE_MAX = -1.0;

/** True when the mel from `computeMel` has no speech-like energy and should not be decoded. */
export function isNoSpeech(data) {
  let loudest = -Infinity;
  for (let i = 0; i < data.length; i += 1) {
    if (data[i] > loudest) loudest = data[i];
  }
  return loudest < SILENCE_MAX;
}
