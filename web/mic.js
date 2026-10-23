/**
 * Record-then-transcribe microphone capture (docs/06 §2): getUserMedia → AudioWorklet PCM →
 * OfflineAudioContext resample to 16 kHz mono. No streaming, no VAD; the caller stops or the
 * 30 s safety limit fires.
 */

const TARGET_RATE = 16000;

async function resampleTo16k(samples, sampleRate) {
  if (samples.length === 0) return new Float32Array(0);
  if (sampleRate === TARGET_RATE) return samples;
  const length = Math.max(1, Math.round((samples.length * TARGET_RATE) / sampleRate));
  const offline = new OfflineAudioContext(1, length, TARGET_RATE);
  const buffer = offline.createBuffer(1, samples.length, sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

/**
 * @param {{maxSeconds?: number, onLimit?: (() => void) | null}} options
 * @returns {Promise<{stop: () => Promise<Float32Array>, stream: MediaStream, sampleRate: number}>}
 */
export async function startMicrophone({ maxSeconds = 30, onLimit = null } = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("getUserMedia non disponibile in questo browser");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });

  const context = new AudioContext();
  const chunks = [];
  let total = 0;
  let finished = false;
  let limitFired = false;
  let stopPromise = null;

  try {
    await context.audioWorklet.addModule(new URL("./mic-worklet.js", import.meta.url));
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, "moth-mic", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      channelCountMode: "explicit",
    });
    // Keep the graph alive without sending the microphone to the speakers.
    const mute = context.createGain();
    mute.gain.value = 0;
    source.connect(node);
    node.connect(mute);
    mute.connect(context.destination);

    node.port.onmessage = ({ data }) => {
      if (finished) return;
      chunks.push(data);
      total += data.length;
      if (
        maxSeconds > 0 &&
        !limitFired &&
        total >= maxSeconds * context.sampleRate
      ) {
        limitFired = true;
        onLimit?.();
      }
    };

    async function stop() {
      if (stopPromise) return stopPromise;
      finished = true;
      stopPromise = (async () => {
        try {
          source.disconnect();
          node.disconnect();
          mute.disconnect();
          node.port.close();
        } finally {
          for (const track of stream.getTracks()) track.stop();
        }
        const sampleRate = context.sampleRate;
        await context.close();
        if (total === 0) return new Float32Array(0);
        const merged = new Float32Array(total);
        let offset = 0;
        for (const chunk of chunks) {
          merged.set(chunk, offset);
          offset += chunk.length;
        }
        return resampleTo16k(merged, sampleRate);
      })();
      return stopPromise;
    }

    return { stop, stream, sampleRate: context.sampleRate };
  } catch (error) {
    finished = true;
    for (const track of stream.getTracks()) track.stop();
    await context.close();
    throw error;
  }
}
