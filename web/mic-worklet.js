/**
 * AudioWorklet sink for the microphone demo: forwards mono float frames to the main thread.
 * The page stops after 30 s and resamples the accumulated PCM to 16 kHz.
 */
class MothMicProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel && channel.length) this.port.postMessage(channel.slice());
    return true;
  }
}

registerProcessor("moth-mic", MothMicProcessor);
