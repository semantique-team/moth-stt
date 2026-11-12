import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createContext, SourceTextModule } from "node:vm";

const source = await readFile(new URL("../mic.js", import.meta.url), "utf8");

async function microphone({ sampleRate = 16000, fail = null } = {}) {
  const calls = { tracks: 0, close: 0, disconnect: 0, port: 0, permission: 0 };
  const stream = { getTracks: () => [{ stop() { calls.tracks += 1; } }] };
  let node;
  const graphNode = () => ({
    connect() {},
    disconnect() { calls.disconnect += 1; },
  });
  const context = createContext({
    Float32Array, URL,
    navigator: { mediaDevices: { async getUserMedia() {
      calls.permission += 1;
      return stream;
    } } },
    AudioContext: class {
      constructor() {
        if (fail === "constructor") throw new Error(fail);
        this.sampleRate = sampleRate;
        this.audioWorklet = { async addModule() {
          if (fail === "worklet") throw new Error(fail);
        } };
      }
      createMediaStreamSource() { return graphNode(); }
      createGain() { return { ...graphNode(), gain: {} }; }
      async close() {
        calls.close += 1;
        if (fail === "close") throw new Error(fail);
      }
    },
    AudioWorkletNode: class {
      constructor() {
        Object.assign(this, graphNode());
        this.port = { close() { calls.port += 1; } };
        node = this;
      }
    },
    OfflineAudioContext: class {
      constructor(channels, length) { this.length = length; }
      createBuffer() { return { copyToChannel() {} }; }
      createBufferSource() { return { connect() {}, start() {} }; }
      async startRendering() {
        return { getChannelData: () => new Float32Array(this.length).fill(0.25) };
      }
    },
  });
  const module = new SourceTextModule(source, {
    context,
    initializeImportMeta(meta) { meta.url = new URL("../mic.js", import.meta.url).href; },
  });
  await module.link(() => { throw new Error("unexpected import"); });
  await module.evaluate();
  return {
    start: module.namespace.startMicrophone,
    send: (samples) => node.port.onmessage({ data: samples }),
    calls,
  };
}

test("microphone stops internally at the cap and clamps oversized chunks", async () => {
  for (const sampleRate of [16000, 44100, 48000]) {
    for (const maxSeconds of [0.10004, 30, 60]) {
      const { start, send, calls } = await microphone({ sampleRate });
      const recording = await start({ maxSeconds });
      const limit = Math.min(maxSeconds, 30);
      send(new Float32Array(Math.floor(limit * sampleRate) - 1).fill(0.25));
      send(new Float32Array(128).fill(0.5));
      send(new Float32Array(128).fill(1));
      await Promise.resolve();
      assert.equal(calls.tracks, 1);
      const pcm = await recording.stop();
      assert.equal(pcm.length, Math.floor(limit * 16000));
      if (sampleRate === 16000) assert.equal(pcm.at(-1), 0.5);
      assert.equal(await recording.stop(), pcm);
      assert.equal(calls.close, 1);
      assert.equal(calls.disconnect, 3);
      assert.equal(calls.port, 1);
    }
  }
});

test("limit callback can stop reentrantly and fires once", async () => {
  const { start, send, calls } = await microphone();
  let notifications = 0;
  let callbackStop;
  const recording = await start({ maxSeconds: 0.1, onLimit() {
    notifications += 1;
    callbackStop = recording.stop();
  } });
  send(new Float32Array(1601));
  send(new Float32Array(1601));
  assert.equal(notifications, 1);
  assert.equal(callbackStop, recording.stop());
  assert.equal((await callbackStop).length, 1600);
  assert.equal(calls.tracks, 1);
});

test("manual stop is idempotent and ignores late messages", async () => {
  for (const length of [0, 1000]) {
    const { start, send, calls } = await microphone();
    const recording = await start();
    send(new Float32Array(length).fill(0.25));
    const first = recording.stop();
    assert.equal(recording.stop(), first);
    send(new Float32Array(1000));
    assert.equal((await first).length, length);
    assert.equal(calls.tracks, 1);
    assert.equal(calls.close, 1);
  }
});

test("throwing limit callback cannot prevent internal cleanup", async () => {
  const { start, send, calls } = await microphone();
  const recording = await start({ maxSeconds: 0.1, onLimit() { throw new Error("callback"); } });
  assert.throws(() => send(new Float32Array(1601)), /callback/);
  assert.equal((await recording.stop()).length, 1600);
  assert.equal(calls.tracks, 1);
  assert.equal(calls.close, 1);
});

test("invalid limits reject before requesting microphone permission", async () => {
  const { start, calls } = await microphone();
  for (const maxSeconds of [0, -1, NaN, Infinity, "30", 0.00001]) {
    await assert.rejects(start({ maxSeconds }), /maxSeconds/);
  }
  assert.equal(calls.permission, 0);
});

test("setup failures release microphone tracks and any created context", async () => {
  for (const fail of ["constructor", "worklet"]) {
    const { start, calls } = await microphone({ fail });
    await assert.rejects(start(), new RegExp(fail));
    assert.equal(calls.tracks, 1);
    assert.equal(calls.close, fail === "constructor" ? 0 : 1);
  }
});

test("automatic stop retains cleanup failure without an unhandled rejection", async () => {
  const { start, send, calls } = await microphone({ fail: "close" });
  const recording = await start({ maxSeconds: 0.1 });
  send(new Float32Array(1601));
  await assert.rejects(recording.stop(), /close/);
  await assert.rejects(recording.stop(), /close/);
  assert.equal(calls.tracks, 1);
  assert.equal(calls.close, 1);
});
