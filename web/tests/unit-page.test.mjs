import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createContext, SourceTextModule, SyntheticModule } from "node:vm";

const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
const source = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};

async function page() {
  const elements = Object.fromEntries(["status", "out", "file", "record", "stop", "run", "bench"]
    .map(id => [id, { disabled: true, textContent: "", className: "", listeners: {},
      addEventListener(type, callback) { this.listeners[type] = callback; } }]));
  const microphones = [], files = [], revoked = [];
  const state = { transcribe: async () => ({ text: "ok" }), benchmark: async () => ({ runs: 5 }) };
  const runner = { loadMs: 1,
    transcribe: (...args) => state.transcribe(...args),
    benchmark: (...args) => state.benchmark(...args) };
  class FakeURL extends URL {
    static createObjectURL(file) { return file.name; }
    static revokeObjectURL(url) { revoked.push(url); }
  }
  const context = createContext({ window: {}, document: { getElementById: id => elements[id] },
    location: { href: "https://example.test/web/index.html", search: "" }, URL: FakeURL,
    URLSearchParams });
  const module = new SourceTextModule(source, { context });
  await module.link(specifier => {
    const exports = specifier === "./mel.js" ? { computeMel() {} }
      : specifier === "./mic.js" ? { startMicrophone(options) {
        const request = deferred(); microphones.push({ ...request, options }); return request.promise;
      } } : { createRunner: async () => runner, decodeTo16k(url) {
        const request = deferred(); files.push({ ...request, url }); return request.promise;
      } };
    return new SyntheticModule(Object.keys(exports), function () {
      for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
    }, { context });
  });
  await module.evaluate();
  const click = id => elements[id].listeners.click();
  const select = name => elements.file.listeners.change({ target: { files: name ? [{ name }] : [] } });
  return { elements, microphones, files, revoked, state, click, select, moth: context.window.moth };
}

test("Stop cancels a pending microphone; late grants and rejections cannot replace newer state", async () => {
  for (const rejects of [false, true]) {
    const ui = await page();
    const pending = ui.click("record");
    assert.equal(ui.elements.stop.disabled, false);
    assert.equal(ui.elements.file.disabled, true);
    ui.click("stop");
    assert.equal(ui.elements.record.disabled, false);
    const newer = ui.click("record");
    let tracksStopped = 0, stopped = 0;
    if (rejects) ui.microphones[0].reject(new Error("late permission denial"));
    else ui.microphones[0].resolve({
      stream: { getTracks: () => [{ stop() { tracksStopped++; } }] },
      async stop() { stopped++; throw new Error("late cleanup failure"); },
    });
    await pending;
    assert.equal(tracksStopped, rejects ? 0 : 1);
    assert.equal(stopped, rejects ? 0 : 1);
    assert.equal(ui.moth.getPcm(), null);
    assert.equal(ui.elements.stop.disabled, false);
    assert.equal(ui.elements.record.disabled, true);
    assert.doesNotMatch(ui.elements.status.textContent, /failure|denial/);
    ui.microphones[1].resolve({ async stop() { return new Float32Array(1600); } });
    await newer;
    await ui.elements.stop.listeners.click();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(ui.elements.run.disabled, false);
  }
});

test("file changes invalidate old PCM and only the latest completion publishes state", async () => {
  const ui = await page();
  const initial = ui.select("initial");
  const oldPcm = new Float32Array(1600);
  ui.files[0].resolve(oldPcm);
  await initial;
  const first = ui.select("first"), second = ui.select("second");
  assert.equal(ui.moth.getPcm(), null);
  assert.equal(ui.elements.run.disabled, true);
  assert.equal(ui.elements.record.disabled, true);
  const latest = new Float32Array(3200);
  ui.files[2].resolve(latest);
  await second;
  ui.files[1].reject(new Error("stale error"));
  await first;
  assert.equal(ui.moth.getPcm(), latest);
  assert.match(ui.elements.status.textContent, /second/);
  const oldSelection = ui.select("older"), newSelection = ui.select("newer");
  ui.files[4].resolve(latest);
  await newSelection;
  ui.files[3].resolve(oldPcm);
  await oldSelection;
  assert.equal(ui.moth.getPcm(), latest);
  assert.match(ui.elements.status.textContent, /newer/);
  const failed = ui.select("broken");
  ui.files[5].reject(new Error("bad audio"));
  await failed;
  assert.equal(ui.moth.getPcm(), null);
  assert.equal(ui.elements.run.disabled, true);
  const stale = ui.select("stale"), cleared = ui.select(null);
  await cleared;
  ui.files[6].resolve(oldPcm);
  await stale;
  assert.equal(ui.moth.getPcm(), null);
  assert.deepEqual(ui.revoked.sort(), ["initial", "first", "second", "older", "newer", "broken", "stale"].sort());
});

test("microphone stop keeps competing actions disabled until resampling settles and recovers on error", async () => {
  const ui = await page();
  const pending = ui.click("record");
  const resampling = deferred();
  ui.microphones[0].resolve({ stop: () => resampling.promise });
  await pending;
  ui.click("stop");
  for (const id of ["file", "record", "stop", "run", "bench"]) assert.equal(ui.elements[id].disabled, true);
  await ui.select("ignored");
  assert.equal(ui.files.length, 0);
  resampling.reject(new Error("resample failed"));
  await new Promise(resolve => setImmediate(resolve));
  assert.match(ui.elements.status.textContent, /resample failed/);
  assert.equal(ui.elements.record.disabled, false);
  assert.equal(ui.elements.file.disabled, false);
  assert.equal(ui.moth.getPcm(), null);
});

test("transcription and benchmark rejections are handled, competing actions disabled, and retry works", async () => {
  for (const id of ["run", "bench"]) {
    const ui = await page();
    const loaded = ui.select("clip");
    ui.files[0].resolve(new Float32Array(1600));
    await loaded;
    const work = deferred();
    let calls = 0;
    ui.state[id === "run" ? "transcribe" : "benchmark"] = () => { calls++; return work.promise; };
    const pending = ui.click(id);
    for (const other of ["run", "bench", "record", "file", "stop"]) {
      assert.equal(ui.elements[other].disabled, true, other);
    }
    await ui.click("run");
    await ui.click("bench");
    await ui.click("record");
    await ui.select("ignored");
    assert.equal(calls, 1);
    assert.equal(ui.files.length, 1);
    work.reject(new Error("injected failure"));
    await pending;
    assert.match(ui.elements.status.textContent, /processing failed.*injected failure/);
    for (const other of ["run", "bench", "record", "file"]) assert.equal(ui.elements[other].disabled, false);
    ui.state[id === "run" ? "transcribe" : "benchmark"] = async () => ({ text: "recovered" });
    await ui.click(id);
    assert.match(ui.elements.out.textContent, /recovered/);
  }
});
