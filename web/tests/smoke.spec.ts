import { mkdirSync, writeFileSync } from "node:fs";

import { expect, test } from "@playwright/test";

/**
 * H4 early (Task 3.5): a random-weight int8 release must run through the browser path
 * (WAV → 16 kHz → JS log-mel → onnxruntime-web WASM → greedy decode) and produce RTFx numbers.
 * Transcript correctness is meaningless with random weights; the smoke proves the graph runs,
 * the release hashes verify and the speed bar can be measured before training.
 */
test("random-weight int8 export runs in the browser and measures RTFx", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  await page.goto("/web/index.html");
  await page.waitForFunction(() => (window as any).moth?.ready === true, undefined, {
    timeout: 120_000,
  });

  const result = await page.evaluate(async () => {
    const moth = (window as any).moth;
    const pcm = await moth.decodeTo16k("/web/tests/fixtures/clip.wav");
    const single = await moth.transcribePcm(pcm);
    const bench = await moth.benchmark(pcm, { runs: 5, warmups: 2 });
    return { single, bench, load_ms: moth.runner.loadMs };
  });

  expect(typeof result.single.text).toBe("string");
  expect(result.single.audioSeconds).toBeGreaterThan(1);
  expect(result.single.outFrames).toBeGreaterThan(0);
  expect(result.bench.rtfx_median).toBeGreaterThan(0);
  expect(result.bench.cross_origin_isolated).toBe(true);
  expect(result.bench.simd).toBe(true);
  expect(errors).toEqual([]);

  mkdirSync("test-results", { recursive: true });
  writeFileSync("test-results/rtfx-smoke.json", JSON.stringify(result, null, 2));
  console.log(`RTFx smoke: ${JSON.stringify(result.bench)}`);
  console.log(`transcript: ${JSON.stringify(result.single.text)}`);
});

/**
 * H4 early speed table, the `docs/05` §3 shape at smoke scale: 5/10/20 s (the fixture tiled) on
 * 1 and 4 threads, 3 warm-ups + 10 runs each. Chrome/Safari cross-check and real recordings are
 * the M7 benchmark.
 */
test("RTFx table on 5/10/20 s clips, 1 and 4 threads", async ({ page, context }) => {
  const collect = async (threads: number) => {
    const runnerPage = threads === 4 ? page : await context.newPage();
    await runnerPage.goto(threads === 4 ? "/web/index.html" : "/web/index.html?threads=1");
    await runnerPage.waitForFunction(() => (window as any).moth?.ready === true, undefined, {
      timeout: 120_000,
    });
    const rows = await runnerPage.evaluate(async (threadCount: number) => {
      const moth = (window as any).moth;
      const base = await moth.decodeTo16k("/web/tests/fixtures/clip.wav");
      const tile = (seconds: number) => {
        const target = Math.round(seconds * 16000);
        const pcm = new Float32Array(target);
        for (let index = 0; index < target; index += 1) pcm[index] = base[index % base.length];
        return pcm;
      };
      const table = [];
      for (const seconds of [5, 10, 20]) {
        const bench = await moth.benchmark(tile(seconds), { runs: 10, warmups: 3 });
        table.push({ threads: threadCount, seconds, ...bench });
      }
      return table;
    }, threads);
    if (runnerPage !== page) await runnerPage.close();
    return rows;
  };

  const rows = [...(await collect(4)), ...(await collect(1))];

  for (const row of rows) {
    console.log(
      `${row.threads}T ${row.seconds}s: total ${row.total_ms_median.toFixed(1)} ms ` +
        `(feature ${row.feature_ms_median.toFixed(1)} + inference ${row.inference_ms_median.toFixed(1)} ` +
        `+ decode ${row.decode_ms_median.toFixed(2)}) → RTFx ${row.rtfx_median.toFixed(1)}×`,
    );
    expect(row.rtfx_median).toBeGreaterThan(5);
  }
  for (const row of rows.filter((row: any) => row.threads === 4)) {
    expect(row.rtfx_median).toBeGreaterThan(10); // the docs/04 H4 bar
  }

  mkdirSync("test-results", { recursive: true });
  writeFileSync("test-results/rtfx-table.json", JSON.stringify(rows, null, 2));
});
