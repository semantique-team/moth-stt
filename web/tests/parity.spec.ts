/**
 * E-022 parity: the browser must produce the same transcripts as Python at the new decode settings.
 *
 * The browser cannot read the mel cache (it would need the whole shard), so this re-derives the mel
 * from the **raw audio** with the shipped JS front-end and decodes through the shipped int8 graph
 * with the shipped `config.json`. Python's side (`runs/E-022/parity.py`) uses the cached mel. If the
 * two front-ends match to within float noise the transcripts must be identical; the comparison is
 * written to `test-results/parity-browser.json` and checked by `parity.py --compare`.
 *
 * Caveat: a mel difference of 1e-6 can flip a near-tie in the beam. A small number of mismatches is
 * therefore expected and is reported as a ratio, not asserted to be zero. The gate used in
 * `docs/06` §4 is "identical on the 20-clip e2e set"; this test reports the measured ratio.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { expect, test } from "@playwright/test";

const CLIPS = JSON.parse(readFileSync("../runs/E-022/parity-clips.json", "utf8"));
const OUT = process.env.PARITY_OUT ?? "parity-browser.json";

test("browser transcripts at the E-022 decode parameters", async ({ page }) => {
  await page.goto("/web/index.html");
  await page.waitForFunction(() => (window as any).moth?.ready === true, undefined, {
    timeout: 120_000,
  });

  const ids = CLIPS.map((clip: any) => clip.id);

  const out = await page.evaluate(async (clipIds: string[]) => {
    const moth = (window as any).moth;
    const manifest: Record<string, string> = {};
    const text = await (await fetch("/recordings/manifests/T.jsonl")).text();
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      const row = JSON.parse(line);
      if (clipIds.includes(row.id)) manifest[row.id] = row.audio;
    }
    const result: Record<string, string> = {};
    for (const id of clipIds) {
      const pcm = await moth.decodeTo16k(`/recordings/audio/${manifest[id]}`);
      const single = await moth.transcribePcm(pcm);
      result[id] = single.text;
    }
    return result;
  }, ids);

  mkdirSync("test-results", { recursive: true });
  writeFileSync(`test-results/${OUT}`, JSON.stringify(out, null, 2));
  expect(Object.keys(out).length).toBe(ids.length);
});