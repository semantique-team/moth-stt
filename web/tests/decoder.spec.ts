import { readFileSync } from "node:fs";

import { expect, test } from "@playwright/test";

const LM_SCORES = JSON.parse(readFileSync("tests/fixtures/lm_scores.json", "utf8"));

test("JS beam preserves blank-separated repeated labels", async ({ page }) => {
  await page.goto("/web/index.html");
  const text = await page.evaluate(async () => {
    const { beamSearch } = await import("/web/decode.js");
    // frames: [blank, "l"], [blank], [blank, "l"]; the two labels must stay distinct.
    const probs = new Float32Array([-10, 0, 0, -10, -10, 0]);
    return beamSearch(probs, 3, 2, ["", "l"], { beamWidth: 3 });
  });
  expect(text).toBe("ll");
});

test("JS LM parser matches the Python scores", async ({ page }) => {
  await page.goto("/web/index.html");
  const results = await page.evaluate(async (cases) => {
    const { loadLM } = await import("/web/lm.js");
    const buffer = await (await fetch("/release/lm_bpe3.bin")).arrayBuffer();
    const lm = loadLM(buffer);
    return cases.map((entry: any) => ({
      ...entry,
      js: lm.score(entry.prev2, entry.prev1, entry.token),
    }));
  }, LM_SCORES);

  for (const entry of results) {
    expect(Math.abs(entry.js - entry.score)).toBeLessThan(0.02);
  }
});
