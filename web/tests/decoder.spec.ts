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

test("JS beam tokenBonus matches the Python decoder", async ({ page }) => {
  await page.goto("/web/index.html");
  // The twin case from tests/test_decode.py::test_beam_token_bonus_can_lengthen_the_hypothesis.
  // Python says: bonus 0 -> "ho", bonus 1.0 -> "ho spostato". The two decoders must agree, or the
  // browser ships a different quality mode from the one measured in runs/E-022/.
  const results = await page.evaluate(async () => {
    const { beamSearch } = await import("/web/decode.js");
    // 2 frames x 3 ids (blank, "▁ho", "▁spostato"), row-major.
    const probs = new Float32Array([
      -1.0, -0.5, -10.0, // frame 0: blank -1.0, token 1 -0.5
      -0.5, -10.0, -1.2, // frame 1: blank -0.5, token 2 -1.2
    ]);
    const pieces = ["", "▁ho", "▁spostato"];
    return {
      plain: beamSearch(probs, 2, 3, pieces, { beamWidth: 4 }),
      bonus: beamSearch(probs, 2, 3, pieces, { beamWidth: 4, tokenBonus: 1.0 }),
    };
  });
  expect(results.plain).toBe("ho");
  expect(results.bonus).toBe("ho spostato");
});

test("JS tokenBonus defaults to a no-op", async ({ page }) => {
  await page.goto("/web/index.html");
  const same = await page.evaluate(async () => {
    const { beamSearch } = await import("/web/decode.js");
    const probs = new Float32Array([-10, 0, -10, -10, -10, 0, -10, 0, -10]);
    const pieces = ["", "a", "b"];
    return (
      beamSearch(probs, 3, 3, pieces, { beamWidth: 3 }) ===
      beamSearch(probs, 3, 3, pieces, { beamWidth: 3, tokenBonus: 0 })
    );
  });
  expect(same).toBe(true);
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
