import { expect, test } from "@playwright/test";

test("JS no-speech guard mirrors the Python threshold", async ({ page }) => {
  await page.goto("/web/index.html");
  const result = await page.evaluate(async () => {
    const { isNoSpeech, SILENCE_MAX } = await import("/web/mel.js");
    // Digital silence: every normalized log-mel bin is exactly -1.5.
    const silence = new Float32Array(80 * 100).fill(-1.5);
    // Quiet speech-like clip: one loud bin is enough, even if the mean stays below the threshold.
    const speech = new Float32Array(80 * 100).fill(-1.9);
    speech[40 * 100 + 50] = 0.1;
    return { silent: isNoSpeech(silence), speech: isNoSpeech(speech), threshold: SILENCE_MAX };
  });
  expect(result.threshold).toBe(-1.0);
  expect(result.silent).toBe(true);
  expect(result.speech).toBe(false);
});
