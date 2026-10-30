import { expect, test } from "@playwright/test";

/**
 * D25 (E-011 addendum): the no-speech guard in the browser is Silero VAD, run on the 16 kHz PCM
 * before the mel. The synthetic tone fixture must be gated; real speech must reach the model.
 */
test("VAD guard gates the tone fixture and keeps speech", async ({ page }) => {
  await page.goto("/web/index.html");
  await page.waitForFunction(() => (window as any).moth?.ready === true, undefined, {
    timeout: 120_000,
  });

  const result = await page.evaluate(async () => {
    const moth = (window as any).moth as any;
    const tone = await moth.decodeTo16k("/web/tests/fixtures/clip.wav");
    const speech = await moth.decodeTo16k("/web/tests/fixtures/speech.wav");
    return {
      gate: moth.runner.vad?.gate ?? null,
      toneFraction: await moth.runner.vad.speechFraction(tone),
      speechFraction: await moth.runner.vad.speechFraction(speech),
      tone: await moth.transcribePcm(tone),
      speech: await moth.transcribePcm(speech),
    };
  });

  expect(result.gate).toBeCloseTo(0.1, 5);
  expect(result.toneFraction).toBeLessThanOrEqual(0.1);
  expect(result.tone.guarded).toBe(true);
  expect(result.tone.text).toBe("");
  expect(result.tone.decoder).toBe("vad-guard");

  expect(result.speechFraction).toBeGreaterThan(0.5);
  expect(result.speech.guarded).toBeUndefined();
  expect(result.speech.text.trim().length).toBeGreaterThan(0);
});
