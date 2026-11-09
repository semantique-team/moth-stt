/**
 * Sentence segmentation in the browser path (runs/audit-2026-11-09).
 *
 * The model is trained on sentence-sized clips and reads a recording of several sentences as one
 * unpunctuated stream, which costs 9.4 WER points on five concatenated sentences. The shipped VAD
 * finds the pauses; this suite checks that the JS path uses it the way the Python probe measured.
 */

import { expect, test } from "@playwright/test";

/** Five held-out-speaker clips, concatenated with 300 ms of silence — the probe's shape. */
const CLIPS = [
  "S3/S3_0003.wav",
  "S3/S3_0009.wav",
  "S3/S3_0012.wav",
  "S3/S3_0021.wav",
  "S3/S3_0033.wav",
];

test("the VAD finds five sentences and the transcript is the five transcripts joined", async ({
  page,
}) => {
  await page.goto("/web/index.html");
  await page.waitForFunction(() => (window as any).moth?.ready === true, undefined, {
    timeout: 120_000,
  });

  const result = await page.evaluate(async (clips: string[]) => {
    const moth = (window as any).moth;
    const parts: Float32Array[] = [];
    for (const clip of clips) {
      parts.push(await moth.decodeTo16k(`/recordings/audio/${clip}`));
    }
    const gap = new Float32Array(Math.round(0.3 * 16000));
    const total = parts.reduce((sum, p) => sum + p.length + gap.length, 0);
    const pcm = new Float32Array(total);
    let offset = 0;
    for (const part of parts) {
      pcm.set(part, offset);
      offset += part.length;
      pcm.set(gap, offset);
      offset += gap.length;
    }
    const bounds = await moth.runner.vad.segments(pcm);
    const withSegments = await moth.transcribePcm(pcm);
    const withoutSegments = await moth.runner.transcribe(pcm, { segment: false });
    return { bounds, withSegments, withoutSegments, seconds: pcm.length / 16000 };
  }, CLIPS);

  console.log(`segments: ${result.bounds.length} over ${result.seconds.toFixed(1)} s`);
  console.log(`with segmentation:    ${JSON.stringify(result.withSegments.text)}`);
  console.log(`without segmentation: ${JSON.stringify(result.withoutSegments.text)}`);

  expect(result.bounds.length).toBe(CLIPS.length);
  // Segmenting must not lose content: the joined transcript has at least as many words.
  const wordsWith = result.withSegments.text.split(/\s+/).filter(Boolean).length;
  const wordsWithout = result.withoutSegments.text.split(/\s+/).filter(Boolean).length;
  expect(wordsWith).toBeGreaterThanOrEqual(wordsWithout);
});

test("a single sentence is decoded whole, not segmented", async ({ page }) => {
  await page.goto("/web/index.html");
  await page.waitForFunction(() => (window as any).moth?.ready === true, undefined, {
    timeout: 120_000,
  });

  const result = await page.evaluate(async () => {
    const moth = (window as any).moth;
    const pcm = await moth.decodeTo16k("/recordings/audio/S3/S3_0003.wav");
    const bounds = await moth.runner.vad.segments(pcm);
    const withSegments = await moth.transcribePcm(pcm);
    const withoutSegments = await moth.runner.transcribe(pcm, { segment: false });
    return { bounds, withSegments, withoutSegments };
  });

  // One speech run: the code must take the whole-clip path, so the two transcripts are identical.
  expect(result.bounds.length).toBe(1);
  expect(result.withSegments.text).toBe(result.withoutSegments.text);
  expect(result.withSegments.text).toBe("ho ricevuto un feedback positivo");
});
