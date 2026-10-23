import { expect, test } from "@playwright/test";

test("microphone capture resamples to 16 kHz PCM", async ({ page }) => {
  await page.goto("/web/index.html");
  await page.waitForFunction(() => (window as any).moth?.ready === true, undefined, {
    timeout: 120_000,
  });

  const result = await page.evaluate(async () => {
    const record = document.getElementById("record") as HTMLButtonElement;
    const stop = document.getElementById("stop") as HTMLButtonElement;
    const status = document.getElementById("status") as HTMLElement;
    record.click();
    await new Promise((resolve) => setTimeout(resolve, 1_500));
    stop.click();
    for (let attempt = 0; attempt < 100; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 100));
      const text = status.textContent ?? "";
      if (text.startsWith("microfono:")) {
        return { status: text, pcm: (window as any).moth.getPcm()?.length ?? 0 };
      }
      if (text.startsWith("errore") || text.startsWith("permesso")) break;
    }
    return { status: status.textContent ?? "", pcm: 0 };
  });

  expect(result.status).toContain("microfono:");
  expect(result.pcm).toBeGreaterThan(1600);
});
