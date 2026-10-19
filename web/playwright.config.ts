import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 180_000,
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:8765",
    headless: true,
    launchOptions: { args: ["--enable-features=SharedArrayBuffer"] },
  },
  webServer: {
    command: "python3 server.py --root .. --port 8765",
    url: "http://127.0.0.1:8765/web/index.html",
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
