import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

const root = fileURLToPath(new URL("../", import.meta.url));
const python = `"${process.env.PYTHON || "python"}"`;

export default defineConfig({
  testDir: "./acceptance",
  timeout: 300_000,
  workers: 1,
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  outputDir: "./test-results/acceptance",
  use: { browserName: "chromium", headless: true, trace: "off", screenshot: "off", video: "off" },
  projects: [
    { name: "isolated-auto", use: { baseURL: "http://127.0.0.1:8767" }, metadata: { isolated: true, threads: null } },
    { name: "isolated-1", use: { baseURL: "http://127.0.0.1:8767" }, metadata: { isolated: true, threads: 1 } },
    { name: "isolated-4", use: { baseURL: "http://127.0.0.1:8767" }, metadata: { isolated: true, threads: 4 } },
    { name: "nonisolated-auto", use: { baseURL: "http://127.0.0.1:8768" }, metadata: { isolated: false, threads: null } },
  ],
  webServer: [
    {
      command: `${python} web/server.py --root . --host 127.0.0.1 --port 8767`,
      cwd: root,
      url: "http://127.0.0.1:8767/web/ort.js",
      reuseExistingServer: false,
    },
    {
      command: `${python} -m http.server 8768 --bind 127.0.0.1 --directory .`,
      cwd: root,
      url: "http://127.0.0.1:8768/web/ort.js",
      reuseExistingServer: false,
    },
  ],
});
