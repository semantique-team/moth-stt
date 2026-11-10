// Decode identical float32 acoustic outputs with the existing browser modules, not WASM.
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { beamSearch, greedy } from '../../web/decode.js';
import { loadLM } from '../../web/lm.js';

const started = performance.now();
const root = process.argv[2];
const config = JSON.parse(readFileSync(resolve(root, 'config.json'), 'utf8'));
const vocab = JSON.parse(readFileSync(resolve(root, config.tokenizer.vocab), 'utf8'));
const pieces = [''];
for (const [id, piece] of Object.entries(vocab)) pieces[Number(id)] = piece;
const bytes = readFileSync(resolve(root, config.decode.lm));
const lm = loadLM(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
console.log(JSON.stringify({ ready: true, load_seconds: (performance.now() - started) / 1000 }));
for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
  const request = JSON.parse(line);
  const bytes = Buffer.from(request.float32_base64, 'base64');
  const scores = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  if (scores.length !== request.frames * pieces.length) throw new Error('logit shape mismatch');
  const start = performance.now();
  const greedyText = greedy(scores, request.frames, pieces.length, pieces);
  const greedySeconds = (performance.now() - start) / 1000;
  const beamStart = performance.now();
  const hyp = beamSearch(scores, request.frames, pieces.length, pieces, {
    beamWidth: config.decode.beam, topK: config.decode.top_k,
    lm, lmWeight: config.decode.lm_weight, tokenBonus: config.decode.token_bonus,
  });
  console.log(JSON.stringify({ id: request.id, hyp, greedy: greedyText,
    beam_seconds: (performance.now() - beamStart) / 1000, greedy_seconds: greedySeconds }));
}
