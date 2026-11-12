// E-031 decoder-level parity: run the browser beam search and LM reader on Python logits.
import { readFileSync } from "node:fs";
import { beamSearch } from "../../web/decode.js";
import { loadLM } from "../../web/lm.js";

const [logitsPath, lmPath, vocabPath] = process.argv.slice(2);
const meta = JSON.parse(readFileSync(logitsPath.replace(/logits\.bin$/, "meta.json"), "utf8"));
const bytes = readFileSync(logitsPath);
const logits = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
const vocabJson = JSON.parse(readFileSync(vocabPath, "utf8"));
const maxId = Math.max(...Object.keys(vocabJson).map(Number));
const pieces = new Array(maxId + 1).fill("");
for (const [id, piece] of Object.entries(vocabJson)) pieces[Number(id)] = piece;
const lmBytes = readFileSync(lmPath);
const lmBuffer = lmBytes.buffer.slice(lmBytes.byteOffset, lmBytes.byteOffset + lmBytes.byteLength);
const lm = loadLM(lmBuffer);
const text = beamSearch(logits, meta.frames, meta.vocab, pieces, {
  beamWidth: meta.params.beam,
  topK: meta.params.top_k,
  lm,
  lmWeight: meta.params.lm_weight,
  tokenBonus: meta.params.token_bonus,
});
process.stdout.write(text);
