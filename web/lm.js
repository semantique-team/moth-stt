/**
 * Browser reader for the compact BPE trigram LM (stt/lm.py save_browser).
 *
 * The layout is sparse CSR: unigram float32, bigram rows by previous id, trigram rows keyed by
 * `prev2 * V + prev1` with sorted uint32 keys. Log-probs are int16-quantized; missing entries
 * back off exactly like the Python score().
 */

export const LM_BOS_ID = 1024;
export const LM_EOS_ID = 1025;
export const LM_VOCAB_SIZE = 1026;

function align4(value) {
  return (value + 3) & ~3;
}

function findContext(keys, key) {
  let low = 0;
  let high = keys.length - 1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    const value = keys[middle];
    if (value === key) return middle;
    if (value < key) low = middle + 1;
    else high = middle - 1;
  }
  return -1;
}

export class BPELM {
  constructor(buffer) {
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
    if (magic !== "MOTHLMB1") throw new Error(`bad LM magic: ${JSON.stringify(magic)}`);
    const view = new DataView(buffer);
    let offset = 8;
    const version = view.getUint32(offset, true);
    offset += 4;
    if (version !== 1) throw new Error(`unsupported LM version: ${version}`);
    this.vocab = view.getUint32(offset, true);
    offset += 4;
    this.backoff = view.getFloat32(offset, true);
    offset += 4;
    const unigramLength = view.getUint32(offset, true);
    offset += 4;
    this.logScale = view.getFloat32(offset, true);
    offset += 4;

    if (this.vocab !== LM_VOCAB_SIZE || unigramLength !== LM_VOCAB_SIZE) {
      throw new Error(`LM vocab mismatch: ${this.vocab}/${unigramLength}`);
    }
    if (offset % 4 !== 0) throw new Error("misaligned LM header");
    this.unigram = new Float32Array(buffer, offset, unigramLength);
    offset += unigramLength * 4;

    const bigramContexts = view.getUint32(offset, true);
    offset += 4;
    if (bigramContexts !== LM_VOCAB_SIZE) {
      throw new Error(`LM bigram contexts: ${bigramContexts}`);
    }
    this.bigramOffsets = new Uint32Array(buffer, offset, bigramContexts + 1);
    offset += (bigramContexts + 1) * 4;
    const bigramEntries = this.bigramOffsets[bigramContexts];
    this.bigramTokens = new Uint16Array(buffer, offset, bigramEntries);
    offset += bigramEntries * 2;
    this.bigramLogq = new Int16Array(buffer, offset, bigramEntries);
    offset += bigramEntries * 2;
    offset = align4(offset);

    const trigramContexts = view.getUint32(offset, true);
    offset += 4;
    this.trigramKeys = new Uint32Array(buffer, offset, trigramContexts);
    offset += trigramContexts * 4;
    this.trigramOffsets = new Uint32Array(buffer, offset, trigramContexts + 1);
    offset += (trigramContexts + 1) * 4;
    const trigramEntries = this.trigramOffsets[trigramContexts];
    this.trigramTokens = new Uint16Array(buffer, offset, trigramEntries);
    offset += trigramEntries * 2;
    this.trigramLogq = new Int16Array(buffer, offset, trigramEntries);

    this.unigramFloor = new Float32Array(this.vocab);
    const floor = 2 * Math.log(this.backoff);
    for (let token = 0; token < this.vocab; token += 1) {
      this.unigramFloor[token] = this.unigram[token] + floor;
    }
  }

  score(prev2, prev1, token) {
    const context = prev2 * this.vocab + prev1;
    const index = findContext(this.trigramKeys, context);
    if (index >= 0) {
      const start = this.trigramOffsets[index];
      const end = this.trigramOffsets[index + 1];
      for (let entry = start; entry < end; entry += 1) {
        if (this.trigramTokens[entry] === token) {
          return this.trigramLogq[entry] * this.logScale;
        }
      }
    }
    const start = this.bigramOffsets[prev1];
    const end = this.bigramOffsets[prev1 + 1];
    for (let entry = start; entry < end; entry += 1) {
      if (this.bigramTokens[entry] === token) return this.bigramLogq[entry] * this.logScale;
    }
    return this.unigramFloor[token];
  }
}

export function loadLM(buffer) {
  return new BPELM(buffer);
}
