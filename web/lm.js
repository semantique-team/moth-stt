/**
 * Browser reader for the compact BPE trigram LM (stt/lm.py save_browser).
 *
 * The layout is sparse CSR: unigram float32, bigram rows by previous id, trigram rows keyed by
 * `prev2 * V + prev1` with sorted uint32 keys. Log-probs are int16-quantized; missing entries
 * back off exactly like the Python score().
 * Empty trigram rows may be omitted without changing scores or the MOTHLMB1 layout.
 */

export const LM_BOS_ID = 1024;
export const LM_EOS_ID = 1025;
export const LM_VOCAB_SIZE = 1026;

function align4(value) {
  return Math.ceil(value / 4) * 4;
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
    if (buffer.byteLength < 28) throw new Error("truncated LM header");
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
    if (magic !== "MOTHLMB1") throw new Error(`bad LM magic: ${JSON.stringify(magic)}`);
    const view = new DataView(buffer);
    let offset = 8;
    const read = (Type, length) => {
      const size = Type.BYTES_PER_ELEMENT * length;
      if (size > buffer.byteLength - offset) throw new Error("truncated LM buffer");
      const result = new Type(buffer, offset, length);
      offset += size;
      return result;
    };
    const validateOffsets = (offsets) => {
      if (offsets[0] !== 0) throw new Error("LM offsets must start at zero");
      for (let i = 1; i < offsets.length; i += 1) {
        if (offsets[i] < offsets[i - 1]) throw new Error("LM offsets must be monotonic");
      }
    };
    const validateTokens = (offsets, tokens) => {
      for (let row = 0; row < offsets.length - 1; row += 1) {
        let previous = -1;
        for (let i = offsets[row]; i < offsets[row + 1]; i += 1) {
          if (tokens[i] >= this.vocab || tokens[i] <= previous) {
            throw new Error("LM row tokens must be in range and strictly sorted");
          }
          previous = tokens[i];
        }
      }
    };
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
    if (!Number.isFinite(this.backoff) || this.backoff <= 0) {
      throw new Error("LM backoff must be finite and positive");
    }
    if (!Number.isFinite(this.logScale) || this.logScale <= 0) {
      throw new Error("LM log scale must be finite and positive");
    }
    this.unigram = read(Float32Array, unigramLength);
    if (!this.unigram.every(Number.isFinite)) throw new Error("LM unigram must be finite");

    const bigramContexts = read(Uint32Array, 1)[0];
    if (bigramContexts !== LM_VOCAB_SIZE) {
      throw new Error(`LM bigram contexts: ${bigramContexts}`);
    }
    this.bigramOffsets = read(Uint32Array, bigramContexts + 1);
    validateOffsets(this.bigramOffsets);
    const bigramEntries = this.bigramOffsets[bigramContexts];
    this.bigramTokens = read(Uint16Array, bigramEntries);
    this.bigramLogq = read(Int16Array, bigramEntries);
    validateTokens(this.bigramOffsets, this.bigramTokens);
    offset = align4(offset);

    const trigramContexts = read(Uint32Array, 1)[0];
    if (trigramContexts > this.vocab * this.vocab) throw new Error("too many LM trigram contexts");
    this.trigramKeys = read(Uint32Array, trigramContexts);
    for (let i = 0; i < trigramContexts; i += 1) {
      if (this.trigramKeys[i] >= this.vocab * this.vocab ||
          (i > 0 && this.trigramKeys[i] <= this.trigramKeys[i - 1])) {
        throw new Error("LM trigram keys must be in range and strictly sorted");
      }
    }
    this.trigramOffsets = read(Uint32Array, trigramContexts + 1);
    validateOffsets(this.trigramOffsets);
    const trigramEntries = this.trigramOffsets[trigramContexts];
    this.trigramTokens = read(Uint16Array, trigramEntries);
    this.trigramLogq = read(Int16Array, trigramEntries);
    validateTokens(this.trigramOffsets, this.trigramTokens);
    if (offset !== buffer.byteLength) throw new Error("trailing LM bytes");

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
