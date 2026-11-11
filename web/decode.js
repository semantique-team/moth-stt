/**
 * CTC decoders (docs/02 §6) — the JS twins of `stt/decode.py`.
 *
 * Greedy: argmax per frame → collapse consecutive repeats → drop blanks (id 0) → join pieces.
 * Beam: prefix beam search with an optional BPE trigram LM (web/lm.js). `pieces` is indexed by
 * CTC id, with `pieces[0] === ""` for blank.
 *
 * `tokenBonus` is added once per emitted token. It is not a quality knob on its own: the LM term
 * (`lmWeight * LM(token)`) is always negative, so without a counterweight a positive `lmWeight`
 * biases the search toward emitting fewer tokens than the acoustics support. The two are tuned
 * together — see `runs/E-022/` and the twin implementation in `stt/decode.py`.
 */

import { LM_BOS_ID, LM_EOS_ID } from "./lm.js";

export const BLANK_ID = 0;
const SPACE_MARKER = "▁";

/** Experimental single-word bias. Sequences are CTC IDs from the frozen tokenizer,
 * not strings guessed from the vocabulary. Terms must already be canonically normalized.
 * delta() is only for emitted labels; finish() confirms the last word at EOS.
 */
export class ContextualBias {
  constructor(sequences, pieces, { bonus = 1, prefixBonus = 0 } = {}) {
    if (!Number.isFinite(bonus) || !Number.isFinite(prefixBonus) ||
        prefixBonus < 0 || prefixBonus > bonus) {
      throw new Error("require finite 0 <= prefixBonus <= bonus");
    }
    this.bonus = bonus;
    this.prefixBonus = prefixBonus;
    this.boundaries = new Set(Object.keys(pieces).filter(
      (id) => pieces[id].startsWith(SPACE_MARKER),
    ).map(Number));
    this.terminals = new Set();
    this.prefixes = new Set();
    this.maxLen = 0;
    for (const sequence of sequences) {
      const ids = [...sequence];
      if (!ids.length || ids.some((id) => !Number.isInteger(id) || id <= 0 ||
          typeof pieces[id] !== "string" || !pieces[id])) {
        throw new Error("contextual sequences require known nonblank integer CTC IDs");
      }
      const spelling = ids.map((id) => pieces[id]).join("");
      if (!this.boundaries.has(ids[0]) ||
          !/^[a-z\u00e0\u00e8\u00e9\u00ec\u00ed\u00f2\u00f3\u00f9\u00fa']+$/.test(spelling.slice(1))) {
        throw new Error("contextual terms must be normalized single words with a start marker");
      }
      this.terminals.add(ids.join(","));
      for (let size = 1; size < ids.length; size += 1) {
        this.prefixes.add(ids.slice(0, size).join(","));
      }
      this.maxLen = Math.max(this.maxLen, ids.length);
    }
  }

  word(tokens) {
    for (let size = 1; size <= Math.min(tokens.length, this.maxLen); size += 1) {
      if (this.boundaries.has(tokens[tokens.length - size])) return tokens.slice(-size);
    }
    return [];
  }

  potential(word) {
    const key = word.join(",");
    return this.terminals.has(key) || this.prefixes.has(key) ? this.prefixBonus : 0;
  }

  delta(prefix, token) {
    const word = this.word(prefix);
    const boundary = this.boundaries.has(token);
    const next = boundary ? [token] : word.length ? word.concat(token) : [];
    const completion = boundary && this.terminals.has(word.join(",")) ? this.bonus : 0;
    return completion + this.potential(next) - this.potential(word);
  }

  finish(prefix) {
    const word = this.word(prefix);
    return (this.terminals.has(word.join(",")) ? this.bonus : 0) - this.potential(word);
  }
}

function idsToText(ids, pieces) {
  let text = "";
  for (const id of ids) text += pieces[id] ?? "";
  return text.split(SPACE_MARKER).join(" ").trim();
}

function logAddExp(a, b) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  if (a < b) {
    const swap = a;
    a = b;
    b = swap;
  }
  return a + Math.log1p(Math.exp(b - a));
}

function topKLabels(logProbs, offset, vocabSize, topK) {
  const top = [];
  for (let id = 1; id < vocabSize; id += 1) {
    const value = logProbs[offset + id];
    if (top.length === topK && value <= top[top.length - 1].value) continue;
    let position = top.length;
    while (
      position > 0 &&
      (top[position - 1].value < value ||
        (top[position - 1].value === value && top[position - 1].id > id))
    ) {
      position -= 1;
    }
    top.splice(position, 0, { id, value });
    if (top.length > topK) top.pop();
  }
  return top;
}

function updateBeam(next, ids, pBlank, pNonblank) {
  const key = ids.join(",");
  let entry = next.get(key);
  if (!entry) {
    entry = { ids, pBlank: -Infinity, pNonblank: -Infinity };
    next.set(key, entry);
  }
  if (pBlank !== null) entry.pBlank = logAddExp(entry.pBlank, pBlank);
  if (pNonblank !== null) entry.pNonblank = logAddExp(entry.pNonblank, pNonblank);
}

function lmBonus(lm, ids, token, weight) {
  if (!lm || weight === 0) return 0;
  const prev1 = ids.length ? ids[ids.length - 1] - 1 : LM_BOS_ID;
  const prev2 = ids.length >= 2 ? ids[ids.length - 2] - 1 : LM_BOS_ID;
  return weight * lm.score(prev2, prev1, token - 1);
}

function beamScore(entry) {
  return logAddExp(entry.pBlank, entry.pNonblank);
}

/**
 * @param {Float32Array} logProbs row-major `[frames, vocabSize]` log-probabilities
 * @param {number} frames number of usable frames (the graph returns only real frames)
 * @param {number} vocabSize `V_ctc` (blank included)
 * @param {string[]} pieces piece per CTC id; `pieces[0]` is blank
 */
export function greedy(logProbs, frames, vocabSize, pieces) {
  let text = "";
  let previous = -1;
  for (let t = 0; t < frames; t += 1) {
    const offset = t * vocabSize;
    let best = 0;
    let bestValue = logProbs[offset];
    for (let id = 1; id < vocabSize; id += 1) {
      const value = logProbs[offset + id];
      if (value > bestValue) {
        bestValue = value;
        best = id;
      }
    }
    if (best !== BLANK_ID && best !== previous) text += pieces[best];
    previous = best;
  }
  return text.split(SPACE_MARKER).join(" ").trim();
}

/**
 * CTC prefix beam search, a port of stt/decode.py.
 * `topK` counts nonblank labels; blank is always kept separately. Candidate ties use ascending ID.
 *
 * @param {Float32Array} logProbs row-major `[frames, vocabSize]`
 * @param {number} frames usable frames
 * @param {number} vocabSize `V_ctc`
 * @param {string[]} pieces piece per CTC id; blank at 0
 * Contextual scoring is opt-in and cannot rescue acoustic top-k exclusions.
 * @param {{beamWidth?: number, topK?: number, lm?: object, lmWeight?: number, tokenBonus?: number, contextual?: ContextualBias}} options
 */
export function beamSearch(
  logProbs,
  frames,
  vocabSize,
  pieces,
  { beamWidth = 5, topK = 8, lm = null, lmWeight = 0, tokenBonus = 0, contextual = null } = {},
) {
  if (beamWidth <= 0 || topK <= 0) throw new Error("beamWidth and topK must be positive");
  if (frames <= 0) return "";

  let beams = new Map([["", { ids: [], pBlank: 0, pNonblank: -Infinity }]]);
  for (let frame = 0; frame < frames; frame += 1) {
    const offset = frame * vocabSize;
    const candidates = topKLabels(logProbs, offset, vocabSize, topK);
    const blankLogp = logProbs[offset];
    const next = new Map();

    for (const beam of beams.values()) {
      const { ids, pBlank, pNonblank } = beam;
      const total = logAddExp(pBlank, pNonblank);
      updateBeam(next, ids, total + blankLogp, null);

      const last = ids.length ? ids[ids.length - 1] : -1;
      for (const { id: token, value: logp } of candidates) {
        if (token === last) {
          updateBeam(next, ids, null, pNonblank + logp);
          const extended = ids.concat(token);
          updateBeam(
            next,
            extended,
            null,
            pBlank + logp + tokenBonus + lmBonus(lm, ids, token, lmWeight) +
              (contextual ? contextual.delta(ids, token) : 0),
          );
        } else {
          const extended = ids.concat(token);
          updateBeam(
            next,
            extended,
            null,
            total + logp + tokenBonus + lmBonus(lm, ids, token, lmWeight) +
              (contextual ? contextual.delta(ids, token) : 0),
          );
        }
      }
    }

    const ranked = [...next.values()].sort((a, b) => beamScore(b) - beamScore(a));
    beams = new Map(ranked.slice(0, beamWidth).map((entry) => [entry.ids.join(","), entry]));
  }

  let best = null;
  let bestScore = -Infinity;
  for (const entry of beams.values()) {
    let score = beamScore(entry);
    if (contextual) score += contextual.finish(entry.ids);
    if (lm && lmWeight !== 0) {
      const ids = entry.ids;
      const prev1 = ids.length ? ids[ids.length - 1] - 1 : LM_BOS_ID;
      const prev2 = ids.length >= 2 ? ids[ids.length - 2] - 1 : LM_BOS_ID;
      score += lmWeight * lm.score(prev2, prev1, LM_EOS_ID);
    }
    if (score > bestScore) {
      bestScore = score;
      best = entry;
    }
  }
  return best ? idsToText(best.ids, pieces) : "";
}
