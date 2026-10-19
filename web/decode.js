/**
 * Greedy CTC decode (docs/02 §6) — the JS twin of `stt/decode.py`.
 *
 * Argmax per frame → collapse consecutive repeats → drop blanks (id 0) → join the pieces
 * (`▁` becomes a space). `pieces` is indexed by CTC id, with `pieces[0] === ""` for blank.
 */

export const BLANK_ID = 0;
const SPACE_MARKER = "▁";

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
