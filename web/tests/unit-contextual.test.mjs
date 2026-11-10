import assert from "node:assert/strict";
import test from "node:test";
import { ContextualBias, beamSearch } from "../decode.js";
import { LM_EOS_ID } from "../lm.js";

const pieces = ["", "\u2581meet", "ing", "s", "\u2581il", "\u2581meeting", "\u2581api", "x", "meet"];
const sequences = [[1, 2], [5], [6]];

function reward(bias, ids, finish = true) {
  let total = 0;
  for (let i = 0; i < ids.length; i += 1) total += bias.delta(ids.slice(0, i), ids[i]);
  return total + (finish ? bias.finish(ids) : 0);
}

test("bounded potential, abandonment, boundaries, EOS, repeated occurrences and acronyms", () => {
  for (const prefixBonus of [0, 0.25, 1]) {
    const bias = new ContextualBias(sequences, pieces, { prefixBonus });
    for (const [ids, expected] of [
      [[], 0], [[1], 0], [[1, 7], 0], [[1, 2], 1], [[5], 1],
      [[1, 2, 3], 0], [[4, 8, 2], 0], [[8, 2], 0], [[1, 2, 4], 1],
      [[1, 2, 1, 2], 2], [[5, 5, 6], 3], [[1, 7, 1, 7, 6], 1],
    ]) assert.equal(reward(bias, ids), expected, String(ids));
    assert.equal(reward(bias, [1, 2], false), prefixBonus);
    assert.equal(reward(bias, [1, 2, 4], false), 1);
    assert.equal(bias.delta([1], 2), 0);
  }
});

test("shared prefixes and duplicate terms do not stack", () => {
  const bias = new ContextualBias([[1], [1, 2], [1, 2], [1, 2, 3]], pieces, { prefixBonus: 1 });
  for (const ids of [[1], [1, 2], [1, 2, 3]]) assert.equal(reward(bias, ids), 1);
  assert.equal(reward(bias, [1, 2, 3, 7]), 0);
});

test("invalid sequences and unbounded bonuses are rejected", () => {
  for (const ids of [[], [0], [999], [8, 2], [1, 4], [1, 0, 2], [1.5]]) {
    assert.throws(() => new ContextualBias([ids], pieces));
  }
  for (const [bonus, prefixBonus] of [[-1, 0], [1, -1], [1, 2], [Infinity, 0], [1, NaN]]) {
    assert.throws(() => new ContextualBias(sequences, pieces, { bonus, prefixBonus }));
  }
});

test("EOS chooses an existing lexical candidate, never rescues acoustic top-k", () => {
  const vocab = ["", "\u2581il", "\u2581meeting"];
  const scores = [-10, 0, -0.5];
  const contextual = new ContextualBias([[2]], vocab);
  assert.equal(beamSearch(scores, 1, 3, vocab), "il");
  assert.equal(beamSearch(scores, 1, 3, vocab, { contextual, topK: 2 }), "meeting");
  assert.equal(beamSearch(scores, 1, 3, vocab, { contextual, topK: 1 }), "il");
});

test("prefix potential aids survival but is refunded for incomplete words at EOS", () => {
  const vocab = ["", "\u2581il", "\u2581meet", "ing"];
  const scores = [-100, 0, -0.5, -100, -100, -100, -100, 0];
  const contextual = new ContextualBias([[2, 3]], vocab, { prefixBonus: 1 });
  assert.equal(beamSearch(scores, 2, 4, vocab, { beamWidth: 1, topK: 2 }), "iling");
  assert.equal(beamSearch(scores, 2, 4, vocab, { beamWidth: 1, topK: 2, contextual }), "meeting");
  assert.equal(beamSearch(scores, 1, 4, vocab, { beamWidth: 2, contextual }), "il");
});

test("repeated CTC frames earn nothing; blank-separated words are separate occurrences", () => {
  const vocab = ["", "\u2581api", "\u2581il"];
  const contextual = new ContextualBias([[1]], vocab, { prefixBonus: 1 });
  assert.equal(beamSearch(Array(4).fill([-100, 0, -100]).flat(), 4, 3, vocab, { contextual }), "api");
  assert.equal(beamSearch([-100, 0, -100, 0, -100, -100, -100, 0, -100],
    3, 3, vocab, { contextual }), "api api");
  assert.equal(beamSearch([-100, 0, -100, -100, 0, -100], 2, 3, vocab, { contextual }), "api");
});

test("default, empty, zero and irrelevant glossary are no-ops", () => {
  const scores = [-10, 0, -0.5, -10, -10, -10, -100, -10, -10];
  const expected = beamSearch(scores, 1, pieces.length, pieces, { topK: 2 });
  for (const contextual of [null, new ContextualBias([], pieces),
    new ContextualBias(sequences, pieces, { bonus: 0 }),
    new ContextualBias([[6]], pieces, { prefixBonus: 1 })]) {
    assert.equal(beamSearch(scores, 1, pieces.length, pieces, { topK: 2, contextual }), expected);
  }
});

test("CTC frame repeats cannot accumulate bonus against EOS cost", () => {
  const vocab = ["", "\u2581api"];
  const contextual = new ContextualBias([[1]], vocab, { prefixBonus: 1 });
  const lm = { score: (_, prev1, token) => token === LM_EOS_ID && prev1 === 0 ? -5 : 0 };
  assert.equal(beamSearch(Array(4).fill([0, -0.75]).flat(), 4, 2, vocab,
    { contextual, lm, lmWeight: 1 }), "");
});
