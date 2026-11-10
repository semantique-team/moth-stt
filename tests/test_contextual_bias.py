"""Synthetic SentencePiece spellings, no tokenizer/model assets or new dependencies."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from stt.decode import beam_search
from stt.hotwords import ContextualBias, HotwordIndex
from stt.lm import EOS_ID

PIECES = [
    "",
    "\u2581meet",
    "ing",
    "s",
    "\u2581il",
    "\u2581meeting",
    "\u2581api",
    "x",
    "meet",
    "\u2581",
]
SEQUENCES = [(1, 2), (5,), (6,)]


def reward(bias, ids, *, finish=True):
    prefix = []
    total = 0.0
    for token in ids:
        total += bias.delta(prefix, token)
        prefix.append(token)
    return total + (bias.finish(prefix) if finish else 0.0)


@pytest.mark.parametrize("prefix_bonus", [0, 0.25, 1])
def test_word_rewards_are_bounded_and_abandoned_potential_is_refunded(prefix_bonus):
    bias = ContextualBias(SEQUENCES, PIECES, bonus=1, prefix_bonus=prefix_bonus)
    cases = [
        ([], 0),
        ([1], 0),
        ([1, 7], 0),
        ([1, 2], 1),
        ([5], 1),
        ([1, 2, 3], 0),
        ([4, 8, 2], 0),
        ([8, 2], 0),
        ([9, 8, 2], 0),
        ([1, 2, 4], 1),
        ([1, 2, 1, 2], 2),
        ([5, 5, 6], 3),
        ([1, 7, 1, 7, 6], 1),
        ([1, 2, 7, 1, 2], 1),
    ]
    for ids, expected in cases:
        assert reward(bias, ids) == pytest.approx(expected), ids
    assert reward(bias, [1, 2], finish=False) == prefix_bonus
    assert reward(bias, [1, 2, 4], finish=False) == 1
    assert bias.delta([1], 2) == 0  # extension replaces, rather than adds, potential


def test_shared_prefix_and_duplicate_terms_do_not_stack():
    bias = ContextualBias([(1,), (1, 2), (1, 2), (1, 2, 3)], PIECES, prefix_bonus=1)
    for ids in [[1], [1, 2], [1, 2, 3]]:
        assert reward(bias, ids) == 1
    assert reward(bias, [1, 2, 3, 7]) == 0


def test_normalized_terms_use_supplied_ctc_encoder_and_roundtrip_exactly():
    calls = []

    def encode(word):
        calls.append(word)
        return {"meeting": [1, 2], "api": [6]}[word]

    bias = ContextualBias.from_terms([" Meeting! ", "API"], PIECES, encode)
    assert calls == ["meeting", "api"]
    assert reward(bias, [1, 2, 6]) == 2
    for term in ["two words", "foo-bar", "", "  "]:
        with pytest.raises(ValueError, match="single word"):
            ContextualBias.from_terms([term], PIECES, encode)
    for ids in [[8, 2], [1], [0], [999], []]:
        with pytest.raises(ValueError):
            ContextualBias.from_terms(["meeting"], PIECES, lambda _, ids=ids: ids)


@pytest.mark.parametrize("sequences", [[[]], [[0]], [[999]], [[8, 2]], [[1, 4]], [[1, 0, 2]]])
def test_rejects_empty_blank_unknown_unanchored_or_multiword_sequences(sequences):
    with pytest.raises(ValueError):
        ContextualBias(sequences, PIECES)


@pytest.mark.parametrize(
    "bonus,prefix", [(-1, 0), (1, -1), (1, 2), (float("inf"), 0), (1, float("nan"))]
)
def test_rejects_unbounded_bonuses(bonus, prefix):
    with pytest.raises(ValueError):
        ContextualBias(SEQUENCES, PIECES, bonus=bonus, prefix_bonus=prefix)


def test_eos_completion_and_top_k_does_not_rescue_excluded_labels():
    pieces = ["", "\u2581il", "\u2581meeting"]
    scores = np.array([[-10, 0, -0.5]], dtype=np.float32)
    bias = ContextualBias([(2,)], pieces)
    assert beam_search(scores, 1, pieces) == "il"
    assert beam_search(scores, 1, pieces, contextual=bias, top_k=2) == "meeting"
    assert beam_search(scores, 1, pieces, contextual=bias, top_k=1) == "il"


def test_prefix_potential_keeps_existing_candidate_alive_but_eos_refunds_incomplete_word():
    pieces = ["", "\u2581il", "\u2581meet", "ing"]
    scores = np.array([[-100, 0, -0.5, -100], [-100, -100, -100, 0]], dtype=np.float32)
    bias = ContextualBias([(2, 3)], pieces, prefix_bonus=1)
    assert beam_search(scores, 2, pieces, beam_width=1, top_k=2) == "iling"
    assert beam_search(scores, 2, pieces, beam_width=1, top_k=2, contextual=bias) == "meeting"
    assert beam_search(scores, 1, pieces, beam_width=2, contextual=bias) == "il"


def test_ctc_frame_repeats_are_not_extensions_but_blank_separated_words_are():
    pieces = ["", "\u2581api", "\u2581il"]
    bias = ContextualBias([(1,)], pieces, prefix_bonus=1)
    scores = np.array([[-100, 0, -100]] * 4, dtype=np.float32)
    assert beam_search(scores, 4, pieces, contextual=bias) == "api"
    scores = np.array([[-100, 0, -100], [0, -100, -100], [-100, 0, -100]])
    assert beam_search(scores, 3, pieces, contextual=bias) == "api api"
    assert beam_search(scores[[0, 2]], 2, pieces, contextual=bias) == "api"


def test_ctc_repeats_cannot_accumulate_bonus_against_eos_cost():
    class LM:
        def score(self, prev2, prev1, token):
            return -5 if token == EOS_ID and prev1 == 0 else 0

    pieces = ["", "\u2581api"]
    scores = np.array([[0, -0.75]] * 4)
    bias = ContextualBias([(1,)], pieces, prefix_bonus=1)
    assert beam_search(scores, 4, pieces, contextual=bias, lm=LM(), lm_weight=1) == ""


def test_default_legacy_and_irrelevant_glossary_are_unchanged():
    rng = np.random.default_rng(42)
    empty = ContextualBias([], PIECES)
    unrelated = ContextualBias([(6,)], PIECES, prefix_bonus=1)
    zero = ContextualBias(SEQUENCES, PIECES, bonus=0)
    legacy = {"hotwords": HotwordIndex([(1, 2)]), "hotword_bonus": 1, "hotword_prefix_bonus": 0.5}
    for _ in range(10):
        scores = rng.normal(size=(8, len(PIECES)))
        scores[:, 6] = -100  # unrelated glossary never appears in acoustic candidates
        for options in [{}, legacy]:
            expected = beam_search(scores, 8, PIECES, top_k=3, **options)
            for bias in [None, empty, zero, unrelated]:
                assert (
                    beam_search(scores, 8, PIECES, top_k=3, contextual=bias, **options) == expected
                )


def test_python_js_reward_and_decoder_parity():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for contextual parity")
    rng = np.random.default_rng(123)
    paths = [[], [1], [1, 2], [1, 2, 3], [4, 8, 2], [1, 2, 1, 2], [5, 5, 6]]
    paths += [rng.integers(1, len(PIECES), size=20).tolist() for _ in range(50)]
    matrices = [rng.integers(-8, 1, size=(8, len(PIECES))).tolist() for _ in range(12)]
    fixtures = [
        {"rows": rows, "prefixBonus": p, "topK": k, "useLM": lm}
        for rows in matrices
        for p in [0, 0.5, 1]
        for k in [1, 4]
        for lm in [False, True]
    ]
    script = """
import {readFileSync} from 'node:fs';
import {ContextualBias, beamSearch} from './web/decode.js';
const {pieces, sequences, paths, fixtures} = JSON.parse(readFileSync(0, 'utf8'));
const bias = new ContextualBias(sequences, pieces, {prefixBonus: 0.5});
const rewards = paths.map(ids => {
  let score = 0;
  for (let i = 0; i < ids.length; ++i) score += bias.delta(ids.slice(0, i), ids[i]);
  return score + bias.finish(ids);
});
const outputs = fixtures.map(f => beamSearch(Float32Array.from(f.rows.flat()),
  f.rows.length, pieces.length, pieces, {beamWidth: 8, topK: f.topK,
    contextual: new ContextualBias(sequences, pieces, {prefixBonus: f.prefixBonus}),
    lm: f.useLM ? {score: () => -0.5} : null, lmWeight: 0.5, tokenBonus: 0.25}));
console.log(JSON.stringify({rewards, outputs}));
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        input=json.dumps(
            {"pieces": PIECES, "sequences": SEQUENCES, "paths": paths, "fixtures": fixtures}
        ),
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    actual = json.loads(result.stdout)
    bias = ContextualBias(SEQUENCES, PIECES, prefix_bonus=0.5)
    assert actual["rewards"] == [reward(bias, path) for path in paths]

    class LM:
        def score(self, *args):
            return -0.5

    for fixture, output in zip(fixtures, actual["outputs"], strict=True):
        scores = np.asarray(fixture["rows"], dtype=np.float32)
        assert output == beam_search(
            scores,
            len(scores),
            PIECES,
            beam_width=8,
            top_k=fixture["topK"],
            contextual=ContextualBias(SEQUENCES, PIECES, prefix_bonus=fixture["prefixBonus"]),
            lm=LM() if fixture["useLM"] else None,
            lm_weight=0.5,
            token_bonus=0.25,
        ), fixture
