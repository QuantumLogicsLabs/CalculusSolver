"""Beam search termination, driven by stub models -- no checkpoint needed.

There is no loadable checkpoint in this repo (checkpoints/ is gitignored and
model/model.pkl is a dead legacy artifact: vocab_size 77 vs the current 124,
and an architecture that no longer load_state_dict()s). These tests therefore
drive beam_search with deterministic stubs, which is stronger than it sounds:
an ORACLE model that is certain about every single token MUST yield the gold
sequence. If it doesn't, beam search is broken independently of any model --
and that is exactly what was happening.
"""

import math
from pathlib import Path

import pytest

torch = pytest.importorskip(
    "torch",
    reason=(
        "inference/beam_search.py imports torch (requirements-neural.txt). "
        "CI installs requirements.txt only, so this suite skips there."
    ),
)

from inference.beam_search import beam_search  # noqa: E402
from inference.grammar import EOS_TOKEN, load_vocab  # noqa: E402
from tests.unit.slang_builders import MULTI_TERM, SIMPLE  # noqa: E402
from tokenizer.slang_serializer import deserialize_slang_math  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VOCAB = load_vocab(str(ROOT / "tokenizer" / "vocab.json"))
VOCAB_SIZE = max(VOCAB["id_to_token"]) + 1


class OracleModel:
    """A decoder stub that is certain about every token.

    Puts logit 20.0 on the correct next gold token and -8.0 on everything
    else. Nothing about search width, scoring or tie-breaking should be able
    to lose an answer this unambiguous.
    """

    def __init__(self, gold_ids):
        self.gold_ids = list(gold_ids)
        self.forward_calls = 0

    def __call__(self, src_tokens, tgt):
        self.forward_calls += 1
        step = tgt.shape[1] - 1  # tokens generated after [BOS]
        logits = torch.full((1, tgt.shape[1], VOCAB_SIZE), -8.0)
        if step < len(self.gold_ids):
            logits[0, -1, self.gold_ids[step]] = 20.0
        return (logits, None, None)


class LoopingModel:
    """Biased toward the VAR/EXP cycle the real checkpoint degenerates into.

    Used only as a liveness check here. Tightening this into an assertion
    about run length is the job of the repetition penalty and the hard
    repeat guard (DEV 2 tasks 1 and 2).
    """

    def __call__(self, src_tokens, tgt):
        logits = torch.zeros(1, tgt.shape[1], VOCAB_SIZE)
        for token in ("VAR:u", "EXP:-3", "NODE:FRAC", "STRUCT:OPEN"):
            logits[0, -1, VOCAB["token_to_id"][token]] = 10.0
        return (logits, None, None)


def run_oracle(ast_tokens, beam_size=5, max_len=32):
    """Drive beam_search with an oracle and return (result, strings, model)."""
    gold_ids = [VOCAB["token_to_id"][t] for t in ast_tokens + [EOS_TOKEN]]
    model = OracleModel(gold_ids)
    result = beam_search(
        model=model,
        src_tokens=torch.zeros(1, 8, dtype=torch.long),
        vocab_map=VOCAB,
        beam_size=beam_size,
        max_len=max_len,
    )
    strings = [VOCAB["id_to_token"][t] for t in result["tokens"]]
    if strings and strings[0] == "[BOS]":
        strings = strings[1:]
    return result, strings, model


# -- the regression ----------------------------------------------------------

def test_oracle_reaches_solved():
    """THE regression test.

    Previously every input returned status="partial": [EOS] was rejected by
    the grammar, so a beam that closed a valid AST had every continuation
    masked to -inf and was dropped from the candidate list entirely. The
    correct answer was deleted and only degenerate open beams survived.
    """
    result, _, _ = run_oracle(SIMPLE)
    assert result["status"] == "solved"


def test_oracle_returns_gold_exactly():
    _, strings, _ = run_oracle(SIMPLE)
    assert strings == SIMPLE


# -- the downstream landmine -------------------------------------------------

def test_terminator_is_stripped():
    """[EOS] must not reach the caller.

    tokenizer/slang_serializer.deserialize_slang_math raises "Extra tokens
    found after deserialization" on any token after a closed AST, [EOS]
    included. Before termination worked, [EOS] was unreachable and this path
    had never once been exercised.
    """
    _, strings, _ = run_oracle(SIMPLE)
    assert EOS_TOKEN not in strings


def test_output_deserializes():
    _, strings, _ = run_oracle(SIMPLE)
    ast = deserialize_slang_math(strings)
    assert ast["numi"]["terms"] == [{"coeff": 6, "var": {"x": 1}}]


# -- illegal-candidate padding -----------------------------------------------

def test_returned_score_is_finite():
    """A -inf score means a masked token was selected into a beam.

    torch.topk pads its selection with masked -inf entries whenever fewer
    than beam_size tokens are legal, and ties break by index -- so [EOS]
    (id 2) got picked, producing a "finished" beam holding a truncated AST
    with score=-inf. Every -inf compares equal, so it then won the final sort.
    """
    result, _, _ = run_oracle(SIMPLE)
    assert math.isfinite(result["score"])


def test_every_prefix_of_the_output_is_grammatical():
    """No illegal token may ever be spliced into a beam."""
    from inference.grammar import is_valid_prefix

    _, strings, _ = run_oracle(SIMPLE)
    for i in range(1, len(strings) + 1):
        assert is_valid_prefix(strings[:i]), f"illegal prefix at {i}: {strings[:i]}"


def test_single_legal_continuation_is_not_padded():
    """After NODE:FRAC exactly one token is legal (STRUCT:OPEN).

    With beam_size=5 the old code filled the other four slots with masked
    tokens. Drive one step and confirm the search still lands on gold.
    """
    result, strings, _ = run_oracle(SIMPLE, beam_size=5, max_len=2)
    assert strings == SIMPLE[:2]
    assert math.isfinite(result["score"])


# -- legitimate repetition ---------------------------------------------------

def test_multi_term_answer_survives():
    """3x^2 + 5x - 7 repeats NODE:TERM/COEF/VAR/STRUCT:SEP legitimately."""
    result, strings, _ = run_oracle(MULTI_TERM)
    assert result["status"] == "solved"
    assert strings == MULTI_TERM
    assert len(deserialize_slang_math(strings)["numi"]["terms"]) == 3


# -- search width ------------------------------------------------------------

@pytest.mark.parametrize("beam_size", [1, 2, 4, 5, 8])
def test_all_beam_widths_terminate(beam_size):
    result, strings, _ = run_oracle(SIMPLE, beam_size=beam_size)
    assert result["status"] == "solved"
    assert strings == SIMPLE


# -- early exit --------------------------------------------------------------

def test_search_exits_before_exhausting_max_len():
    """Beams must retire on [EOS] instead of grinding out the full budget.

    Running every beam to max_len is what made beam_size=5 cost ~19s/problem;
    the forward passes, not the grammar mask, dominate that.
    """
    _, _, model = run_oracle(SIMPLE, beam_size=5, max_len=64)
    assert model.forward_calls < 5 * 64


def test_generous_max_len_does_not_change_the_answer():
    _, strings, _ = run_oracle(SIMPLE, max_len=128)
    assert strings == SIMPLE


# -- liveness ----------------------------------------------------------------

def test_looping_model_still_returns():
    """A degenerate model must return a well-formed result, not hang."""
    result = beam_search(
        model=LoopingModel(),
        src_tokens=torch.zeros(1, 8, dtype=torch.long),
        vocab_map=VOCAB,
        beam_size=2,
        max_len=16,
    )
    assert result["status"] in ("solved", "partial")
    assert len(result["tokens"]) <= 17  # [BOS] + max_len


# -- node-type reachability (task 2a follow-up) ------------------------------
# NODE:GRADIENT had no branch in the grammar, so it was masked out at every
# position and no gradient answer could be generated at any beam width. That
# is why gradient scores 0/50 in docs/EVAL_RESULTS.md independent of the
# model. These go red without the grammar's gradient branch.

from tests.unit.slang_builders import GRADIENT_2D  # noqa: E402


def test_gradient_answer_is_reachable():
    # GRADIENT_2D is 38 tokens -- one fraction per variable. max_len is the
    # decoder's step budget, so it must exceed the target length or the
    # search correctly returns the best COMPLETED answer (the x-component
    # alone) rather than a truncated one.
    result, strings, _ = run_oracle(GRADIENT_2D, max_len=64)
    assert result["status"] == "solved"
    assert strings == GRADIENT_2D


def test_gradient_answer_deserializes():
    _, strings, _ = run_oracle(GRADIENT_2D, max_len=64)
    grad = deserialize_slang_math(strings)
    assert sorted(grad) == ["x", "y"]


def test_decorated_op_node_is_reachable():
    """OP:sec COEF:-9 EXP:2 ... -- ~13% of real targets carry a decorator."""
    ast = (
        ["OP:sec", "COEF:-9", "EXP:2", "STRUCT:OPEN"]
        + SIMPLE
        + ["STRUCT:CLOSE"]
    )
    result, strings, _ = run_oracle(ast)
    assert result["status"] == "solved"
    assert strings == ast


# -- deserializable-output fallback ------------------------------------------

def test_salvage_returns_a_closed_sub_ast():
    """When no beam terminates, hand back the longest CLOSED prefix rather
    than a truncated sequence the deserializer cannot parse. Only tokens the
    model actually produced are returned -- nothing is fabricated.
    """
    from inference.decoding import longest_complete_prefix
    from inference.grammar import is_complete

    ids = [VOCAB["token_to_id"]["[BOS]"]] + [
        VOCAB["token_to_id"][t] for t in SIMPLE + ["NODE:FRAC", "STRUCT:OPEN"]
    ]
    salvaged = longest_complete_prefix(ids, VOCAB["id_to_token"])
    strings = [VOCAB["id_to_token"][t] for t in salvaged][1:]
    assert is_complete(strings)
    assert strings == SIMPLE


def test_salvage_returns_empty_when_nothing_closed():
    """A sequence whose outermost node never closes has no complete prefix."""
    from inference.decoding import longest_complete_prefix

    ids = [VOCAB["token_to_id"]["[BOS]"]] + [
        VOCAB["token_to_id"][t] for t in ["NODE:FRAC", "STRUCT:OPEN", "STRUCT:NUMI"]
    ]
    assert longest_complete_prefix(ids, VOCAB["id_to_token"]) == []
