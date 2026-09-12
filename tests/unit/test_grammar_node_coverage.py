"""Node types the grammar must accept, derived from tokenizer/slang_serializer.py.

The grammar drives the decode mask, so anything it rejects is not merely
scored badly -- it is UNREACHABLE. Before this, is_complete() accepted only
80.9% of real targets: NODE:GRADIENT had no parser branch at all, and
op-nodes could not carry the optional COEF/EXP/POINT decorators that
serialize_op_node emits. That made every gradient answer impossible to
generate, which is why gradient scores 0/50 in docs/EVAL_RESULTS.md
independent of model quality.

Acceptance is now 100.00% of all 155,000 targets in data/splits/.

Pure stdlib -- runs in CI.
"""

import pytest

from inference.grammar import is_complete, is_valid_prefix
from tests.unit.slang_builders import (
    GRADIENT_2D, SIMPLE, frac, gradient, term, term_list,
)

ONE = frac(term_list(term(1)), term_list(term(1)))


# ── gradient node ────────────────────────────────────────────────────────────

def test_gradient_node_is_accepted():
    assert is_complete(GRADIENT_2D) is True


def test_gradient_node_is_reachable_as_a_first_token():
    """The mask must offer NODE:GRADIENT at position 0, or no gradient
    answer can ever be generated."""
    assert is_valid_prefix(["NODE:GRADIENT"]) is True


def test_single_variable_gradient():
    assert is_complete(gradient(("x", ONE))) is True


def test_three_variable_gradient():
    assert is_complete(gradient(("x", ONE), ("y", ONE), ("z", ONE))) is True


def test_gradient_rejects_duplicate_opvar():
    """Gradient serializes a {var: expr} DICT -- a variable cannot repeat."""
    assert is_complete(gradient(("x", ONE), ("x", ONE))) is False


def test_empty_gradient_rejected():
    assert is_complete(["NODE:GRADIENT", "STRUCT:OPEN", "STRUCT:CLOSE"]) is False


def test_gradient_missing_child_rejected():
    assert is_complete(
        ["NODE:GRADIENT", "STRUCT:OPEN", "OPVAR:x", "STRUCT:CLOSE"]
    ) is False


def test_gradient_partial_is_a_valid_prefix():
    for i in range(1, len(GRADIENT_2D)):
        assert is_valid_prefix(GRADIENT_2D[:i]), f"prefix {i} rejected"


# ── op-node decorators ───────────────────────────────────────────────────────

def op(fn, *decorators_and_children):
    head = [f"OP:{fn}"] + list(decorators_and_children)
    return head + ["STRUCT:OPEN"] + ONE + ["STRUCT:CLOSE"]


def test_plain_op_node():
    assert is_complete(op("diff", "OPVAR:x")) is True


def test_op_node_with_coef_decorator():
    """-sin(x) -- {"coeff": -1, "op": "sin"}"""
    assert is_complete(op("sin", "COEF:-8")) is True


def test_op_node_with_coef_and_exp_decorators():
    """sec^2(x) with a scale -- the OP:sec COEF:-9 EXP:2 shape in real data."""
    assert is_complete(op("sec", "COEF:-9", "EXP:2")) is True


def test_op_node_with_point_decorator():
    assert is_complete(op("tangent_line", "POINT:3", "OPVAR:x")) is True


def test_op_node_with_all_decorators():
    assert is_complete(op("sec", "COEF:2", "EXP:2", "POINT:3", "OPVAR:x")) is True


def test_decorator_order_is_enforced():
    """serialize_op_node emits COEF, then EXP, then POINT -- a fixed order
    chosen so parse_op_node stays unambiguous. Accepting permutations would
    loosen the mask beyond what the deserializer handles."""
    assert is_complete(op("sec", "EXP:2", "COEF:-9")) is False
    assert is_complete(op("sec", "POINT:3", "COEF:2")) is False


def test_duplicate_decorator_rejected():
    assert is_complete(op("sin", "COEF:2", "COEF:3")) is False


# ── OOV placeholder tokens ───────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["COEF:OTHER", "EXP:OTHER"])
def test_oov_placeholder_tokens_are_rejected(bad):
    """COEF:OTHER / EXP:OTHER are vocabulary OOV placeholders. They match the
    token prefix but deserialize_slang_math does float() on the value and
    raises, so a decoder allowed to emit one produces an unparseable answer.
    Found by fuzzing the mask against the deserializer.
    """
    prefix = bad.split(":")[0]
    if prefix == "COEF":
        seq = frac(term_list(["NODE:TERM", bad]), term_list(term(1)))
    else:
        seq = frac(term_list(["NODE:TERM", "COEF:1", "VAR:x", bad]), term_list(term(1)))
    assert is_complete(seq) is False


def test_numeric_coef_and_exp_still_accepted():
    assert is_complete(SIMPLE) is True
    assert is_complete(
        frac(term_list(term(-10, ("x", -3))), term_list(term(1)))
    ) is True
