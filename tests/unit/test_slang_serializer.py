"""
Unit tests for tokenizer/slang_serializer.py
Tests the serialize_slang_math and deserialize_slang_math round-trip.
"""

import pytest
from tokenizer.slang_serializer import serialize_slang_math, deserialize_slang_math


# ── AST fixtures ───────────────────────────────────────────────────────────────

CONSTANT_FRACTION = {
    "numi": {"terms": [{"coeff": 5}]},
    "deno": 1
}

SIMPLE_FRACTION = {
    "numi": {"terms": [{"coeff": 3, "var": {"x": 2}}]},
    "deno": 1
}

MULTI_TERM_FRACTION = {
    "numi": {"terms": [
        {"coeff": 3, "var": {"x": 2}},
        {"coeff": -1, "var": {"x": 1}},
        {"coeff": 5}
    ]},
    "deno": 1
}

MULTI_VAR_FRACTION = {
    "numi": {"terms": [{"coeff": 2, "var": {"x": 1, "y": 2}}]},
    "deno": 1
}

ZERO_COEFF_FRACTION = {
    "numi": {"terms": [{"coeff": 0}]},
    "deno": 1
}

NEGATIVE_COEFF_FRACTION = {
    "numi": {"terms": [{"coeff": -4, "var": {"x": 3}}]},
    "deno": 1
}

LARGE_COEFF_FRACTION = {
    "numi": {"terms": [{"coeff": 100, "var": {"x": 1}}]},
    "deno": 1
}

OP_NODE_DIFF = {
    "op": "diff",
    "var": "x",
    "expr": {
        "numi": {"terms": [{"coeff": 3, "var": {"x": 2}}]},
        "deno": 1
    }
}

OP_NODE_INTEGRATE = {
    "op": "integrate",
    "var": "x",
    "expr": {
        "numi": {"terms": [{"coeff": 6, "var": {"x": 1}}]},
        "deno": 1
    }
}


# ── Round-trip tests: AST → tokens → AST → tokens (compare token lists) ───────
#
# We compare round-trips by re-serializing the deserialized output and checking
# the token list matches the original. This is more stable than deep dict equality
# because it avoids int/float coeff representation differences.

@pytest.mark.parametrize("ast", [
    pytest.param(CONSTANT_FRACTION,       id="constant"),
    pytest.param(SIMPLE_FRACTION,         id="simple_fraction"),
    pytest.param(MULTI_TERM_FRACTION,     id="multi_term"),
    pytest.param(MULTI_VAR_FRACTION,      id="multi_var"),
    pytest.param(ZERO_COEFF_FRACTION,     id="zero_coeff"),
    pytest.param(NEGATIVE_COEFF_FRACTION, id="negative_coeff"),
    pytest.param(LARGE_COEFF_FRACTION,    id="large_coeff"),
])
def test_fraction_round_trip(ast):
    """serialize → deserialize → re-serialize must produce identical token list."""
    tokens = serialize_slang_math(ast)
    reconstructed = deserialize_slang_math(tokens)
    retokenized = serialize_slang_math(reconstructed)
    assert retokenized == tokens, (
        f"Round-trip failed.\n"
        f"  Original tokens:  {tokens}\n"
        f"  Retokenized:      {retokenized}"
    )


def test_op_node_diff_round_trip():
    """Op node (diff) round-trip preserves op, var, and expr."""
    tokens = serialize_slang_math(OP_NODE_DIFF)
    result = deserialize_slang_math(tokens)
    assert result["op"] == "diff"
    assert result["var"] == "x"
    # Verify the inner expr round-trips correctly too
    expr_tokens_original = serialize_slang_math(OP_NODE_DIFF["expr"])
    expr_tokens_result = serialize_slang_math(result["expr"])
    assert expr_tokens_result == expr_tokens_original


def test_op_node_integrate_round_trip():
    """Op node (integrate) round-trip preserves op, var, and expr."""
    tokens = serialize_slang_math(OP_NODE_INTEGRATE)
    result = deserialize_slang_math(tokens)
    assert result["op"] == "integrate"
    assert result["var"] == "x"
    expr_tokens_original = serialize_slang_math(OP_NODE_INTEGRATE["expr"])
    expr_tokens_result = serialize_slang_math(result["expr"])
    assert expr_tokens_result == expr_tokens_original


# ── Token-level structural assertions ─────────────────────────────────────────

def test_fraction_token_starts_with_node_frac():
    tokens = serialize_slang_math(SIMPLE_FRACTION)
    assert tokens[0] == "NODE:FRAC"


def test_fraction_contains_struct_open():
    # STRUCT:OPEN is the bracket token used by the serializer as the opening
    # bracket inside fraction and op-node structures.
    # It is defined as OPEN = "STRUCT:OPEN" in slang_serializer.py and is
    # present in vocab.json at ID 23 (assigned in vocab v1.1, Fix 3).
    tokens = serialize_slang_math(SIMPLE_FRACTION)
    assert "STRUCT:OPEN" in tokens


def test_fraction_contains_numi_and_deno():
    tokens = serialize_slang_math(SIMPLE_FRACTION)
    assert "STRUCT:NUMI" in tokens
    assert "STRUCT:DENO" in tokens


def test_fraction_contains_close():
    tokens = serialize_slang_math(SIMPLE_FRACTION)
    assert "STRUCT:CLOSE" in tokens


def test_single_term_token_content():
    tokens = serialize_slang_math(SIMPLE_FRACTION)
    assert "NODE:TERM" in tokens
    assert "COEF:3" in tokens
    assert "VAR:x" in tokens
    assert "EXP:2" in tokens


def test_multi_term_has_correct_term_count():
    tokens = serialize_slang_math(MULTI_TERM_FRACTION)
    assert tokens.count("NODE:TERM") == 4


def test_multi_term_has_separators():
    tokens = serialize_slang_math(MULTI_TERM_FRACTION)
    # Two SEPs separating three terms
    assert tokens.count("STRUCT:SEP") >= 2


def test_op_node_token_starts_with_op_prefix():
    tokens = serialize_slang_math(OP_NODE_DIFF)
    assert tokens[0] == "OP:diff"


def test_op_node_token_second_is_opvar():
    tokens = serialize_slang_math(OP_NODE_DIFF)
    assert tokens[1] == "OPVAR:x"


def test_op_node_contains_inner_fraction_tokens():
    tokens = serialize_slang_math(OP_NODE_DIFF)
    # The inner expr is a fraction, so its tokens must appear inside
    assert "NODE:FRAC" in tokens
    assert "NODE:TERM" in tokens
    assert "COEF:3" in tokens


# ── Variable sort order ────────────────────────────────────────────────────────

def test_multi_var_variables_sorted_alphabetically():
    """Variables in a term must be serialized in alphabetical order (x before y)."""
    tokens = serialize_slang_math(MULTI_VAR_FRACTION)
    x_pos = tokens.index("VAR:x")
    y_pos = tokens.index("VAR:y")
    assert x_pos < y_pos, (
        f"Expected VAR:x before VAR:y in token list, got positions {x_pos} and {y_pos}"
    )


# ── Coefficient normalization ──────────────────────────────────────────────────

def test_float_whole_number_coeff_serializes_as_int():
    """A coeff of 3.0 (float but whole) must produce COEF:3, not COEF:3.0."""
    ast = {"numi": {"terms": [{"coeff": 3.0, "var": {"x": 1}}]}, "deno": 1}
    tokens = serialize_slang_math(ast)
    assert "COEF:3" in tokens
    assert "COEF:3.0" not in tokens


def test_fractional_coeff_serializes_as_float():
    """A coeff of 1.5 (non-whole float) must produce COEF:1.5."""
    ast = {"numi": {"terms": [{"coeff": 1.5, "var": {"x": 1}}]}, "deno": 1}
    tokens = serialize_slang_math(ast)
    assert "COEF:1.5" in tokens


def test_negative_coeff_serializes_correctly():
    tokens = serialize_slang_math(NEGATIVE_COEFF_FRACTION)
    assert "COEF:-4" in tokens


def test_zero_coeff_serializes_as_coef_zero():
    tokens = serialize_slang_math(ZERO_COEFF_FRACTION)
    assert "COEF:0" in tokens


# ── Exponent normalization ─────────────────────────────────────────────────────

def test_float_whole_number_exp_serializes_as_int():
    """An exponent of 2.0 must produce EXP:2, not EXP:2.0."""
    ast = {"numi": {"terms": [{"coeff": 1, "var": {"x": 2.0}}]}, "deno": 1}
    tokens = serialize_slang_math(ast)
    assert "EXP:2" in tokens
    assert "EXP:2.0" not in tokens


# ── Token list is a flat list of strings ──────────────────────────────────────

def test_serialize_returns_list():
    result = serialize_slang_math(SIMPLE_FRACTION)
    assert isinstance(result, list)


def test_serialize_returns_list_of_strings():
    result = serialize_slang_math(SIMPLE_FRACTION)
    assert all(isinstance(t, str) for t in result)


def test_serialize_returns_nonempty_list():
    result = serialize_slang_math(SIMPLE_FRACTION)
    assert len(result) > 0


# ── Error cases: serialize ────────────────────────────────────────────────────

def test_serialize_none_raises():
    with pytest.raises((ValueError, TypeError)):
        serialize_slang_math(None)


def test_serialize_unknown_dict_raises():
    """A dict that doesn't match any known SLaNg node shape must raise ValueError."""
    with pytest.raises(ValueError):
        serialize_slang_math({"unknown_key": 42})


def test_serialize_bare_string_raises():
    with pytest.raises((ValueError, TypeError, AttributeError)):
        serialize_slang_math("3x^2")


# ── Error cases: deserialize ──────────────────────────────────────────────────

def test_deserialize_returns_dict():
    tokens = serialize_slang_math(SIMPLE_FRACTION)
    result = deserialize_slang_math(tokens)
    assert isinstance(result, dict)


def test_deserialize_truncated_tokens_raises():
    """An incomplete token list must raise ValueError, not silently return garbage."""
    with pytest.raises(ValueError):
        deserialize_slang_math(["NODE:FRAC", "STRUCT:OPEN"])


def test_deserialize_extra_trailing_tokens_raises():
    """Extra tokens after a complete node must raise ValueError."""
    tokens = serialize_slang_math(SIMPLE_FRACTION)
    with pytest.raises(ValueError):
        deserialize_slang_math(tokens + ["NODE:TERM", "COEF:1"])


def test_deserialize_empty_list_raises():
    with pytest.raises((ValueError, IndexError)):
        deserialize_slang_math([])


def test_deserialize_wrong_opening_token_raises():
    with pytest.raises(ValueError):
        deserialize_slang_math(["COEF:3", "VAR:x", "EXP:2"])


def test_deserialize_non_list_raises():
    with pytest.raises((ValueError, TypeError)):
        deserialize_slang_math("NODE:FRAC STRUCT:OPEN")


# ── Idempotency ───────────────────────────────────────────────────────────────

def test_double_round_trip_is_stable():
    """Two full round-trips must produce the same token list as one."""
    tokens_1 = serialize_slang_math(SIMPLE_FRACTION)
    round_1 = deserialize_slang_math(tokens_1)
    tokens_2 = serialize_slang_math(round_1)
    round_2 = deserialize_slang_math(tokens_2)
    tokens_3 = serialize_slang_math(round_2)
    assert tokens_1 == tokens_2 == tokens_3
