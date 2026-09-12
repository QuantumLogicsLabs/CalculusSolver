"""Structural bounds on the grammar.

Without these the grammar admits infinitely long valid prefixes, so a
degenerate decoder can keep emitting grammatical tokens forever and be
hard-truncated at max_len into an undeserializable sequence. The beam-level
repeat guards cannot prevent that on their own: a model that cycles through
distinct tokens repeats neither a token nor a bigram.

Every bound is set from the dataset with large margin. The companion test at
the bottom fails loudly if real data ever outgrows one.
"""

import json
from pathlib import Path

import pytest

from inference.grammar import (
    MAX_NESTING_DEPTH,
    MAX_OPVARS_PER_OP,
    MAX_SIBLINGS,
    is_complete,
    is_valid_prefix,
)
from tests.unit.slang_builders import frac, term, term_list

ROOT = Path(__file__).resolve().parents[2]
NEST_UNIT = ["NODE:FRAC", "STRUCT:OPEN", "STRUCT:NUMI", "STRUCT:OPEN"]


# ── OPVAR run ────────────────────────────────────────────────────────────────

def test_opvar_run_within_cap_is_allowed():
    assert is_valid_prefix(["OP:hessian"] + ["OPVAR:x"] * MAX_OPVARS_PER_OP) is True


def test_opvar_run_beyond_cap_is_rejected():
    assert is_valid_prefix(
        ["OP:hessian"] + ["OPVAR:x"] * (MAX_OPVARS_PER_OP + 1)
    ) is False


def test_runaway_opvar_run_is_rejected():
    assert is_valid_prefix(["OP:hessian"] + ["OPVAR:x"] * 40) is False


# ── nesting depth ────────────────────────────────────────────────────────────

def test_nesting_within_cap_is_allowed():
    assert is_valid_prefix(NEST_UNIT * (MAX_NESTING_DEPTH - 1)) is True


def test_nesting_beyond_cap_is_rejected():
    assert is_valid_prefix(NEST_UNIT * (MAX_NESTING_DEPTH + 1)) is False


def test_runaway_nesting_is_rejected():
    assert is_valid_prefix(NEST_UNIT * 40) is False


# ── siblings ─────────────────────────────────────────────────────────────────

def test_term_list_within_sibling_cap():
    ast = frac(term_list(*[term(i) for i in range(MAX_SIBLINGS)]), term_list(term(1)))
    assert is_complete(ast) is True


def test_term_list_beyond_sibling_cap_is_rejected():
    ast = frac(
        term_list(*[term(i) for i in range(MAX_SIBLINGS + 1)]), term_list(term(1))
    )
    assert is_complete(ast) is False


# ── the bounds must not touch real data ──────────────────────────────────────

@pytest.mark.parametrize("split", ["test", "val"])
def test_real_targets_fit_inside_every_bound(split):
    """Observed maxima across all 155,000 targets: OPVAR run 1, nesting depth
    3, terms per list 2. If a future generator exceeds a cap, the decoder
    would silently mask out legitimate answers -- fail here instead."""
    path = ROOT / "data" / "splits" / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not present")
    from tokenizer.slang_serializer import serialize_slang_math

    checked = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ast = json.loads(line).get("tgt_output_tokens")
            if ast is None:
                continue
            assert is_complete(serialize_slang_math(ast)), (
                "a real target is no longer accepted by the grammar"
            )
            checked += 1
    assert checked > 0
