"""Task 2a: a variable may appear at most once per term.

SLaNg stores a term's variables as a JSON dict --
{"coeff": 3, "var": {"x": 2}} -- so a variable can appear at most once per
term by construction. x * x is x^2, not two VAR:x entries. A repeated VAR
is therefore not merely improbable, it is un-representable: verified zero
occurrences across all 155,000 targets in data/splits/.

Enforcing that in the grammar makes the degenerate decode loop observed in
docs/ROOT_CAUSE_REPORT.md -- VAR:u EXP:-3 VAR:u EXP:-3 ... -- structurally
impossible at the mask level, before any scoring runs. That is strictly
stronger than a soft repetition penalty, which can only make the loop
expensive rather than impossible.

Pure stdlib -- runs in CI.
"""

import json
from pathlib import Path

import pytest

from inference.grammar import is_complete, is_valid_prefix
from tests.unit.slang_builders import frac, term, term_list

ROOT = Path(__file__).resolve().parents[2]

OPEN_TERM = ["NODE:FRAC", "STRUCT:OPEN", "STRUCT:NUMI", "STRUCT:OPEN",
             "NODE:TERM", "COEF:1"]


# ── the tightening ───────────────────────────────────────────────────────────

def test_repeated_var_in_one_term_is_rejected():
    assert is_valid_prefix(OPEN_TERM + ["VAR:u", "EXP:-3", "VAR:u"]) is False


def test_repeated_var_kills_the_loop_early():
    """Rejected at the SECOND occurrence, not after it has run away."""
    one = OPEN_TERM + ["VAR:u", "EXP:-3"]
    assert is_valid_prefix(one) is True
    assert is_valid_prefix(one + ["VAR:u", "EXP:-3"]) is False


def test_the_observed_degenerate_loop_is_rejected():
    """The exact shape beam search produced before the guard."""
    loop = OPEN_TERM + ["VAR:u", "EXP:-3"] * 9
    assert is_valid_prefix(loop) is False


def test_repeat_with_a_different_exponent_is_still_rejected():
    """x^2 * x^3 must be written x^5 -- the dict cannot hold both."""
    assert is_valid_prefix(OPEN_TERM + ["VAR:x", "EXP:2", "VAR:x", "EXP:3"]) is False


# ── legitimate repetition must survive ───────────────────────────────────────

def test_distinct_vars_in_one_term_are_fine():
    """3x^2*y*z -- a multivariate term is ordinary."""
    ast = frac(term_list(term(3, ("x", 2), ("y", 1), ("z", 3))), term_list(term(1)))
    assert is_complete(ast) is True


def test_same_var_in_different_terms_is_fine():
    """3x^2 + 5x repeats VAR:x across terms. Scope is per-term, not global."""
    ast = frac(term_list(term(3, ("x", 2)), term(5, ("x", 1))), term_list(term(1)))
    assert is_complete(ast) is True


def test_same_var_in_numerator_and_denominator_is_fine():
    ast = frac(term_list(term(3, ("x", 2))), term_list(term(1, ("x", 1))))
    assert is_complete(ast) is True


# ── the safety net: real data must be unaffected ─────────────────────────────

@pytest.mark.parametrize("split", ["test", "val"])
def test_real_targets_contain_no_duplicate_vars(split):
    """Guards the premise. If a future data generator ever emits a repeated
    VAR in one term, this tightening would start rejecting real targets --
    fail loudly here rather than silently masking them out at decode time.
    """
    path = ROOT / "data" / "splits" / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not present (data/ is not tracked in git)")

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
            tokens = serialize_slang_math(ast)
            seen, in_term = set(), False
            for tok in tokens:
                if tok == "NODE:TERM":
                    in_term, seen = True, set()
                elif tok.startswith("VAR:"):
                    assert not (in_term and tok in seen), (
                        f"duplicate {tok} in one term of {tokens}"
                    )
                    seen.add(tok)
                elif tok in ("STRUCT:SEP", "STRUCT:CLOSE"):
                    in_term, seen = False, set()
            checked += 1
    assert checked > 0
