"""Termination semantics of the SLaNg grammar.

Pure stdlib -- no torch -- so this runs in CI, where only requirements.txt
is installed. This is the regression guard for the bug that made every
neural prediction unusable:

  is_valid_prefix() required the parse to consume every token, so EVERY
  continuation of a closed AST parsed as invalid -- [EOS] included. The
  decoder therefore had no legal way to terminate. In beam_search the
  grammar mask went all -inf the moment a beam got the answer right, and
  that beam was discarded, leaving only degenerate still-open beams. Every
  input returned status="partial" and a truncated sequence that failed
  deserialization with "reached end of tokens".
"""

from inference.grammar import EOS_TOKEN, is_complete, is_valid_prefix
from tests.unit.slang_builders import MULTI_TERM, SIMPLE


# ── the regression ───────────────────────────────────────────────────────────

def test_eos_accepted_after_complete_ast():
    """THE regression. Without this the decoder can never terminate."""
    assert is_valid_prefix(SIMPLE + [EOS_TOKEN]) is True


def test_closed_ast_admits_eos_and_nothing_else():
    """A closed AST must have exactly one legal continuation: [EOS].

    This is the invariant beam_search depends on -- if it ever admits zero
    continuations again, finished beams get masked to all -inf and dropped.
    """
    candidates = [
        EOS_TOKEN, "[PAD]", "[BOS]", "[MASK]", "NODE:TERM", "NODE:FRAC",
        "STRUCT:OPEN", "STRUCT:CLOSE", "STRUCT:SEP", "STRUCT:NUMI",
        "STRUCT:DENO", "COEF:1", "VAR:x", "EXP:1", "OP:diff", "OPVAR:x",
    ]
    legal = [c for c in candidates if is_valid_prefix(SIMPLE + [c])]
    assert legal == [EOS_TOKEN]


# ── is_valid_prefix ──────────────────────────────────────────────────────────

def test_complete_ast_is_a_valid_prefix():
    assert is_valid_prefix(SIMPLE) is True


def test_partial_ast_is_a_valid_prefix():
    assert is_valid_prefix(SIMPLE[:6]) is True


def test_eos_rejected_after_incomplete_ast():
    assert is_valid_prefix(SIMPLE[:6] + [EOS_TOKEN]) is False


def test_bare_eos_rejected():
    assert is_valid_prefix([EOS_TOKEN]) is False


def test_nothing_may_follow_eos():
    assert is_valid_prefix(SIMPLE + [EOS_TOKEN, "NODE:TERM"]) is False
    assert is_valid_prefix(SIMPLE + [EOS_TOKEN, EOS_TOKEN]) is False


def test_trailing_garbage_after_complete_ast_rejected():
    assert is_valid_prefix(SIMPLE + ["NODE:TERM"]) is False


def test_empty_is_a_valid_prefix():
    assert is_valid_prefix([]) is True


def test_garbage_rejected():
    assert is_valid_prefix(["COEF:1"]) is False
    assert is_valid_prefix(["STRUCT:CLOSE"]) is False


# ── is_complete ──────────────────────────────────────────────────────────────

def test_is_complete_on_closed_ast():
    assert is_complete(SIMPLE) is True


def test_is_complete_tolerates_trailing_eos():
    """is_complete(x) and is_valid_prefix(x + [EOS]) must agree."""
    assert is_complete(SIMPLE + [EOS_TOKEN]) is True
    assert is_complete(SIMPLE) == is_valid_prefix(SIMPLE + [EOS_TOKEN])


def test_is_complete_false_on_partial():
    assert is_complete(SIMPLE[:6]) is False


def test_is_complete_false_on_empty():
    assert is_complete([]) is False


# ── legitimate repetition must survive ───────────────────────────────────────

def test_multi_term_ast_is_complete():
    """3x^2 + 5x - 7 repeats NODE:TERM/COEF/VAR/STRUCT:SEP legitimately.

    Guards against an over-tightened grammar or over-eager repetition
    penalty breaking ordinary multi-term answers.
    """
    assert is_complete(MULTI_TERM) is True
    assert is_valid_prefix(MULTI_TERM + [EOS_TOKEN]) is True


# ── RULE: prefix ─────────────────────────────────────────────────────────────

def test_rule_prefixed_ast_completes():
    assert is_complete(["RULE:power_rule"] + SIMPLE) is True
    assert is_valid_prefix(["RULE:power_rule"] + SIMPLE + [EOS_TOKEN]) is True


def test_bare_rule_token_is_a_valid_prefix():
    assert is_valid_prefix(["RULE:power_rule"]) is True
