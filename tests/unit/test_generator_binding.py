"""Invariants for the diff/partial problem generators.

Driven by a failure analysis of checkpoints/final/best.pt on the 300-problem
benchmark (diff 86.2%, partial 35.0%):

  * All 11 diff failures were multi-term; 0 of 50 single-term problems failed.
    Failure rate rose with term count (2 terms 28.6%, 3 terms 43.8%).
  * 10 of 11 had every wrong coefficient explained by c_i * p_j with i != j --
    the model paired a term's coefficient with ANOTHER term's exponent. The
    11th had the right coefficients in the wrong pairing.
  * 27 of 39 partial failures used the wrong variable's term entirely.

The data cause: 14.06% of multi-term diff rows were binding-ambiguous (a
cross-pairing reproduced the same answer), and the differentiated variable's
term sat first in 100% of partial rows, so position alone identified it.
These tests keep both properties fixed.
"""

import collections
import itertools
import random

import pytest

import problem_generator as G

VARS = G.VARIABLES


def _is_ambiguous(pairs):
    pairs = [(c, p) for c, p in pairs if p]
    if len(pairs) < 2:
        return False
    coeffs = [c for c, _ in pairs]
    powers = [p for _, p in pairs]
    correct = sorted((c * p, p - 1) for c, p in pairs)
    for perm in itertools.permutations(range(len(pairs))):
        if all(perm[i] == i for i in range(len(pairs))):
            continue
        alt = sorted((coeffs[i] * powers[perm[i]], powers[perm[i]] - 1)
                     for i in range(len(pairs)))
        if alt == correct:
            return True
    return False


# -- the helper itself --------------------------------------------------------

def test_shared_coefficient_is_ambiguous():
    """Two terms with the same coefficient: swapping exponents is undetectable."""
    assert G._binding_is_unambiguous([(3, 2), (3, 4)]) is False


def test_distinct_coefficients_are_unambiguous():
    assert G._binding_is_unambiguous([(3, 2), (5, 4)]) is True


def test_single_term_is_trivially_unambiguous():
    assert G._binding_is_unambiguous([(3, 2)]) is True
    assert G._binding_is_unambiguous([]) is True


# -- generator invariants -----------------------------------------------------

@pytest.fixture(autouse=True)
def _seed():
    random.seed(1234)


def test_multi_term_diff_never_emits_an_ambiguous_binding():
    checked = 0
    for _ in range(600):
        var = random.choice(VARS)
        result = G.generate_multi_term_diff(var)
        if result is None:
            continue  # rejected by design
        checked += 1
        pairs = G._term_pairs(result[0][0]["numi"]["terms"], var)
        assert not _is_ambiguous(pairs), f"ambiguous sample emitted: {pairs}"
    assert checked > 100, "generator produced too few samples to be meaningful"


def test_multivar_diff_never_emits_an_ambiguous_binding():
    """The binding that determines the answer is coefficient -> exponent of
    the DIFFERENTIATED variable. A mixed term's second variable rides along
    untouched and is not part of it, so it is excluded from the check."""
    checked = 0
    for _ in range(600):
        result = G.generate_multivar_diff()
        if result is None:
            continue
        src, _, var, _ = result
        checked += 1
        terms = src[0]["numi"]["terms"]
        pairs = G._term_pairs(terms, var)
        assert not _is_ambiguous(pairs), f"ambiguous sample emitted: {pairs}"
    assert checked > 100


def test_multivar_diff_does_not_always_put_the_target_variable_first():
    """Previously 100% first, so the model never had to read the op's var."""
    positions = []
    for _ in range(600):
        result = G.generate_multivar_diff()
        if result is None:
            continue
        src, _, var, _ = result
        terms = src[0]["numi"]["terms"]
        idx = [i for i, t in enumerate(terms) if var in t.get("var", {})]
        if idx:
            positions.append(idx[0])
    assert positions
    first_share = positions.count(0) / len(positions)
    assert first_share < 0.6, f"target variable is first in {first_share:.0%} of rows"
    assert len(set(positions)) >= 3, "target variable never moves beyond two slots"


def test_multivar_diff_emits_mixed_variable_terms():
    """A term carrying two variables (3*x^2*y) forces the model to keep the
    other variable in the answer. The dataset previously had zero such terms
    anywhere, so this generalisation was never trained or tested."""
    mixed = 0
    seen = 0
    for _ in range(600):
        result = G.generate_multivar_diff()
        if result is None:
            continue
        seen += 1
        terms = result[0][0]["numi"]["terms"]
        if any(len(t.get("var", {})) > 1 for t in terms):
            mixed += 1
    assert seen > 100
    assert mixed / seen > 0.05, f"only {mixed}/{seen} rows carry a mixed term"


def test_mixed_term_derivative_keeps_the_other_variable():
    """d/dx(3*x^2*y) = 6*x*y, not 6x."""
    term = {"coeff": 3, "var": {"x": 2, "y": 1}}
    assert G._differentiate_term(term, "x") == {"coeff": 6, "var": {"x": 1, "y": 1}}
    # exponent falling to zero drops the variable, others survive
    assert G._differentiate_term({"coeff": 4, "var": {"x": 1, "y": 2}}, "x") == {
        "coeff": 4, "var": {"y": 2}
    }
    # a term without the variable vanishes
    assert G._differentiate_term({"coeff": 5, "var": {"y": 2}}, "x") is None


def test_multivar_diff_sometimes_puts_the_variable_in_two_terms():
    """Previously the differentiated variable appeared in exactly one term in
    100% of rows, so the answer was always a single term."""
    counts = collections.Counter()
    for _ in range(600):
        result = G.generate_multivar_diff()
        if result is None:
            continue
        src, _, var, _ = result
        terms = src[0]["numi"]["terms"]
        counts[sum(1 for t in terms if var in t.get("var", {}))] += 1
    assert counts[2] > 0, f"variable never appears in two terms: {dict(counts)}"


def test_multivar_diff_always_has_at_least_two_variables():
    for _ in range(400):
        result = G.generate_multivar_diff()
        if result is None:
            continue
        terms = result[0][0]["numi"]["terms"]
        assert len({v for t in terms for v in t.get("var", {})}) >= 2


def test_multivar_diff_coefficients_and_exponents_are_distinct():
    """The constructive guarantee behind unambiguous binding: with all
    coefficients and exponents distinct, c_i * p_j can never equal c_i * p_i."""
    for _ in range(400):
        result = G.generate_multivar_diff()
        if result is None:
            continue
        terms = result[0][0]["numi"]["terms"]
        coeffs = [t.get("coeff", 0) for t in terms]
        exps = [p for t in terms for p in t.get("var", {}).values()]
        assert len(set(coeffs)) == len(coeffs)
        assert len(set(exps)) == len(exps)


def test_multivar_diff_varies_term_count():
    counts = set()
    for _ in range(600):
        result = G.generate_multivar_diff()
        if result is None:
            continue
        counts.add(len(result[0][0]["numi"]["terms"]))
    assert counts >= {2, 3, 4}, f"term counts seen: {sorted(counts)}"


def test_multivar_diff_answer_matches_the_named_variable():
    """Guards the wrong-variable failure mode directly."""
    for _ in range(300):
        result = G.generate_multivar_diff()
        if result is None:
            continue
        src, ans, var, rule = result
        terms = src[0]["numi"]["terms"]
        # Independent recomputation -- deliberately not calling the
        # generator's own _differentiate_term, so this stays a real check.
        expected = []
        for t in terms:
            powers = t.get("var", {})
            p = powers.get(var, 0)
            if not p:
                continue
            remaining = {v: q for v, q in powers.items() if v != var}
            if p - 1 != 0:
                remaining[var] = p - 1
            term = {"coeff": t["coeff"] * p}
            if remaining:
                term["var"] = dict(sorted(remaining.items()))
            expected.append(term)
        if not expected:
            expected = [{"coeff": 0}]
        assert ans[0]["numi"]["terms"] == expected
        assert rule == G.RULE_ID_PARTIAL


def test_generated_outputs_stay_in_vocabulary():
    for _ in range(300):
        result = G.generate_multivar_diff()
        if result is None:
            continue
        for t in result[1][0]["numi"]["terms"]:
            assert t.get("coeff", 0) in G.SAFE_COEFFS
            for p in t.get("var", {}).values():
                assert p in G.SAFE_EXPONENTS


# -- rule labelling (task 3) --------------------------------------------------

def test_rule_ids_match_vocab_classifier_indices():
    """rule_ids are classifier indices; train.py derives labels by sorting
    rule_tokens on vocab id. These must not drift apart."""
    import json

    with open("tokenizer/vocab.json", encoding="utf-8") as fh:
        rule_tokens = json.load(fh)["rule_tokens"]
    order = [name for name, _ in sorted(rule_tokens.items(), key=lambda kv: kv[1])]
    expected = {
        G.RULE_ID_POWER: "RULE:power_rule",
        G.RULE_ID_SUM: "RULE:sum_rule",
        G.RULE_ID_CONSTANT: "RULE:constant_rule",
        G.RULE_ID_INTEGRAL: "RULE:power_rule_integral",
        G.RULE_ID_PARTIAL: "RULE:partial_derivative",
        G.RULE_ID_TRIG: "RULE:trig_rule",
        G.RULE_ID_EXP: "RULE:exp_rule",
        G.RULE_ID_LOG: "RULE:log_rule",
        G.RULE_ID_GRADIENT: "RULE:gradient",
        G.RULE_ID_TANGENT_LINE: "RULE:tangent_line",
    }
    for rule_id, token in expected.items():
        assert order[rule_id] == token, (
            f"rule_id {rule_id} should be {token}, vocab ordering gives {order[rule_id]}"
        )
