"""Binding-contrast pairs: same expression, different variable, consistent targets.

The expected derivative is computed here by an independent implementation
(not the generator's helper) so the test cannot agree with a bug by sharing it.
"""

import json
import random
from collections import Counter

import pytest

import problem_generator as G
from training.contrast import ContrastBatchSource, build_pair_index, merge_batches

import torch


def _independent_derivative(expr, var):
    """d/dvar of a polynomial given as {'numi': {'terms': [...]}}, as {powers: coeff}."""
    out = Counter()
    for term in expr["numi"]["terms"]:
        powers = dict(term.get("var", {}))
        p = powers.get(var, 0)
        if p == 0:
            continue
        coeff = term["coeff"] * p
        powers[var] = p - 1
        key = tuple(sorted((v, q) for v, q in powers.items() if q != 0))
        out[key] += coeff
    return {k: c for k, c in out.items() if c != 0}


def _as_poly(expr):
    out = Counter()
    for term in expr["numi"]["terms"]:
        key = tuple(sorted((v, q) for v, q in term.get("var", {}).items() if q != 0))
        out[key] += term["coeff"]
    return {k: c for k, c in out.items() if c != 0}


def _pairs(n=300, seed=7):
    rng = random.Random(seed)
    made = []
    while len(made) < n:
        item = G.generate_binding_contrast_pair(rng)
        if item is not None:
            made.append(item)
    return made


def test_pair_keeps_expression_and_changes_variable_and_target():
    for item in _pairs():
        (var_a, ans_a), (var_b, ans_b) = item["pair"]
        assert var_a != var_b
        assert ans_a != ans_b
        expr_vars = {v for t in item["expr"]["numi"]["terms"] for v in t.get("var", {})}
        assert {var_a, var_b} <= expr_vars


def test_targets_are_the_correct_derivatives():
    for item in _pairs():
        for var, ans in item["pair"]:
            assert _as_poly(ans) == _independent_derivative(item["expr"], var), (item["expr"], var)


def test_target_vanishing_terms_follow_the_bound_variable():
    """Terms free of the bound variable must not leak into the answer."""
    for item in _pairs():
        for var, ans in item["pair"]:
            n_dependent = sum(1 for t in item["expr"]["numi"]["terms"] if var in t.get("var", {}))
            assert 1 <= len(ans["numi"]["terms"]) <= n_dependent


def test_pair_answers_stay_inside_the_vocabulary():
    for item in _pairs():
        for _, ans in item["pair"]:
            assert all(G._term_in_vocab(t) for t in ans["numi"]["terms"])


def test_pair_covers_shapes_and_mixed_terms():
    pairs = _pairs(600)
    assert {p["shape"] for p in pairs} == set(G.CONTRAST_SHAPES)
    assert any(
        len(t.get("var", {})) == 2 for p in pairs for t in p["expr"]["numi"]["terms"]
