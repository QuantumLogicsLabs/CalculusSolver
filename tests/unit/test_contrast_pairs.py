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
    ), "no mixed-variable terms generated"


def test_target_term_position_varies():
    positions = []
    for item in _pairs(400):
        for var, _ in item["pair"]:
            idx = [i for i, t in enumerate(item["expr"]["numi"]["terms"]) if var in t.get("var", {})]
            positions.append(idx[0])
    assert positions.count(0) / len(positions) < 0.65
    assert len(set(positions)) >= 3


def test_generation_is_deterministic_for_a_seed():
    a = [(p["expr"], p["pair"]) for p in _pairs(50, seed=3)]
    b = [(p["expr"], p["pair"]) for p in _pairs(50, seed=3)]
    assert a == b


def test_dataset_rows_are_grouped_pairs_and_splits_do_not_share_expressions(tmp_path):
    result = G.generate_contrast_dataset(
        train_pairs=60, val_pairs=20, seed=11, splits_dir=str(tmp_path), exclude_main_splits=False
    )
    for split, rows in result.items():
        assert len(rows) % 2 == 0
        groups = Counter(r["contrast_group"] for r in rows)
        assert set(groups.values()) == {2}
        by_group = {}
        for r in rows:
            by_group.setdefault(r["contrast_group"], []).append(r)
        for a, b in by_group.values():
            assert a["src_tokens"]["expr"] == b["src_tokens"]["expr"]
            assert a["src_tokens"]["var"] != b["src_tokens"]["var"]
            assert a["src_tokens"]["var"] == a["contrast_var"]
            assert a["tgt_output_tokens"] != b["tgt_output_tokens"]
            assert a["rule_ids"] == b["rule_ids"] == G.RULE_ID_PARTIAL

    def expr_keys(rows):
        return {json.dumps(r["src_tokens"]["expr"], sort_keys=True) for r in rows}

    assert not expr_keys(result["train"]) & expr_keys(result["val"])
    assert (tmp_path / "contrast_train.jsonl").is_file()
    assert (tmp_path / "contrast_val.jsonl").is_file()


# -- batching -----------------------------------------------------------------

def test_build_pair_index_rejects_half_pairs():
    assert build_pair_index([0, 0, 1, 1, None]) == [(0, 1), (2, 3)]
    with pytest.raises(ValueError):
        build_pair_index([0, 0, 1])


class _ToyDataset:
    def __init__(self, groups):
        self.contrast_groups = groups

    def __len__(self):
        return len(self.contrast_groups)

    def __getitem__(self, i):
        return {"src_seq": torch.tensor([i, i]), "op": "partial", "row": torch.tensor(i)}


def test_contrast_source_always_emits_complete_pairs_and_is_deterministic():
    ds = _ToyDataset([0, 0, 1, 1, 2, 2, 3, 3])
    a = ContrastBatchSource(ds, pairs_per_batch=2, seed=5)
    b = ContrastBatchSource(ds, pairs_per_batch=2, seed=5)
    seen = []
    for _ in range(6):  # wraps the 4-pair pool
        batch_a, batch_b = a.next_batch(), b.next_batch()
        assert torch.equal(batch_a["row"], batch_b["row"])
        rows = batch_a["row"].tolist()
        assert len(rows) == 4
        for i in range(0, 4, 2):
            assert ds.contrast_groups[rows[i]] == ds.contrast_groups[rows[i + 1]]
        seen.extend(rows)
    assert set(seen) == set(range(8))


def test_merge_batches_concatenates_tensors_and_lists():
    main = {"x": torch.zeros(3, 2), "op": ["a", "b", "c"]}
    extra = {"x": torch.ones(2, 2), "op": ["partial", "partial"]}
    merged = merge_batches(main, extra)
    assert merged["x"].shape == (5, 2)
    assert merged["op"] == ["a", "b", "c", "partial", "partial"]
