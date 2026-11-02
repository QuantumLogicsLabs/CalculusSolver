import pytest
import sympy as sp
from function_sequences import CATALOG, KIND, analyze, run_example, x, n


@pytest.mark.parametrize("key", list(CATALOG))
def test_catalog_verdicts(key):
    r = run_example(key)
    assert r.kind == KIND
    assert r.pointwise_ok
    assert r.uniform == CATALOG[key]["uniform"]


def test_xn_discontinuous_limit_disproves_uniform():
    r = run_example("xn_on_0_1")
    assert r.fn_continuous and not r.limit_continuous
    assert "CANNOT be uniform" in r.continuity_verdict
    assert r.sup_norm_limit == "1"


def test_continuous_limit_is_not_enough():
    r = run_example("n_x_exp_on_0_inf")
    assert r.limit_continuous and not r.uniform
    assert "does NOT" in r.continuity_verdict


def test_auto_limit_and_string_input():
    r = analyze("x/(n+1)", sp.Interval(0, 1))
    assert r.uniform and r.limit_function.startswith("0")


def test_wrong_limit_is_rejected():
    r = analyze(x / n, sp.Interval(0, 1), f=1)
    assert not r.pointwise_ok and not r.uniform
    assert r.verdict == "Not convergent to the given f"


def test_numeric_check_matches_symbolic():
    r = run_example("xn_on_0_half")
    assert abs(r.numeric_check[1][1] - 0.5 ** 10) < 1e-9
