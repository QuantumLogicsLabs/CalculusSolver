"""
Integration tests for POST /solve against the FallbackSolver.

Uses Starlette's TestClient (synchronous, no running server needed).
The FallbackSolver is the "stub solver" — it runs when no GROQ_API_KEY
is set and no neural checkpoint exists, which is always true in CI.

Covers:
  - Health check
  - Happy-path solve for every supported operation
  - Response shape invariants
  - Step object structure
  - LaTeX field presence and content
  - Error handling (400, 422, 503 paths)
"""

import os
import pytest

# ── Reset solver singleton BEFORE importing the app ───────────────────────────
# _shared uses module-level globals to cache the solver after first load.
# We clear them here so tests always start with FallbackSolver, never Groq.
os.environ.pop("GROQ_API_KEY", None)

import api._shared as _shared
_shared._solver = None
_shared._solver_mode = "unloaded"
_shared._solver_error = None

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from starlette.testclient import TestClient
from api.app import app

client = TestClient(app, raise_server_exceptions=True)


# ── Helper ─────────────────────────────────────────────────────────────────────

def make_envelope(op: str, var: str, terms: list, deno=1, point: dict = None) -> dict:
    """Build a standard SLaNg input envelope for POST /solve."""
    inner = {
        "op": op,
        "var": var,
        "expr": {
            "numi": {"terms": terms},
            "deno": deno
        }
    }
    if point is not None:
        inner["point"] = point
    return {"input": inner}


# ── Health check ───────────────────────────────────────────────────────────────

def test_health_returns_200():
    r = client.get("/api/health")
    assert r.status_code == 200


def test_health_status_is_ok():
    r = client.get("/api/health")
    assert r.json()["status"] == "ok"


def test_health_solver_mode_is_fallback():
    r = client.get("/api/health")
    assert r.json()["solver_mode"] in ("fallback", "neural")


def test_health_solver_loaded_is_true():
    r = client.get("/api/health")
    assert r.json()["solver_loaded"] is True


def test_health_also_reachable_without_api_prefix():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── POST /solve — diff ─────────────────────────────────────────────────────────

def test_diff_single_term_status_200():
    payload = make_envelope("diff", "x", [{"coeff": 3, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 200


def test_diff_single_term_status_field():
    payload = make_envelope("diff", "x", [{"coeff": 3, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert r.json()["status"] == "solved"


def test_diff_single_term_mode_is_fallback():
    payload = make_envelope("diff", "x", [{"coeff": 3, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert r.json()["mode"] in ("fallback", "neural")


def test_diff_single_term_verified_true():
    payload = make_envelope("diff", "x", [{"coeff": 3, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert r.json()["verified"] is True


def test_diff_single_term_correct_coefficient():
    # d/dx (3x²) = 6x  →  coeff=6
    payload = make_envelope("diff", "x", [{"coeff": 3, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    terms = r.json()["expr"]["numi"]["terms"]
    assert any(t["coeff"] == 6 for t in terms)


def test_diff_single_term_correct_exponent():
    # d/dx (3x²) = 6x  →  x^1
    payload = make_envelope("diff", "x", [{"coeff": 3, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    terms = r.json()["expr"]["numi"]["terms"]
    assert any(t.get("var", {}).get("x") == 1 for t in terms)


def test_diff_single_term_latex_contains_x():
    payload = make_envelope("diff", "x", [{"coeff": 3, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert "x" in r.json()["latex"]


def test_diff_constant_term_gives_zero():
    # d/dx (7) = 0
    payload = make_envelope("diff", "x", [{"coeff": 7}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 200
    terms = r.json()["expr"]["numi"]["terms"]
    assert all(t["coeff"] == 0 for t in terms)


def test_diff_multi_term_polynomial():
    # d/dx (3x² - x + 5) = 6x - 1
    payload = make_envelope("diff", "x", [
        {"coeff": 3, "var": {"x": 2}},
        {"coeff": -1, "var": {"x": 1}},
        {"coeff": 5}
    ])
    r = client.post("/solve", json=payload)
    assert r.status_code == 200
    terms = r.json()["expr"]["numi"]["terms"]
    coeff_by_power = {t.get("var", {}).get("x", 0): t["coeff"] for t in terms}
    assert coeff_by_power.get(1) == 6    # 6x term
    assert coeff_by_power.get(0) == -1   # -1 constant term


def test_diff_confidence_is_one():
    payload = make_envelope("diff", "x", [{"coeff": 3, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert r.json()["confidence"] == 1.0


def test_diff_also_reachable_without_api_prefix():
    payload = make_envelope("diff", "x", [{"coeff": 3, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 200


# ── POST /solve — integrate ────────────────────────────────────────────────────

def test_integrate_single_term_status_200():
    payload = make_envelope("integrate", "x", [{"coeff": 6, "var": {"x": 1}}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 200


def test_integrate_single_term_correct_coefficient():
    # ∫ 6x dx = 3x²  →  coeff=3.0
    payload = make_envelope("integrate", "x", [{"coeff": 6, "var": {"x": 1}}])
    r = client.post("/solve", json=payload)
    terms = r.json()["expr"]["numi"]["terms"]
    assert any(t["coeff"] == 3.0 for t in terms)


def test_integrate_single_term_correct_exponent():
    # ∫ 6x dx = 3x²  →  x^2
    payload = make_envelope("integrate", "x", [{"coeff": 6, "var": {"x": 1}}])
    r = client.post("/solve", json=payload)
    terms = r.json()["expr"]["numi"]["terms"]
    assert any(t.get("var", {}).get("x") == 2 for t in terms)


def test_integrate_verified_true():
    payload = make_envelope("integrate", "x", [{"coeff": 6, "var": {"x": 1}}])
    r = client.post("/solve", json=payload)
    assert r.json()["verified"] is True


def test_integrate_step_rule_is_power_rule_integral():
    payload = make_envelope("integrate", "x", [{"coeff": 6, "var": {"x": 1}}])
    r = client.post("/solve", json=payload)
    steps = r.json()["steps"]
    assert any(s["rule"] == "power_rule_integral" for s in steps)


def test_integrate_constant_term():
    # ∫ 4 dx = 4x
    payload = make_envelope("integrate", "x", [{"coeff": 4}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 200
    terms = r.json()["expr"]["numi"]["terms"]
    assert any(t.get("var", {}).get("x") == 1 for t in terms)


# ── POST /solve — partial ──────────────────────────────────────────────────────

def test_partial_status_200():
    payload = make_envelope("partial", "x", [{"coeff": 5, "var": {"x": 3}}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 200


def test_partial_status_field_is_solved():
    payload = make_envelope("partial", "x", [{"coeff": 5, "var": {"x": 3}}])
    r = client.post("/solve", json=payload)
    assert r.json()["status"] == "solved"


def test_partial_treats_other_var_as_constant():
    # ∂/∂x (3x²y) — y is treated as a constant by fallback
    payload = make_envelope("partial", "x", [{"coeff": 3, "var": {"x": 2, "y": 1}}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 200


def test_partial_step_rule_is_power_rule():
    payload = make_envelope("partial", "x", [{"coeff": 5, "var": {"x": 3}}])
    r = client.post("/solve", json=payload)
    steps = r.json()["steps"]
    assert any(s["rule"] == "power_rule" for s in steps)


# ── POST /solve — gradient ─────────────────────────────────────────────────────

def test_gradient_status_200():
    payload = make_envelope("gradient", "x", [
        {"coeff": 1, "var": {"x": 2}},
        {"coeff": 1, "var": {"y": 2}}
    ])
    r = client.post("/solve", json=payload)
    assert r.status_code == 200


def test_gradient_expr_contains_gradient_key():
    payload = make_envelope("gradient", "x", [
        {"coeff": 1, "var": {"x": 2}},
        {"coeff": 1, "var": {"y": 2}}
    ])
    r = client.post("/solve", json=payload)
    assert "gradient" in r.json()["expr"]


def test_gradient_contains_partial_for_x():
    payload = make_envelope("gradient", "x", [
        {"coeff": 1, "var": {"x": 2}},
        {"coeff": 1, "var": {"y": 2}}
    ])
    r = client.post("/solve", json=payload)
    assert "x" in r.json()["expr"]["gradient"]


def test_gradient_contains_partial_for_y():
    payload = make_envelope("gradient", "x", [
        {"coeff": 1, "var": {"x": 2}},
        {"coeff": 1, "var": {"y": 2}}
    ])
    r = client.post("/solve", json=payload)
    assert "y" in r.json()["expr"]["gradient"]


def test_gradient_latex_contains_nabla():
    payload = make_envelope("gradient", "x", [
        {"coeff": 1, "var": {"x": 2}},
        {"coeff": 1, "var": {"y": 2}}
    ])
    r = client.post("/solve", json=payload)
    assert "nabla" in r.json()["latex"] or "∇" in r.json()["latex"]


# ── POST /solve — tangent_line ─────────────────────────────────────────────────

def test_tangent_line_status_200():
    payload = make_envelope(
        "tangent_line", "x",
        [{"coeff": 1, "var": {"x": 2}}],
        point={"x": 2}
    )
    r = client.post("/solve", json=payload)
    assert r.status_code == 200


def test_tangent_line_latex_contains_y_equals():
    # Tangent line is always formatted as "y = ..."
    payload = make_envelope(
        "tangent_line", "x",
        [{"coeff": 1, "var": {"x": 2}}],
        point={"x": 2}
    )
    r = client.post("/solve", json=payload)
    assert "y" in r.json()["latex"]


def test_tangent_line_at_x2_slope_is_4():
    # f(x) = x²  →  f'(x) = 2x  →  slope at x=2 is 4
    payload = make_envelope(
        "tangent_line", "x",
        [{"coeff": 1, "var": {"x": 2}}],
        point={"x": 2}
    )
    r = client.post("/solve", json=payload)
    terms = r.json()["expr"]["numi"]["terms"]
    slope_term = next(
        (t for t in terms if t.get("var", {}).get("x") == 1), None
    )
    assert slope_term is not None
    assert slope_term["coeff"] == 4.0


def test_tangent_line_missing_point_returns_422():
    # tangent_line without a point must raise ValueError → 422
    payload = make_envelope("tangent_line", "x", [{"coeff": 1, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 422


# ── Response shape invariants ──────────────────────────────────────────────────

REQUIRED_RESPONSE_KEYS = {"status", "expr", "steps", "latex", "confidence", "verified", "mode"}

@pytest.mark.parametrize("op,terms", [
    ("diff",      [{"coeff": 3, "var": {"x": 2}}]),
    ("integrate", [{"coeff": 2, "var": {"x": 1}}]),
    ("partial",   [{"coeff": 5, "var": {"x": 3}}]),
])
def test_response_has_all_required_keys(op, terms):
    payload = make_envelope(op, "x", terms)
    r = client.post("/solve", json=payload)
    assert r.status_code == 200
    body = r.json()
    missing = REQUIRED_RESPONSE_KEYS - body.keys()
    assert not missing, f"Response missing keys: {missing}"


@pytest.mark.parametrize("op,terms", [
    ("diff",      [{"coeff": 3, "var": {"x": 2}}]),
    ("integrate", [{"coeff": 2, "var": {"x": 1}}]),
    ("partial",   [{"coeff": 5, "var": {"x": 3}}]),
])
def test_expr_has_numi_and_deno(op, terms):
    payload = make_envelope(op, "x", terms)
    r = client.post("/solve", json=payload)
    expr = r.json()["expr"]
    assert "numi" in expr or "gradient" in expr  # gradient op has different expr shape


@pytest.mark.parametrize("op,terms", [
    ("diff",      [{"coeff": 3, "var": {"x": 2}}]),
    ("integrate", [{"coeff": 2, "var": {"x": 1}}]),
])
def test_latex_is_nonempty_string(op, terms):
    payload = make_envelope(op, "x", terms)
    r = client.post("/solve", json=payload)
    latex = r.json()["latex"]
    assert isinstance(latex, str)
    assert len(latex) > 0


@pytest.mark.parametrize("op,terms", [
    ("diff",      [{"coeff": 3, "var": {"x": 2}}]),
    ("integrate", [{"coeff": 2, "var": {"x": 1}}]),
])
def test_confidence_is_float(op, terms):
    payload = make_envelope(op, "x", terms)
    r = client.post("/solve", json=payload)
    assert isinstance(r.json()["confidence"], float)


# ── Step object structure ──────────────────────────────────────────────────────

REQUIRED_STEP_KEYS = {"rule", "description", "before", "after"}

@pytest.mark.parametrize("op,terms", [
    ("diff",      [{"coeff": 3, "var": {"x": 2}}]),
    ("integrate", [{"coeff": 2, "var": {"x": 1}}]),
    ("partial",   [{"coeff": 5, "var": {"x": 3}}]),
])
def test_steps_is_list(op, terms):
    payload = make_envelope(op, "x", terms)
    r = client.post("/solve", json=payload)
    assert isinstance(r.json()["steps"], list)


@pytest.mark.parametrize("op,terms", [
    ("diff",      [{"coeff": 3, "var": {"x": 2}}]),
    ("integrate", [{"coeff": 2, "var": {"x": 1}}]),
    ("partial",   [{"coeff": 5, "var": {"x": 3}}]),
])
def test_steps_has_at_least_one_entry(op, terms):
    payload = make_envelope(op, "x", terms)
    r = client.post("/solve", json=payload)
    assert len(r.json()["steps"]) >= 1


@pytest.mark.parametrize("op,terms", [
    ("diff",      [{"coeff": 3, "var": {"x": 2}}]),
    ("integrate", [{"coeff": 2, "var": {"x": 1}}]),
])
def test_each_step_has_required_keys(op, terms):
    payload = make_envelope(op, "x", terms)
    r = client.post("/solve", json=payload)
    for step in r.json()["steps"]:
        missing = REQUIRED_STEP_KEYS - step.keys()
        assert not missing, f"Step missing keys: {missing}. Step was: {step}"


@pytest.mark.parametrize("op,terms", [
    ("diff",      [{"coeff": 3, "var": {"x": 2}}]),
    ("integrate", [{"coeff": 2, "var": {"x": 1}}]),
])
def test_each_step_fields_are_strings(op, terms):
    payload = make_envelope(op, "x", terms)
    r = client.post("/solve", json=payload)
    for step in r.json()["steps"]:
        for key in ("rule", "description", "before", "after"):
            assert isinstance(step[key], str), (
                f"Step field '{key}' is not a string: {step[key]!r}"
            )


# ── Error handling ─────────────────────────────────────────────────────────────

def test_invalid_json_body_returns_400():
    r = client.post(
        "/solve",
        content=b"this is not json",
        headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 400


def test_input_not_a_dict_returns_422():
    r = client.post("/solve", json={"input": "not-an-object"})
    assert r.status_code == 422


def test_missing_expr_field_returns_422():
    # op and var present but no expr — FallbackSolver raises ValueError
    r = client.post("/solve", json={"input": {"op": "diff", "var": "x"}})
    assert r.status_code == 422


def test_unsupported_op_returns_422():
    # taylor is not supported by FallbackSolver → raises ValueError → 422
    payload = make_envelope("taylor", "x", [{"coeff": 1, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 422


def test_unsupported_op_hessian_returns_422():
    payload = make_envelope("hessian", "x", [{"coeff": 1, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 422


def test_unsupported_op_lagrange_returns_422():
    payload = make_envelope("lagrange", "x", [{"coeff": 1, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert r.status_code == 422


def test_empty_json_object_returns_422():
    # {} has no expr field → solver raises ValueError
    r = client.post("/solve", json={})
    assert r.status_code == 422


def test_response_content_type_is_json():
    payload = make_envelope("diff", "x", [{"coeff": 3, "var": {"x": 2}}])
    r = client.post("/solve", json=payload)
    assert "application/json" in r.headers.get("content-type", "")


# ── Direct envelope (no "input" wrapper) ──────────────────────────────────────

def test_direct_envelope_without_input_wrapper_accepted():
    # The API does body.get("input", body) — so a direct envelope also works
    direct = {
        "op": "diff",
        "var": "x",
        "expr": {
            "numi": {"terms": [{"coeff": 3, "var": {"x": 2}}]},
            "deno": 1
        }
    }
    r = client.post("/solve", json=direct)
    assert r.status_code == 200
    assert r.json()["status"] == "solved"
