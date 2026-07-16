"""
Regression test scaffolding for CalculusSolver FallbackSolver.

Design:
  - Every fixture in tests/regression/fixtures/*.json is auto-discovered.
  - Each fixture contains an "input" envelope and an "expected" subset.
  - The test runs solver.solve(input) and asserts each field in "expected".
  - Adding a new regression case = drop a new .json file. No code changes needed.

Comparison strategy for "expr":
  - Never compare expr dicts with == (int/float coeff differences cause false failures).
  - Instead, re-serialize both sides with serialize_slang_math and compare token lists.

Gradient fixtures use "gradient_partials" instead of "expr" because the gradient
result has a different shape: {"gradient": {"x": ..., "y": ...}}.
"""

import json
import os

import pytest

# ── Force FallbackSolver — no Groq, no neural ─────────────────────────────────
os.environ.pop("GROQ_API_KEY", None)

from inference.fallback_solver import FallbackSolver
from tokenizer.slang_serializer import serialize_slang_math

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

solver = FallbackSolver()


# ── Fixture loader ─────────────────────────────────────────────────────────────

def load_fixtures():
    """
    Auto-discover every .json file in the fixtures directory.
    Returns a list of pytest.param objects, one per fixture file.
    Each param is labelled with the filename (minus .json) for clear test IDs.
    """
    cases = []
    for fname in sorted(os.listdir(FIXTURES_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(FIXTURES_DIR, fname)
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        cases.append(pytest.param(data, id=fname.replace(".json", "")))
    return cases


# ── Helpers ────────────────────────────────────────────────────────────────────

def _assert_expr(result_expr: dict, expected_expr: dict, fixture_id: str) -> None:
    """
    Compare two SLaNg expr dicts by re-serializing both to token lists.
    This avoids int vs float coeff false failures (e.g. 3 vs 3.0).
    """
    try:
        result_tokens = serialize_slang_math(result_expr)
        expected_tokens = serialize_slang_math(expected_expr)
    except Exception as exc:
        raise AssertionError(
            f"[{fixture_id}] serialize_slang_math failed during expr comparison: {exc}\n"
            f"  result_expr:   {result_expr}\n"
            f"  expected_expr: {expected_expr}"
        ) from exc
    assert result_tokens == expected_tokens, (
        f"[{fixture_id}] expr mismatch after token comparison.\n"
        f"  result tokens:   {result_tokens}\n"
        f"  expected tokens: {expected_tokens}\n"
        f"  result expr:     {result_expr}\n"
        f"  expected expr:   {expected_expr}"
    )


def _assert_gradient_partials(
    result_expr: dict,
    gradient_partials: dict,
    fixture_id: str
) -> None:
    """
    For gradient results, result_expr has shape {"gradient": {"x": ..., "y": ...}}.
    Compare each partial derivative separately using token-list comparison.
    """
    assert "gradient" in result_expr, (
        f"[{fixture_id}] Expected 'gradient' key in expr, got: {result_expr}"
    )
    result_gradient = result_expr["gradient"]
    for var_name, expected_partial in gradient_partials.items():
        assert var_name in result_gradient, (
            f"[{fixture_id}] Expected partial for variable '{var_name}' "
            f"in gradient result. Found keys: {list(result_gradient.keys())}"
        )
        _assert_expr(result_gradient[var_name], expected_partial, fixture_id)


# ── Main regression test ───────────────────────────────────────────────────────

@pytest.mark.parametrize("case", load_fixtures())
def test_regression(case: dict, request) -> None:
    """
    For every fixture file, run the solver and assert all expected fields.

    Fields checked (when present in "expected"):
      status            — always checked
      verified          — always checked
      latex             — checked when present (exact string match)
      rule              — checked when present (exact string match)
      confidence        — checked when present (exact value match)
      expr              — checked when present (token-list comparison)
      gradient_partials — checked when present (per-variable token-list comparison)
    """
    fixture_id = request.node.callspec.id

    input_env = case.get("input")
    expected = case.get("expected")

    assert input_env is not None, (
        f"[{fixture_id}] Fixture is missing 'input' key."
    )
    assert expected is not None, (
        f"[{fixture_id}] Fixture is missing 'expected' key."
    )

    # ── Run solver ────────────────────────────────────────────────────────────
    try:
        result = solver.solve(input_env)
    except Exception as exc:
        raise AssertionError(
            f"[{fixture_id}] solver.solve() raised an unexpected exception: {exc}\n"
            f"  input: {input_env}"
        ) from exc

    # ── Assert status (always required) ───────────────────────────────────────
    assert "status" in expected, (
        f"[{fixture_id}] Fixture 'expected' must contain 'status'."
    )
    assert result["status"] == expected["status"], (
        f"[{fixture_id}] status mismatch.\n"
        f"  got:      {result['status']}\n"
        f"  expected: {expected['status']}"
    )

    # ── Assert verified (always required) ─────────────────────────────────────
    assert "verified" in expected, (
        f"[{fixture_id}] Fixture 'expected' must contain 'verified'."
    )
    assert result["verified"] == expected["verified"], (
        f"[{fixture_id}] verified mismatch.\n"
        f"  got:      {result['verified']}\n"
        f"  expected: {expected['verified']}"
    )

    # ── Assert latex (optional) ────────────────────────────────────────────────
    if "latex" in expected:
        assert result.get("latex") == expected["latex"], (
            f"[{fixture_id}] latex mismatch.\n"
            f"  got:      {result.get('latex')!r}\n"
            f"  expected: {expected['latex']!r}"
        )

    # ── Assert rule (optional) ────────────────────────────────────────────────
    if "rule" in expected:
        assert result.get("rule") == expected["rule"], (
            f"[{fixture_id}] rule mismatch.\n"
            f"  got:      {result.get('rule')!r}\n"
            f"  expected: {expected['rule']!r}"
        )

    # ── Assert confidence (optional) ──────────────────────────────────────────
    if "confidence" in expected:
        assert result.get("confidence") == expected["confidence"], (
            f"[{fixture_id}] confidence mismatch.\n"
            f"  got:      {result.get('confidence')}\n"
            f"  expected: {expected['confidence']}"
        )

    # ── Assert expr (optional, token-list comparison) ─────────────────────────
    if "expr" in expected:
        assert "expr" in result, (
            f"[{fixture_id}] Expected 'expr' in result but it was missing.\n"
            f"  result keys: {list(result.keys())}"
        )
        _assert_expr(result["expr"], expected["expr"], fixture_id)

    # ── Assert gradient_partials (optional, gradient-specific) ────────────────
    if "gradient_partials" in expected:
        assert "expr" in result, (
            f"[{fixture_id}] Expected 'expr' in result for gradient check, "
            f"but it was missing.\n  result keys: {list(result.keys())}"
        )
        _assert_gradient_partials(
            result["expr"],
            expected["gradient_partials"],
            fixture_id
        )


# ── Scaffolding sanity checks ──────────────────────────────────────────────────

def test_fixtures_directory_exists():
    """The fixtures directory must exist and be a directory."""
    assert os.path.isdir(FIXTURES_DIR), (
        f"Fixtures directory not found: {FIXTURES_DIR}"
    )


def test_at_least_one_fixture_exists():
    """There must be at least one .json fixture file present."""
    fixtures = [f for f in os.listdir(FIXTURES_DIR) if f.endswith(".json")]
    assert len(fixtures) >= 1, (
        f"No .json fixture files found in {FIXTURES_DIR}"
    )


def test_all_fixtures_are_valid_json():
    """Every .json file in the fixtures directory must be valid JSON."""
    for fname in os.listdir(FIXTURES_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(FIXTURES_DIR, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Fixture file {fname} contains invalid JSON: {exc}")


def test_all_fixtures_have_input_and_expected_keys():
    """Every fixture must have both 'input' and 'expected' top-level keys."""
    for fname in sorted(os.listdir(FIXTURES_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(FIXTURES_DIR, fname)
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        assert "input" in data, (
            f"Fixture {fname} is missing the 'input' key."
        )
        assert "expected" in data, (
            f"Fixture {fname} is missing the 'expected' key."
        )


def test_all_fixtures_expected_has_status_and_verified():
    """Every fixture's 'expected' block must contain 'status' and 'verified'."""
    for fname in sorted(os.listdir(FIXTURES_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(FIXTURES_DIR, fname)
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        expected = data.get("expected", {})
        assert "status" in expected, (
            f"Fixture {fname} 'expected' block is missing 'status'."
        )
        assert "verified" in expected, (
            f"Fixture {fname} 'expected' block is missing 'verified'."
        )
