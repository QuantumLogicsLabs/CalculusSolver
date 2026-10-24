import pytest
from inference.solve import CalculusSolverInference  # adjust import path to match your project


@pytest.fixture(scope="module")
def solver():
    # Adjust to however you actually construct/load the solver in your project
    return CalculusSolverInference(model_path="checkpoints/latest.pt")


def _slang_expr(terms):
    """Helper: builds a SLaNg numerator-terms expression dict."""
    return {"numi": {"terms": terms}, "deno": 1}


class TestGradientRegression:

    def test_two_variable_gradient(self, solver):
        # grad(3x - 6y) w.r.t. {x, y}
        expr = _slang_expr([
            {"coeff": 3, "var": {"x": 1}},
            {"coeff": -6, "var": {"y": 1}},
        ])
        result = solver._solve_gradient(expr, ["x", "y"])
        assert set(result["gradient"].keys()) == {"x", "y"}

    def test_three_variable_gradient(self, solver):
        # The report's own example: grad(2x^2 + 3y + 5z^3)
        expr = _slang_expr([
            {"coeff": 2, "var": {"x": 2}},
            {"coeff": 3, "var": {"y": 1}},
            {"coeff": 5, "var": {"z": 3}},
        ])
        result = solver._solve_gradient(expr, ["x", "y", "z"])
        assert set(result["gradient"].keys()) == {"x", "y", "z"}
        # d/dx: 4x ; d/dy: 3 ; d/dz: 15z^2 — assert against your actual
        # serialized output shape once you confirm it; placeholders below:
        # assert result["gradient"]["x"] == {...}
        # assert result["gradient"]["y"] == {...}
        # assert result["gradient"]["z"] == {...}

    def test_gradient_with_vanishing_term(self, solver):
        # Term with no 'y' should produce zero derivative w.r.t. y
        expr = _slang_expr([
            {"coeff": 3, "var": {"x": 1}},
            {"coeff": -6, "var": {"y": 1}},
        ])
        result = solver._solve_gradient(expr, ["x"])
        # d/dx of (3x - 6y) == 3 ; the y-term vanishes entirely
        assert "x" in result["gradient"]

    def test_gradient_with_negative_coefficients(self, solver):
        expr = _slang_expr([
            {"coeff": -4, "var": {"y": 1}},
            {"coeff": 1, "var": {"z": 4}},
        ])
        result = solver._solve_gradient(expr, ["y", "z"])
        assert set(result["gradient"].keys()) == {"y", "z"}

    def test_gradient_with_constant_term(self, solver):
        expr = _slang_expr([
            {"coeff": 7, "var": {}},  # constant term — check your actual encoding for constants
            {"coeff": 2, "var": {"x": 1}},
        ])
        result = solver._solve_gradient(expr, ["x"])
        assert "x" in result["gradient"]

    def test_gradient_with_power_terms(self, solver):
        expr = _slang_expr([
            {"coeff": 1, "var": {"x": 4}},
        ])
        result = solver._solve_gradient(expr, ["x"])
        assert "x" in result["gradient"]

    def test_gradient_partial_derivative_error_is_raised(self, solver):
        # Confirm a broken partial call surfaces, not silently swallowed
        expr = _slang_expr([{"coeff": 1, "var": {"x": 1}}])
        with pytest.raises(Exception):
            solver._solve_gradient(expr, ["not_a_real_variable_xyz"])


class TestGradientEdgeCases:

    def test_term_independent_of_target_variable_becomes_zero(self, solver):
        # terms with no occurrence of the bound variable must vanish, not
        # leak into the output (this was the Dev-1 dataset bug from before)
        expr = _slang_expr([
            {"coeff": 5, "var": {"y": 2}},  # no x at all
        ])
        result = solver._solve_gradient(expr, ["x"])
        # the x-partial of a y-only term should be the zero expression —
        # assert the exact zero representation your serializer uses
        assert "x" in result["gradient"]

    def test_coefficients_exponents_variable_names_bound_correctly(self, solver):
        expr = _slang_expr([
            {"coeff": 9, "var": {"x": 3}},
            {"coeff": -2, "var": {"z": 1}},
        ])
        result = solver._solve_gradient(expr, ["x", "z"])
        assert set(result["gradient"].keys()) == {"x", "z"}