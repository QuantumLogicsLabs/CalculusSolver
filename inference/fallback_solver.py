"""
FallbackSolver — pure-Python deterministic calculus solver.
No neural model, no torch, no checkpoint required.

Supports:
  op=diff / partial  — polynomial differentiation (power rule)
  op=integrate       — polynomial integration (power rule)
  op=gradient        — partial derivatives w.r.t. each variable in expr
  op=tangent_line    — tangent line at a point (1-variable)

Input envelope (SLaNg):
  {
    "op":   "diff" | "partial" | "integrate" | "gradient" | "tangent_line",
    "var":  "x",                  # variable of differentiation / integration
    "expr": {                     # SLaNg fraction object
        "numi": {"terms": [{"coeff": 3, "var": {"x": 2}}, ...]},
        "deno": 1
    },
    "point": {"x": 2}            # required for tangent_line
  }
"""

import json
from typing import Any, Dict, List, Optional


# ── Internal helpers ──────────────────────────────────────────────────────────


def _copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _norm_term(term: dict) -> dict:
    coeff = term.get("coeff", 0)
    if isinstance(coeff, float) and coeff.is_integer():
        coeff = int(coeff)
    clean: dict = {"coeff": coeff}
    variables = {k: v for k, v in term.get("var", {}).items() if v != 0}
    if variables:
        clean["var"] = variables
    return clean


def _diff_fraction(expr: dict, variable: str) -> dict:
    """Differentiate a SLaNg fraction w.r.t. variable (power rule)."""
    if expr.get("deno", 1) != 1:
        raise ValueError(
            "Fallback solver only handles denominator=1. "
            "Train a checkpoint for quotient-rule differentiation."
        )
    terms: List[dict] = []
    for term in expr.get("numi", {}).get("terms", []):
        power = term.get("var", {}).get(variable, 0)
        if power == 0:
            continue  # constant → 0
        t = _copy(term)
        t["coeff"] = t.get("coeff", 0) * power
        t.setdefault("var", {})[variable] = power - 1
        terms.append(_norm_term(t))
    return {"numi": {"terms": terms or [{"coeff": 0}]}, "deno": 1}


def _integrate_fraction(expr: dict, variable: str) -> dict:
    """Integrate a SLaNg fraction w.r.t. variable (power rule)."""
    if expr.get("deno", 1) != 1:
        raise ValueError(
            "Fallback solver only handles denominator=1. "
            "Train a checkpoint for integration of rational functions."
        )
    terms: List[dict] = []
    for term in expr.get("numi", {}).get("terms", []):
        power = term.get("var", {}).get(variable, 0)
        if power == -1:
            raise ValueError(
                "Fallback solver does not handle 1/x (logarithmic) integrals. "
                "Train a checkpoint for full integration support."
            )
        t = _copy(term)
        np1 = power + 1
        new_c = t.get("coeff", 0) / np1
        if float(new_c).is_integer():
            new_c = int(new_c)
        t["coeff"] = new_c
        t.setdefault("var", {})[variable] = np1
        terms.append(_norm_term(t))
    return {"numi": {"terms": terms or [{"coeff": 0}]}, "deno": 1}


def _eval_fraction(expr: dict, point: dict) -> float:
    """Numerically evaluate a SLaNg fraction at a given point."""
    numerator = 0.0
    for term in expr.get("numi", {}).get("terms", []):
        val = float(term.get("coeff", 0))
        for var_name, power in term.get("var", {}).items():
            val *= float(point.get(var_name, 0)) ** power
        numerator += val
    deno = float(expr.get("deno", 1))
    if deno == 0:
        raise ValueError("Division by zero evaluating expression at point.")
    return numerator / deno


def _term_to_latex(term: dict) -> str:
    coeff = term.get("coeff", 0)
    variables = term.get("var", {})
    if not variables:
        return str(coeff)
    parts = []
    if coeff == -1:
        parts.append("-")
    elif coeff != 1:
        parts.append(str(coeff))
    for name, power in variables.items():
        parts.append(name if power == 1 else f"{name}^{{{power}}}")
    return "".join(parts)


def _fraction_to_latex(expr: dict) -> str:
    terms = expr.get("numi", {}).get("terms", [])
    if not terms:
        return "0"
    parts = []
    for t in terms:
        s = _term_to_latex(t)
        if parts and not s.startswith("-"):
            parts.append("+")
        parts.append(s)
    numerator = " ".join(parts)
    deno = expr.get("deno", 1)
    return numerator if deno == 1 else f"\\frac{{{numerator}}}{{{deno}}}"


# ── Public solver class ───────────────────────────────────────────────────────


class FallbackSolver:
    """
    Deterministic polynomial solver used when no neural checkpoint is available.
    Implements the same .solve(payload) interface as CalculusSolverInference.
    """

    mode = "fallback"
    stage = "none"

    def solve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        op = payload.get("op", "")
        var = payload.get("var", "x")
        expr = payload.get("expr")
        point = payload.get("point", {})

        if not isinstance(expr, dict):
            raise ValueError(
                "Input must include 'expr' as a SLaNg fraction object "
                'e.g. {"numi":{"terms":[{"coeff":3,"var":{"x":2}}]},"deno":1}'
            )

        # ── diff / partial ────────────────────────────────────────────────────
        if op in ("diff", "partial"):
            result_expr = _diff_fraction(expr, var)
            steps = [
                {
                    "rule": "power_rule",
                    "description": f"Differentiated with respect to {var} using the power rule.",
                    "before": _fraction_to_latex(expr),
                    "after": _fraction_to_latex(result_expr),
                }
            ]
            return _response(op, result_expr, steps)

        # ── integrate ─────────────────────────────────────────────────────────
        if op == "integrate":
            result_expr = _integrate_fraction(expr, var)
            steps = [
                {
                    "rule": "power_rule_integral",
                    "description": f"Integrated with respect to {var} using the reverse power rule.",
                    "before": _fraction_to_latex(expr),
                    "after": _fraction_to_latex(result_expr) + " + C",
                }
            ]
            return _response(op, result_expr, steps)

        # ── gradient ──────────────────────────────────────────────────────────
        if op == "gradient":
            # Find all variables present in the expression
            all_vars: List[str] = []
            for term in expr.get("numi", {}).get("terms", []):
                for v in term.get("var", {}).keys():
                    if v not in all_vars:
                        all_vars.append(v)
            if not all_vars:
                all_vars = [var]

            partials = {}
            steps = []
            for v in all_vars:
                pd = _diff_fraction(expr, v)
                partials[v] = pd
                steps.append(
                    {
                        "rule": "partial_derivative",
                        "description": f"∂/∂{v}",
                        "before": _fraction_to_latex(expr),
                        "after": _fraction_to_latex(pd),
                    }
                )
            # Return gradient as a dict of partials
            result_expr = {"gradient": {v: partials[v] for v in all_vars}}
            return {
                "status": "solved",
                "expr": result_expr,
                "steps": steps,
                "latex": "\\nabla f = ("
                + ", ".join(_fraction_to_latex(partials[v]) for v in all_vars)
                + ")",
                "confidence": 1.0,
                "verified": True,
                "warning": "Fallback mode — no neural checkpoint loaded.",
                "rule": "gradient",
            }

        # ── tangent_line ──────────────────────────────────────────────────────
        if op == "tangent_line":
            if not point:
                raise ValueError("tangent_line requires a 'point' dict e.g. {\"x\": 2}")
            slope_expr = _diff_fraction(expr, var)
            slope = _eval_fraction(slope_expr, point)
            y0 = _eval_fraction(expr, point)
            x0 = float(point.get(var, 0))
            # y - y0 = slope*(x - x0)  →  y = slope*x + (y0 - slope*x0)
            intercept = y0 - slope * x0
            tangent_latex = (
                f"y = {slope}x + {intercept}"
                if intercept >= 0
                else f"y = {slope}x - {abs(intercept)}"
            )
            steps = [
                {
                    "rule": "power_rule",
                    "description": f"Differentiated to find slope at {var}={x0}",
                    "before": _fraction_to_latex(expr),
                    "after": _fraction_to_latex(slope_expr),
                },
                {
                    "rule": "point_slope",
                    "description": f"slope={slope}, point=({x0},{y0}) → {tangent_latex}",
                    "before": f"slope={slope}, ({var}₀,y₀)=({x0},{y0})",
                    "after": tangent_latex,
                },
            ]
            if float(slope).is_integer():
                slope = int(slope)
            if float(intercept).is_integer():
                intercept = int(intercept)
            tangent_terms = []
            if slope != 0:
                tangent_terms.append({"coeff": slope, "var": {var: 1}})
            tangent_terms.append({"coeff": intercept})
            result_expr = {
                "numi": {
                    "terms": tangent_terms
                },
                "deno": 1,
            }
            return {
                "status": "solved",
                "expr": result_expr,
                "steps": steps,
                "latex": tangent_latex,
                "confidence": 1.0,
                "verified": True,
                "warning": "Fallback mode — no neural checkpoint loaded.",
                "rule": "tangent_line",
            }

        # ── unsupported ───────────────────────────────────────────────────────
        raise ValueError(
            f"Fallback solver does not support op='{op}'. "
            f"Supported: diff, partial, integrate, gradient, tangent_line. "
            f"Train a checkpoint for: chain_rule, product_rule, quotient_rule, "
            f"limits, lagrange, series, dir_deriv, hessian, optimize."
        )


def _response(op: str, result_expr: dict, steps: list) -> dict:
    return {
        "status": "solved",
        "expr": result_expr,
        "steps": steps,
        "latex": _fraction_to_latex(result_expr),
        "confidence": 1.0,
        "verified": True,
        "warning": "Fallback mode — no neural checkpoint loaded.",
        "rule": steps[0]["rule"] if steps else op,
    }
