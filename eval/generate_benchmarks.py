"""
Benchmark generator for CalculusSolver eval pipeline.

Generates labeled SLaNg problems across all supported operations
(diff, partial, integrate, gradient, tangent_line) using FallbackSolver
to compute provably-correct ground truth answers.

Output: eval/benchmarks/*.json files, ~300 total problems.
"""

import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.fallback_solver import FallbackSolver

solver = FallbackSolver()

# Vocab-safe coefficient and exponent ranges
SAFE_COEFFS = list(range(-10, 11)) + [12]
SAFE_POS_COEFFS = [c for c in SAFE_COEFFS if c > 0]
SAFE_NONZERO_COEFFS = [c for c in SAFE_COEFFS if c != 0]
SAFE_POS_EXPONENTS = list(range(1, 6))  # 1..5


def _make_fraction(terms):
    """Wrap a list of term dicts into a SLaNg fraction."""
    return {"numi": {"terms": terms}, "deno": 1}


def _safe_diff_term(var="x"):
    """Generate a term where differentiation output stays in vocab range."""
    for _ in range(200):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(SAFE_POS_EXPONENTS)
        out_coeff = coeff * power
        out_exp = power - 1
        if out_coeff in SAFE_COEFFS and out_exp in range(-3, 6):
            return {"coeff": coeff, "var": {var: power}}
    return {"coeff": 2, "var": {var: 2}}


def _safe_integrate_term(var="x"):
    """Generate a term where integration output stays in vocab range."""
    for _ in range(200):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(SAFE_POS_EXPONENTS)
        new_power = power + 1
        if new_power > 5:
            continue
        new_coeff = coeff / new_power
        if float(new_coeff).is_integer() and int(new_coeff) in SAFE_COEFFS:
            return {"coeff": coeff, "var": {var: power}}
    return {"coeff": 2, "var": {var: 1}}


def generate_diff_benchmarks(n=80):
    """Generate differentiation benchmark problems."""
    problems = []
    random.seed(100)

    for i in range(n):
        var = "x"
        # Mix single and multi-term
        if i < 50:
            # Single term
            term = _safe_diff_term(var)
            expr = _make_fraction([term])
        else:
            # Multi-term (2-3 terms)
            num_terms = random.randint(2, 3)
            terms = []
            used_powers = set()
            for _ in range(num_terms):
                for __ in range(50):
                    t = _safe_diff_term(var)
                    p = list(t.get("var", {}).values())[0] if t.get("var") else 0
                    if p not in used_powers:
                        used_powers.add(p)
                        terms.append(t)
                        break
            if not terms:
                terms = [_safe_diff_term(var)]
            expr = _make_fraction(terms)

        payload = {"op": "diff", "var": var, "expr": expr}
        try:
            result = solver.solve(payload)
            problems.append({
                "expr": payload,
                "target": result["expr"],
                "operation": "diff",
                "expected_rule": "power_rule",
            })
        except Exception as e:
            print(f"  Skipping diff problem {i}: {e}")

    return problems


def generate_integrate_benchmarks(n=60):
    """Generate integration benchmark problems."""
    problems = []
    random.seed(200)

    for i in range(n):
        var = "x"
        if i < 40:
            term = _safe_integrate_term(var)
            expr = _make_fraction([term])
        else:
            terms = []
            used_powers = set()
            for _ in range(2):
                for __ in range(50):
                    t = _safe_integrate_term(var)
                    p = list(t.get("var", {}).values())[0] if t.get("var") else 0
                    if p not in used_powers:
                        used_powers.add(p)
                        terms.append(t)
                        break
            if not terms:
                terms = [_safe_integrate_term(var)]
            expr = _make_fraction(terms)

        payload = {"op": "integrate", "var": var, "expr": expr}
        try:
            result = solver.solve(payload)
            problems.append({
                "expr": payload,
                "target": result["expr"],
                "operation": "integrate",
                "expected_rule": "power_rule_integral",
            })
        except Exception as e:
            print(f"  Skipping integrate problem {i}: {e}")

    return problems


def generate_partial_benchmarks(n=60):
    """Generate partial derivative benchmark problems (multi-variable)."""
    problems = []
    random.seed(300)
    variables = ["x", "y", "z"]

    for i in range(n):
        diff_var = random.choice(variables)
        terms = []
        # Term with the differentiation variable
        terms.append(_safe_diff_term(diff_var))
        # Term with a different variable (constant w.r.t. diff_var)
        other_var = random.choice([v for v in variables if v != diff_var])
        coeff = random.choice(SAFE_POS_COEFFS)
        power = random.choice(SAFE_POS_EXPONENTS)
        terms.append({"coeff": coeff, "var": {other_var: power}})

        expr = _make_fraction(terms)
        payload = {"op": "partial", "var": diff_var, "expr": expr}
        try:
            result = solver.solve(payload)
            problems.append({
                "expr": payload,
                "target": result["expr"],
                "operation": "partial",
                "expected_rule": "power_rule",
            })
        except Exception as e:
            print(f"  Skipping partial problem {i}: {e}")

    return problems


def generate_gradient_benchmarks(n=50):
    """Generate gradient benchmark problems."""
    problems = []
    random.seed(400)

    for i in range(n):
        # 2-variable polynomial
        terms = []
        for var in ["x", "y"]:
            for _ in range(50):
                t = _safe_diff_term(var)
                terms.append(t)
                break

        expr = _make_fraction(terms)
        payload = {"op": "gradient", "var": "x", "expr": expr}
        try:
            result = solver.solve(payload)
            problems.append({
                "expr": payload,
                "target": result["expr"],
                "operation": "gradient",
                "expected_rule": "gradient",
            })
        except Exception as e:
            print(f"  Skipping gradient problem {i}: {e}")

    return problems


def generate_tangent_line_benchmarks(n=50):
    """Generate tangent line benchmark problems."""
    problems = []
    random.seed(500)

    for i in range(n):
        var = "x"
        term = _safe_diff_term(var)
        expr = _make_fraction([term])
        x_val = random.choice([1, 2, -1, -2, 3])

        payload = {"op": "tangent_line", "var": var, "expr": expr, "point": {var: x_val}}
        try:
            result = solver.solve(payload)
            problems.append({
                "expr": payload,
                "target": result["expr"],
                "operation": "tangent_line",
                "expected_rule": "tangent_line",
            })
        except Exception as e:
            print(f"  Skipping tangent_line problem {i}: {e}")

    return problems


def main():
    benchmark_dir = Path("eval/benchmarks")
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("benchmark_diff.json", generate_diff_benchmarks),
        ("benchmark_integrate.json", generate_integrate_benchmarks),
        ("benchmark_partial.json", generate_partial_benchmarks),
        ("benchmark_gradient.json", generate_gradient_benchmarks),
        ("benchmark_tangent_line.json", generate_tangent_line_benchmarks),
    ]

    total = 0
    for filename, gen_fn in generators:
        problems = gen_fn()
        filepath = benchmark_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(problems, f, indent=2)
        print(f"  {filename}: {len(problems)} problems")
        total += len(problems)

    print(f"\nTotal benchmark problems: {total}")
    print(f"Output directory: {benchmark_dir}")


if __name__ == "__main__":
    main()
