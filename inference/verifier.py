"""
Verifier module in pure Python.
Replaces verifier.js and eliminates subprocess Node.js dependency.
Provides algebraic evaluation, differentiation, integration, and equivalence testing.
"""

import math
import random
from typing import Any, List, Dict, Tuple, Union


def evaluate_fraction(expr: Any, point: Dict[str, float]) -> float:
    """Numerically evaluate a SLaNg expression at a given point."""
    if isinstance(expr, (int, float)):
        return float(expr)
    if isinstance(expr, dict) and "numi" in expr and "deno" in expr:
        num_val = evaluate_container(expr["numi"], point)
        den_val = evaluate_container(expr["deno"], point)
        if den_val == 0:
            raise ZeroDivisionError("Division by zero in SLaNg evaluation.")
        return num_val / den_val
    if isinstance(expr, dict) and "coeff" in expr:
        return evaluate_term(expr, point)
    if isinstance(expr, list):
        return sum(evaluate_term(t, point) for t in expr)
    raise ValueError(f"Unknown expression format: {expr}")


def evaluate_container(container: Any, point: Dict[str, float]) -> float:
    if container is None:
        return 0.0
    if isinstance(container, (int, float)):
        return float(container)
    if isinstance(container, list):
        return sum(evaluate_term(t, point) for t in container)
    if isinstance(container, dict):
        if "terms" in container:
            return sum(evaluate_term(t, point) for t in container["terms"])
        if "coeff" in container:
            return evaluate_term(container, point)
    raise ValueError(f"Unknown container format: {container}")


def evaluate_term(term: Dict[str, Any], point: Dict[str, float]) -> float:
    coeff = float(term.get("coeff", 0.0))
    val = coeff
    var_dict = term.get("var", {})
    if isinstance(var_dict, dict):
        for var_name, power in var_dict.items():
            val *= float(point.get(var_name, 0.0)) ** float(power)
    return val


# ── Algebraic Operations ──────────────────────────────────────────────────────


def extract_poly(container: Any) -> List[Dict[str, Any]]:
    if container is None:
        return [{"coeff": 0.0}]
    if isinstance(container, list):
        return container
    if isinstance(container, dict):
        if "terms" in container:
            return container["terms"]
        if "coeff" in container:
            return [container]
    if isinstance(container, (int, float)):
        return [{"coeff": float(container)}]
    raise ValueError(f"Cannot extract polynomial from: {container}")


def add_terms(t1: List[dict], t2: List[dict]) -> List[dict]:
    res = []
    def var_key(var_dict):
        return tuple(sorted((k, v) for k, v in var_dict.items() if v != 0))
    
    merged = {}
    for term in t1 + t2:
        coeff = term.get("coeff", 0.0)
        v_dict = term.get("var", {})
        vk = var_key(v_dict)
        merged[vk] = merged.get(vk, 0.0) + coeff
    
    for vk, coeff in merged.items():
        if abs(coeff) > 1e-12:
            term = {"coeff": coeff}
            if vk:
                term["var"] = dict(vk)
            res.append(term)
    if not res:
        res = [{"coeff": 0.0}]
    return res


def multiply_terms(t1: List[dict], t2: List[dict]) -> List[dict]:
    merged = {}
    for term1 in t1:
        c1 = term1.get("coeff", 0.0)
        v1 = term1.get("var", {})
        for term2 in t2:
            c2 = term2.get("coeff", 0.0)
            v2 = term2.get("var", {})
            
            coeff = c1 * c2
            new_vars = {}
            for k, v in v1.items():
                new_vars[k] = new_vars.get(k, 0.0) + v
            for k, v in v2.items():
                new_vars[k] = new_vars.get(k, 0.0) + v
            
            new_vars = {k: v for k, v in new_vars.items() if v != 0}
            
            def var_key(var_dict):
                return tuple(sorted(var_dict.items()))
            vk = var_key(new_vars)
            merged[vk] = merged.get(vk, 0.0) + coeff

    res = []
    for vk, coeff in merged.items():
        if abs(coeff) > 1e-12:
            term = {"coeff": coeff}
            if vk:
                term["var"] = dict(vk)
            res.append(term)
    if not res:
        res = [{"coeff": 0.0}]
    return res


def diff_poly(poly: List[dict], variable: str) -> List[dict]:
    res = []
    for term in poly:
        coeff = term.get("coeff", 0.0)
        v_dict = term.get("var", {})
        power = v_dict.get(variable, 0.0)
        if power == 0.0:
            continue
        new_coeff = coeff * power
        new_vars = v_dict.copy()
        new_vars[variable] = power - 1
        new_vars = {k: v for k, v in new_vars.items() if v != 0}
        
        new_term = {"coeff": new_coeff}
        if new_vars:
            new_term["var"] = new_vars
        res.append(new_term)
    return res


def differentiate_fraction(expr: dict, variable: str) -> dict:
    """Differentiate SLaNg fraction u/v using quotient rule."""
    u = extract_poly(expr.get("numi"))
    v = extract_poly(expr.get("deno", 1))
    
    du = diff_poly(u, variable)
    dv = diff_poly(v, variable)
    
    du_v = multiply_terms(du, v)
    neg_u = [{"coeff": -t.get("coeff", 0.0), "var": t.get("var", {})} for t in u]
    u_dv = multiply_terms(neg_u, dv)
    
    new_numi = add_terms(du_v, u_dv)
    new_deno = multiply_terms(v, v)
    
    return {"numi": {"terms": new_numi}, "deno": {"terms": new_deno}}


def integrate_fraction(expr: dict, variable: str) -> dict:
    """Integrate polynomial fraction w.r.t variable (denominator must be constant)."""
    u = extract_poly(expr.get("numi"))
    v = extract_poly(expr.get("deno", 1))
    
    for term in v:
        if term.get("var"):
            raise ValueError("Integration of general rational functions is not supported.")
            
    integrated_num = []
    for term in u:
        coeff = term.get("coeff", 0.0)
        v_dict = term.get("var", {})
        power = v_dict.get(variable, 0.0)
        if power == -1.0:
            raise ValueError("Reverse power rule cannot integrate 1/x to logarithmic form.")
        new_power = power + 1
        new_coeff = coeff / new_power
        new_vars = v_dict.copy()
        new_vars[variable] = new_power
        new_vars = {k: v for k, v in new_vars.items() if v != 0}
        
        new_term = {"coeff": new_coeff}
        if new_vars:
            new_term["var"] = new_vars
        integrated_num.append(new_term)
        
    return {"numi": {"terms": integrated_num}, "deno": {"terms": v}}


def definite_integrate_fraction(expr: dict, lower: float, upper: float, variable: str) -> float:
    integrated = integrate_fraction(expr, variable)
    val_upper = evaluate_fraction(integrated, {variable: upper})
    val_lower = evaluate_fraction(integrated, {variable: lower})
    return val_upper - val_lower


def product_rule_differentiate(u: dict, v: dict, variable: str) -> dict:
    """Differentiate u * v using product rule."""
    u_frac = {"numi": u.get("numi", u), "deno": u.get("deno", 1)}
    v_frac = {"numi": v.get("numi", v), "deno": v.get("deno", 1)}
    
    du = differentiate_fraction(u_frac, variable)
    dv = differentiate_fraction(v_frac, variable)
    
    du_v_numi = multiply_terms(extract_poly(du["numi"]), extract_poly(v_frac["numi"]))
    du_v_deno = multiply_terms(extract_poly(du["deno"]), extract_poly(v_frac["deno"]))
    
    u_dv_numi = multiply_terms(extract_poly(u_frac["numi"]), extract_poly(dv["numi"]))
    u_dv_deno = multiply_terms(extract_poly(u_frac["deno"]), extract_poly(dv["deno"]))
    
    # du_v + u_dv = (A/B) + (C/D) = (A*D + B*C) / (B*D)
    numi1 = multiply_terms(du_v_numi, u_dv_deno)
    numi2 = multiply_terms(du_v_deno, u_dv_numi)
    numi = add_terms(numi1, numi2)
    deno = multiply_terms(du_v_deno, u_dv_deno)
    
    return {"numi": {"terms": numi}, "deno": {"terms": deno}}


def quotient_rule_differentiate(u: dict, v: dict, variable: str) -> dict:
    """Differentiate u / v using quotient rule."""
    u_frac = {"numi": u.get("numi", u), "deno": u.get("deno", 1)}
    v_frac = {"numi": v.get("numi", v), "deno": v.get("deno", 1)}
    
    du = differentiate_fraction(u_frac, variable)
    dv = differentiate_fraction(v_frac, variable)
    
    du_v_numi = multiply_terms(extract_poly(du["numi"]), extract_poly(v_frac["numi"]))
    du_v_deno = multiply_terms(extract_poly(du["deno"]), extract_poly(v_frac["deno"]))
    
    u_dv_numi = multiply_terms(extract_poly(u_frac["numi"]), extract_poly(dv["numi"]))
    u_dv_deno = multiply_terms(extract_poly(u_frac["deno"]), extract_poly(dv["deno"]))
    
    # A/B - C/D = (A*D - B*C) / (B*D)
    numi1 = multiply_terms(du_v_numi, u_dv_deno)
    neg_u_dv_numi = [{"coeff": -t.get("coeff", 0.0), "var": t.get("var", {})} for t in u_dv_numi]
    numi2 = multiply_terms(du_v_deno, neg_u_dv_numi)
    
    numi_diff = add_terms(numi1, numi2)
    deno_diff = multiply_terms(du_v_deno, u_dv_deno)
    
    # Divide by v^2
    v_numi_sq = multiply_terms(extract_poly(v_frac["numi"]), extract_poly(v_frac["numi"]))
    v_deno_sq = multiply_terms(extract_poly(v_frac["deno"]), extract_poly(v_frac["deno"]))
    
    final_numi = multiply_terms(numi_diff, v_deno_sq)
    final_deno = multiply_terms(deno_diff, v_numi_sq)
    
    return {"numi": {"terms": final_numi}, "deno": {"terms": final_deno}}


def gradient_oracle(expr: dict, variables: List[str]) -> Dict[str, dict]:
    res = {}
    for var in variables:
        res[var] = differentiate_fraction(expr, var)
    return res


# ── Equivalence Verification ──────────────────────────────────────────────────


def test_points(variables: List[str], n: int = 50) -> List[Dict[str, float]]:
    points = []
    for _ in range(n):
        point = {}
        for var in variables:
            # Random floats in [-4.9, 4.9]
            point[var] = random.random() * 9.8 - 4.9
        points.append(point)
    return points


def compare_expressions(a: Any, b: Any, variables: List[str], tol: float = 1e-6) -> dict:
    points = test_points(variables, 50)
    tested = 0
    passed = 0

    for point in points:
        try:
            va = evaluate_fraction(a, point)
            vb = evaluate_fraction(b, point)
            if not math.isfinite(va) or not math.isfinite(vb):
                continue
            tested += 1
            if abs(va - vb) <= tol:
                passed += 1
        except Exception:
            continue

    return {
        "tested": tested,
        "passed": passed,
        "confidence": passed / tested if tested > 0 else 0.0,
        "equivalent": tested > 0 and passed == tested,
    }


def get_variables(input_env: dict) -> List[str]:
    if isinstance(input_env.get("vars"), list) and len(input_env["vars"]) > 0:
        return input_env["vars"]
    v = input_env.get("var")
    if isinstance(v, str):
        return [v]
    return ["x"]


def verify(input_env: Dict[str, Any], output_tokens: List[str]) -> Dict[str, Any]:
    """Verify SLaNg output tokens against an input math operation envelope."""
    try:
        from tokenizer.slang_serializer import deserialize_slang_math
        output = deserialize_slang_math(output_tokens)
    except Exception as error:
        return {
            "status": "unverified",
            "verified": False,
            "confidence": 0.0,
            "error": f"Output deserialization failed: {error}",
        }

    op = input_env.get("op")
    if op == "undefined":
        return {
            "status": "unsolvable",
            "verified": False,
            "confidence": 0.0,
            "output": output,
        }

    if op == "diff" or op == "partial":
        oracle_fn = lambda inp: differentiate_fraction(inp["expr"], inp["var"])
    elif op == "integrate":
        oracle_fn = lambda inp: integrate_fraction(inp["expr"], inp["var"])
    elif op == "def_integrate":
        def def_int_fn(inp):
            val = definite_integrate_fraction(inp["expr"], float(inp["lower"]), float(inp["upper"]), inp["var"])
            return {"numi": {"terms": [{"coeff": val}]}, "deno": 1}
        oracle_fn = def_int_fn
    elif op == "gradient":
        oracle_fn = lambda inp: gradient_oracle(inp["expr"], get_variables(inp))
    elif op == "product_rule":
        oracle_fn = lambda inp: product_rule_differentiate(inp["u"], inp["v"], inp["var"])
    elif op == "quotient_rule":
        oracle_fn = lambda inp: quotient_rule_differentiate(inp["u"], inp["v"], inp["var"])
    else:
        return {
            "status": "unverified",
            "verified": False,
            "confidence": 0.0,
            "error": f"Unsupported operation for verifier: {op}",
            "output": output,
        }

    try:
        oracle = oracle_fn(input_env)
    except Exception as error:
        return {
            "status": "unverified",
            "verified": False,
            "confidence": 0.0,
            "error": f"Oracle execution failed: {error}",
            "output": output,
        }

    variables = get_variables(input_env)
    if op == "gradient" and isinstance(oracle, dict) and oracle is not None:
        if not isinstance(output, dict):
            return {
                "status": "unverified",
                "verified": False,
                "confidence": 0.0,
                "output": output,
            }
        for key in oracle.keys():
            if key not in output:
                return {
                    "status": "unverified",
                    "verified": False,
                    "confidence": 0.0,
                    "output": output,
                }
            result = compare_expressions(oracle[key], output[key], variables)
            if not result["equivalent"]:
                return {
                    "status": "unverified",
                    "verified": False,
                    "confidence": result["confidence"],
                    "output": output,
                }
        return {
            "status": "solved",
            "verified": True,
            "confidence": 1.0,
            "output": output,
        }

    result = compare_expressions(oracle, output, variables)
    return {
        "status": "solved" if result["equivalent"] else "unverified",
        "verified": result["equivalent"],
        "confidence": result["confidence"],
        "output": output,
    }
