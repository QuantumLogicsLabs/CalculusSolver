import json
from typing import Any, Dict
from tokenizer.slang_serializer import serialize_slang_math


def _canonicalize_slang(expr: Any) -> Any:
    if not isinstance(expr, dict):
        return expr
    # Unwrap {"gradient": {...}}
    if set(expr.keys()) == {"gradient"} and isinstance(expr["gradient"], dict):
        return {k: _canonicalize_slang(v) for k, v in expr["gradient"].items()}
    # Gradient dict {var: expr}
    if (
        expr
        and all(isinstance(v, (dict, list, int, float)) for v in expr.values())
        and "numi" not in expr
        and "op" not in expr
        and "coeff" not in expr
    ):
        return {k: _canonicalize_slang(v) for k, v in expr.items()}
    # Bare term -> fraction
    if "coeff" in expr and "numi" not in expr and "op" not in expr:
        return {"numi": {"terms": [expr]}, "deno": 1}
    # Bare terms dict -> fraction
    if "terms" in expr and "numi" not in expr:
        return {"numi": expr, "deno": 1}
    # Fraction normalization
    if "numi" in expr and "deno" in expr:
        deno = expr["deno"]
        if isinstance(deno, dict) and "terms" in deno:
            terms = deno.get("terms", [])
            if len(terms) == 1 and terms[0].get("coeff") == 1 and not terms[0].get("var"):
                deno = 1
        numi = expr["numi"]
        if isinstance(numi, dict) and "terms" in numi:
            terms = numi.get("terms", [])
            non_zero = [t for t in terms if isinstance(t, dict) and t.get("coeff", 0) != 0]
            if non_zero:
                numi = {"terms": non_zero}
            else:
                numi = {"terms": [{"coeff": 0}]}
        elif isinstance(numi, list):
            non_zero = [t for t in numi if isinstance(t, dict) and t.get("coeff", 0) != 0]
            numi = {"terms": non_zero if non_zero else [{"coeff": 0}]}
        return {"numi": numi, "deno": deno}
    return expr


def is_equivalent(a, b):
    if a == b:
        return True
    try:
        if isinstance(a, str) and a.strip().startswith("{"):
            try:
                a = json.loads(a)
            except Exception:
                pass
        if isinstance(b, str) and b.strip().startswith("{"):
            try:
                b = json.loads(b)
            except Exception:
                pass

        can_a = _canonicalize_slang(a)
        can_b = _canonicalize_slang(b)
        if can_a == can_b:
            return True

        tok_a = serialize_slang_math(can_a) if isinstance(can_a, dict) else can_a
        tok_b = serialize_slang_math(can_b) if isinstance(can_b, dict) else can_b
        if tok_a == tok_b:
            return True

        # Mathematical equivalence fallback
        if isinstance(a, dict) and isinstance(b, dict):
            from inference.verifier import compare_expressions
            vars_list = ["x", "y", "z", "t", "r"]
            if (
                isinstance(can_a, dict)
                and isinstance(can_b, dict)
                and "numi" not in can_a
                and "numi" not in can_b
            ):
                if set(can_a.keys()) == set(can_b.keys()) and can_a:
                    return all(
                        compare_expressions(can_a[k], can_b[k], vars_list).get("equivalent", False)
                        for k in can_a
                    )
            res = compare_expressions(can_a, can_b, vars_list)
            if res.get("equivalent", False):
                return True
    except Exception:
        pass
    return False

def exact_match_accuracy(predictions, references):
    if not references:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references) if is_equivalent(p, r))
    return correct / len(references)

def eval_step_trace(generated_traces, expected_traces):
    if not expected_traces:
        return 0.0
    score = sum(1 for g, e in zip(generated_traces, expected_traces) if g == e)
    return score / len(expected_traces)

def compare_models_v1_placeholder(model_out, fallback_out, groq_out, ground_truth):
    return {
        "model_exact_match": is_equivalent(model_out, ground_truth),
        "fallback_exact_match": is_equivalent(fallback_out, ground_truth),
        "groq_exact_match": is_equivalent(groq_out, ground_truth)
    }

def check_syntax(expr):
    if not isinstance(expr, dict):
        return False
    if "gradient" in expr:
        grad = expr["gradient"]
        if not isinstance(grad, dict):
            return False
        for k, v in grad.items():
            if not check_syntax(v):
                return False
        return True
    if "numi" not in expr:
        return False
    numi = expr["numi"]
    if isinstance(numi, dict):
        terms = numi.get("terms")
        if not isinstance(terms, list):
            return False
        for term in terms:
            if not isinstance(term, dict) or "coeff" not in term:
                return False
            v = term.get("var", {})
            if not isinstance(v, dict):
                return False
    elif isinstance(numi, list):
        for term in numi:
            if not isinstance(term, dict) or "coeff" not in term:
                return False
            v = term.get("var", {})
            if not isinstance(v, dict):
                return False
    else:
        return False
    return True

def categorize_error_v1(prediction, reference):
    if not prediction:
        return "empty_output"
        
    try:
        if isinstance(prediction, dict):
            serialize_slang_math(prediction)
            valid_syntax = check_syntax(prediction)
        elif isinstance(prediction, str):
            try:
                parsed = json.loads(prediction)
                serialize_slang_math(parsed)
                valid_syntax = check_syntax(parsed)
            except Exception:
                valid_syntax = False
        else:
            valid_syntax = False
    except Exception:
        valid_syntax = False
        
    if not valid_syntax:
        return "syntax_error"
        
    try:
        tok_pred = serialize_slang_math(prediction) if isinstance(prediction, dict) else prediction
        tok_ref = serialize_slang_math(reference) if isinstance(reference, dict) else reference
        
        if isinstance(tok_pred, str) and tok_pred.strip().startswith("{"):
            try:
                tok_pred = serialize_slang_math(json.loads(tok_pred))
            except Exception:
                pass
        if isinstance(tok_ref, str) and tok_ref.strip().startswith("{"):
            try:
                tok_ref = serialize_slang_math(json.loads(tok_ref))
            except Exception:
                pass
        
        len_p = len(tok_pred)
        len_r = len(tok_ref)
        if len_p < len_r / 2:
            return "severe_truncation"
    except Exception:
        pass
        
    return "logic_error"

def run_error_analysis_v1(predictions, references):
    report = {}
    for p, r in zip(predictions, references):
        if not is_equivalent(p, r):
            category = categorize_error_v1(p, r)
            report[category] = report.get(category, 0) + 1
    return report

def categorize_error_v1_placeholder(prediction, reference):
    return categorize_error_v1(prediction, reference)

def run_error_analysis_v1_placeholder(predictions, references):
    return run_error_analysis_v1(predictions, references)