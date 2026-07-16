import json
from tokenizer.slang_serializer import serialize_slang_math

def is_equivalent(a, b):
    if a == b:
        return True
    try:
        tok_a = serialize_slang_math(a) if isinstance(a, dict) else a
        tok_b = serialize_slang_math(b) if isinstance(b, dict) else b
        # If string but looks like json, try parsing
        if isinstance(tok_a, str) and tok_a.strip().startswith("{"):
            tok_a = serialize_slang_math(json.loads(tok_a))
        if isinstance(tok_b, str) and tok_b.strip().startswith("{"):
            tok_b = serialize_slang_math(json.loads(tok_b))
        return tok_a == tok_b
    except Exception:
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