def exact_match_accuracy(predictions, references):
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(references) if references else 0.0

def eval_step_trace(generated_traces, expected_traces):
    score = sum(1 for g, e in zip(generated_traces, expected_traces) if g == e)
    return score / len(expected_traces) if expected_traces else 0.0

def compare_models(model_out, fallback_out, groq_out, ground_truth):
    return {
        "model_exact_match": model_out == ground_truth,
        "fallback_exact_match": fallback_out == ground_truth,
        "groq_exact_match": groq_out == ground_truth
    }

def categorize_error(prediction, reference):
    if not prediction:
        return "empty_output"
    if len(prediction) < len(reference) / 2:
        return "severe_truncation"
    if prediction.lower() == reference.lower():
        return "casing_mismatch"
    return "logic_or_hallucination"

def run_error_analysis(predictions, references):
    report = {}
    for p, r in zip(predictions, references):
        if p != r:
            category = categorize_error(p, r)
            report[category] = report.get(category, 0) + 1
    return report