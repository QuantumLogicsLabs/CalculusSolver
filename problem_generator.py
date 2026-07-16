import json
import random
from pathlib import Path

# ── Vocab-safe ranges ────────────────────────────────────────────────────────
# These must match the tokenizer/vocab.json ranges exactly.
# Coefficients: COEF:-10 to COEF:12 (integers only, skip COEF:OTHER/COEF:100)
SAFE_COEFFS = list(range(-10, 11)) + [12]
# Positive coefficients only (for cases where we need non-zero positive)
SAFE_POS_COEFFS = [c for c in SAFE_COEFFS if c > 0]
# Non-zero coefficients
SAFE_NONZERO_COEFFS = [c for c in SAFE_COEFFS if c != 0]
# Exponents: EXP:-3 to EXP:5 (integers only, skip EXP:OTHER)
SAFE_EXPONENTS = list(range(-3, 6))
# Positive exponents for power rule differentiation (need power >= 1 for non-trivial result)
SAFE_POS_EXPONENTS = [e for e in SAFE_EXPONENTS if e >= 1]
# Variables
VARIABLES = ["x", "y", "z"]


def _output_in_vocab(coeff, power):
    """Check that the derivative output (coeff*power, power-1) stays in vocab range."""
    out_coeff = coeff * power
    out_exp = power - 1
    return out_coeff in SAFE_COEFFS and out_exp in SAFE_EXPONENTS


def _integral_in_vocab(coeff, power):
    """Check that the integral output (coeff/(power+1), power+1) stays in vocab range."""
    new_power = power + 1
    if new_power == 0:
        return False  # Would be ln|x|, not supported
    new_coeff = coeff / new_power
    # Must be an integer to tokenize cleanly
    if not float(new_coeff).is_integer():
        return False
    return int(new_coeff) in SAFE_COEFFS and new_power in SAFE_EXPONENTS


def generate_single_term_diff(var="x"):
    """Generate a single-term power-rule differentiation problem."""
    for _ in range(100):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(SAFE_POS_EXPONENTS)
        if _output_in_vocab(coeff, power):
            src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
            ans = {"numi": {"terms": [{"coeff": coeff * power, "var": {var: power - 1}}]}, "deno": 1}
            # Clean zero-exponent vars
            if ans["numi"]["terms"][0]["var"][var] == 0:
                ans = {"coeff": coeff * power}
            return src, ans, 0  # rule_id 0 = power_rule
    # Fallback safe pair
    return {"numi": {"terms": [{"coeff": 2, "var": {var: 2}}]}, "deno": 1}, {"numi": {"terms": [{"coeff": 4, "var": {var: 1}}]}, "deno": 1}, 0


def generate_constant_term():
    """Generate a constant differentiation problem (derivative = 0)."""
    coeff = random.choice(SAFE_NONZERO_COEFFS)
    src = {"numi": {"terms": [{"coeff": coeff}]}, "deno": 1}
    ans = {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
    return src, ans, 5  # rule_id 5 = constant_rule


def generate_multi_term_diff(var="x", num_terms=None):
    """Generate a multi-term polynomial differentiation problem (sum rule)."""
    if num_terms is None:
        num_terms = random.randint(2, 3)

    src_terms = []
    ans_terms = []

    for i in range(num_terms):
        # Mix: some power-rule terms, optionally a constant
        if i == num_terms - 1 and random.random() < 0.3:
            # Add a constant term
            c_src, c_ans, _ = generate_constant_term()
            src_terms.append(c_src)
            # Constant differentiates to 0, so we skip adding to ans
        else:
            t_src, t_ans, _ = generate_single_term_diff(var)
            # Avoid duplicate exponents in the same polynomial
            src_exps = {list(t["numi"]["terms"][0].get("var", {}).values())[0] for t in src_terms if t["numi"]["terms"][0].get("var")}
            t_exp = list(t_src["numi"]["terms"][0].get("var", {}).values())[0] if t_src["numi"]["terms"][0].get("var") else None
            if t_exp in src_exps:
                # Try a different exponent
                t_src, t_ans, _ = generate_single_term_diff(var)
            src_terms.append(t_src)
            ans_terms.append(t_ans)

    if not ans_terms:
        ans_terms = [{"numi": {"terms": [{"coeff": 0}]}, "deno": 1}]

    return src_terms, ans_terms, 4  # rule_id 4 = sum_rule


def generate_negative_exp_diff(var="x"):
    """Generate a differentiation problem with negative exponents."""
    neg_exps = [e for e in SAFE_EXPONENTS if e < 0]
    for _ in range(100):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(neg_exps)
        if _output_in_vocab(coeff, power):
            src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
            new_exp = power - 1
            ans = {"numi": {"terms": [{"coeff": coeff * power, "var": {var: new_exp}}]}, "deno": 1}
            return src, ans, 0  # power_rule
    return {"numi": {"terms": [{"coeff": 1, "var": {var: -1}}]}, "deno": 1}, {"numi": {"terms": [{"coeff": -1, "var": {var: -2}}]}, "deno": 1}, 0


def generate_multivar_diff():
    """Generate a multi-variable partial differentiation problem."""
    var = random.choice(VARIABLES)
    other_vars = [v for v in VARIABLES if v != var]

    src_terms = []
    ans_terms = []

    # Term with the target variable
    t_src, t_ans, _ = generate_single_term_diff(var)
    src_terms.append(t_src)
    ans_terms.append(t_ans)

    # Term with another variable (treated as constant → differentiates to 0)
    if other_vars:
        ov = random.choice(other_vars)
        c = random.choice(SAFE_NONZERO_COEFFS)
        p = random.choice(SAFE_POS_EXPONENTS)
        src_terms.append({"numi": {"terms": [{"coeff": c, "var": {ov: p}}]}, "deno": 1})
        # This term vanishes under d/d(var)

    if not ans_terms:
        ans_terms = [{"numi": {"terms": [{"coeff": 0}]}, "deno": 1}]

    return src_terms, ans_terms, var, 7  # rule_id 7 = partial_derivative


def generate_slang_dataset():
    print("[Dataset Engine] Programmatically synthesizing 100k-row expanded SLaNg dataset...")
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    random.seed(42)  # Reproducible

    # ── Distribution of problem types ─────────────────────────────────────────
    # 35k single-term power rule
    # 25k multi-term polynomial (sum rule)
    # 10k constant terms
    # 10k negative exponent
    # 20k multi-variable partial derivatives
    # Total: 100k

    # 1. Single-term power rule (35k)
    for _ in range(35000):
        var = random.choice(VARIABLES[:1])  # mostly x for single-term
        src, ans, rule_id = generate_single_term_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 2. Multi-term polynomial / sum rule (25k)
    for _ in range(25000):
        var = random.choice(VARIABLES[:1])
        src_terms, ans_terms, rule_id = generate_multi_term_diff(var)
        # Wrap as single expression: use first term as the src expr
        # For multi-term, we use the first term as op node expr
        # and the answer is the first derivative term
        # (The model sees individual term-level examples)
        src_op = {"op": "diff", "var": var, "expr": src_terms[0]}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1},
            "tgt_output_tokens": ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1},
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 3. Constant terms (10k)
    for _ in range(10000):
        src, ans, rule_id = generate_constant_term()
        var = random.choice(VARIABLES[:1])
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 4. Negative exponents (10k)
    for _ in range(10000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_negative_exp_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 5. Multi-variable partial derivatives (20k)
    for _ in range(20000):
        src_terms, ans_terms, var, rule_id = generate_multivar_diff()
        src_op = {"op": "diff", "var": var, "expr": src_terms[0]}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1},
            "tgt_output_tokens": ans_terms[0] if ans_terms else {"coeff": 0},
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    random.shuffle(dataset)

    with open("data/slang_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    for name, split_data in [("train", dataset[:90000]), ("val", dataset[90000:95000]), ("test", dataset[95000:])]:
        with open(splits_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")

    # Print coverage stats
    rule_counts = {}
    for item in dataset:
        rid = item["rule_ids"]
        rule_counts[rid] = rule_counts.get(rid, 0) + 1

    print(f"[Dataset Engine] 100,000 expanded lines generated successfully.")
    print(f"   Rule distribution: {rule_counts}")
    print(f"   Coefficient range: {min(SAFE_COEFFS)} to {max(SAFE_COEFFS)}")
    print(f"   Exponent range: {min(SAFE_EXPONENTS)} to {max(SAFE_EXPONENTS)}")
    print(f"   Variables: {VARIABLES}")


if __name__ == "__main__":
    generate_slang_dataset()
