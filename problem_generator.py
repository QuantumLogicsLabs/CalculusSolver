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
# Symmetric subset: only used where a value AND its negation must both exist
# as vocab tokens (the coefficient range is asymmetric: -10 to 12, so 12's
# negation, -12, is not a valid token). Needed for cos's derivative, which
# introduces a negative sign via the coeff decorator.
SAFE_SYMMETRIC_NONZERO_COEFFS = [c for c in SAFE_NONZERO_COEFFS if -c in SAFE_COEFFS]
# Exponents: EXP:-3 to EXP:5 (integers only, skip EXP:OTHER)
SAFE_EXPONENTS = list(range(-3, 6))
# Positive exponents for power rule differentiation (need power >= 1 for non-trivial result)
SAFE_POS_EXPONENTS = [e for e in SAFE_EXPONENTS if e >= 1]
# Variables
VARIABLES = ["x", "y", "z"]

# ── Rule IDs ─────────────────────────────────────────────────────────────────
# IMPORTANT: these are NOT vocab token IDs. train.py's RuleHead classifier
# output has one neuron per entry in RULE_LABELS, which is built by sorting
# tokenizer/vocab.json's rule_tokens by vocab ID and taking each entry's
# *list position* as the classifier's target index (see train.py's
# flatten_vocab / RULE_LABELS construction). rule_ids written here must be
# that 0-based classifier index, not the raw vocab ID.
#
# Current rule_tokens (sorted by vocab ID) and their resulting classifier index:
#   RULE:power_rule (90)           -> index 0
#   RULE:chain_rule (91)           -> index 1
#   RULE:product_rule (92)         -> index 2
#   RULE:quotient_rule (93)        -> index 3
#   RULE:sum_rule (94)             -> index 4
#   RULE:constant_rule (95)        -> index 5
#   RULE:power_rule_integral (96)  -> index 6
#   RULE:partial_derivative (97)   -> index 7
#   RULE:lagrange_multiplier (98)  -> index 8
#   RULE:integration_by_parts (99) -> index 9
#   RULE:trig_rule (102)           -> index 10
#   RULE:exp_rule (103)            -> index 11
#   RULE:log_rule (104)            -> index 12

RULE_ID_TRIG = 10   # index of RULE:trig_rule -- sin, cos, tan
RULE_ID_EXP = 11    # index of RULE:exp_rule -- exp
RULE_ID_LOG = 12    # index of RULE:log_rule -- ln


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
        if i == num_terms - 1 and random.random() < 0.3:
            c_src, c_ans, _ = generate_constant_term()
            src_terms.append(c_src)
        else:
            t_src, t_ans, _ = generate_single_term_diff(var)
            src_exps = {list(t["numi"]["terms"][0].get("var", {}).values())[0] for t in src_terms if t["numi"]["terms"][0].get("var")}
            t_exp = list(t_src["numi"]["terms"][0].get("var", {}).values())[0] if t_src["numi"]["terms"][0].get("var") else None
            if t_exp in src_exps:
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

    t_src, t_ans, _ = generate_single_term_diff(var)
    src_terms.append(t_src)
    ans_terms.append(t_ans)

    if other_vars:
        ov = random.choice(other_vars)
        c = random.choice(SAFE_NONZERO_COEFFS)
        p = random.choice(SAFE_POS_EXPONENTS)
        src_terms.append({"numi": {"terms": [{"coeff": c, "var": {ov: p}}]}, "deno": 1})

    if not ans_terms:
        ans_terms = [{"numi": {"terms": [{"coeff": 0}]}, "deno": 1}]

    return src_terms, ans_terms, var, 7  # rule_id 7 = partial_derivative


def generate_sin_diff(var="x"):
    """d/dvar[sin(k*var)] = k*cos(k*var)."""
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "sin", "expr": inner}
    ans = {"op": "cos", "expr": inner}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_TRIG


def generate_cos_diff(var="x"):
    """d/dvar[cos(k*var)] = -k*sin(k*var)."""
    k = random.choice(SAFE_SYMMETRIC_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "cos", "expr": inner}
    ans = {"op": "sin", "expr": inner, "coeff": -k}
    return src, ans, RULE_ID_TRIG


def generate_tan_diff(var="x"):
    """d/dvar[tan(k*var)] = k*sec^2(k*var)."""
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "tan", "expr": inner}
    ans = {"op": "sec", "expr": inner, "power": 2}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_TRIG


def generate_exp_diff(var="x"):
    """d/dvar[exp(k*var)] = k*exp(k*var)."""
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "exp", "expr": inner}
    ans = {"op": "exp", "expr": inner}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_EXP


def generate_ln_diff(var="x"):
    """d/dvar[ln(k*var)] = 1/var."""
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "ln", "expr": inner}
    ans = {"numi": {"terms": [{"coeff": 1}]}, "deno": {"terms": [{"coeff": 1, "var": {var: 1}}]}}
    return src, ans, RULE_ID_LOG


def generate_integrate_diff(var="x"):
    """Generate a single-term power-rule integration problem."""
    for _ in range(100):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(SAFE_EXPONENTS)
        if power == -1:
            continue
        if _integral_in_vocab(coeff, power):
            new_power = power + 1
            new_coeff = int(coeff / new_power)
            if power == 0:
                src = {"numi": {"terms": [{"coeff": coeff}]}, "deno": 1}
            else:
                src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
            ans = {"numi": {"terms": [{"coeff": new_coeff, "var": {var: new_power}}]}, "deno": 1}
            return src, ans, 6  # rule_id 6 = power_rule_integral
    return (
        {"numi": {"terms": [{"coeff": 4, "var": {var: 3}}]}, "deno": 1},
        {"numi": {"terms": [{"coeff": 1, "var": {var: 4}}]}, "deno": 1},
        6,
    )


def generate_gradient_diff():
    """Generate a two-variable gradient problem."""
    vx, vy = "x", "y"

    cx = random.choice(SAFE_NONZERO_COEFFS)
    px = random.choice(SAFE_POS_EXPONENTS)
    cy = random.choice(SAFE_NONZERO_COEFFS)
    py = random.choice(SAFE_POS_EXPONENTS)

    for _ in range(100):
        if _output_in_vocab(cx, px) and _output_in_vocab(cy, py):
            break
        cx = random.choice(SAFE_NONZERO_COEFFS)
        px = random.choice(SAFE_POS_EXPONENTS)
        cy = random.choice(SAFE_NONZERO_COEFFS)
        py = random.choice(SAFE_POS_EXPONENTS)

    expr = {
        "numi": {
            "terms": [
                {"coeff": cx, "var": {vx: px}},
                {"coeff": cy, "var": {vy: py}},
            ]
        },
        "deno": 1,
    }

    dx = {"numi": {"terms": [{"coeff": cx * px, "var": {vx: px - 1}}]}, "deno": 1}
    dy = {"numi": {"terms": [{"coeff": cy * py, "var": {vy: py - 1}}]}, "deno": 1}

    ans = {"gradient": {vx: dx, vy: dy}}
    return expr, ans, 7  # rule_id 7 = partial_derivative


def generate_tangent_line_diff(var="x"):
    """Generate a tangent-line problem: find y = f'(x0)(x-x0) + f(x0) for a
    single-term polynomial f, evaluated at integer x0 in [-5, 5]."""
    for _ in range(100):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(SAFE_POS_EXPONENTS)
        x0 = random.choice(range(-5, 6))
        if not _output_in_vocab(coeff, power):
            continue

        slope = coeff * power * (x0 ** (power - 1)) if power >= 1 else 0
        y0 = coeff * (x0 ** power)
        intercept = y0 - slope * x0

        # Keep answer coefficients in vocab range
        if slope not in SAFE_COEFFS or intercept not in SAFE_COEFFS:
            continue

        src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
        ans_terms = []
        if slope != 0:
            ans_terms.append({"coeff": int(slope), "var": {var: 1}})
        ans_terms.append({"coeff": int(intercept)})
        ans = {"numi": {"terms": ans_terms}, "deno": 1}

        return src, ans, x0, 0  # rule_id 0 = power_rule

    # Fallback: f(x)=x^2 at x0=1 -> tangent line y = 2x - 1
    return (
        {"numi": {"terms": [{"coeff": 1, "var": {var: 2}}]}, "deno": 1},
        {"numi": {"terms": [{"coeff": 2, "var": {var: 1}}, {"coeff": -1}]}, "deno": 1},
        1,
        0,
    )


def generate_slang_dataset():
    print("[Dataset Engine] Programmatically synthesizing expanded SLaNg dataset...")
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    random.seed(42)  # Reproducible

    # 1. Single-term power rule (35k)
    for _ in range(35000):
        var = random.choice(VARIABLES[:1])
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

    # 6. Trig — sin (5k)
    for _ in range(5000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_sin_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 7. Trig — cos (5k)
    for _ in range(5000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_cos_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 8. Trig — tan (5k)
    for _ in range(5000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_tan_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 9. Exponential — exp (5k)
    for _ in range(5000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_exp_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 10. Logarithmic — ln (5k)
    for _ in range(5000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_ln_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 11. Integration — power rule integral (10k)
    for _ in range(10000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_integrate_diff(var)
        src_op = {"op": "integrate", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 12. Gradient (10k)
    for _ in range(10000):
        expr, ans, rule_id = generate_gradient_diff()
        src_op = {"op": "gradient", "var": "x", "expr": expr}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 13. Tangent line (10k)
    for _ in range(10000):
        var = random.choice(VARIABLES[:1])
        src, ans, x0, rule_id = generate_tangent_line_diff(var)
        src_op = {"op": "tangent_line", "var": var, "expr": src, "point": x0}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    random.shuffle(dataset)

    with open("data/slang_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    total = len(dataset)
    train_end = int(total * 0.90)
    val_end = int(total * 0.95)
    for name, split_data in [("train", dataset[:train_end]), ("val", dataset[train_end:val_end]), ("test", dataset[val_end:])]:
        with open(splits_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")

    # Print coverage stats
    rule_counts = {}
    for item in dataset:
        rid = item["rule_ids"]
        rule_counts[rid] = rule_counts.get(rid, 0) + 1

    print(f"[Dataset Engine] {total} expanded lines generated successfully.")
    print(f"   Rule distribution: {rule_counts}")
    print(f"   Coefficient range: {min(SAFE_COEFFS)} to {max(SAFE_COEFFS)}")
    print(f"   Exponent range: {min(SAFE_EXPONENTS)} to {max(SAFE_EXPONENTS)}")
    print(f"   Variables: {VARIABLES}")


if __name__ == "__main__":
    generate_slang_dataset()
