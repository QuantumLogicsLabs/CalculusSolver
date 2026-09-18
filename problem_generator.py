import json
import random
from pathlib import Path

SAFE_COEFFS = list(range(-10, 11)) + [12]
SAFE_POS_COEFFS = [c for c in SAFE_COEFFS if c > 0]
SAFE_NONZERO_COEFFS = [c for c in SAFE_COEFFS if c != 0]
SAFE_SYMMETRIC_NONZERO_COEFFS = [c for c in SAFE_NONZERO_COEFFS if -c in SAFE_COEFFS]
SAFE_EXPONENTS = list(range(-3, 6))
SAFE_POS_EXPONENTS = [e for e in SAFE_EXPONENTS if e >= 1]
VARIABLES = ["x", "y", "z"]


RULE_ID_TRIG = 10
RULE_ID_EXP = 11
RULE_ID_LOG = 12
RULE_ID_GRADIENT = 13
RULE_ID_TANGENT_LINE = 14


def _output_in_vocab(coeff, power):
    """Check that the derivative output (coeff*power, power-1) stays in vocab range."""
    out_coeff = coeff * power
    out_exp = power - 1
    return out_coeff in SAFE_COEFFS and out_exp in SAFE_EXPONENTS


def _integral_in_vocab(coeff, power):
    """Check that the integral output (coeff/(power+1), power+1) stays in vocab range."""
    new_power = power + 1
    if new_power == 0:
        return False
    new_coeff = coeff / new_power
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
            if ans["numi"]["terms"][0]["var"][var] == 0:
                ans = {"coeff": coeff * power}
            return src, ans, 0
    return {"numi": {"terms": [{"coeff": 2, "var": {var: 2}}]}, "deno": 1}, {"numi": {"terms": [{"coeff": 4, "var": {var: 1}}]}, "deno": 1}, 0


def generate_constant_term():
    """Generate a constant differentiation problem (derivative = 0)."""
    coeff = random.choice(SAFE_NONZERO_COEFFS)
    src = {"numi": {"terms": [{"coeff": coeff}]}, "deno": 1}
    ans = {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
    return src, ans, 5


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

    return src_terms, ans_terms, 4


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
            return src, ans, 0
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

    return src_terms, ans_terms, var, 7


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
            return src, ans, 6
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
    return expr, ans, RULE_ID_GRADIENT


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

        if slope not in SAFE_COEFFS or intercept not in SAFE_COEFFS:
            continue

        src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
        ans_terms = []
        if slope != 0:
            ans_terms.append({"coeff": int(slope), "var": {var: 1}})
        ans_terms.append({"coeff": int(intercept)})
        ans = {"numi": {"terms": ans_terms}, "deno": 1}

        src_op = {"op": "tangent_line", "var": var, "expr": src, "point": {var: x0}}

        return src_op, ans, x0, RULE_ID_TANGENT_LINE

    fallback_src = {"numi": {"terms": [{"coeff": 1, "var": {var: 2}}]}, "deno": 1}
    fallback_ans = {"numi": {"terms": [{"coeff": 2, "var": {var: 1}}, {"coeff": -1}]}, "deno": 1}
    fallback_src_op = {"op": "tangent_line", "var": var, "expr": fallback_src, "point": {var: 1}}

    return (
        fallback_src_op,
        fallback_ans,
        1,
        RULE_ID_TANGENT_LINE,
    )


def generate_slang_dataset():
    print("[Dataset Engine] Programmatically synthesizing expanded SLaNg dataset...")
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    random.seed(42)

    for _ in range(35000):
        var = random.choice(VARIABLES)
        src, ans, rule_id = generate_single_term_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(25000):
        var = random.choice(VARIABLES)
        src_terms, ans_terms, rule_id = generate_multi_term_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src_terms[0]}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1},
            "tgt_output_tokens": ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1},
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(10000):
        src, ans, rule_id = generate_constant_term()
        var = random.choice(VARIABLES)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(10000):
        var = random.choice(VARIABLES)
        src, ans, rule_id = generate_negative_exp_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(20000):
        src_terms, ans_terms, var, rule_id = generate_multivar_diff()
        src_op = {"op": "partial", "var": var, "expr": src_terms[0]}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1},
            "tgt_output_tokens": ans_terms[0] if ans_terms else {"coeff": 0},
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(5000):
        var = random.choice(VARIABLES)
        src, ans, rule_id = generate_sin_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(5000):
        var = random.choice(VARIABLES)
        src, ans, rule_id = generate_cos_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(5000):
        var = random.choice(VARIABLES)
        src, ans, rule_id = generate_tan_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(5000):
        var = random.choice(VARIABLES)
        src, ans, rule_id = generate_exp_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(5000):
        var = random.choice(VARIABLES)
        src, ans, rule_id = generate_ln_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(10000):
        var = random.choice(VARIABLES)
        src, ans, rule_id = generate_integrate_diff(var)
        src_op = {"op": "integrate", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(30000):
        expr, ans, rule_id = generate_gradient_diff()
        src_op = {"op": "gradient", "var": "x", "expr": expr}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    for _ in range(10000):
        var = random.choice(VARIABLES)
        src_op, ans, _, rule_id = generate_tangent_line_diff(var)
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
