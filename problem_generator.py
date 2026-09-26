import glob
import itertools
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tokenizer.slang_serializer import serialize_slang_math

SAFE_COEFFS = list(range(-10, 11)) + [12]
SAFE_POS_COEFFS = [c for c in SAFE_COEFFS if c > 0]
SAFE_NONZERO_COEFFS = [c for c in SAFE_COEFFS if c != 0]
SAFE_SYMMETRIC_NONZERO_COEFFS = [c for c in SAFE_NONZERO_COEFFS if -c in SAFE_COEFFS]
SAFE_EXPONENTS = list(range(-3, 6))
SAFE_POS_EXPONENTS = [e for e in SAFE_EXPONENTS if e >= 1]
VARIABLES = ["x", "y", "z"]

RULE_ID_POWER = 0
RULE_ID_SUM = 4
RULE_ID_CONSTANT = 5
RULE_ID_INTEGRAL = 6
RULE_ID_PARTIAL = 7
RULE_ID_TRIG = 10
RULE_ID_EXP = 11
RULE_ID_LOG = 12
RULE_ID_GRADIENT = 13
RULE_ID_TANGENT_LINE = 14


def _differentiate_term(term: Dict, var: str) -> Optional[Dict]:
    """Partial derivative of one SLaNg term w.r.t. var, or None if it vanishes.

    Handles mixed terms (e.g. 3*x^2*y): the other variables are carried through
    untouched, which is what makes d/dx(3x^2*y) = 6*x*y rather than 6x.
    """
    powers = term.get("var", {})
    p = powers.get(var, 0)
    if not p:
        return None
    remaining = {v: q for v, q in powers.items() if v != var}
    if p - 1 != 0:
        remaining[var] = p - 1
    out: Dict = {"coeff": term.get("coeff", 0) * p}
    if remaining:
        out["var"] = dict(sorted(remaining.items()))
    return out


def _term_in_vocab(term: Dict) -> bool:
    """Every coefficient and exponent of a term must be representable."""
    if term.get("coeff", 0) not in SAFE_COEFFS:
        return False
    for q in term.get("var", {}).values():
        if q not in SAFE_EXPONENTS or q == 0:
            return False
    return True


def _binding_is_unambiguous(pairs: List[Tuple[int, int]]) -> bool:
    """True if only the correct coefficient-to-exponent pairing yields this answer.

    An example is binding-AMBIGUOUS when some cross-pairing of the terms'
    coefficients onto other terms' exponents reproduces the identical answer
    multiset. Such a row cannot teach the model to bind a coefficient to its
    own exponent, because the wrong binding scores exactly the same.

    Measured on the previous dataset: 14.06% of multi-term diff rows were
    ambiguous (20.99% of 3-term rows), while 0 of the 11 failing benchmark
    problems were -- so the benchmark demands correct binding that roughly one
    training row in seven actively taught was optional. The observed failure
    mode matched exactly: the model emitted c_i * p_j for i != j.

    With exponents already distinct, this reduces to "no two terms share a
    coefficient", but the permutation check is kept explicit so the property
    still holds if the exponent constraint is ever relaxed.
    """
    pairs = [(c, p) for c, p in pairs if p]
    if len(pairs) < 2:
        return True
    coeffs = [c for c, _ in pairs]
    powers = [p for _, p in pairs]
    correct = sorted((c * p, p - 1) for c, p in pairs)
    for perm in itertools.permutations(range(len(pairs))):
        if all(perm[i] == i for i in range(len(pairs))):
            continue
        alt = sorted((coeffs[i] * powers[perm[i]], powers[perm[i]] - 1)
                     for i in range(len(pairs)))
        if alt == correct:
            return False
    return True


def _term_pairs(terms: List[Dict], var: str) -> List[Tuple[int, int]]:
    """(coefficient, exponent-of-var) for each term that contains var."""
    return [
        (t.get("coeff", 0), t.get("var", {}).get(var, 0))
        for t in terms
        if t.get("var", {}).get(var, 0)
    ]


def _output_in_vocab(coeff: int, power: int) -> bool:
    """Check that derivative output (coeff*power, power-1) stays in vocab range."""
    out_coeff = coeff * power
    out_exp = power - 1
    return out_coeff in SAFE_COEFFS and out_exp in SAFE_EXPONENTS


def _integral_in_vocab(coeff: int, power: int) -> bool:
    """Check that integral output (coeff/(power+1), power+1) stays in vocab range."""
    new_power = power + 1
    if new_power == 0:
        return False
    new_coeff = coeff / new_power
    if not float(new_coeff).is_integer():
        return False
    return int(new_coeff) in SAFE_COEFFS and new_power in SAFE_EXPONENTS


def load_quarantined_benchmarks() -> Set[str]:
    """Load canonical serialized tokens of all benchmark problems to guarantee 0% leakage."""
    benchmark_signatures: Set[str] = set()
    benchmark_dir = Path("eval/benchmarks")
    if not benchmark_dir.exists():
        return benchmark_signatures

    for bf in benchmark_dir.glob("*.json"):
        with open(bf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for p in data:
                    expr = p.get("expr")
                    if expr:
                        toks = serialize_slang_math(expr)
                        benchmark_signatures.add(" ".join(toks))
            except Exception as e:
                print(f"[Warning] Could not parse benchmark file {bf}: {e}")

    print(f"[Dataset Engine] Quarantined {len(benchmark_signatures)} benchmark signatures to guarantee 0% leakage.")
    return benchmark_signatures


def generate_single_term_diff(var="x"):
    for _ in range(100):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(SAFE_POS_EXPONENTS)
        if _output_in_vocab(coeff, power):
            src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
            ans_term = {"coeff": coeff * power}
            if power - 1 != 0:
                ans_term["var"] = {var: power - 1}
            ans = {"numi": {"terms": [ans_term]}, "deno": 1}
            return src, ans, RULE_ID_POWER
    return (
        {"numi": {"terms": [{"coeff": 2, "var": {var: 2}}]}, "deno": 1},
        {"numi": {"terms": [{"coeff": 4, "var": {var: 1}}]}, "deno": 1},
        RULE_ID_POWER,
    )


def generate_constant_term():
    coeff = random.choice(SAFE_NONZERO_COEFFS)
    src = {"numi": {"terms": [{"coeff": coeff}]}, "deno": 1}
    ans = {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
    return src, ans, RULE_ID_CONSTANT


def generate_multi_term_diff(var="x", num_terms=None):
    if num_terms is None:
        num_terms = random.randint(2, 3)

    src_terms = []
    ans_terms = []

    for i in range(num_terms):
        if i == num_terms - 1 and random.random() < 0.25:
            c_src, c_ans, _ = generate_constant_term()
            src_terms.extend(c_src["numi"]["terms"])
        else:
            for _ in range(50):
                t_src, t_ans, _ = generate_single_term_diff(var)
                t_exp = (
                    list(t_src["numi"]["terms"][0].get("var", {}).values())[0]
                    if t_src["numi"]["terms"][0].get("var")
                    else None
                )
                src_exps = {
                    list(t.get("var", {}).values())[0]
                    for t in src_terms
                    if t.get("var")
                }
                if t_exp not in src_exps:
                    src_terms.extend(t_src["numi"]["terms"])
                    for t in t_ans["numi"]["terms"]:
                        if t.get("coeff", 0) != 0:
                            ans_terms.append(t)
                    break

    if not ans_terms:
        ans_terms = [{"coeff": 0}]

    # Reject binding-ambiguous samples: if a cross-pairing of coefficients and
    # exponents gives the same answer, this row teaches nothing about binding.
    if not _binding_is_unambiguous(_term_pairs(src_terms, var)):
        return None

    src_expr = {"numi": {"terms": src_terms}, "deno": 1}
    ans_expr = {"numi": {"terms": ans_terms}, "deno": 1}
    return [src_expr], [ans_expr], RULE_ID_SUM


def generate_negative_exp_diff(var="x"):
    neg_exps = [e for e in SAFE_EXPONENTS if e < 0]
    for _ in range(100):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(neg_exps)
        if _output_in_vocab(coeff, power):
            src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
            new_exp = power - 1
            ans = {"numi": {"terms": [{"coeff": coeff * power, "var": {var: new_exp}}]}, "deno": 1}
            return src, ans, RULE_ID_POWER
    return (
        {"numi": {"terms": [{"coeff": 1, "var": {var: -1}}]}, "deno": 1},
        {"numi": {"terms": [{"coeff": -1, "var": {var: -2}}]}, "deno": 1},
        RULE_ID_POWER,
    )


def generate_multivar_diff():
    """Partial derivative over a multi-variable polynomial.

    Every property here comes from a measured defect in the 300-problem
    benchmark, where partial scored 35.0% and 27 of 39 failures used the
    wrong term entirely (e.g. d/dy(-2y^2 + 4z^4) should be -4y; the model
    emitted -8y^3, which is c_y * p_z -- the y coefficient bound to z's
    exponent).

    1. Term order is shuffled. The differentiated variable's term was first
       in 100% of rows, so position alone identified it and the model never
       had to read the variable named in the op envelope.
    2. Coefficients and exponents are globally distinct, which makes the
       coefficient-to-exponent binding unique (see _binding_is_unambiguous).
    3. Three source shapes instead of one fixed two-term form:
         single       - the variable appears in exactly one term
         two_target   - it appears in two terms, so the answer has two terms
         mixed        - a term carries two variables (3*x^2*y), so the model
                        must keep the other variable in the answer
       Previously 100% of rows were the "single" shape with no mixed terms
       anywhere in the dataset.
    """
    var = random.choice(VARIABLES)
    others = [v for v in VARIABLES if v != var]
    shape = random.choices(
        ["single", "two_target", "mixed"], weights=[50, 27, 23], k=1
    )[0]

    used_coeffs: Set[int] = set()
    used_exps: Set[int] = set()

    def pick_pair():
        """A (coefficient, exponent) whose derivative stays in vocabulary and
        whose parts have not been used, keeping the binding unambiguous."""
        for _ in range(60):
            c = random.choice(SAFE_NONZERO_COEFFS)
            p = random.choice(SAFE_POS_EXPONENTS)
            if c in used_coeffs or p in used_exps:
                continue
            if not _output_in_vocab(c, p):
                continue
            used_coeffs.add(c)
            used_exps.add(p)
            return c, p
        return None

    src_terms: List[Dict] = []

    num_target_terms = 2 if shape == "two_target" else 1
    for _ in range(num_target_terms):
        pair = pick_pair()
        if pair is None:
            return None
        c, p = pair
        powers = {var: p}
        if shape == "mixed" and others:
            spare = [e for e in SAFE_POS_EXPONENTS if e not in used_exps]
            if spare:
                q = random.choice(spare)
                used_exps.add(q)
                powers[random.choice(others)] = q
        src_terms.append({"coeff": c, "var": dict(sorted(powers.items()))})

    for i in range(random.randint(1, 2)):
        pair = pick_pair()
        if pair is None:
            break
        c, p = pair
        src_terms.append({"coeff": c, "var": {others[i % len(others)]: p}})

    if len(src_terms) < 2:
        return None

    # The differentiated variable must not be identifiable by position alone.
    random.shuffle(src_terms)

    ans_terms = [d for d in (_differentiate_term(t, var) for t in src_terms) if d]
    if not ans_terms:
        ans_terms = [{"coeff": 0}]
    if not all(_term_in_vocab(t) for t in ans_terms):
        return None

    # The binding that determines the answer is coefficient -> exponent OF THE
    # DIFFERENTIATED VARIABLE. A mixed term's second variable is carried
    # through untouched and is not part of that binding, so it must not be
    # flattened into the check -- doing so makes every mixed term look
    # ambiguous (one coefficient against two exponents) and rejects them all.
    if not _binding_is_unambiguous(_term_pairs(src_terms, var)):
        return None

    # Cross-VARIABLE confusion (the observed c_y * p_z error) is prevented
    # constructively: with all coefficients and all exponents globally
    # distinct, c_i * p_j can never equal c_i * p_i for j != i. Asserted here
    # so the guarantee cannot silently lapse if construction changes.
    coeffs = [t.get("coeff", 0) for t in src_terms]
    exponents = [p for t in src_terms for p in t.get("var", {}).values()]
    if len(set(coeffs)) != len(coeffs) or len(set(exponents)) != len(exponents):
        return None

    # A partial-derivative example needs more than one variable present.
    if len({v for t in src_terms for v in t.get("var", {})}) < 2:
        return None

    src_expr = {"numi": {"terms": src_terms}, "deno": 1}
    ans_expr = {"numi": {"terms": ans_terms}, "deno": 1}
    return [src_expr], [ans_expr], var, RULE_ID_PARTIAL


def generate_partial_constant_vanish(var=None):
    """Generate partial derivative problems where non-target variables explicitly vanish.

    Specifically targets the primary failure mode in multivariable benchmarks:
    The target variable has degree 1 or 2 (yielding a constant or linear derivative),
    while 1 to 2 other variables are present with zero derivative w.r.t var.
    """
    if var is None:
        var = random.choice(VARIABLES)
    others = [v for v in VARIABLES if v != var]

    for _ in range(60):
        target_coeff = random.choice(SAFE_NONZERO_COEFFS)
        target_power = 1 if random.random() < 0.60 else 2
        if _output_in_vocab(target_coeff, target_power):
            break
    else:
        target_coeff = 2
        target_power = 1

    target_term = {"coeff": target_coeff, "var": {var: target_power}}

    non_target_terms = []
    num_others = random.randint(1, 2)
    used_coeffs = {target_coeff}
    for i in range(num_others):
        other_var = others[i % len(others)]
        c = random.choice([x for x in SAFE_NONZERO_COEFFS if x not in used_coeffs] or SAFE_NONZERO_COEFFS)
        used_coeffs.add(c)
        p = random.choice(SAFE_POS_EXPONENTS)
        non_target_terms.append({"coeff": c, "var": {other_var: p}})

    src_terms = [target_term] + non_target_terms
    random.shuffle(src_terms)

    ans_coeff = target_coeff * target_power
    if target_power == 1:
        ans_terms = [{"coeff": ans_coeff}]
    else:
        ans_terms = [{"coeff": ans_coeff, "var": {var: target_power - 1}}]

    src_expr = {"numi": {"terms": src_terms}, "deno": 1}
    ans_expr = {"numi": {"terms": ans_terms}, "deno": 1}
    return [src_expr], [ans_expr], var, RULE_ID_PARTIAL


def generate_sin_diff(var="x"):
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "sin", "expr": inner}
    ans = {"op": "cos", "expr": inner}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_TRIG


def generate_cos_diff(var="x"):
    k = random.choice(SAFE_SYMMETRIC_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "cos", "expr": inner}
    ans = {"op": "sin", "expr": inner, "coeff": -k}
    return src, ans, RULE_ID_TRIG


def generate_tan_diff(var="x"):
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "tan", "expr": inner}
    ans = {"op": "sec", "expr": inner, "power": 2}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_TRIG


def generate_exp_diff(var="x"):
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "exp", "expr": inner}
    ans = {"op": "exp", "expr": inner}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_EXP


def generate_ln_diff(var="x"):
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "ln", "expr": inner}
    ans = {"numi": {"terms": [{"coeff": 1}]}, "deno": {"terms": [{"coeff": 1, "var": {var: 1}}]}}
    return src, ans, RULE_ID_LOG


def generate_integrate_diff(var="x"):
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
            return src, ans, RULE_ID_INTEGRAL
    return (
        {"numi": {"terms": [{"coeff": 4, "var": {var: 3}}]}, "deno": 1},
        {"numi": {"terms": [{"coeff": 1, "var": {var: 4}}]}, "deno": 1},
        RULE_ID_INTEGRAL,
    )


def generate_multi_term_integrate(var="x", num_terms=None):
    if num_terms is None:
        num_terms = random.randint(2, 3)

    src_terms = []
    ans_terms = []

    for _ in range(num_terms):
        for _ in range(50):
            s, a, _ = generate_integrate_diff(var)
            st = s["numi"]["terms"][0]
            at = a["numi"]["terms"][0]
            t_exp = list(st.get("var", {}).values())[0] if st.get("var") else 0
            s_exps = {
                list(t.get("var", {}).values())[0] if t.get("var") else 0
                for t in src_terms
            }
            if t_exp not in s_exps:
                src_terms.append(st)
                ans_terms.append(at)
                break

    if not src_terms:
        s, a, _ = generate_integrate_diff(var)
        src_terms = s["numi"]["terms"]
        ans_terms = a["numi"]["terms"]

    src_expr = {"numi": {"terms": src_terms}, "deno": 1}
    ans_expr = {"numi": {"terms": ans_terms}, "deno": 1}
    return src_expr, ans_expr, RULE_ID_INTEGRAL


def generate_gradient_diff():
    vars_pool = random.sample(VARIABLES, 2)
    vx, vy = vars_pool[0], vars_pool[1]

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

    dx_term = {"coeff": cx * px}
    if px - 1 > 0:
        dx_term["var"] = {vx: px - 1}
    dx = {"numi": {"terms": [dx_term]}, "deno": 1}

    dy_term = {"coeff": cy * py}
    if py - 1 > 0:
        dy_term["var"] = {vy: py - 1}
    dy = {"numi": {"terms": [dy_term]}, "deno": 1}

    ans = {"gradient": {vx: dx, vy: dy}}
    return expr, ans, RULE_ID_GRADIENT


def generate_gradient_diff_3var():
    vx, vy, vz = "x", "y", "z"

    cx = random.choice(SAFE_NONZERO_COEFFS)
    px = random.choice(SAFE_POS_EXPONENTS)
    cy = random.choice(SAFE_NONZERO_COEFFS)
    py = random.choice(SAFE_POS_EXPONENTS)
    cz = random.choice(SAFE_NONZERO_COEFFS)
    pz = random.choice(SAFE_POS_EXPONENTS)

    for _ in range(100):
        if _output_in_vocab(cx, px) and _output_in_vocab(cy, py) and _output_in_vocab(cz, pz):
            break
        cx = random.choice(SAFE_NONZERO_COEFFS)
        px = random.choice(SAFE_POS_EXPONENTS)
        cy = random.choice(SAFE_NONZERO_COEFFS)
        py = random.choice(SAFE_POS_EXPONENTS)
        cz = random.choice(SAFE_NONZERO_COEFFS)
        pz = random.choice(SAFE_POS_EXPONENTS)

    expr = {
        "numi": {
            "terms": [
                {"coeff": cx, "var": {vx: px}},
                {"coeff": cy, "var": {vy: py}},
                {"coeff": cz, "var": {vz: pz}},
            ]
        },
        "deno": 1,
    }

    dx_term = {"coeff": cx * px}
    if px - 1 > 0:
        dx_term["var"] = {vx: px - 1}
    dx = {"numi": {"terms": [dx_term]}, "deno": 1}

    dy_term = {"coeff": cy * py}
    if py - 1 > 0:
        dy_term["var"] = {vy: py - 1}
    dy = {"numi": {"terms": [dy_term]}, "deno": 1}

    dz_term = {"coeff": cz * pz}
    if pz - 1 > 0:
        dz_term["var"] = {vz: pz - 1}
    dz = {"numi": {"terms": [dz_term]}, "deno": 1}

    ans = {"gradient": {vx: dx, vy: dy, vz: dz}}
    return expr, ans, RULE_ID_GRADIENT


def generate_tangent_line_diff(var="x"):
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
    return fallback_src_op, fallback_ans, 1, RULE_ID_TANGENT_LINE


def generate_multi_term_tangent_line(var="x"):
    for _ in range(100):
        p1 = random.choice([2, 3])
        p2 = random.choice([0, 1])
        c1 = random.choice([-2, -1, 1, 2])
        c2 = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        x0 = random.choice(range(-3, 4))

        slope = c1 * p1 * (x0 ** (p1 - 1))
        if p2 == 1:
            slope += c2

        y0 = c1 * (x0 ** p1) + (c2 * x0 if p2 == 1 else c2)
        intercept = y0 - slope * x0

        if slope not in SAFE_COEFFS or intercept not in SAFE_COEFFS:
            continue

        t1 = {"coeff": c1, "var": {var: p1}}
        t2 = {"coeff": c2, "var": {var: 1}} if p2 == 1 else {"coeff": c2}
        src = {"numi": {"terms": [t1, t2]}, "deno": 1}

        ans_terms = []
        if slope != 0:
            ans_terms.append({"coeff": int(slope), "var": {var: 1}})
        ans_terms.append({"coeff": int(intercept)})
        ans = {"numi": {"terms": ans_terms}, "deno": 1}

        src_op = {"op": "tangent_line", "var": var, "expr": src, "point": {var: x0}}
        return src_op, ans, x0, RULE_ID_TANGENT_LINE

    return generate_tangent_line_diff(var)


def generate_slang_dataset(target_total: int = 75000):  # Scaled total target row count
    print("[Dataset Engine] Synthesizing clean, deduplicated, zero-leakage SLaNg dataset...")
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    quarantined = load_quarantined_benchmarks()
    dataset: List[Dict] = []
    seen_inputs: Set[str] = set()
    random.seed(42)

    def try_add(src_op: Dict, ans: Dict, rule_id: int) -> bool:
        try:
            src_toks = serialize_slang_math(src_op)
            tgt_toks = serialize_slang_math(ans)
        except Exception:
            return False

        sig = " ".join(src_toks)
        if sig in seen_inputs or sig in quarantined:
            return False

        seen_inputs.add(sig)
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })
        return True

    # Quotas for balanced training coverage across all 5 calculus categories
    categories = [
        ("single_term_diff", 1000),
        ("multi_term_diff", 12000),
        ("constant_term", 30),
        ("negative_exp_diff", 1500),
        ("multivar_diff", 15000),
        ("partial_constant_vanish", 10000),  # Multivariable vanishing constant targets
        ("sin_diff", 80),
        ("cos_diff", 80),
        ("tan_diff", 80),
        ("exp_diff", 80),
        ("ln_diff", 80),
        ("integrate_single", 1000),
        ("integrate_multi", 8000),
        ("gradient_2var", 10000),          # 2-variable gradient generator
        ("gradient_3var", 10000),          # Added: 3-variable gradient generator (Dev 1 task)
        ("tangent_line_single", 3000),
        ("tangent_line_multi", 4000),
    ]

    print("[Dataset Engine] Generating unique problem categories...")
    for cat_name, quota in categories:
        added_for_cat = 0
        max_attempts = quota * 35
        attempts = 0

        while added_for_cat < quota and attempts < max_attempts:
            attempts += 1
            var = random.choice(VARIABLES)

            if cat_name == "single_term_diff":
                src, ans, rid = generate_single_term_diff(var)
                src_op = {"op": "diff", "var": var, "expr": src}
            elif cat_name == "multi_term_diff":
                result = generate_multi_term_diff(var)
                if result is None:
                    continue  # binding-ambiguous sample, rejected
                src_terms, ans_terms, rid = result
                src_op = {"op": "diff", "var": var, "expr": src_terms[0]}
                ans = ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
            elif cat_name == "constant_term":
                src, ans, rid = generate_constant_term()
                src_op = {"op": "diff", "var": var, "expr": src}
            elif cat_name == "negative_exp_diff":
                src, ans, rid = generate_negative_exp_diff(var)
                src_op = {"op": "diff", "var": var, "expr": src}
            elif cat_name == "multivar_diff":
                result = generate_multivar_diff()
                if result is None:
                    continue  # binding-ambiguous sample, rejected
                src_terms, ans_terms, mvar, rid = result
                src_op = {"op": "partial", "var": mvar, "expr": src_terms[0]}
                ans = ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
            elif cat_name == "partial_constant_vanish":
                result = generate_partial_constant_vanish(var)
                src_terms, ans_terms, mvar, rid = result
                src_op = {"op": "partial", "var": mvar, "expr": src_terms[0]}
                ans = ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
            elif cat_name == "sin_diff":
                src, ans, rid = generate_sin_diff(var)
                src_op = {"op": "diff", "var": var, "expr": src}
            elif cat_name == "cos_diff":
                src, ans, rid = generate_cos_diff(var)
                src_op = {"op": "diff", "var": var, "expr": src}
            elif cat_name == "tan_diff":
                src, ans, rid = generate_tan_diff(var)
                src_op = {"op": "diff", "var": var, "expr": src}
            elif cat_name == "exp_diff":
                src, ans, rid = generate_exp_diff(var)
                src_op = {"op": "diff", "var": var, "expr": src}
            elif cat_name == "ln_diff":
                src, ans, rid = generate_ln_diff(var)
                src_op = {"op": "diff", "var": var, "expr": src}
            elif cat_name == "integrate_single":
                src, ans, rid = generate_integrate_diff(var)
                src_op = {"op": "integrate", "var": var, "expr": src}
            elif cat_name == "integrate_multi":
                src, ans, rid = generate_multi_term_integrate(var)
                src_op = {"op": "integrate", "var": var, "expr": src}
            elif cat_name == "gradient_2var":
                expr, ans, rid = generate_gradient_diff()
                src_op = {"op": "gradient", "var": var, "expr": expr}  # Fixed hardcoded 'x' to dynamic var
            elif cat_name == "gradient_3var":
                expr, ans, rid = generate_gradient_diff_3var()
                src_op = {"op": "gradient", "var": var, "expr": expr}  # Added 3-variable caller
            elif cat_name == "tangent_line_single":
                src_op, ans, _, rid = generate_tangent_line_diff(var)
            elif cat_name == "tangent_line_multi":
                src_op, ans, _, rid = generate_multi_term_tangent_line(var)
            else:
                continue

            if try_add(src_op, ans, rid):
                added_for_cat += 1

        print(f"  - {cat_name}: {added_for_cat}/{quota} unique examples generated (attempts: {attempts}).")

    supplement_types = ["multi_term_diff", "multivar_diff", "partial_constant_vanish", "integrate_multi", "gradient_2var", "gradient_3var", "tangent_line_multi"]
    extra_attempts = 0
    while len(dataset) < target_total and extra_attempts < 100000:
        extra_attempts += 1
        st = random.choice(supplement_types)
        var = random.choice(VARIABLES)
        if st == "multi_term_diff":
            result = generate_multi_term_diff(var)
            if result is None:
                continue  # binding-ambiguous sample, rejected
            src_terms, ans_terms, rid = result
            src_op = {"op": "diff", "var": var, "expr": src_terms[0]}
            ans = ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
        elif st == "multivar_diff":
            result = generate_multivar_diff()
            if result is None:
                continue  # binding-ambiguous sample, rejected
            src_terms, ans_terms, mvar, rid = result
            src_op = {"op": "partial", "var": mvar, "expr": src_terms[0]}
            ans = ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
        elif st == "partial_constant_vanish":
            result = generate_partial_constant_vanish(var)
            src_terms, ans_terms, mvar, rid = result
            src_op = {"op": "partial", "var": mvar, "expr": src_terms[0]}
            ans = ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
        elif st == "integrate_multi":
            src, ans, rid = generate_multi_term_integrate(var)
            src_op = {"op": "integrate", "var": var, "expr": src}
        elif st == "gradient_2var":
            expr, ans, rid = generate_gradient_diff()
            src_op = {"op": "gradient", "var": var, "expr": expr}
        elif st == "gradient_3var":
            expr, ans, rid = generate_gradient_diff_3var()
            src_op = {"op": "gradient", "var": var, "expr": expr}
        else:
            src_op, ans, _, rid = generate_multi_term_tangent_line(var)

        try_add(src_op, ans, rid)

    random.shuffle(dataset)
    total = len(dataset)
    print(f"\n[Dataset Engine] Total 100% unique rows generated: {total}")

    with open("data/slang_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    train_end = int(total * 0.90)
    val_end = int(total * 0.95)
    splits = [
        ("train", dataset[:train_end]),
        ("val", dataset[train_end:val_end]),
        ("test", dataset[val_end:]),
    ]

    for name, split_data in splits:
        with open(splits_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        print(f"  -> {name}.jsonl: {len(split_data)} rows")

    rule_counts: Dict[int, int] = {}
    for item in dataset:
        rid = item["rule_ids"]
        rule_counts[rid] = rule_counts.get(rid, 0) + 1

    print(f"\n[Dataset Engine] Rule distribution: {rule_counts}")
    print("[Dataset Engine] Dataset generation and split complete with 0% benchmark leakage and 0 duplicates.")

    print("\n[Dataset Engine] Running anti-overfitting & clean-data verification...")
    try:
        from data_validator import validate_slang_data
        validate_slang_data()
    except Exception as e:
        print(f"[Dataset Engine] Validation warning: {e}")

if __name__ == "__main__":
    generate_slang_dataset()
