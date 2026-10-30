import itertools
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tokenizer.slang_serializer import serialize_slang_math

# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Strict vocabulary loading
# ---------------------------------------------------------------------------
with open("tokenizer/vocab.json", "r", encoding="utf-8") as f:
    RAW_VOCAB = json.load(f)

VOCAB_SET: Set[str] = set()
for category in RAW_VOCAB.values():
    if isinstance(category, dict):
        VOCAB_SET.update(category.keys())
    elif isinstance(category, list):
        VOCAB_SET.update(category)


def is_valid_num(n: int) -> bool:
    """True only when the integer produces a token that exists in vocab.json."""
    return f"COEF:{n}" in VOCAB_SET or str(n) in VOCAB_SET


def safe_nonzero_coeffs() -> List[int]:
    return [c for c in SAFE_NONZERO_COEFFS if is_valid_num(c)]


def safe_pos_exponents() -> List[int]:
    return [e for e in SAFE_POS_EXPONENTS if is_valid_num(e)]


def _output_in_vocab(coeff: int, power: int) -> bool:
    """Both the product coeff*power and the residual exponent must be in vocab."""
    out_coeff = coeff * power
    out_exp = power - 1
    return is_valid_num(out_coeff) and (out_exp in SAFE_EXPONENTS)


def _integral_in_vocab(coeff: int, power: int) -> bool:
    new_power = power + 1
    if new_power == 0:
        return False
    new_coeff = coeff / new_power
    if not float(new_coeff).is_integer():
        return False
    return is_valid_num(int(new_coeff)) and new_power in SAFE_EXPONENTS


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def _differentiate_term(term: Dict, var: str) -> Optional[Dict]:
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
    if not is_valid_num(term.get("coeff", 0)):
        return False
    for q in term.get("var", {}).values():
        if q not in SAFE_EXPONENTS or q == 0:
            return False
    return True


def _binding_is_unambiguous(pairs: List[Tuple[int, int]]) -> bool:
    pairs = [(c, p) for c, p in pairs if p]
    if len(pairs) < 2:
        return True
    coeffs = [c for c, _ in pairs]
    powers = [p for _, p in pairs]
    correct = sorted((c * p, p - 1) for c, p in pairs)
    for perm in itertools.permutations(range(len(pairs))):
        if all(perm[i] == i for i in range(len(pairs))):
            continue
        alt = sorted(
            (coeffs[i] * powers[perm[i]], powers[perm[i]] - 1)
            for i in range(len(pairs))
        )
        if alt == correct:
            return False
    return True


def _term_pairs(terms: List[Dict], var: str) -> List[Tuple[int, int]]:
    return [
        (t.get("coeff", 0), t.get("var", {}).get(var, 0))
        for t in terms
        if t.get("var", {}).get(var, 0)
    ]


def load_quarantined_benchmarks() -> Set[str]:
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

    print(
        f"[Dataset Engine] Quarantined {len(benchmark_signatures)} "
        "benchmark signatures to guarantee 0% leakage."
    )
    return benchmark_signatures


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def generate_single_term_diff(var: str = "x"):
    valid_coeffs = safe_nonzero_coeffs()
    valid_exps = safe_pos_exponents()
    for _ in range(200):
        coeff = random.choice(valid_coeffs)
        power = random.choice(valid_exps)
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
    coeff = random.choice(safe_nonzero_coeffs())
    src = {"numi": {"terms": [{"coeff": coeff}]}, "deno": 1}
    ans = {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
    return src, ans, RULE_ID_CONSTANT


def generate_multi_term_diff(var: str = "x", num_terms: Optional[int] = None):
    if num_terms is None:
        num_terms = random.randint(2, 3)

    src_terms: List[Dict] = []
    ans_terms: List[Dict] = []

    for i in range(num_terms):
        if i == num_terms - 1 and random.random() < 0.25:
            c_src, _, _ = generate_constant_term()
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

    if not _binding_is_unambiguous(_term_pairs(src_terms, var)):
        return None

    src_expr = {"numi": {"terms": src_terms}, "deno": 1}
    ans_expr = {"numi": {"terms": ans_terms}, "deno": 1}
    return [src_expr], [ans_expr], RULE_ID_SUM


def generate_negative_exp_diff(var: str = "x"):
    neg_exps = [e for e in SAFE_EXPONENTS if e < 0]
    valid_coeffs = safe_nonzero_coeffs()
    for _ in range(100):
        coeff = random.choice(valid_coeffs)
        power = random.choice(neg_exps)
        if _output_in_vocab(coeff, power):
            src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
            new_exp = power - 1
            ans = {
                "numi": {"terms": [{"coeff": coeff * power, "var": {var: new_exp}}]},
                "deno": 1,
            }
            return src, ans, RULE_ID_POWER
    return (
        {"numi": {"terms": [{"coeff": 1, "var": {var: -1}}]}, "deno": 1},
        {"numi": {"terms": [{"coeff": -1, "var": {var: -2}}]}, "deno": 1},
        RULE_ID_POWER,
    )


def generate_multivar_diff():
    var = random.choice(VARIABLES)
    others = [v for v in VARIABLES if v != var]
    shape = random.choices(
        ["single", "two_target", "mixed"], weights=[35, 35, 30], k=1
    )[0]

    used_coeffs: Set[int] = set()
    used_exps: Set[int] = set()
    valid_coeffs = safe_nonzero_coeffs()
    valid_exps = safe_pos_exponents()

    def pick_pair():
        for _ in range(60):
            c = random.choice(valid_coeffs)
            p = random.choice(valid_exps)
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
            spare = [e for e in valid_exps if e not in used_exps]
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

    random.shuffle(src_terms)

    ans_terms = [d for d in (_differentiate_term(t, var) for t in src_terms) if d]
    if not ans_terms:
        ans_terms = [{"coeff": 0}]
    if not all(_term_in_vocab(t) for t in ans_terms):
        return None

    if not _binding_is_unambiguous(_term_pairs(src_terms, var)):
        return None

    coeffs = [t.get("coeff", 0) for t in src_terms]
    exponents = [p for t in src_terms for p in t.get("var", {}).values()]
    if len(set(coeffs)) != len(coeffs) or len(set(exponents)) != len(exponents):
        return None

    if len({v for t in src_terms for v in t.get("var", {})}) < 2:
        return None

    src_expr = {"numi": {"terms": src_terms}, "deno": 1}
    ans_expr = {"numi": {"terms": ans_terms}, "deno": 1}
    return [src_expr], [ans_expr], var, RULE_ID_PARTIAL


def generate_partial_constant_vanish(var: Optional[str] = None):
    """
    Generate two binding-contrast examples of partial differentiation where
    terms that are constant w.r.t. the differentiation variable vanish.

    Returns a list of two problems (same source terms, different bound vars)
    so the caller can extend the dataset with both rows.
    """
    # Build a multi-variable expression whose terms each involve a distinct variable.
    # This guarantees that differentiating w.r.t. any one of them makes the others vanish.
    vars_pool = list(VARIABLES)
    random.shuffle(vars_pool)

    # Decide how many terms (2 or 3) so we always have at least two distinct variables.
    num_terms = random.randint(2, 3)
    chosen_vars = vars_pool[:num_terms]

    valid_coeffs = safe_nonzero_coeffs()
    valid_exps = safe_pos_exponents()

    terms = []
    used_coeffs = set()
    for v in chosen_vars:
        candidates = [c for c in valid_coeffs if c not in used_coeffs] or valid_coeffs
        coeff = random.choice(candidates)
        used_coeffs.add(coeff)

        # Prefer power 1 (most common vanishing-case) but allow higher powers
        # only when the resulting coefficient stays inside the existing vocab.
        power = 1 if random.random() < 0.60 else random.choice(valid_exps)
        for _ in range(40):
            # CRITICAL: pass (coeff, power) — the helper multiplies them itself
            if _output_in_vocab(coeff, power):
                break
            power = 1 if random.random() < 0.60 else random.choice(valid_exps)
        else:
            # Absolute fallback that is guaranteed to be in vocab
            coeff, power = 2, 1

        terms.append({"coeff": coeff, "var": {v: power}})

    # Step 4.1 – shuffle so the surviving term is not always first.
    random.shuffle(terms)

    variables_in_expr = [list(t["var"].keys())[0] for t in terms]

    def _build_partial_problem(src_terms, bind_var):
        # Locate the (unique) term that depends on bind_var; all others vanish.
        target = next(t for t in src_terms if bind_var in t["var"])
        power = target["var"][bind_var]
        ans_coeff = target["coeff"] * power

        # Final safety net – never emit a coefficient absent from vocab.json
        if not is_valid_num(ans_coeff):
            # Should never happen after the loop above, but guard anyway
            ans_coeff = target["coeff"]  # power == 1 case is always safe
            power = 1

        if power == 1:
            ans_terms = [{"coeff": ans_coeff}]
        else:
            ans_terms = [{"coeff": ans_coeff, "var": {bind_var: power - 1}}]

        src_expr = {"numi": {"terms": src_terms}, "deno": 1}
        ans_expr = {"numi": {"terms": ans_terms}, "deno": 1}
        return [src_expr], [ans_expr], bind_var, RULE_ID_PARTIAL

    # Step 4.2 – binding-contrast twin: same terms, different bound variable.
    bind_var = random.choice(variables_in_expr)
    other_var = random.choice([v for v in variables_in_expr if v != bind_var])

    problem_a = _build_partial_problem(terms, bind_var)
    problem_b = _build_partial_problem(terms, other_var)

    return [problem_a, problem_b]


# ---------------------------------------------------------------------------
# Variable-binding contrast pairs
# ---------------------------------------------------------------------------
# One multivariable expression, differentiated with respect to two different
# variables. Everything about the input is identical except ``var``; the
# target changes only as dictated by that variable. Training minibatches carry
# both members together (see training/contrast.py).
CONTRAST_SHAPES = ["split", "split_plus", "bystander", "mixed"]
CONTRAST_SHAPE_WEIGHTS = [25, 25, 20, 30]
# Longest serialized expression/answer we allow: max_len (80) minus BOS, EOS
# and the rule prefix token that train.py adds to decoder sequences.
CONTRAST_MAX_TOKENS = 77


def _derivative_expr(terms: List[Dict], var: str) -> Dict:
    ans_terms = [d for d in (_differentiate_term(t, var) for t in terms) if d]
    return {"numi": {"terms": ans_terms or [{"coeff": 0}]}, "deno": 1}


def _expr_signature(expr: Dict) -> str:
    return " ".join(serialize_slang_math(expr))


def generate_binding_contrast_pair(rng=None) -> Optional[Dict]:
    """Return one contrast pair, or None when the random draw is rejected.

    Result: ``{"expr", "shape", "pair": [(var_a, ans_a), (var_b, ans_b)]}``.
    The two answers are derivatives of the SAME ``expr`` with respect to two
    different variables, so they differ only through the bound variable.
    """
    rng = rng or random
    va, vb = rng.sample(VARIABLES, 2)
    vc = next(v for v in VARIABLES if v not in (va, vb))
    valid_coeffs = safe_nonzero_coeffs()
    valid_exps = safe_pos_exponents()
    used_coeffs: Set[int] = set()
    used_exps: Set[int] = set()

    def pick_pair():
        for _ in range(60):
            c = rng.choice(valid_coeffs)
            p = rng.choice(valid_exps)
            if c in used_coeffs or p in used_exps or not _output_in_vocab(c, p):
                continue
            used_coeffs.add(c)
            used_exps.add(p)
            return c, p
        return None

    shape = rng.choices(CONTRAST_SHAPES, weights=CONTRAST_SHAPE_WEIGHTS, k=1)[0]
    if shape == "split":
        layout = [(va,), (vb,)]
    elif shape == "split_plus":
        layout = rng.choice([[(va,), (va,), (vb,)], [(va,), (vb,), (vb,)]])
    elif shape == "bystander":
        layout = [(va,), (vb,), (vc,)]  # the vc term vanishes under either binding
    else:
        layout = rng.choice(
            [[(va, vb), (va,)], [(va, vb), (vb,)], [(va, vb), (va,), (vb,)]]
        )

    terms: List[Dict] = []
    for term_vars in layout:
        pair = pick_pair()
        if pair is None:
            return None
        c, p = pair
        if len(term_vars) == 1:
            powers = {term_vars[0]: p}
        else:
            spare = [e for e in valid_exps if e not in used_exps and is_valid_num(c * e)]
            if not spare:
                return None
            q = rng.choice(spare)
            used_exps.add(q)
            powers = {term_vars[0]: p, term_vars[1]: q}
        terms.append({"coeff": c, "var": dict(sorted(powers.items()))})

    rng.shuffle(terms)

    for v in (va, vb):
        if not _binding_is_unambiguous(_term_pairs(terms, v)):
            return None

    ans_a = _derivative_expr(terms, va)
    ans_b = _derivative_expr(terms, vb)
    for ans in (ans_a, ans_b):
        if not all(_term_in_vocab(t) for t in ans["numi"]["terms"]):
            return None
    if ans_a == ans_b:
        return None

    expr = {"numi": {"terms": terms}, "deno": 1}
    try:
        lengths = [len(serialize_slang_math(x)) for x in (expr, ans_a, ans_b)]
    except Exception:
        return None
    if max(lengths) > CONTRAST_MAX_TOKENS:
        return None

    pair = [(va, ans_a), (vb, ans_b)]
    rng.shuffle(pair)
    return {"expr": expr, "shape": shape, "pair": pair}


def _heldout_signatures(paths) -> Tuple[Set[str], Set[str]]:
    """(full-input signatures, partial-expression signatures) of existing splits."""
    inputs: Set[str] = set()
    exprs: Set[str] = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    src = json.loads(line)["src_tokens"]
                    inputs.add(" ".join(serialize_slang_math(src)))
                    if isinstance(src, dict) and src.get("op") == "partial":
                        exprs.add(_expr_signature(src["expr"]))
                except Exception:
                    continue
    return inputs, exprs


def _build_contrast_split(
    n_pairs: int,
    rng: random.Random,
    blocked_inputs: Set[str],
    blocked_exprs: Set[str],
    label: str,
) -> Tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    seen_exprs: Set[str] = set()
    stats = {"shapes": {}, "var_pairs": {}, "attempts": 0, "target_first": 0, "target_slots": 0}
    max_attempts = max(n_pairs * 200, 1000)

    group = 0
    while group < n_pairs and stats["attempts"] < max_attempts:
        stats["attempts"] += 1
        made = generate_binding_contrast_pair(rng)
        if made is None:
            continue
        expr = made["expr"]
        expr_sig = _expr_signature(expr)
        if expr_sig in seen_exprs or expr_sig in blocked_exprs:
            continue

        candidate = []
        clash = False
        for var, ans in made["pair"]:
            src_op = {"op": "partial", "var": var, "expr": expr}
            try:
                sig = " ".join(serialize_slang_math(src_op))
                serialize_slang_math(ans)
            except Exception:
                clash = True
                break
            if sig in blocked_inputs:
                clash = True
                break
            candidate.append((var, src_op, ans))
        if clash:
            continue

        seen_exprs.add(expr_sig)
        for var, src_op, ans in candidate:
            rows.append(
                {
                    "src_tokens": src_op,
                    "tgt_input_tokens": ans,
                    "tgt_output_tokens": ans,
                    "rule_ids": RULE_ID_PARTIAL,
                    "verification_state": 1,
                    "contrast_group": group,
                    "contrast_var": var,
                }
            )
            idx = next(
                (i for i, t in enumerate(expr["numi"]["terms"]) if var in t.get("var", {})), None
            )
            stats["target_slots"] += 1
            stats["target_first"] += int(idx == 0)
        stats["shapes"][made["shape"]] = stats["shapes"].get(made["shape"], 0) + 1
        key = "".join(sorted(v for v, _, _ in candidate))
        stats["var_pairs"][key] = stats["var_pairs"].get(key, 0) + 1
        group += 1

    blocked_exprs |= seen_exprs
    print(
        f"  - contrast_{label}: {group}/{n_pairs} pairs ({len(rows)} rows), "
        f"attempts={stats['attempts']}, shapes={stats['shapes']}, "
        f"var_pairs={stats['var_pairs']}, "
        f"target_term_first={stats['target_first'] / max(stats['target_slots'], 1):.0%}"
    )
    return rows, stats


def generate_contrast_dataset(
    train_pairs: int = 8000,
    val_pairs: int = 400,
    seed: int = 1234,
    splits_dir: str = "data/splits",
    exclude_main_splits: bool = True,
    write: bool = True,
) -> Dict[str, List[Dict]]:
    """Build ``contrast_train.jsonl`` / ``contrast_val.jsonl``.

    Uses its own seeded ``random.Random`` so it never disturbs (or depends on)
    the global RNG stream that produces the main dataset. Validation pairs are
    drawn first and blocked from every other split by expression, so no
    validation expression is ever seen in training; benchmark inputs are
    quarantined exactly as in the main builder.
    """
    print("[Contrast Engine] Building variable-binding contrast pairs...")
    rng = random.Random(seed)
    splits_path = Path(splits_dir)
    main_dir = Path("data/splits")

    quarantined = load_quarantined_benchmarks()
    blocked_inputs: Set[str] = set(quarantined)
    train_exprs: Set[str] = set()
    val_exprs: Set[str] = set()
    if exclude_main_splits:
        held_in, held_ex = _heldout_signatures(
            [main_dir / "val.jsonl", main_dir / "test.jsonl"]
        )
        blocked_inputs |= held_in
        train_exprs |= held_ex
        val_exprs |= held_ex
        _, main_train_ex = _heldout_signatures([main_dir / "train.jsonl"])
        val_exprs |= main_train_ex

    val_rows, _ = _build_contrast_split(val_pairs, rng, blocked_inputs, val_exprs, "val")
    # Contrast-val expressions stay out of contrast-train.
    train_exprs |= {_expr_signature(r["src_tokens"]["expr"]) for r in val_rows}
    train_rows, _ = _build_contrast_split(train_pairs, rng, blocked_inputs, train_exprs, "train")

    result = {"train": train_rows, "val": val_rows}
    if write:
        splits_path.mkdir(parents=True, exist_ok=True)
        for name, rows in result.items():
            with open(splits_path / f"contrast_{name}.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"  -> contrast_{name}.jsonl: {len(rows)} rows")
    return result


def generate_sin_diff(var: str = "x"):
    k = random.choice(safe_nonzero_coeffs())
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "sin", "expr": inner}
    ans = {"op": "cos", "expr": inner}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_TRIG


def generate_cos_diff(var: str = "x"):
    k = random.choice([c for c in SAFE_SYMMETRIC_NONZERO_COEFFS if is_valid_num(c)])
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "cos", "expr": inner}
    ans = {"op": "sin", "expr": inner, "coeff": -k}
    return src, ans, RULE_ID_TRIG


def generate_tan_diff(var: str = "x"):
    k = random.choice(safe_nonzero_coeffs())
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "tan", "expr": inner}
    ans = {"op": "sec", "expr": inner, "power": 2}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_TRIG


def generate_exp_diff(var: str = "x"):
    k = random.choice(safe_nonzero_coeffs())
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "exp", "expr": inner}
    ans = {"op": "exp", "expr": inner}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_EXP


def generate_ln_diff(var: str = "x"):
    k = random.choice(safe_nonzero_coeffs())
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "ln", "expr": inner}
    ans = {
        "numi": {"terms": [{"coeff": 1}]},
        "deno": {"terms": [{"coeff": 1, "var": {var: 1}}]},
    }
    return src, ans, RULE_ID_LOG


def generate_integrate_diff(var: str = "x"):
    valid_coeffs = safe_nonzero_coeffs()
    for _ in range(100):
        coeff = random.choice(valid_coeffs)
        power = random.choice(SAFE_EXPONENTS)
        if power == -1:
            continue
        if _integral_in_vocab(coeff, power):
            new_power = power + 1
            new_coeff = int(coeff / new_power)
            if power == 0:
                src = {"numi": {"terms": [{"coeff": coeff}]}, "deno": 1}
            else:
                src = {
                    "numi": {"terms": [{"coeff": coeff, "var": {var: power}}]},
                    "deno": 1,
                }
            ans = {
                "numi": {"terms": [{"coeff": new_coeff, "var": {var: new_power}}]},
                "deno": 1,
            }
            return src, ans, RULE_ID_INTEGRAL
    return (
        {"numi": {"terms": [{"coeff": 4, "var": {var: 3}}]}, "deno": 1},
        {"numi": {"terms": [{"coeff": 1, "var": {var: 4}}]}, "deno": 1},
        RULE_ID_INTEGRAL,
    )


def generate_multi_term_integrate(var: str = "x", num_terms: Optional[int] = None):
    if num_terms is None:
        num_terms = random.randint(2, 3)

    src_terms: List[Dict] = []
    ans_terms: List[Dict] = []

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


def _choose_2var_pair():
    # even split across all three pairs — matches benchmark distribution
    return random.choice([("x", "y"), ("x", "z"), ("y", "z")])


def generate_gradient_diff():
    vx, vy = _choose_2var_pair()
    valid_coeffs = safe_nonzero_coeffs()
    valid_exps = safe_pos_exponents()

    for _ in range(200):
        cx = random.choice(valid_coeffs)
        px = random.choice(valid_exps)
        cy = random.choice(valid_coeffs)
        py = random.choice(valid_exps)
        if _output_in_vocab(cx, px) and _output_in_vocab(cy, py):
            break
    else:
        cx, px, cy, py = 2, 2, 3, 1

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
    valid_coeffs = safe_nonzero_coeffs()
    valid_exps = safe_pos_exponents()

    for _ in range(200):
        cx = random.choice(valid_coeffs)
        px = random.choice(valid_exps)
        cy = random.choice(valid_coeffs)
        py = random.choice(valid_exps)
        cz = random.choice(valid_coeffs)
        pz = random.choice(valid_exps)
        if (
            _output_in_vocab(cx, px)
            and _output_in_vocab(cy, py)
            and _output_in_vocab(cz, pz)
        ):
            break
    else:
        cx, px, cy, py, cz, pz = 2, 2, 3, 1, 1, 3

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

    def make_partial(c: int, p: int, v: str) -> Dict:
        term = {"coeff": c * p}
        if p - 1 > 0:
            term["var"] = {v: p - 1}
        return {"numi": {"terms": [term]}, "deno": 1}

    ans = {
        "gradient": {
            vx: make_partial(cx, px, vx),
            vy: make_partial(cy, py, vy),
            vz: make_partial(cz, pz, vz),
        }
    }
    return expr, ans, RULE_ID_GRADIENT


def generate_gradient_diff_1var():
    """Single-variable gradient (forces the model to handle 1-component output)."""
    v = random.choice(VARIABLES)
    valid_coeffs = safe_nonzero_coeffs()
    valid_exps = safe_pos_exponents()

    for _ in range(500):
        c = random.choice(valid_coeffs)
        p = random.choice(valid_exps)
        if is_valid_num(c * p) and is_valid_num(max(1, p - 1)):
            break
    else:
        c, p = 2, 2

    expr = {"numi": {"terms": [{"coeff": c, "var": {v: p}}]}, "deno": 1}

    if p - 1 == 0:
        d = {"numi": {"terms": [{"coeff": c * p}]}, "deno": 1}
    else:
        d = {"numi": {"terms": [{"coeff": c * p, "var": {v: p - 1}}]}, "deno": 1}

    return expr, {"gradient": {v: d}}, RULE_ID_GRADIENT


def generate_tangent_line_diff(var: str = "x"):
    valid_coeffs = safe_nonzero_coeffs()
    valid_exps = safe_pos_exponents()

    for _ in range(300):
        coeff = random.choice(valid_coeffs)
        power = random.choice(valid_exps)
        x0 = random.choice(range(-5, 6))
        if not _output_in_vocab(coeff, power):
            continue

        slope = coeff * power * (x0 ** (power - 1)) if power >= 1 else 0
        y0 = coeff * (x0 ** power)
        intercept = y0 - slope * x0

        if not (is_valid_num(int(slope)) and is_valid_num(int(intercept))):
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
    fallback_ans = {
        "numi": {"terms": [{"coeff": 2, "var": {var: 1}}, {"coeff": -1}]},
        "deno": 1,
    }
    fallback_src_op = {
        "op": "tangent_line",
        "var": var,
        "expr": fallback_src,
        "point": {var: 1},
    }
    return fallback_src_op, fallback_ans, 1, RULE_ID_TANGENT_LINE


def generate_multi_term_tangent_line(var: str = "x"):
    for _ in range(300):
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

        if not (is_valid_num(int(slope)) and is_valid_num(int(intercept))):
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


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------
def generate_slang_dataset(target_total: int = 75000):
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
            _ = serialize_slang_math(ans)
        except Exception:
            return False

        sig = " ".join(src_toks)
        if sig in seen_inputs or sig in quarantined:
            return False

        seen_inputs.add(sig)
        dataset.append(
            {
                "src_tokens": src_op,
                "tgt_input_tokens": ans,
                "tgt_output_tokens": ans,
                "rule_ids": rule_id,
                "verification_state": 1,
            }
        )
        return True

    categories = [
        ("single_term_diff", 1000),
        ("multi_term_diff", 8000),
        ("constant_term", 30),
        ("negative_exp_diff", 1500),
        ("multivar_diff", 16000),
        ("partial_constant_vanish", 14000),
        ("sin_diff", 80),
        ("cos_diff", 80),
        ("tan_diff", 80),
        ("exp_diff", 80),
        ("ln_diff", 80),
        ("integrate_single", 1000),
        ("integrate_multi", 8000),
        ("gradient_2var", 12000),
        ("gradient_3var", 10000),
        ("gradient_1var", 4000),
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
                    continue
                src_terms, ans_terms, rid = result
                src_op = {"op": "diff", "var": var, "expr": src_terms[0]}
                ans = (
                    ans_terms[0]
                    if ans_terms
                    else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
                )
            elif cat_name == "constant_term":
                src, ans, rid = generate_constant_term()
                src_op = {"op": "diff", "var": var, "expr": src}
            elif cat_name == "negative_exp_diff":
                src, ans, rid = generate_negative_exp_diff(var)
                src_op = {"op": "diff", "var": var, "expr": src}
            elif cat_name == "multivar_diff":
                result = generate_multivar_diff()
                if result is None:
                    continue
                src_terms, ans_terms, mvar, rid = result
                src_op = {"op": "partial", "var": mvar, "expr": src_terms[0]}
                ans = (
                    ans_terms[0]
                    if ans_terms
                    else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
                )
            elif cat_name == "partial_constant_vanish":
                # Function now returns a list of two (src_terms, ans_terms, mvar, rid) tuples
                results = generate_partial_constant_vanish(var)
                added_this_call = 0
                for result in results:
                    src_terms, ans_terms, mvar, rid = result
                    src_op = {"op": "partial", "var": mvar, "expr": src_terms[0]}
                    ans = (
                        ans_terms[0]
                        if ans_terms
                        else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
                    )
                    if try_add(src_op, ans, rid):
                        added_this_call += 1
                added_for_cat += added_this_call
                continue  # already counted via try_add; skip the common try_add below
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
                var_primary = list(ans["gradient"].keys())[0] if "gradient" in ans else var
                src_op = {"op": "gradient", "var": var_primary, "expr": expr}
            elif cat_name == "gradient_3var":
                expr, ans, rid = generate_gradient_diff_3var()
                var_primary = list(ans["gradient"].keys())[0] if "gradient" in ans else var
                src_op = {"op": "gradient", "var": var_primary, "expr": expr}
            elif cat_name == "gradient_1var":
                expr, ans, rid = generate_gradient_diff_1var()
                var_primary = list(ans["gradient"].keys())[0] if "gradient" in ans else var
                src_op = {"op": "gradient", "var": var_primary, "expr": expr}
            elif cat_name == "tangent_line_single":
                src_op, ans, _, rid = generate_tangent_line_diff(var)
            elif cat_name == "tangent_line_multi":
                src_op, ans, _, rid = generate_multi_term_tangent_line(var)
            else:
                continue

            if try_add(src_op, ans, rid):
                added_for_cat += 1

        print(
            f"  - {cat_name}: {added_for_cat}/{quota} unique examples "
            f"generated (attempts: {attempts})."
        )

    # Fill remaining quota
    supplement_types = [
        "multi_term_diff",
        "multivar_diff",
        "partial_constant_vanish",
        "integrate_multi",
        "gradient_2var",
        "gradient_3var",
        "gradient_1var",
        "tangent_line_multi",
    ]
    extra_attempts = 0
    while len(dataset) < target_total and extra_attempts < 100000:
        extra_attempts += 1
        st = random.choice(supplement_types)
        var = random.choice(VARIABLES)

        if st == "multi_term_diff":
            result = generate_multi_term_diff(var)
            if result is None:
                continue
            src_terms, ans_terms, rid = result
            src_op = {"op": "diff", "var": var, "expr": src_terms[0]}
            ans = (
                ans_terms[0]
                if ans_terms
                else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
            )
            try_add(src_op, ans, rid)
        elif st == "multivar_diff":
            result = generate_multivar_diff()
            if result is None:
                continue
            src_terms, ans_terms, mvar, rid = result
            src_op = {"op": "partial", "var": mvar, "expr": src_terms[0]}
            ans = (
                ans_terms[0]
                if ans_terms
                else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
            )
            try_add(src_op, ans, rid)
        elif st == "partial_constant_vanish":
            # Function now returns a list of two problems
            results = generate_partial_constant_vanish(var)
            for result in results:
                src_terms, ans_terms, mvar, rid = result
                src_op = {"op": "partial", "var": mvar, "expr": src_terms[0]}
                ans = (
                    ans_terms[0]
                    if ans_terms
                    else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
                )
                try_add(src_op, ans, rid)
        elif st == "integrate_multi":
            src, ans, rid = generate_multi_term_integrate(var)
            src_op = {"op": "integrate", "var": var, "expr": src}
            try_add(src_op, ans, rid)
        elif st == "gradient_2var":
            expr, ans, rid = generate_gradient_diff()
            var_primary = list(ans["gradient"].keys())[0] if "gradient" in ans else var
            src_op = {"op": "gradient", "var": var_primary, "expr": expr}
            try_add(src_op, ans, rid)
        elif st == "gradient_3var":
            expr, ans, rid = generate_gradient_diff_3var()
            var_primary = list(ans["gradient"].keys())[0] if "gradient" in ans else var
            src_op = {"op": "gradient", "var": var_primary, "expr": expr}
            try_add(src_op, ans, rid)
        elif st == "gradient_1var":
            expr, ans, rid = generate_gradient_diff_1var()
            var_primary = list(ans["gradient"].keys())[0] if "gradient" in ans else var
            src_op = {"op": "gradient", "var": var_primary, "expr": expr}
            try_add(src_op, ans, rid)
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
    print(
        "[Dataset Engine] Dataset generation and split complete "
        "with 0% benchmark leakage and 0 duplicates."
    )

    print("\n[Dataset Engine] Running anti-overfitting & clean-data verification...")
    try:
        from data_validator import validate_slang_data
        validate_slang_data()
    except Exception as e:
        print(f"[Dataset Engine] Validation warning: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLaNg dataset generator")
    parser.add_argument(
        "--contrast-only",
        action="store_true",
        help="Only (re)build data/splits/contrast_{train,val}.jsonl; leaves the main splits untouched",
    )
    parser.add_argument("--contrast-train-pairs", type=int, default=8000)
    parser.add_argument("--contrast-val-pairs", type=int, default=400)
    parser.add_argument("--contrast-seed", type=int, default=1234)
    cli_args = parser.parse_args()

    if cli_args.contrast_only:
        generate_contrast_dataset(
            train_pairs=cli_args.contrast_train_pairs,
            val_pairs=cli_args.contrast_val_pairs,
            seed=cli_args.contrast_seed,
        )
    else:
        generate_slang_dataset()