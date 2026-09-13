"""Small builders for well-formed SLaNg token sequences, shared by the
grammar and beam-search suites. Pure stdlib -- safe to import in CI.
"""


def term(coeff, *var_exp_pairs):
    """NODE:TERM COEF:<c> [VAR:<v> EXP:<e>]*"""
    tokens = ["NODE:TERM", f"COEF:{coeff}"]
    for name, power in var_exp_pairs:
        tokens += [f"VAR:{name}", f"EXP:{power}"]
    return tokens


def term_list(*terms):
    """term (STRUCT:SEP term)*"""
    tokens = []
    for i, t in enumerate(terms):
        if i:
            tokens.append("STRUCT:SEP")
        tokens += t
    return tokens


def frac(numerator, denominator):
    """A complete SLaNg fraction node."""
    return (
        ["NODE:FRAC", "STRUCT:OPEN", "STRUCT:NUMI", "STRUCT:OPEN"]
        + numerator
        + ["STRUCT:CLOSE", "STRUCT:SEP", "STRUCT:DENO", "STRUCT:OPEN"]
        + denominator
        + ["STRUCT:CLOSE", "STRUCT:CLOSE"]
    )


# 6x / 1
SIMPLE = frac(term_list(term(6, ("x", 1))), term_list(term(1)))

# (3x^2 + 5x - 7) / 1 -- legitimately repeats NODE:TERM, COEF, VAR, STRUCT:SEP
MULTI_TERM = frac(
    term_list(term(3, ("x", 2)), term(5, ("x", 1)), term(-7)),
    term_list(term(1)),
)


def gradient(*var_expr_pairs):
    """NODE:GRADIENT OPEN OPVAR:<v> <node> (SEP OPVAR:<v> <node>)* CLOSE"""
    tokens = ["NODE:GRADIENT", "STRUCT:OPEN"]
    for i, (name, expr) in enumerate(var_expr_pairs):
        if i:
            tokens.append("STRUCT:SEP")
        tokens += [f"OPVAR:{name}"] + expr
    return tokens + ["STRUCT:CLOSE"]


# grad of a 2-variable function: one fraction per variable
GRADIENT_2D = gradient(
    ("x", frac(term_list(term(10, ("x", 4))), term_list(term(1)))),
    ("y", frac(term_list(term(-4, ("y", 0))), term_list(term(1)))),
)
