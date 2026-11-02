"""
function_sequences.py
Module A - Sequences of Functions
Topic 1 - Pointwise and Uniform Convergence

Covers:
  * definitions (pointwise / uniform)
  * supremum-norm check:  M_n = sup_{x in D} |f_n(x) - f(x)|,  uniform  <=>  M_n -> 0
  * uniform-limit continuity theorem (and its contrapositive as a disproof tool)
  * a catalog of standard counterexamples

This module is deliberately independent from the numerical-series tests
(ratio / root / comparison / ...). It never imports them. Series tests work on
a_n (numbers); this module works on f_n(x) (functions of x on a domain D).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

KIND = "function_sequence"  # use this to route requests (NOT the series tests)

x = sp.Symbol("x", real=True)
n = sp.Symbol("n", positive=True, integer=True)

Piece = Tuple[sp.Expr, sp.Set]
LimitSpec = Union[sp.Expr, int, float, List[Piece], None]

DEFINITIONS = {
    "pointwise": (
        "f_n -> f pointwise on D if for every x in D and every eps > 0 there is "
        "N = N(eps, x) such that |f_n(x) - f(x)| < eps for all n >= N."
    ),
    "uniform": (
        "f_n -> f uniformly on D if for every eps > 0 there is N = N(eps) "
        "(independent of x) such that |f_n(x) - f(x)| < eps for all n >= N and all x in D."
    ),
    "sup_norm": (
        "f_n -> f uniformly on D  <=>  M_n = sup_{x in D} |f_n(x) - f(x)| -> 0 as n -> oo."
    ),
    "continuity_theorem": (
        "If every f_n is continuous on D and f_n -> f uniformly on D, then f is continuous on D."
    ),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bounds(I: sp.Set):
    """Return (a, b, left_open, right_open) for an Interval or single point."""
    if isinstance(I, sp.FiniteSet):
        p = list(I)[0]
        return p, p, False, False
    return I.start, I.end, bool(I.left_open), bool(I.right_open)


def _pieces(f: LimitSpec, domain: sp.Set) -> List[Piece]:
    if isinstance(f, list):
        return [(sp.sympify(e), I) for e, I in f]
    return [(sp.sympify(f), domain)]


def _fl(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def _sample_points(I: sp.Set) -> List[sp.Expr]:
    """A few exact sample points that lie INSIDE the piece."""
    a, b, lo, ro = _bounds(I)
    if a == b:
        return [a]
    pts: List[sp.Expr] = []
    a_inf, b_inf = a == -sp.oo, b == sp.oo
    if a_inf and b_inf:
        return [sp.Integer(v) for v in (-5, -1, 0, 1, 5)]
    if b_inf:
        base = a
        pts = [base + d for d in (0, 1, 2, 5)]
    elif a_inf:
        pts = [b - d for d in (0, 1, 2, 5)]
    else:
        w = b - a
        pts = [a, a + w / 4, a + w / 2, a + 3 * w / 4, a + sp.Rational(9, 10) * w, b]
    out = []
    for p in pts:
        if (p == a and lo) or (p == b and ro):
            continue
        out.append(sp.nsimplify(p))
    return out


def _inside_open(v: float, a, b) -> bool:
    lo = -np.inf if a == -sp.oo else float(a)
    hi = np.inf if b == sp.oo else float(b)
    return lo < v < hi


# --------------------------------------------------------------------------- #
# 1. pointwise convergence
# --------------------------------------------------------------------------- #
def pointwise_limit_at(fn: sp.Expr, x0) -> sp.Expr:
    """lim_{n->oo} f_n(x0). Substituting a concrete x0 makes sympy reliable."""
    return sp.simplify(sp.limit(fn.subs(x, x0), n, sp.oo))


def guess_pointwise_limit(fn: sp.Expr, domain: sp.Set) -> Optional[sp.Expr]:
    """Try the generic limit in n; accept it only if it survives a sample check."""
    try:
        cand = sp.limit(fn, n, sp.oo)
    except Exception:
        return None
    if cand.has(sp.Limit) or cand.has(sp.nan) or cand.has(sp.AccumBounds):
        return None
    ok, _ = check_pointwise(fn, [(cand, domain)])
    return cand if ok else None


def check_pointwise(fn: sp.Expr, pieces: List[Piece]) -> Tuple[bool, List[Dict[str, Any]]]:
    rows, all_ok = [], True
    for expr, I in pieces:
        for pt in _sample_points(I):
            try:
                lim = pointwise_limit_at(fn, pt)
                exp = sp.simplify(expr.subs(x, pt))
                ok = bool(sp.simplify(lim - exp) == 0)
            except Exception:
                lim, exp, ok = sp.nan, sp.nan, False
            rows.append({"x": str(pt), "lim f_n(x)": str(lim), "f(x)": str(exp), "ok": ok})
            all_ok &= ok
    return all_ok, rows


# --------------------------------------------------------------------------- #
# 2. supremum norm
# --------------------------------------------------------------------------- #
def _sup_candidates(diff: sp.Expr, I: sp.Set, n_probe: int = 60) -> List[sp.Expr]:
    """Values of |diff| at endpoints (or one-sided limits) and interior critical points."""
    a, b, lo, ro = _bounds(I)
    if a == b:
        return [sp.simplify(diff.subs(x, a))]
    vals: List[sp.Expr] = []
    for end, side, is_open in ((a, "+", lo), (b, "-", ro)):
        if end in (sp.oo, -sp.oo):
            v = sp.limit(diff, x, end)
        elif is_open:
            v = sp.limit(diff, x, end, side)
        else:
            v = diff.subs(x, end)
        vals.append(sp.simplify(v))
    try:
        crit = sp.solve(sp.diff(diff, x), x)
    except Exception:
        crit = []
    for c in crit:
        try:
            cv = complex(sp.N(c.subs(n, n_probe)))
        except Exception:
            continue
        if abs(cv.imag) > 1e-12 or not _inside_open(cv.real, a, b):
            continue
        vals.append(sp.simplify(diff.subs(x, c)))
    return vals


def sup_norm(fn: sp.Expr, pieces: List[Piece], n_probe: int = 60) -> sp.Expr:
    """M_n = sup |f_n - f| over D as an expression in n."""
    best, best_num = None, -1.0
    for expr, I in pieces:
        for v in _sup_candidates(sp.simplify(fn - expr), I, n_probe):
            av = sp.Abs(v)
            num = _fl(sp.N(av.subs(n, n_probe)), None)
            if num is None or np.isnan(num):
                continue
            if num > best_num:
                best, best_num = av, num
    if best is None:
        raise ValueError("could not compute the supremum norm symbolically")
    return sp.simplify(best)


def sup_norm_numeric(fn: sp.Expr, pieces: List[Piece], n_val: int,
                     grid: int = 4001, cap: float = 50.0) -> float:
    """Independent numeric cross-check of M_n (grid search, clustered near endpoints)."""
    fnn = fn.subs(n, n_val)
    best = 0.0
    for expr, I in pieces:
        a, b, lo, ro = _bounds(I)
        a_f = -cap if a == -sp.oo else float(a)
        b_f = cap if b == sp.oo else float(b)
        if a_f == b_f:
            xs = np.array([a_f])
        else:
            t = np.linspace(0, 1, grid)
            k = np.linspace(1, 12, 60)
            t = np.unique(np.concatenate([t, 10.0 ** -k, 1 - 10.0 ** -k]))
            xs = a_f + (b_f - a_f) * t
            if lo and a != -sp.oo:
                xs = xs[xs > a_f]
            if ro and b != sp.oo:
                xs = xs[xs < b_f]
        d = sp.lambdify(x, fnn - expr, "numpy")(xs)
        d = np.broadcast_to(np.asarray(d, dtype=float), xs.shape)
        best = max(best, float(np.nanmax(np.abs(d))))
    return best


# --------------------------------------------------------------------------- #
# 3. continuity of the limit
# --------------------------------------------------------------------------- #
def _continuous_on(expr: sp.Expr, I: sp.Set) -> bool:
    if isinstance(I, sp.FiniteSet):
        return True
    try:
        from sympy.calculus.util import continuous_domain
        return continuous_domain(expr, x, I) == I
    except Exception:
        return False


def is_continuous_pieces(pieces: List[Piece]) -> Tuple[bool, List[str]]:
    """Check continuity of a piecewise limit; returns (ok, list of discontinuity points)."""
    bad: List[str] = []
    for expr, I in pieces:
        if not _continuous_on(expr, I):
            bad.append(f"inside {I}")
    bounds = set()
    for _, I in pieces:
        a, b, _, _ = _bounds(I)
        for p in (a, b):
            if p not in (sp.oo, -sp.oo):
                bounds.add(p)
    for p in sorted(bounds, key=float):
        vals = []
        for expr, I in pieces:
            a, b, lo, ro = _bounds(I)
            inside = (a <= p <= b) and not (p == a and lo) and not (p == b and ro)
            if inside:
                vals.append(sp.simplify(expr.subs(x, p)))
            if a != b:
                if b == p:
                    vals.append(sp.simplify(sp.limit(expr, x, p, "-")))
                if a == p:
                    vals.append(sp.simplify(sp.limit(expr, x, p, "+")))
        if len({sp.nsimplify(v) for v in vals}) > 1:
            bad.append(f"x = {p}")
    return (not bad), bad


def fn_continuous(fn: sp.Expr, domain: sp.Set) -> bool:
    return _continuous_on(fn, domain)


# --------------------------------------------------------------------------- #
# 4. full analysis
# --------------------------------------------------------------------------- #
@dataclass
class ConvergenceReport:
    kind: str
    f_n: str
    domain: str
    limit_function: str
    pointwise_ok: bool
    pointwise_table: List[Dict[str, Any]]
    sup_norm: str
    sup_norm_limit: str
    uniform: bool
    fn_continuous: bool
    limit_continuous: bool
    discontinuities: List[str]
    continuity_verdict: str
    numeric_check: List[Tuple[int, float]]
    verdict: str
    steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def analyze(fn: Union[str, sp.Expr], domain: sp.Set, f: LimitSpec = None) -> ConvergenceReport:
    """
    fn      f_n(x) as a sympy expression in x and n (or a string like "x**n")
    domain  a sympy Interval, e.g. sp.Interval(0, 1)
    f       the pointwise limit: an expression, or a list of (expr, Interval) pieces.
            If None, the code tries to derive it.
    """
    if isinstance(fn, str):
        fn = sp.sympify(fn, locals={"x": x, "n": n})
    pieces = _pieces(f, domain) if f is not None else None
    if pieces is None:
        guess = guess_pointwise_limit(fn, domain)
        if guess is None:
            raise ValueError(
                "Could not derive the pointwise limit automatically. "
                "Pass it explicitly as f=expr or f=[(expr, Interval), ...]."
            )
        pieces = [(guess, domain)]

    steps = [
        "Step 1 - Definitions",
        "  Pointwise: " + DEFINITIONS["pointwise"],
        "  Uniform:   " + DEFINITIONS["uniform"],
    ]

    p_ok, table = check_pointwise(fn, pieces)
    f_txt = "; ".join(f"{e} on {I}" for e, I in pieces)
    steps += [
        f"Step 2 - Pointwise limit: fix x in D and let n -> oo.  f(x) = {f_txt}",
        "  Verified at sample points: " + ("all agree" if p_ok else "MISMATCH - check f"),
    ]

    M = sup_norm(fn, pieces)
    M_lim = sp.limit(M, n, sp.oo)
    uniform = bool(p_ok and M_lim == 0)
    steps += [
        f"Step 3 - Supremum norm: M_n = sup_D |f_n - f| = {M}",
        f"  lim M_n = {M_lim}  ->  " + ("0, so convergence is UNIFORM" if uniform
                                       else "not 0, so convergence is NOT uniform"),
    ]

    numeric = [(k, sup_norm_numeric(fn, pieces, k)) for k in (1, 10, 100, 1000)]
    steps.append("  Numeric cross-check of M_n: " + ", ".join(f"n={k}: {v:.4g}" for k, v in numeric))

    fn_ok = fn_continuous(fn, domain)
    f_ok, bad = is_continuous_pieces(pieces)
    if fn_ok and not f_ok:
        cont = ("Each f_n is continuous but the limit f is not -> by the uniform-limit "
                "theorem the convergence CANNOT be uniform (counterexample method).")
    elif fn_ok and f_ok:
        cont = ("f is continuous, which is consistent with uniform convergence but does NOT "
                "prove it (converse is false) - rely on the sup-norm result.")
    else:
        cont = "Some f_n are not continuous on D, so the continuity theorem does not apply."
    steps += ["Step 4 - Uniform-limit continuity theorem: " + DEFINITIONS["continuity_theorem"], "  " + cont]

    verdict = ("Uniformly convergent on D" if uniform
               else ("Pointwise but NOT uniformly convergent on D" if p_ok
                     else "Not convergent to the given f"))
    steps.append("Verdict: " + verdict)

    return ConvergenceReport(
        kind=KIND, f_n=str(fn), domain=str(domain), limit_function=f_txt,
        pointwise_ok=p_ok, pointwise_table=table, sup_norm=str(M),
        sup_norm_limit=str(M_lim), uniform=uniform, fn_continuous=fn_ok,
        limit_continuous=f_ok, discontinuities=bad, continuity_verdict=cont,
        numeric_check=numeric, verdict=verdict, steps=steps,
    )


# --------------------------------------------------------------------------- #
# 5. counterexample catalog
# --------------------------------------------------------------------------- #
R = sp.Interval(-sp.oo, sp.oo)
CATALOG: Dict[str, Dict[str, Any]] = {
    "xn_on_0_1": dict(
        fn=x ** n, domain=sp.Interval(0, 1),
        f=[(sp.Integer(0), sp.Interval.Ropen(0, 1)), (sp.Integer(1), sp.FiniteSet(1))],
        uniform=False, note="Continuous f_n, discontinuous limit at x=1."),
    "xn_on_0_half": dict(
        fn=x ** n, domain=sp.Interval(0, sp.Rational(1, 2)), f=0,
        uniform=True, note="Same sequence, domain bounded away from 1: M_n = (1/2)^n."),
    "n_x_exp_on_0_inf": dict(
        fn=n * x * sp.exp(-n * x ** 2), domain=sp.Interval(0, sp.oo), f=0,
        uniform=False, note="Limit is continuous, yet M_n -> oo: converse of the theorem is false."),
    "x_over_n_on_R": dict(
        fn=x / n, domain=R, f=0,
        uniform=False, note="Pointwise only on an unbounded domain; M_n = oo."),
    "x_over_n_on_0_1": dict(
        fn=x / n, domain=sp.Interval(0, 1), f=0,
        uniform=True, note="Bounded domain rescues uniformity: M_n = 1/n."),
    "sin_nx_over_n_on_R": dict(
        fn=sp.sin(n * x) / n, domain=R, f=0,
        uniform=True, note="|sin|<=1 gives M_n = 1/n."),
    "x_over_1_nx2_on_R": dict(
        fn=x / (1 + n * x ** 2), domain=R, f=0,
        uniform=True, note="Max at x = +-1/sqrt(n): M_n = 1/(2 sqrt(n))."),
}


def run_example(key: str) -> ConvergenceReport:
    ex = CATALOG[key]
    return analyze(ex["fn"], ex["domain"], ex["f"])


if __name__ == "__main__":
    for k in CATALOG:
        r = run_example(k)
        print(f"{k:22s} -> {r.verdict:45s} M_n = {r.sup_norm}")
