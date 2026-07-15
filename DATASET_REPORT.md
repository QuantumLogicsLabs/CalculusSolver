# SLaNg Dataset Report

## 1. Dataset Scale
- **Total Records:** 125,000 unique calculus strings.
- **Data Splits:** 90% Train (112,500), 5% Val (6,250), 5% Test (6,250).

## 2. Rule Coverage
- **Expanded Polynomials:** The synthesizer generates operations representing the calculus power rule, sum rule, constant rule, and partial derivatives.
  - 35,000 single-term power rule problems
  - 25,000 multi-term polynomials (sum rule)
  - 10,000 constant terms
  - 10,000 negative exponent problems
  - 20,000 multi-variable partial derivatives (using `x`, `y`, `z`)
- **Trig/Exp/Log Support (Phase 2):** Added support for non-polynomial functions using vocabulary v1.3's `OP:sin` / `OP:cos` / `OP:tan` / `OP:exp` / `OP:ln` / `OP:sec` tokens, plus new optional `coeff` and `power` decorator fields on op-nodes (see `docs/KNOWN_ISSUES.md` for the full schema change).
  - 5,000 trigonometric `sin(k·x)` differentiation problems (derivative: `k·cos(k·x)`)
  - 5,000 trigonometric `cos(k·x)` differentiation problems (derivative: `-k·sin(k·x)`)
  - 5,000 trigonometric `tan(k·x)` differentiation problems (derivative: `k·sec²(k·x)`)
  - 5,000 exponential `exp(k·x)` differentiation problems (derivative: `k·exp(k·x)`)
  - 5,000 logarithmic `ln(k·x)` differentiation problems (derivative: `1/x`)
  - Each of the above varies the inner argument multiplier `k` (e.g. `sin(2x)`, `sin(-5x)`, not just `sin(x)`) so the model sees a range of coefficients rather than a single fixed pattern. Verified via exhaustive serialization test: 0 failures across all 375,000 generated node instances (125,000 records × 3 fields each).
- **Envelope Format:** Real SLaNg representation (no legacy `"type"` or `"terms"` wrappers on input math expressions).
- **Constraints:**
  - Coefficient ($\text{coeff}$) range: $[-10, 12]$ (asymmetric — negation-dependent cases, e.g. `cos`'s derivative sign, are restricted to the symmetric subset $[-10, 10]$ to keep the negated value in-vocab)
  - Power/Exponent ($\text{power}$) range: $[-3, 5]$
  - Variables: `x`, `y`, `z`

## 3. Limitations & Gaps
- All five core transcendental functions (`sin`, `cos`, `tan`, `exp`, `ln`) are now covered for first-derivative differentiation with varied coefficients. Chain rule composition beyond a linear inner argument (e.g. `sin(x^2)`, nested functions like `sin(cos(x))`) is not yet covered and would require additional template work.
- `rule_ids` for all Phase 2 trig/exp/log records currently reuse `RULE:chain_rule` (rule_id 1) as the closest existing label — there is no dedicated `RULE:trig_rule` / `RULE:exp_rule` / `RULE:log_rule` token yet. Flagged for team review as a possible follow-up (next free rule ID would need to be added to `vocab.json`'s `rule_tokens` block).