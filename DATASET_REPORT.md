# SLaNg Dataset Report

## 1. Dataset Scale
- **Total Records:** 100,000 unique calculus strings.
- **Data Splits:** 90% Train (90,000), 5% Val (5,000), 5% Test (5,000).

## 2. Rule Coverage
- **Expanded Polynomials:** The synthesizer generates operations representing the calculus power rule, sum rule, constant rule, and partial derivatives.
  - 35,000 single-term power rule problems
  - 25,000 multi-term polynomials (sum rule)
  - 10,000 constant terms
  - 10,000 negative exponent problems
  - 20,000 multi-variable partial derivatives (using `x`, `y`, `z`)
- **Envelope Format:** Real SLaNg representation (no legacy `"type"` or `"terms"` wrappers on input math expressions).
- **Constraints:**
  - Coefficient ($\text{coeff}$) range: $[-10, 12]$
  - Power/Exponent ($\text{power}$) range: $[-3, 5]$
  - Variables: `x`, `y`, `z`

## 3. Limitations & Gaps
- **Trig/Exp/Log:** Currently supports polynomials only. Trigonometric derivatives, exponential rules, and logarithmic rules are not covered because the current vocabulary does not contain tokens for `sin`, `cos`, `tan`, `exp`, or `ln`. Expanding to these would require changes to `vocab.json` and retraining from scratch.