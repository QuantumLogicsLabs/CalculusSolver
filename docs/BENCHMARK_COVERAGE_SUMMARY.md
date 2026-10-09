# Benchmark Coverage & Partial Regression Protection Summary

**Developer 4 Report**  
**Project Scope:** Own benchmark quality and the already-resolved partial category. Ensure future evaluation covers the training distribution and partial does not regress.  
**Source Basis:** `GRADIENT_PARTIAL_ANALYSIS.pdf` (Sections 2, 3.4, 3.5, and 4)

---

## 1. Executive Summary & Expected Outcome

- **Benchmark Expansion Does Not Alter Model Quality:**  
  Expanding benchmark coverage improves **measurement fidelity**, not the underlying model weights. It ensures evaluation reflects the true multi-variable training distribution rather than reporting performance on an artificially narrow slice.
- **Partial Derivative Status:**  
  Partial differentiation scored **60/60 (100.0%)** in the established baseline. It is classified as a **regression-risk area** requiring continuous automated protection, **not an active shortfall**.
- **Model Retraining Note:**  
  Any downstream conclusions regarding model performance changes on the expanded variable sets require dataset regeneration and checkpoint retraining.

---

## 2. Gradient Benchmark Expansion

### Prior State vs. Expanded State

Prior evaluation used a narrow 18-problem benchmark restricted entirely to the $\{x, y\}$ variable pair. The expanded suite introduces full multi-variable coverage across $\{x, z\}$, $\{y, z\}$, and $\{x, y, z\}$.

| Metric / Attribute | Prior Benchmark | Expanded Benchmark | Notes |
|---|---:|---:|---|
| **Total Gradient Problems** | 18 | **72** | 4x expansion with balanced distribution |
| **$\{x, y\}$ Problem Count** | 18 (100%) | 18 (25.0%) | All 18 legacy cases preserved intact at indices 0–17 |
| **$\{x, z\}$ Problem Count** | 0 (0%) | **18 (25.0%)** | Added 2-variable variant |
| **$\{y, z\}$ Problem Count** | 0 (0%) | **18 (25.0%)** | Added 2-variable variant |
| **$\{x, y, z\}$ Problem Count** | 0 (0%) | **18 (25.0%)** | Added 3-variable polynomial gradient |
| **Vocabulary Compliance** | 100% | **100%** | Zero missing tokens; verified against `vocab.json` |
| **Verifier Acceptance** | 100% | **100%** | All records verified via `inference.verifier` |

### Key Files Updated
1. [eval/benchmarks/benchmark_gradient.json](file:///d:/QuantumLogics/CalculusSolver/eval/benchmarks/benchmark_gradient.json):
   - Expanded from 18 to 72 problems.
   - Preserved original 18 records at indices 0–17.
   - Added 18 records for $\{x, z\}$ (indices 18–35), 18 for $\{y, z\}$ (indices 36–53), and 18 for $\{x, y, z\}$ (indices 54–71).
2. [eval/generate_benchmarks.py](file:///d:/QuantumLogics/CalculusSolver/eval/generate_benchmarks.py):
   - Updated `generate_gradient_benchmarks(n=72)` to cycle across variable sets `[("x", "y"), ("x", "z"), ("y", "z"), ("x", "y", "z")]`.
3. [tests/unit/test_generator_gradient.py](file:///d:/QuantumLogics/CalculusSolver/tests/unit/test_generator_gradient.py):
   - Added `test_gradient_benchmark_variable_set_coverage()` to assert that `benchmark_gradient.json` covers $\{x, y\}$, $\{x, z\}$, $\{y, z\}$, and $\{x, y, z\}$ with at least 72 records.
4. [tests/regression/fixtures/](file:///d:/QuantumLogics/CalculusSolver/tests/regression/fixtures/):
   - Added [gradient_xz.json](file:///d:/QuantumLogics/CalculusSolver/tests/regression/fixtures/gradient_xz.json).
   - Added [gradient_yz.json](file:///d:/QuantumLogics/CalculusSolver/tests/regression/fixtures/gradient_yz.json).
   - Added [gradient_xyz.json](file:///d:/QuantumLogics/CalculusSolver/tests/regression/fixtures/gradient_xyz.json).

---

## 3. Partial Regression Protection & Hardening

The prior defect where partial differentiation dropped to 35.0% was previously resolved by removing variable binding ambiguity and ensuring the differentiated variable does not exclusively sit in position 0.

### Retained Training Data Invariants
All established test guards in [tests/unit/test_generator_binding.py](file:///d:/QuantumLogics/CalculusSolver/tests/unit/test_generator_binding.py) were retained:
1. **Unambiguous Binding Guard:**
   - `test_shared_coefficient_is_ambiguous`
   - `test_distinct_coefficients_are_unambiguous`
   - `test_single_term_is_trivially_unambiguous`
   - `test_multi_term_diff_never_emits_an_ambiguous_binding`
   - `test_multivar_diff_never_emits_an_ambiguous_binding`
   - `test_multivar_diff_coefficients_and_exponents_are_distinct`
2. **Variable Position Invariance Guard:**
   - `test_multivar_diff_does_not_always_put_the_target_variable_first` (ensures target variable is first in < 60% of cases and spans multiple slots)
   - `test_partial_constant_vanish_emits_valid_vanishing_derivatives`

### Strengthened Regression Coverage
