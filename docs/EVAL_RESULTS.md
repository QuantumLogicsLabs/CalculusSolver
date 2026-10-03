# Evaluation Results

## Current reproducible baseline

- **Checkpoint:** `checkpoints/final/best.pt`
- **Checkpoint SHA-256:** `f5166332750d6260007e8efe3e0018e01e9ce0cafbaa91456b5f8b1f16afcb81`
- **Evaluation mode:** hybrid inference (neural beam search + verifier-gated fallback)
- **Decoder:** beam size 3, max length 64
- **Seed:** 0

| Operation | Problems | Exact match | Verification rate |
|---|---:|---:|---:|
| diff | 80 | 77/80 (96.2%) | 80/80 (100.0%) |
| gradient | 18 | 12/18 (66.7%) | 18/18 (100.0%) |
| integrate | 60 | 60/60 (100.0%) | 60/60 (100.0%) |
| partial | 60 | 60/60 (100.0%) | 60/60 (100.0%) |
| tangent_line | 50 | 50/50 (100.0%) | 50/50 (100.0%) |
| **Overall** | **268** | **259/268 (96.6%)** | **268/268 (100.0%)** |

> Partial is not an active accuracy shortfall in this baseline: it is 60/60 (100.0%). Gradient is 12/18 (66.7%), and the gradient benchmark contains 18 records.

> Exact match and verification rate are distinct measurements. In particular, the current gradient verifier accepts predictions with an extra component, so verification can exceed exact match. This report records both without treating verifier acceptance as exact correctness.

## Correction of the stale report

The previous table was generated for an older 300-record suite and older checkpoint behavior. Commit `e9b7923` replaced the 50-record gradient benchmark with the current 18-record file, but the table was not regenerated. The 32-record reduction explains the full total mismatch: 300 - (50 - 18) = 268. The resolution is to report the benchmark files that actually exist, not to synthesize 32 missing cases.

| Measurement | Before (stale document) | After (current inputs) |
|---|---:|---:|
| Benchmark total | 300 | 268 |
| Partial exact match | 31/60 (51.7%) | 60/60 (100.0%) |
| Gradient exact match | 20/50 (40.0%) | 12/18 (66.7%) |
| Overall exact match | 219/300 (73.0%) | 259/268 (96.6%) |

## Benchmark manifest

The hashes below define the exact benchmark inputs. Record counts are read from the JSON arrays, not copied from documentation.

| File | Records | SHA-256 |
|---|---:|---|
| `eval/benchmarks/benchmark_diff.json` | 80 | `d3de159d4f4d5e18568debf736039ab4a1679893829253016e4c68ba2ad6a09a` |
| `eval/benchmarks/benchmark_gradient.json` | 18 | `babafba316c4c5362bdbb0ae0d207f9c934a8b26c1a3bf3ea8c86c73dfc0e09b` |
| `eval/benchmarks/benchmark_integrate.json` | 60 | `21c659bc24dcb4b5827b10661407f26203fb1143e4cd4196385e46c106a845b4` |
| `eval/benchmarks/benchmark_partial.json` | 60 | `1aadeb957ae3934ff4c87a3c00ebf83e0a7eb51c5b00e33fd396b62390a9fad9` |
| `eval/benchmarks/benchmark_tangent_line.json` | 50 | `fe537963b53320f7c7b2bb8c2d47ccb3cc01ac8c61047d472ab3b6bb8cee4ef5` |
| **Total** | **268** | |

## Reproduce this evaluation

Run from the repository root with Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-neural.txt
.\.venv\Scripts\python.exe eval/run_eval.py --checkpoint checkpoints/final/best.pt --beam-size 3 --max-len 64
```

The command regenerates this file and `docs/eval_results.json`. A matching run must use the checkpoint and benchmark hashes above and should finish with zero exceptions. For a checkpoint-only diagnostic, pass `--no-fallback` together with non-canonical `--output` and `--json-output` paths; do not compare that diagnostic directly with this hybrid baseline.

## Recorded environment

- **Git commit:** `00e8d3eba8db6bad5572f646e16935f3e947b2eb`
- **Git working tree dirty:** `True`
- **Evaluator SHA-256:** `a3c3ed0cdb7a6e34a47270cf6608294acfe2eea18455ac52838722dc2c500ccd`
- **Python:** `3.12.14`
- **PyTorch:** `2.3.1+cpu`
- **NumPy:** `1.26.4`
- **Platform:** `Windows-11-10.0.26200-SP0`
- **Evaluation exceptions:** `0`

The machine-readable before/after evidence for future comparisons is committed in `docs/eval_results.json`; the console output reports the same per-category numerators and denominators.
