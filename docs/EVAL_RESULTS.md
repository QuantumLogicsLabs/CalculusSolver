# Evaluation Results

## Current reproducible baseline

- **Checkpoint:** `checkpoints/final/best.pt`
- **Checkpoint SHA-256:** `f36c3798ba1e8aba62fb662c50bcb8a2ac5b3572140de5a6800c831876c7ff3c`
- **Evaluation mode:** hybrid inference (neural beam search + verifier-gated fallback)
- **Decoder:** beam size 3, max length 64
- **Seed:** 0

| Operation | Problems | Exact match | Verification rate |
|---|---:|---:|---:|
| diff | 80 | 80/80 (100.0%) | 80/80 (100.0%) |
| gradient | 72 | 72/72 (100.0%) | 72/72 (100.0%) |
| integrate | 60 | 59/60 (98.3%) | 60/60 (100.0%) |
| partial | 60 | 60/60 (100.0%) | 60/60 (100.0%) |
| tangent_line | 50 | 50/50 (100.0%) | 50/50 (100.0%) |
| **Overall** | **322** | **321/322 (99.7%)** | **322/322 (100.0%)** |

> Partial is not an active accuracy shortfall in this baseline: it is 60/60 (100.0%). Gradient is 72/72 (100.0%), and the gradient benchmark contains 72 records.

> Exact match and verification rate are distinct measurements. In particular, the current gradient verifier accepts predictions with an extra component, so verification can exceed exact match. This report records both without treating verifier acceptance as exact correctness.

## Correction of the stale report

The previous table was generated for an older 300-record suite and older checkpoint behavior. Commit `e9b7923` replaced the 50-record gradient benchmark with the current 18-record file, but the table was not regenerated. The 32-record reduction explains the full total mismatch: 300 - (50 - 18) = 268. The resolution is to report the benchmark files that actually exist, not to synthesize 32 missing cases.

| Measurement | Before (stale document) | After (current inputs) |
|---|---:|---:|
| Benchmark total | 300 | 322 |
| Partial exact match | 31/60 (51.7%) | 60/60 (100.0%) |
| Gradient exact match | 20/50 (40.0%) | 72/72 (100.0%) |
| Overall exact match | 219/300 (73.0%) | 321/322 (99.7%) |

## Benchmark manifest

The hashes below define the exact benchmark inputs. Record counts are read from the JSON arrays, not copied from documentation.

| File | Records | SHA-256 |
|---|---:|---|
| `eval/benchmarks/benchmark_diff.json` | 80 | `83d553720fea733c72b0a6ae28edbc6ed1c129c262cd5dfadb67d6482e9372a1` |
| `eval/benchmarks/benchmark_gradient.json` | 72 | `2159d04e6c5ba874845e68bca23357e8a1fd3d425cd42497e710aa60d427c698` |
| `eval/benchmarks/benchmark_integrate.json` | 60 | `061f1e69bb289fd55cba5979c35179592963c82f7a7faf52fa61d01a18f275de` |
| `eval/benchmarks/benchmark_partial.json` | 60 | `83588d1fbd90602515ca43e75d07a6db78b9791f9c52df15a61f2b240430f73d` |
| `eval/benchmarks/benchmark_tangent_line.json` | 50 | `5f06315bbde35e01f9baa87cccbcc57d5a7591c7ed72bc5216daae9964b6a3b0` |
| **Total** | **322** | |

## Reproduce this evaluation

Run from the repository root with Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-neural.txt
.\.venv\Scripts\python.exe eval/run_eval.py --checkpoint checkpoints/final/best.pt --beam-size 3 --max-len 64
```

The command regenerates this file and `docs/eval_results.json`. A matching run must use the checkpoint and benchmark hashes above and should finish with zero exceptions. For a checkpoint-only diagnostic, pass `--no-fallback` together with non-canonical `--output` and `--json-output` paths; do not compare that diagnostic directly with this hybrid baseline.

## Recorded environment

- **Git commit:** `bd6ec5882eea04aaaac6d4bad43b3c84cb69dd38`
- **Git working tree dirty:** `True`
- **Evaluator SHA-256:** `a3c3ed0cdb7a6e34a47270cf6608294acfe2eea18455ac52838722dc2c500ccd`
- **Python:** `3.13.15`
- **PyTorch:** `2.11.0+cu130`
- **NumPy:** `2.1.3`
- **Platform:** `Linux-6.6.122+-x86_64-with-glibc2.39`
- **Evaluation exceptions:** `0`

The machine-readable before/after evidence for future comparisons is committed in `docs/eval_results.json`; the console output reports the same per-category numerators and denominators.
