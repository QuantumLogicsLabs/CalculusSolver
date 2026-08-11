"""
Developer 4 deliverable: "a measured decision (not a guess)" + "proof the
converted model still produces correct answers."

This script is the proof. It:
  1. Runs every eval/benchmarks/*.json problem through BOTH the original
     PyTorch solver (inference/solve.py) and the new ONNX solver
     (deployment/onnx_solve.py), and confirms they produce identical output
     token sequences (not just "both roughly right" — literally the same
     generation, since it's the same weights through the same algorithm).
  2. Measures the actual installed size of the ONNX Runtime deployment
     bundle (onnx artifact files + the onnxruntime package on disk) against
     Vercel's ~250MB unzipped limit.
  3. Writes docs/EXPORT_DECISION.md from these real numbers.

Run from the repo root, AFTER deployment/export_onnx.py:
    python -u deployment/verify_export.py [checkpoint_path]

Needs both requirements-neural.txt (torch, for the "before" side of the
comparison) and requirements-onnx.txt (onnxruntime, for the "after" side)
installed locally. This script itself never ships to production — it's a
one-time (or re-run-on-every-new-checkpoint, see the re-validation rule)
verification step.
"""
import glob
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

VERCEL_LIMIT_MB = 250


def get_installed_package_size_mb(package_name: str) -> float:
    """Sum the on-disk size of an installed package's directory."""
    try:
        mod = importlib.import_module(package_name)
        pkg_dir = Path(mod.__file__).resolve().parent
        total = sum(f.stat().st_size for f in pkg_dir.rglob("*") if f.is_file())
        return total / 1e6
    except Exception as e:
        print(f"  (could not measure {package_name}: {e})")
        return -1.0


def main():
    checkpoint_arg = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final/best.pt"
    artifacts_dir = ROOT / "deployment" / "onnx_artifacts"

    if not artifacts_dir.exists():
        print("Error: deployment/onnx_artifacts/ not found. Run deployment/export_onnx.py first.")
        sys.exit(1)

    print("Loading PyTorch solver (original, for comparison)...")
    from inference.solve import CalculusSolverInference
    torch_solver = CalculusSolverInference(model_path=checkpoint_arg)

    print("Loading ONNX solver (Option A candidate)...")
    from deployment.onnx_solve import OnnxCalculusSolverInference
    onnx_solver = OnnxCalculusSolverInference(str(artifacts_dir))

    benchmark_files = sorted(glob.glob(str(ROOT / "eval" / "benchmarks" / "*.json")))
    total = 0
    exact_token_matches = 0
    mismatches = []
    torch_times = []
    onnx_times = []

    for fp in benchmark_files:
        op_name = Path(fp).stem.replace("benchmark_", "")
        with open(fp) as f:
            problems = json.load(f)
        print(f"[{op_name}] {len(problems)} problems...")
        for p in problems:
            expr = p["expr"]
            total += 1

            t0 = time.time()
            torch_result = torch_solver.solve(expr)
            torch_times.append(time.time() - t0)

            t0 = time.time()
            onnx_result = onnx_solver.solve(expr)
            onnx_times.append(time.time() - t0)

            if torch_result["output_tokens"] == onnx_result["output_tokens"]:
                exact_token_matches += 1
            else:
                mismatches.append({
                    "operation": op_name,
                    "input": expr,
                    "torch_tokens": torch_result["output_tokens"],
                    "onnx_tokens": onnx_result["output_tokens"],
                })

    torch_solver.close()
    onnx_solver.close()

    match_rate = exact_token_matches / total if total else 0.0
    avg_torch_ms = 1000 * sum(torch_times) / len(torch_times) if torch_times else 0.0
    avg_onnx_ms = 1000 * sum(onnx_times) / len(onnx_times) if onnx_times else 0.0

    print(f"\nToken-exact match: {exact_token_matches}/{total} ({match_rate:.1%})")
    print(f"Avg latency — torch: {avg_torch_ms:.1f}ms, onnx: {avg_onnx_ms:.1f}ms")

    if mismatches:
        mismatch_path = ROOT / "docs" / "onnx_export_mismatches.json"
        with open(mismatch_path, "w") as f:
            json.dump(mismatches, f, indent=2)
        print(f"Wrote {len(mismatches)} mismatches to {mismatch_path}")

    # ---- Size measurement ----
    onnx_artifact_mb = sum(f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file()) / 1e6
    onnxruntime_mb = get_installed_package_size_mb("onnxruntime")
    numpy_mb = get_installed_package_size_mb("numpy")
    total_bundle_mb = onnx_artifact_mb + max(onnxruntime_mb, 0) + max(numpy_mb, 0)

    fits = total_bundle_mb <= VERCEL_LIMIT_MB
    verdict = "FITS" if fits else "DOES NOT FIT"

    print(f"\nONNX artifacts: {onnx_artifact_mb:.1f} MB")
    print(f"onnxruntime package (installed, this machine): {onnxruntime_mb:.1f} MB")
    print(f"numpy package (installed, this machine): {numpy_mb:.1f} MB")
    print(f"Estimated total bundle: {total_bundle_mb:.1f} MB vs {VERCEL_LIMIT_MB}MB limit -> {verdict}")

    # ---- Write docs/EXPORT_DECISION.md from these real numbers ----
    decision = "Option A (ONNX export)" if (fits and match_rate == 1.0) else \
               "Option B (separate hosted service) — see reasoning below"

    doc = f"""# Export Decision — Developer 4 (measured, {time.strftime('%Y-%m-%d')})

## Checkpoint evaluated
`{checkpoint_arg}`

## Measured bundle size (Option A: ONNX export)
| Component | Size |
|---|---|
| ONNX artifacts (encoder.onnx + decoder.onnx + rule_head weights) | {onnx_artifact_mb:.1f} MB |
| onnxruntime package (installed size, this machine) | {onnxruntime_mb:.1f} MB |
| numpy package (installed size, this machine) | {numpy_mb:.1f} MB |
| **Total estimated bundle** | **{total_bundle_mb:.1f} MB** |
| Vercel unzipped limit | {VERCEL_LIMIT_MB} MB |
| **Result** | **{verdict}** |

Note: package sizes above were measured on the machine that ran this
script. Vercel's actual deployed size can differ slightly by platform
(manylinux wheel size for onnxruntime), so re-check post-deploy.

## Correctness verification
Ran all {total} eval/benchmarks/*.json problems through both the original
PyTorch checkpoint and the exported ONNX artifacts, using the same beam
search algorithm (deployment/onnx_beam_search.py mirrors
inference/beam_search.py line-for-line, numpy in place of torch).

- **Token-exact match: {exact_token_matches}/{total} ({match_rate:.1%})**
- Average latency — PyTorch: {avg_torch_ms:.1f}ms/problem, ONNX: {avg_onnx_ms:.1f}ms/problem
- Mismatches (if any) logged to `docs/onnx_export_mismatches.json`

## Decision
**{decision}**

{"ONNX export fits comfortably under the Vercel limit and produces byte-for-byte identical generations to the original checkpoint. No need for a separate hosted service — deploy deployment/onnx_solve.py directly, gated behind requirements-onnx.txt (no torch in production)." if (fits and match_rate == 1.0) else
 "Either the bundle does not fit under the limit, or the ONNX outputs diverge from the original PyTorch outputs (see docs/onnx_export_mismatches.json). Do not ship Option A as-is — fall back to Option B (self-hosted model service, e.g. Render/Fly.io, with the Vercel frontend calling it over HTTP) as described in docs/HOSTING_DECISION.md's 'When to revisit' section."}

## Re-validation rule
Per the task doc: any time Developer 3 delivers a new signed-off checkpoint,
re-run this exact script (`deployment/export_onnx.py` then
`deployment/verify_export.py`) against it before deploying. A different
checkpoint can convert differently even if the previous one converted cleanly.
"""

    with open(ROOT / "docs" / "EXPORT_DECISION.md", "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\nWrote docs/EXPORT_DECISION.md — decision: {decision}")


if __name__ == "__main__":
    main()