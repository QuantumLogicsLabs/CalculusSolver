import json
import os
import sys
import glob
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.solve import CalculusSolverInference
from inference.eval_harness import is_equivalent

def main():
    checkpoint_path = ROOT / "checkpoints" / "final" / "best.pt"
    if not checkpoint_path.exists():
        print(f"Error: checkpoint {checkpoint_path} does not exist.")
        sys.exit(1)

    print("Loading neural model...")
    solver = CalculusSolverInference(model_path=str(checkpoint_path))

    benchmark_dir = ROOT / "eval" / "benchmarks"
    benchmark_files = glob.glob(str(benchmark_dir / "*.json"))
    if not benchmark_files:
        print("No benchmark files found in eval/benchmarks/")
        sys.exit(1)

    report_lines = [
        "# Evaluation Results",
        "",
        f"**Checkpoint Evaluated:** `{checkpoint_path.relative_to(ROOT)}`",
        "",
        "| Operation | Total Problems | Exact Match (Accuracy) | Verification Rate |",
        "|---|---|---|---|",
    ]

    total_problems = 0
    total_exact_match = 0
    total_verified = 0

    for filepath in sorted(benchmark_files):
        op_name = Path(filepath).stem.replace("benchmark_", "")
        with open(filepath, "r", encoding="utf-8") as f:
            problems = json.load(f)

        exact_match_count = 0
        verified_count = 0
        op_total = len(problems)

        for p in problems:
            expr = p["expr"]
            target = p["target"]

            try:
                res = solver.solve(expr)
                pred = res.get("output") or res.get("expr") or {}
                
                # Check equivalence
                if is_equivalent(pred, target):
                    exact_match_count += 1
                if res.get("verified", False):
                    verified_count += 1
            except Exception as e:
                print(f"Error evaluating problem: {e}")

        accuracy = exact_match_count / op_total if op_total > 0 else 0.0
        ver_rate = verified_count / op_total if op_total > 0 else 0.0

        report_lines.append(f"| {op_name} | {op_total} | {exact_match_count}/{op_total} ({accuracy:.1%}) | {verified_count}/{op_total} ({ver_rate:.1%}) |")

        total_problems += op_total
        total_exact_match += exact_match_count
        total_verified += verified_count

    overall_accuracy = total_exact_match / total_problems if total_problems > 0 else 0.0
    overall_ver_rate = total_verified / total_problems if total_problems > 0 else 0.0

    report_lines.append(f"| **Overall** | **{total_problems}** | **{total_exact_match}/{total_problems} ({overall_accuracy:.1%})** | **{total_verified}/{total_problems} ({overall_ver_rate:.1%})** |")

    # Write report
    eval_results_path = ROOT / "docs" / "EVAL_RESULTS.md"
    with open(eval_results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Saved evaluation results to {eval_results_path}")

if __name__ == "__main__":
    main()
