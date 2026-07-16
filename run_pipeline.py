"""
End-to-end evaluation pipeline for CalculusSolver.

Runs all available solvers (fallback, neural, groq) against the benchmark
dataset and produces docs/MODEL_COMPARISON.md with per-operation accuracy.
"""

import os
import sys
import json
import glob
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from inference.eval_harness import (
    exact_match_accuracy,
    is_equivalent,
    compare_models_v1_placeholder,
    run_error_analysis_v1_placeholder,
)


def load_benchmark_data(benchmark_dir="eval/benchmarks"):
    """Load all benchmark JSON files, returning items grouped by operation."""
    all_items = []

    if not os.path.exists(benchmark_dir):
        print(f"WARNING: {benchmark_dir} does not exist. No benchmarks to evaluate.")
        return all_items

    for filepath in sorted(glob.glob(os.path.join(benchmark_dir, "*.json"))):
        with open(filepath, 'r') as f:
            data = json.load(f)
            for item in data:
                all_items.append(item)

    return all_items


def run_solver(solver, items, solver_name):
    """Run a solver against all benchmark items. Returns list of (prediction, ground_truth, operation, correct)."""
    results = []
    for item in items:
        payload = item["expr"]
        ground_truth = item["target"]
        operation = item.get("operation", "unknown")

        try:
            result = solver.solve(payload)
            prediction = result.get("expr", {})
        except Exception as e:
            prediction = {"error": str(e)}

        correct = is_equivalent(prediction, ground_truth)
        results.append({
            "prediction": prediction,
            "ground_truth": ground_truth,
            "operation": operation,
            "correct": correct,
        })
    return results


def compute_accuracy_by_op(results):
    """Compute accuracy grouped by operation type."""
    by_op = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        op = r["operation"]
        by_op[op]["total"] += 1
        if r["correct"]:
            by_op[op]["correct"] += 1

    accuracies = {}
    for op, counts in sorted(by_op.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        accuracies[op] = {
            "accuracy": acc,
            "correct": counts["correct"],
            "total": counts["total"],
        }
    return accuracies


def write_model_comparison(solver_results, output_path="docs/MODEL_COMPARISON.md"):
    """Write the model comparison report to markdown."""
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    lines = [
        "# Model Comparison Report",
        "",
        "Comparison of all available solvers against the benchmark dataset.",
        "",
        "## Overall Accuracy",
        "",
        "| Solver | Total Problems | Correct | Accuracy |",
        "|--------|---------------|---------|----------|",
    ]

    for solver_name, results in solver_results.items():
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        acc = correct / total if total > 0 else 0.0
        lines.append(f"| {solver_name} | {total} | {correct} | {acc:.1%} |")

    lines.extend(["", "## Accuracy by Operation", ""])

    # Collect all operations
    all_ops = set()
    for results in solver_results.values():
        for r in results:
            all_ops.add(r["operation"])
    all_ops = sorted(all_ops)

    # Header
    header = "| Operation |"
    separator = "|-----------|"
    for solver_name in solver_results:
        header += f" {solver_name} |"
        separator += "----------|"
    lines.append(header)
    lines.append(separator)

    # Per-operation rows
    for op in all_ops:
        row = f"| {op} |"
        for solver_name, results in solver_results.items():
            acc_data = compute_accuracy_by_op(results)
            if op in acc_data:
                a = acc_data[op]
                row += f" {a['accuracy']:.1%} ({a['correct']}/{a['total']}) |"
            else:
                row += " N/A |"
        lines.append(row)

    # Error analysis
    lines.extend(["", "## Error Analysis", ""])
    for solver_name, results in solver_results.items():
        predictions = [r["prediction"] for r in results]
        ground_truths = [r["ground_truth"] for r in results]
        errors = run_error_analysis_v1_placeholder(predictions, ground_truths)
        if errors:
            lines.append(f"### {solver_name}")
            lines.append("")
            lines.append("| Error Type | Count |")
            lines.append("|------------|-------|")
            for err_type, count in sorted(errors.items()):
                lines.append(f"| {err_type} | {count} |")
            lines.append("")
        else:
            lines.append(f"### {solver_name}")
            lines.append("")
            lines.append("No errors detected.")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Model comparison written to {output_path}")


def run_end_to_end_pipeline():
    items = load_benchmark_data()
    if not items:
        print("No benchmark data found. Run eval/generate_benchmarks.py first.")
        sys.exit(1)

    print(f"Loaded {len(items)} benchmark problems")

    solver_results = {}

    # 1. Fallback solver (always available)
    print("\n--- Running Fallback Solver ---")
    from inference.fallback_solver import FallbackSolver
    fallback = FallbackSolver()
    solver_results["Fallback"] = run_solver(fallback, items, "Fallback")
    fallback_correct = sum(1 for r in solver_results["Fallback"] if r["correct"])
    print(f"Fallback accuracy: {fallback_correct}/{len(items)} = {fallback_correct/len(items):.1%}")

    # 2. Neural solver (only if checkpoint exists)
    checkpoint_path = Path("checkpoints/final/best.pt")
    if checkpoint_path.exists():
        print("\n--- Running Neural Solver ---")
        try:
            from inference.solve import CalculusSolverInference
            neural = CalculusSolverInference(model_path=str(checkpoint_path))
            solver_results["Neural"] = run_solver(neural, items, "Neural")
            neural_correct = sum(1 for r in solver_results["Neural"] if r["correct"])
            print(f"Neural accuracy: {neural_correct}/{len(items)} = {neural_correct/len(items):.1%}")
        except Exception as e:
            print(f"Neural solver failed to load: {e}")
            # Mark all as incorrect
            solver_results["Neural"] = [
                {"prediction": {"error": str(e)}, "ground_truth": item["target"],
                 "operation": item.get("operation", "unknown"), "correct": False}
                for item in items
            ]
    else:
        print(f"\n--- Neural Solver: SKIPPED (no checkpoint at {checkpoint_path}) ---")
        solver_results["Neural"] = [
            {"prediction": "N/A", "ground_truth": item["target"],
             "operation": item.get("operation", "unknown"), "correct": False}
            for item in items
        ]

    # 3. Groq solver (only if API key is set)
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        print("\n--- Running Groq Solver ---")
        try:
            from inference.groq_solver import GroqSolver
            groq = GroqSolver()
            solver_results["Groq"] = run_solver(groq, items, "Groq")
            groq_correct = sum(1 for r in solver_results["Groq"] if r["correct"])
            print(f"Groq accuracy: {groq_correct}/{len(items)} = {groq_correct/len(items):.1%}")
        except Exception as e:
            print(f"Groq solver failed: {e}")
            solver_results["Groq"] = [
                {"prediction": {"error": str(e)}, "ground_truth": item["target"],
                 "operation": item.get("operation", "unknown"), "correct": False}
                for item in items
            ]
    else:
        print("\n--- Groq Solver: SKIPPED (no GROQ_API_KEY set) ---")
        solver_results["Groq"] = [
            {"prediction": "N/A", "ground_truth": item["target"],
             "operation": item.get("operation", "unknown"), "correct": False}
            for item in items
        ]

    # 4. Run compare_models_v1_placeholder for per-item comparison
    print("\n--- Per-item model comparison (compare_models_v1_placeholder) ---")
    comparison_summary = {"model_wins": 0, "fallback_wins": 0, "groq_wins": 0, "all_correct": 0, "none_correct": 0}
    for i, item in enumerate(items):
        neural_pred = solver_results["Neural"][i]["prediction"] if "Neural" in solver_results else "N/A"
        fallback_pred = solver_results["Fallback"][i]["prediction"]
        groq_pred = solver_results["Groq"][i]["prediction"] if "Groq" in solver_results else "N/A"

        cmp = compare_models_v1_placeholder(
            neural_pred, fallback_pred, groq_pred, item["target"]
        )
        if cmp["model_exact_match"]:
            comparison_summary["model_wins"] += 1
        if cmp["fallback_exact_match"]:
            comparison_summary["fallback_wins"] += 1
        if cmp["groq_exact_match"]:
            comparison_summary["groq_wins"] += 1
        if all(cmp.values()):
            comparison_summary["all_correct"] += 1
        if not any(cmp.values()):
            comparison_summary["none_correct"] += 1

    print(f"Comparison: {json.dumps(comparison_summary, indent=2)}")

    # 5. Write report
    write_model_comparison(solver_results)


if __name__ == "__main__":
    run_end_to_end_pipeline()