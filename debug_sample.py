"""
Quick sanity check: run the trained model on a few real benchmark problems
and print exactly what it generates, next to the expected answer.

Run this AFTER training + eval, from the project root:
    python debug_sample.py

This tells us whether the model is producing structured-but-wrong output
(= just needs more training) or empty/garbage output (= still a bug).
"""
import json
import glob
from pathlib import Path

from inference.solve import CalculusSolverInference

ROOT = Path(__file__).resolve().parent
checkpoint_path = ROOT / "checkpoints" / "final" / "best.pt"

if not checkpoint_path.exists():
    raise SystemExit(f"No checkpoint found at {checkpoint_path} — train first.")

solver = CalculusSolverInference(model_path=str(checkpoint_path))

benchmark_files = sorted(glob.glob(str(ROOT / "eval" / "benchmarks" / "*.json")))
sample_count = 0

for filepath in benchmark_files:
    op_name = Path(filepath).stem
    with open(filepath, "r", encoding="utf-8") as f:
        problems = json.load(f)

    # just take the first problem from each category
    if not problems:
        continue
    p = problems[0]
    expr = p["expr"]
    target = p["target"]

    print(f"\n{'='*60}")
    print(f"Category: {op_name}")
    print(f"Input expr: {json.dumps(expr)[:200]}")
    print(f"Expected target: {json.dumps(target)[:200]}")

    try:
        result = solver.solve(expr)
        print(f"Model output_tokens: {result.get('output_tokens')}")
        print(f"Model output (parsed): {json.dumps(result.get('output'))[:200]}")
        print(f"Status: {result.get('status')}, Verified: {result.get('verified')}")
    except Exception as e:
        print(f"ERROR during solve: {e}")

    sample_count += 1
    if sample_count >= 5:
        break

solver.close()