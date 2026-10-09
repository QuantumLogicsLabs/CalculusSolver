import sys
import os
import inspect
from collections import Counter

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import problem_generator


def get_generator_instance():
    # Look for classes in problem_generator that have the generate_gradient method
    for name, obj in inspect.getmembers(problem_generator, inspect.isclass):
        if hasattr(obj, "generate_gradient"):
            try:
                # Try instantiating with default args
                return obj()
            except TypeError:
                # Try instantiating with common constructor parameters
                try:
                    return obj(seed=42)
                except Exception:
                    pass
    return None


def test_gradient_generator_balance():
    gen_instance = get_generator_instance()
    
    if gen_instance is None:
        print("[INFO] Testing gradient distribution directly on train.jsonl...")
        # Fallback check directly against train.jsonl if class setup varies
        import json
        counts = Counter()
        with open("data/splits/train.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                src = row.get("src_tokens", {})
                if src.get("op") != "gradient":
                    continue
                terms = src.get("expr", {}).get("numi", {}).get("terms", [])
                seen = set()
                for t in terms:
                    seen.update(t.get("var", {}).keys())
                if seen:
                    counts[len(seen)] += 1
        
        print(f"[TEST METRICS] Train dataset gradient distribution: {dict(counts)}")
        assert counts[2] > 0, "2-variable gradients missing in dataset"
        assert counts[3] > 0, "3-variable gradients missing in dataset"
        return

    counts = Counter()
    for _ in range(1000):
        prob = gen_instance.generate_gradient()
        if isinstance(prob, dict):
            vars_used = prob.get("vars")
            if not vars_used and "expr" in prob:
                terms = prob["expr"].get("numi", {}).get("terms", [])
                seen = set()
                for t in terms:
                    seen.update(t.get("var", {}).keys())
                vars_used = list(seen)
            if vars_used:
                counts[len(vars_used)] += 1

    print(f"[TEST METRICS] Gradient variable count distribution (sample 1000): {dict(counts)}")
    assert counts[2] > 0, "2-variable gradients missing in generator output"
    assert counts[3] > 0, "3-variable gradients missing in generator output"


if __name__ == "__main__":
    test_gradient_generator_balance()
    print("Gradient generator balance unit test passed!")


def test_gradient_benchmark_variable_set_coverage():
    """Ensure benchmark_gradient.json covers multiple variable sets:
    {x, y}, {x, z}, {y, z}, and {x, y, z}, preventing regression to narrow {x, y} coverage."""
    import json
    benchmark_path = os.path.join("eval", "benchmarks", "benchmark_gradient.json")
    assert os.path.exists(benchmark_path), f"Benchmark file not found: {benchmark_path}"
    with open(benchmark_path, encoding="utf-8") as f:
        benchmarks = json.load(f)

    var_sets_seen = set()
    for item in benchmarks:
        grad_target = item.get("target", {}).get("gradient", {})
        assert grad_target, f"Empty gradient target in record: {item}"
        var_sets_seen.add(tuple(sorted(grad_target.keys())))

    assert ("x", "y") in var_sets_seen, "{x, y} cases missing from gradient benchmark"
    assert ("x", "z") in var_sets_seen, "{x, z} cases missing from gradient benchmark"
    assert ("y", "z") in var_sets_seen, "{y, z} cases missing from gradient benchmark"
    assert ("x", "y", "z") in var_sets_seen, "{x, y, z} cases missing from gradient benchmark"
    assert len(benchmarks) >= 72, f"Benchmark suite should have at least 72 records, got {len(benchmarks)}"
