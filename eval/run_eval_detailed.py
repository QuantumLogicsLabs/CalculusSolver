import json
import sys
import glob
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.solve import CalculusSolverInference
from inference.eval_harness import is_equivalent, categorize_error_v1
from tokenizer.slang_serializer import serialize_slang_math


def main(checkpoint_rel, max_len, out_json, only_op=None):
    checkpoint_path = ROOT / checkpoint_rel
    if not checkpoint_path.exists():
        print(f"MISSING_CHECKPOINT:{checkpoint_path}")
        sys.exit(2)

    print(f"Loading model from {checkpoint_path} (max_len={max_len})...")
    solver = CalculusSolverInference(model_path=str(checkpoint_path), beam_size=3, max_len=max_len)
    print("Model loaded.")

    benchmark_dir = ROOT / "eval" / "benchmarks"
    benchmark_files = sorted(glob.glob(str(benchmark_dir / "*.json")))
    if only_op:
        benchmark_files = [f for f in benchmark_files if Path(f).stem == f"benchmark_{only_op}"]

    all_results = []
    summary = {}

    for filepath in benchmark_files:
        op_name = Path(filepath).stem.replace("benchmark_", "")
        with open(filepath, "r", encoding="utf-8") as f:
            problems = json.load(f)

        n_exact = 0
        n_verified = 0
        t0 = time.time()
        for i, p in enumerate(problems):
            expr = p["expr"]
            target = p["target"]
            expected_rule = p.get("expected_rule")

            row = {
                "operation": op_name,
                "index": i,
                "input_expr": json.dumps(expr),
                "target": json.dumps(target),
                "expected_rule": expected_rule,
            }
            try:
                res = solver.solve(expr)
                pred_out = res.get("output")
                exact = is_equivalent(pred_out, target)
                verified = bool(res.get("verified", False))
                out_tokens = res.get("output_tokens", [])
                try:
                    target_tokens = serialize_slang_math(target) if isinstance(target, dict) else target
                except Exception:
                    target_tokens = None

                hit_max_len = len(out_tokens) >= max_len
                err_cat = None
                if not exact:
                    err_cat = categorize_error_v1(pred_out, target)

                row.update({
                    "predicted_output": json.dumps(pred_out) if pred_out is not None else None,
                    "predicted_rule": res.get("rule"),
                    "status": res.get("status"),
                    "verified": verified,
                    "exact_match": exact,
                    "confidence": res.get("confidence"),
                    "warning": res.get("warning"),
                    "output_token_count": len(out_tokens),
                    "target_token_count": (len(target_tokens) if isinstance(target_tokens, list) else None),
                    "hit_max_len_cap": hit_max_len,
                    "error_category": err_cat,
                })
                if exact:
                    n_exact += 1
                if verified:
                    n_verified += 1
            except Exception as e:
                row.update({
                    "predicted_output": None,
                    "predicted_rule": None,
                    "status": "exception",
                    "verified": False,
                    "exact_match": False,
                    "confidence": None,
                    "warning": str(e),
                    "output_token_count": None,
                    "target_token_count": None,
                    "hit_max_len_cap": None,
                    "error_category": "exception",
                })
            all_results.append(row)
            if i % 10 == 0:
                print(f"[{op_name}] {i}/{len(problems)} elapsed={time.time()-t0:.1f}s")

        summary[op_name] = {
            "total": len(problems),
            "exact_match": n_exact,
            "verified": n_verified,
            "accuracy": n_exact / len(problems) if problems else 0.0,
        }
        print(f"[{op_name}] DONE acc={n_exact}/{len(problems)} verified={n_verified}/{len(problems)} time={time.time()-t0:.1f}s")

    solver.close()

    output = {"checkpoint": str(checkpoint_rel), "max_len": max_len, "summary": summary, "results": all_results}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1)
    print(f"Wrote {out_json}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final/best.pt"
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    out = sys.argv[3] if len(sys.argv) > 3 else "eval_detailed.json"
    only_op = sys.argv[4] if len(sys.argv) > 4 else None
    main(ckpt, max_len, out, only_op)