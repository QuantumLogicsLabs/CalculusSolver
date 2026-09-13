import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.solve import CalculusSolverInference

def run_cli():
    print("=" * 60)
    print("      CalculusSolver Interactive Model Tester")
    print("=" * 60)
    
    candidate_paths = [
        ROOT / "model" / "model.pkl",
        ROOT / "checkpoints" / "final" / "best.pt",
        ROOT / "checkpoints" / "checkpoint_epoch_1.pt",
    ]
    checkpoint_path = None
    for p in candidate_paths:
        if p.exists():
            checkpoint_path = p
            break

    if checkpoint_path is None:
        print(f"Error: No model checkpoint found. Checked: {candidate_paths}")
        return

    try:
        print(f"Loading model checkpoint ({checkpoint_path.relative_to(ROOT)})...")
        solver = CalculusSolverInference(model_path=str(checkpoint_path), beam_size=5)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Presets available:")
    print(" 1) d/dx (3x^2)")
    print(" 2) d/dx (5x^3 + 2x)")
    print(" 3) d/dx (sin(x))")
    print(" 4) ∫ (x^2) dx")
    print(" 5) ∂/∂x (x^2 * y^3)\n")

    while True:
        try:
            user_input = input("Enter preset (1-5), JSON expression, or 'exit': ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        expr_dict = None
        if user_input == "1":
            expr_dict = {"op": "diff", "var": "x", "expr": {"coeff": 3, "var": {"x": 2}}}
        elif user_input == "2":
            expr_dict = {"op": "diff", "var": "x", "expr": {"numi": {"terms": [{"coeff": 5, "var": {"x": 3}}, {"coeff": 2, "var": {"x": 1}}]}, "deno": 1}}
        elif user_input == "3":
            expr_dict = {"op": "diff", "var": "x", "expr": {"op": "sin", "arg": {"var": {"x": 1}}}}
        elif user_input == "4":
            expr_dict = {"op": "integrate", "var": "x", "expr": {"var": {"x": 2}}}
        elif user_input == "5":
            expr_dict = {"op": "partial", "var": "x", "expr": {"coeff": 1, "var": {"x": 2, "y": 3}}}
        else:
            try:
                expr_dict = json.loads(user_input)
            except Exception as e:
                print(f"Invalid input. Enter 1-5 or valid JSON dict. Error: {e}\n")
                continue

        print("\n------------------------------------------------------------")
        print(f"Input SLaNg Dict: {json.dumps(expr_dict)}")
        print("Running model beam search...")
        
        try:
            res = solver.solve(expr_dict)
            print(f"\nPredicted Rule:    {res.get('rule')}")
            print(f"Status:            {res.get('status')}")
            print(f"Verified:          {res.get('verified')}")
            print(f"Output AST:        {res.get('output')}")
            print(f"Output Tokens:     {res.get('output_tokens')}")
            if res.get("warning"):
                print(f"Warning:           {res.get('warning')}")
        except Exception as err:
            print(f"Error during solve: {err}")

        print("------------------------------------------------------------\n")

if __name__ == "__main__":
    run_cli()
