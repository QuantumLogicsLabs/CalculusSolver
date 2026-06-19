import os
import json
import argparse
from tqdm import tqdm
from inference.solve import CalculusSolverInference

def run_benchmark(dataset_path, checkpoint_path, vocab_path):
    # 1. Verify dataset file existence
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Hand-crafted evaluation dataset missing at: {dataset_path}")
        
    # 2. Load dataset expressions
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Expecting data format to be a list of dicts: [{"expr": "diff x pow x 2", "target": "mul 2 x"}, ...]
    if not isinstance(data, list):
        print("Error: Dataset format must be a JSON array containing evaluation targets.")
        return

    print(f"Initializing Inference module using checkpoint: {checkpoint_path}")
    # 3. Instantiate the solution solver framework
    try:
        solver = CalculusSolverInference(model_path=checkpoint_path, vocab_path=vocab_path)
    except FileNotFoundError as e:
        print(f"Benchmark initialization skipped: {str(e)}")
        print("Note: If the training pipeline has not run yet, this is expected behavior.")
        return

    exact_matches = 0
    total_records = len(data)
    
    print(f"Starting evaluation benchmark on {total_records} hand-crafted expressions...")
    
    # 4. Iterate through test expressions and evaluate matching accuracy
    for item in tqdm(data, desc="Evaluating"):
        input_expression = item.get("expr", "")
        ground_truth = item.get("target", "").strip()
        
        # Call inference solve method
        payload = {"expr": input_expression}
        response = solver.solve(payload)
        
        if response.get("status") == "success":
            predicted_output = response.get("expr", "").strip()
            # Perform Exact Match (EM) comparison
            if predicted_output == ground_truth:
                exact_matches += 1
        else:
            # Logs warnings if runtime processing issues occur
            pass

    # 5. Compute metrics
    em_accuracy = (exact_matches / total_records) * 100 if total_records > 0 else 0.0
    
    print("\n" + "="*40)
    print("        BENCHMARK EVALUATION RESULTS        ")
    print("="*40)
    print(f"Total Evaluated Records : {total_records}")
    print(f"Exact Matches (EM) Count: {exact_matches}")
    print(f"Exact Match Accuracy    : {em_accuracy:.2f}%")
    print("="*40 + "\n")

    # Save metrics log file for tracking
    report = {
        "total_records": total_records,
        "exact_matches": exact_matches,
        "exact_match_accuracy_percentage": em_accuracy
    }
    with open("data/benchmark_report.json", "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=4)
    print("Benchmark evaluation metric report saved inside 'data/benchmark_report.json'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculus Solver Model Benchmark Script")
    parser.add_argument("--dataset", type=str, default="data/hand_crafted_test.json", help="Path to evaluation file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to weights file")
    parser.add_argument("--vocab", type=str, default="data/vocab.json", help="Path to vocabulary file")
    
    args = parser.parse_args()
    run_benchmark(args.dataset, args.checkpoint, args.vocab)