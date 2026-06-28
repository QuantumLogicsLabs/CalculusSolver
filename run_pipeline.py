import os
import sys
import json
import glob
from model.checkpoint_utils import create_dummy_checkpoint, validate_checkpoint
from inference.router import standalone_inference
from inference.eval_harness import (
    exact_match_accuracy,
    compare_models_v1_placeholder,
    run_error_analysis_v1_placeholder,
)

def load_benchmark_data(benchmark_dir="eval/benchmarks"):
    inputs = []
    ground_truths = []
    
    if not os.path.exists(benchmark_dir):
        return inputs, ground_truths
        
    for filepath in glob.glob(os.path.join(benchmark_dir, "*.json")):
        with open(filepath, 'r') as f:
            data = json.load(f)
            for item in data:
                inputs.append(item.get("expr", ""))
                ground_truths.append(item.get("target", ""))
                
    return inputs, ground_truths

def run_end_to_end_pipeline(checkpoint_path):
    is_dummy = "dummy" in checkpoint_path
    
    if is_dummy:
        expected_shapes = {
            "linear.weight": (10, 10),
            "linear.bias": (10,)
        }
        if not os.path.exists(checkpoint_path):
            create_dummy_checkpoint(checkpoint_path)
    else:
        expected_shapes = {
            "encoder.embedding.weight": (1000, 256),
            "decoder.fc_out.weight": (256, 1000),
            "rule_head.classifier.weight": (50, 256),
            "step_tracer.classifier.weight": (100, 256)
        }
        
    validate_checkpoint(checkpoint_path, expected_shapes)
    
    inputs, ground_truths = load_benchmark_data()
    
    predictions = []
    
    for x in inputs:
        output = standalone_inference(checkpoint_path, {"expr": x}, strategy="beam")
        if isinstance(output, dict):
            predictions.append(output.get("expr", ""))
        else:
            predictions.append(output)
            
    em_score = exact_match_accuracy(predictions, ground_truths)
    error_report = run_error_analysis_v1_placeholder(predictions, ground_truths)
    
    print(f"Accuracy: {em_score}")
    print(f"Errors: {error_report}")

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/dummy_model.pt"
    run_end_to_end_pipeline(target_path)