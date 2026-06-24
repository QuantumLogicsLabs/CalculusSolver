import os
from model.checkpoint_utils import create_dummy_checkpoint, validate_checkpoint
from inference.router import standalone_inference
from inference.eval_harness import (
    exact_match_accuracy,
    eval_step_trace,
    compare_models,
    run_error_analysis,
)

def run_end_to_end_pipeline():
    checkpoint_path = "checkpoints/dummy_model.pt"
    
    expected_shapes = {
        "linear.weight": (10, 10),
        "linear.bias": (10,)
    }
    
    if not os.path.exists(checkpoint_path):
        create_dummy_checkpoint(checkpoint_path)
        
    validate_checkpoint(checkpoint_path, expected_shapes)
    
    mock_inputs = ["int(x) dx", "diff(x^2) dx"]
    mock_ground_truths = ["0.5*x^2", "2*x"]
    mock_expected_traces = [["step1", "step2"], ["step1"]]
    
    predictions = []
    generated_traces = []
    
    for x in mock_inputs:
        output = standalone_inference(checkpoint_path, x, strategy="beam")
        predictions.append(output.get("prediction", ""))
        generated_traces.append(output.get("trace", []))
        
    em_score = exact_match_accuracy(predictions, mock_ground_truths)
    trace_score = eval_step_trace(generated_traces, mock_expected_traces)
    error_report = run_error_analysis(predictions, mock_ground_truths)
    
    print(f"Accuracy: {em_score}")
    print(f"Trace Score: {trace_score}")
    print(f"Errors: {error_report}")

if __name__ == "__main__":
    run_end_to_end_pipeline()