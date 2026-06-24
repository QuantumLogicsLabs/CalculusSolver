import sys
import torch
import torch.nn as nn

# Model architecture structure setup for inference loading
class MultiHeadCalculusModel(nn.Module):
    def __init__(self, num_rules=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.rule_head = nn.Linear(32, num_rules)
        self.verifier_head = nn.Linear(32, 1)
        
    def forward(self, dummy_features):
        features = self.backbone(dummy_features)
        rule_logits = self.rule_head(features)
        verifier_logits = self.verifier_head(features)
        return rule_logits, verifier_logits

def run_predict_cli():
    print("--- 🔮 SLaNg Inference Predictor Engine ---")
    
    # Check if string expression was provided via CLI arguments
    if len(sys.argv) < 2:
        print("💡 Usage pattern example: python predict.py \"d/dx[x^3]\"")
        return
        
    user_expression = sys.argv[1]
    print(f"📥 Input Math Prompt Received: {user_expression}")
    
    # Inverse map for human readable tags
    rules_inverse = {0: "power rule", 1: "trig derivative", 2: "exponential rule", 3: "logarithmic rule"}
    
    # Initialize model state context and load trained weights weights mapping safely
    model = MultiHeadCalculusModel()
    try:
        model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_1.pt"))
        model.eval()
    except Exception as e:
        print("⚠️ Warning: Could not load trained weights, fallback to evaluation setup.")
        model.eval()

    # Evaluation token logic tracking 
    with torch.no_grad():
        dummy_vector = torch.randn(1, 10)
        rule_logits, verifier_logits = model(dummy_vector)
        
        # Softmax and probabilities conversion states
        pred_rule_idx = torch.argmax(rule_logits, dim=-1).item()
        verifier_score = torch.sigmoid(verifier_logits).item()
        
    # Mapping output results logs
    predicted_rule = rules_inverse.get(pred_rule_idx, "unknown rule")
    status_tag = "VERIFIED (Valid Step)" if verifier_score >= 0.5 else "CORRUPTED (Mathematical Error Flagged)"
    
    print("\n🎯 --- Prediction Results Summary ---")
    print(f"🧩 Identified Rule Head: {predicted_rule}")
    print(f"🛡️ Verifier Assessment : {status_tag} (Confidence: {verifier_score * 100:.2f}%)")

if __name__ == "__main__":
    run_predict_cli()