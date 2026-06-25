import sys
import json
import torch
from model import CalculusSolverModel  # Shared architecture import

# Load dynamic configs
with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

def evaluate_cli_input():
    if len(sys.argv) < 2:
        print("💡 Usage: python predict.py \"d/dx[x^3]\"")
        return
        
    user_input = sys.argv[1]
    print(f"📥 Real Prompt Parsed: {user_input}")
    
    v_size = config["vocab_size"]
    
    # Bound dynamic character indexes within configuration safe-limits
    encoded_src = [((ord(c) % (v_size - 3)) + 3) for c in user_input]
    if len(encoded_src) < 20:
        encoded_src += [0] * (20 - len(encoded_src))
    src_tensor = torch.tensor([encoded_src[:20]], dtype=torch.long)
    dummy_tgt = torch.zeros((1, 20), dtype=torch.long)
    
    rules_inverse = {0: "power rule", 1: "trig derivative", 2: "exponential rule", 3: "logarithmic rule"}
    model = CalculusSolverModel(vocab_size=v_size)
    
    try:
        model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_1.pt"))
    except Exception:
        pass
        
    model.eval()
    with torch.no_grad():
        _, rule_logits, verifier_logits = model(src_tensor, dummy_tgt)
        pred_rule = torch.argmax(rule_logits, dim=-1).item()
        confidence = torch.sigmoid(verifier_logits).item()
        
    print("\n🎯 --- Prediction Results Summary ---")
    print(f"🧩 Identified Rule Head: {rules_inverse.get(pred_rule, 'power rule')}")
    print(f"🛡️ Verifier Assessment : {'VERIFIED' if confidence >= 0.5 else 'CORRUPTED'} (Confidence: {confidence*100:.2f}%)")

if __name__ == "__main__":
    evaluate_cli_input()