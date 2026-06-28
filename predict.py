import sys
import json
import torch
from pathlib import Path
from solver_model import CalculusSolverModel
from tokenizer.slang_serializer import serialize_slang_math

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab_mapping = json.load(f)
REAL_VOCAB_SIZE = len(vocab_mapping)

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

def evaluate_cli_input():
    if len(sys.argv) < 2:
        print("💡 Usage: python predict.py '{\"op\": \"diff\", \"var\": \"x\", \"expr\": {\"numi\": {\"terms\": [{\"coeff\": 3, \"var\": {\"x\": 2}}]}, \"deno\": 1}}'")
        return
        
    try:
        user_envelope = json.loads(sys.argv[1])
    except Exception:
        print("❌ Error: Command prompt input string must be valid serialized JSON matrix pattern.")
        return
        
    token_output = serialize_slang_math(user_envelope)
    tokens = token_output.split() if isinstance(token_output, str) else list(token_output)
    
    encoded_src = [vocab_mapping.get(t, vocab_mapping.get("<unk>", 3)) for t in tokens]
    if len(encoded_src) < 20:
        encoded_src += [0] * (20 - len(encoded_src))
        
    src_tensor = torch.tensor([encoded_src[:20]], dtype=torch.long)
    dummy_tgt = torch.zeros((1, 20), dtype=torch.long)
    
    model = CalculusSolverModel(vocab_size=REAL_VOCAB_SIZE, hidden_dim=config["hidden_dim"])
    try:
        model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_1.pt"))
    except Exception:
        pass
        
    model.eval()
    with torch.no_grad():
        _, rule_logits, verifier_logits = model(src_tensor, dummy_tgt)
        pred_rule = torch.argmax(rule_logits, dim=-1).item()
        confidence = torch.sigmoid(verifier_logits).item()
        
    print(f"\n🎯 Output Rule Class: {pred_rule} | Verification State Confidence: {confidence*100:.2f}%")

if __name__ == "__main__":
    evaluate_cli_input()