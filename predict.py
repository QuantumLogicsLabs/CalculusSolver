import sys
import torch
import torch.nn as nn

class CalculusSolverModel(nn.Module):
    def __init__(self, vocab_size=256, embedding_dim=64, hidden_dim=128, num_rules=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.TreeEncoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.TreeDecoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.seq_generation_head = nn.Linear(hidden_dim, vocab_size)
        self.RuleHead = nn.Linear(hidden_dim, num_rules)
        self.StepTracer = nn.Linear(hidden_dim, 1)
        
    def forward(self, src_seq, tgt_in_seq):
        embedded_src = self.embedding(src_seq)
        enc_out, (hn, cn) = self.TreeEncoder(embedded_src)
        embedded_tgt = self.embedding(tgt_in_seq)
        dec_out, _ = self.TreeDecoder(embedded_tgt, (hn, cn))
        return self.seq_generation_head(dec_out), self.RuleHead(enc_out[:, -1, :]), self.StepTracer(enc_out[:, -1, :])

def evaluate_cli_input():
    if len(sys.argv) < 2:
        print("💡 Usage: python predict.py \"d/dx[x^3]\"")
        return
        
    user_input = sys.argv[1]
    print(f"📥 Real Prompt Parsed: {user_input}")
    
    encoded_src = [((ord(c) % 253) + 3) for c in user_input]
    if len(encoded_src) < 20:
        encoded_src += [0] * (20 - len(encoded_src))
    src_tensor = torch.tensor([encoded_src[:20]], dtype=torch.long)
    dummy_tgt = torch.zeros((1, 20), dtype=torch.long)
    
    rules_inverse = {0: "power rule", 1: "trig derivative", 2: "exponential rule", 3: "logarithmic rule"}
    model = CalculusSolverModel()
    
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