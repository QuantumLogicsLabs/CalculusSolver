import sys
import torch
import torch.nn as nn
from pathlib import Path

# Team ke original structure model tracking folder ko path system mein allow karein
sys.path.append(str(Path(__file__).parent.resolve()))

try:
    # 🎯 FIX 1: Direct team ke actual Transformer layout se import karne ki koshish
    from model.transformer import CalculusSolverModel
    print("🎯 [Shared Architecture] Successfully hooked into the official team Transformer layout!")

except ImportError:
    # 🎯 FIX 2: Agar module external testing ya subfolders mein na miley, 
    # toh yeh identical structure state_dict keys ki mapping ko track rakhta hai.
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
            
            token_logits = self.seq_generation_head(dec_out)
            pooled_features = enc_out[:, -1, :]
            
            rule_logits = self.RuleHead(pooled_features)
            verifier_logits = self.StepTracer(pooled_features)
            
            return token_logits, rule_logits, verifier_logits