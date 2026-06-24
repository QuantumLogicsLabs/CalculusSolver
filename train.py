import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Load configurations
with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

class SlangTrainingDataset(Dataset):
    def __init__(self, file_path, vocab_size=256):
        self.data = []
        self.vocab_size = vocab_size
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def _pad_or_truncate(self, tokens, max_len=20):
        encoded = []
        for c in tokens:
            # Handle special sequence boundary tokens securely
            if c == "<s>":
                encoded.append(1)
            elif c == "</s>":
                encoded.append(2)
            else:
                encoded.append((ord(c) % (self.vocab_size - 3)) + 3)
                
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        return torch.tensor(encoded[:max_len], dtype=torch.long)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "src_seq": self._pad_or_truncate(item["src_tokens"]),
            "tgt_in_seq": self._pad_or_truncate(item["tgt_input_tokens"]),
            "tgt_out_seq": self._pad_or_truncate(item["tgt_output_tokens"]),
            "rule_id": torch.tensor(item["rule_ids"], dtype=torch.long),
            "v_state": torch.tensor(item["verification_state"], dtype=torch.float)
        }

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

def main():
    print("--- 🏋️ Running Corrected 3-Head Shared Architecture Pipeline ---")
    train_loader = DataLoader(SlangTrainingDataset("data/splits/train.jsonl"), batch_size=config["batch_size"], shuffle=True)
    
    model = CalculusSolverModel(embedding_dim=config["embedding_dim"], hidden_dim=config["hidden_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    criterion_sequence = nn.CrossEntropyLoss()
    criterion_rule = nn.CrossEntropyLoss()
    criterion_verify = nn.BCEWithLogitsLoss()
    
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        token_logits, rule_logits, verifier_logits = model(batch["src_seq"], batch["tgt_in_seq"])
        
        loss_seq = criterion_sequence(token_logits.view(-1, 256), batch["tgt_out_seq"].view(-1))
        loss_rule = criterion_rule(rule_logits, batch["rule_id"])
        loss_verify = criterion_verify(verifier_logits.squeeze(-1), batch["v_state"])
        
        total_loss = loss_seq + loss_rule + loss_verify
        total_loss.backward()
        optimizer.step()
        
        if batch_idx % 500 == 0:
            print(f"[Placeholder Log System] Step {batch_idx}/{config['max_steps']} | Consolidated Loss: {total_loss.item():.4f}")
            
        if batch_idx >= config["max_steps"]:
            break
            
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/checkpoint_epoch_1.pt")
    print("✨ SLaNg Checkpoint successfully saved inside checkpoints/ folder.")

if __name__ == "__main__":
    main()