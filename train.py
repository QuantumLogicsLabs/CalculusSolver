import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from model import CalculusSolverModel  # Dynamically synced through shared module

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

class SlangTrainingDataset(Dataset):
    def __init__(self, file_path, vocab_size):
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
            if c == "<s>": encoded.append(1)
            elif c == "</s>": encoded.append(2)
            else: encoded.append((ord(c) % (self.vocab_size - 3)) + 3)
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

def main():
    print("--- 🏋️ Running Masked Token-Loss Architecture System ---")
    v_size = config["vocab_size"]
    
    train_loader = DataLoader(
        SlangTrainingDataset("data/splits/train.jsonl", vocab_size=v_size), 
        batch_size=config["batch_size"], 
        shuffle=True
    )
    
    model = CalculusSolverModel(
        vocab_size=v_size,
        embedding_dim=config["embedding_dim"], 
        hidden_dim=config["hidden_dim"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    criterion_sequence = nn.CrossEntropyLoss(reduction='none') # Element-wise matrix for dynamic masking
    criterion_rule = nn.CrossEntropyLoss()
    criterion_verify = nn.BCEWithLogitsLoss()
    
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        token_logits, rule_logits, verifier_logits = model(batch["src_seq"], batch["tgt_in_seq"])
        
        # 1. Raw Sequence Loss matrix computation
        raw_loss_seq = criterion_sequence(token_logits.view(-1, v_size), batch["tgt_out_seq"].view(-1))
        raw_loss_seq = raw_loss_seq.view(batch["src_seq"].size(0), -1).mean(dim=-1)
        
        # 🎯 FIX 3: Masking incorrect sequence data! Loss will ONLY train generation head when verification_state == 1
        mask_correct_steps = (batch["v_state"] == 1.0).float()
        loss_seq = (raw_loss_seq * mask_correct_steps).sum() / (mask_correct_steps.sum() + 1e-8)
        
        # 2. Rule classification and binary validation loss loops
        loss_rule = criterion_rule(rule_logits, batch["rule_id"])
        loss_verify = criterion_verify(verifier_logits.squeeze(-1), batch["v_state"])
        
        total_loss = loss_seq + loss_rule + loss_verify
        total_loss.backward()
        optimizer.step()
        
        # NOTE: Logging utilizes bare prints intentionally as a designated placeholder system.
        if batch_idx % 500 == 0:
            print(f"[Placeholder Log System] Step {batch_idx}/{config['max_steps']} | Consolidated Loss: {total_loss.item():.4f}")
            
        if batch_idx >= config["max_steps"]:
            break
            
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/checkpoint_epoch_1.pt")
    print("✨ SLaNg Checkpoint successfully synchronized and saved.")

if __name__ == "__main__":
    main()