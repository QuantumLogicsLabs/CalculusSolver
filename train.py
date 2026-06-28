import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from solver_model import CalculusSolverModel

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 🎯 FIX: Use explicit absolute target structural path configuration files
vocab_path = Path("tokenizer/vocab.json")
if not vocab_path.exists():
    vocab_path = Path("vocab.json")

# Explicit crash loop if the reference is unresolvable rather than silent generation
if not vocab_path.exists():
    raise FileNotFoundError(f"❌ Production error: Token repository schema not found at: {vocab_path}")

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_mapping = json.load(f)
REAL_VOCAB_SIZE = len(vocab_mapping)

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

from tokenizer.slang_serializer import serialize_slang_math

class SlangTrainingDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def _serialize_and_map_tokens(self, envelope_dict, max_len=20, is_target=False):
        token_output = serialize_slang_math(envelope_dict)
        tokens = token_output.split() if isinstance(token_output, str) else list(token_output)
            
        if is_target:
            tokens = ["<s>"] + tokens + ["</s>"]
            
        encoded_ids = []
        for t in tokens:
            if t not in vocab_mapping:
                # Direct strict identification trace to capture bugs instantly
                raise KeyError(f"❌ Training Exception: Token entity '{t}' can not be mapped within target vocabulary maps.")
            encoded_ids.append(vocab_mapping[t])
            
        if len(encoded_ids) < max_len:
            encoded_ids += [vocab_mapping.get("<pad>", 0)] * (max_len - len(encoded_ids))
        return torch.tensor(encoded_ids[:max_len], dtype=torch.long)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "src_seq": self._serialize_and_map_tokens(item["src_tokens"], is_target=False),
            "tgt_in_seq": self._serialize_and_map_tokens(item["tgt_input_tokens"], is_target=True),
            "tgt_out_seq": self._serialize_and_map_tokens(item["tgt_output_tokens"], is_target=True),
            "rule_id": torch.tensor(item["rule_ids"], dtype=torch.long),
            "v_state": torch.tensor(item["verification_state"], dtype=torch.float)
        }

def main():
    print(f"--- 🏋️ Running Tokenizer-Verified Production Framework (Vocab: {REAL_VOCAB_SIZE}) ---")
    train_loader = DataLoader(SlangTrainingDataset("data/splits/train.jsonl"), batch_size=config["batch_size"], shuffle=True)
    
    model = CalculusSolverModel(vocab_size=REAL_VOCAB_SIZE, hidden_dim=config["hidden_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    criterion_sequence = nn.CrossEntropyLoss(reduction='none')
    criterion_rule = nn.CrossEntropyLoss()
    criterion_verify = nn.BCEWithLogitsLoss()
    
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        token_logits, rule_logits, verifier_logits = model(batch["src_seq"], batch["tgt_in_seq"])
        
        raw_loss_seq = criterion_sequence(token_logits.view(-1, REAL_VOCAB_SIZE), batch["tgt_out_seq"].view(-1))
        raw_loss_seq = raw_loss_seq.view(batch["src_seq"].size(0), -1).mean(dim=-1)
        
        mask_correct_steps = (batch["v_state"] == 1.0).float()
        loss_seq = (raw_loss_seq * mask_correct_steps).sum() / (mask_correct_steps.sum() + 1e-8)
        
        loss_rule = criterion_rule(rule_logits, batch["rule_id"])
        loss_verify = criterion_verify(verifier_logits.squeeze(-1), batch["v_state"])
        
        total_loss = loss_seq + loss_rule + loss_verify
        total_loss.backward()
        optimizer.step()
        break
            
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/checkpoint_epoch_1.pt")
    print("✨ Model pipeline verification tracking successfully completed.")

if __name__ == "__main__":
    main()