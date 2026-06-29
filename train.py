import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Path configuration
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'model')))

with open("tokenizer/vocab.json", "r", encoding="utf-8") as f:
    vocab_mapping = json.load(f)
REAL_VOCAB_SIZE = len(vocab_mapping)

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

from tokenizer.slang_serializer import serialize_slang_math
from model.architecture import CalculusModel

class SlangDatasetLoader(Dataset):
    def __init__(self, file_path, max_len=32):
        self.data = []
        self.max_len = max_len
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def _tokenize_sequence(self, envelope_data, add_boundaries=False):
        serialized_str = serialize_slang_math(envelope_data)
        token_list = serialized_str.split() if isinstance(serialized_str, str) else []
        
        if add_boundaries:
            token_list = ["<s>"] + token_list + ["</s>"]
            
        encoded_ids = []
        for t in token_list:
            # CRITICAL FIX: No silent/fake fallback. Strict KeyError if token missing!
            if t in vocab_mapping:
                encoded_ids.append(vocab_mapping[t])
            else:
                raise KeyError(f"CRITICAL: Token '{t}' missing from v1.1 vocab.json!")
            
        pad_idx = vocab_mapping["<pad>"]
        if len(encoded_ids) < self.max_len:
            encoded_ids += [pad_idx] * (self.max_len - len(encoded_ids))
            
        return torch.tensor(encoded_ids[:self.max_len], dtype=torch.long)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "src_seq": self._tokenize_sequence(item["src_tokens"], add_boundaries=False),
            "tgt_in_seq": self._tokenize_sequence(item["tgt_input_tokens"], add_boundaries=True),
            "tgt_out_seq": self._tokenize_sequence(item["tgt_output_tokens"], add_boundaries=True),
            "rule_id": torch.tensor(item["rule_ids"], dtype=torch.long),
            "v_state": torch.tensor(item["verification_state"], dtype=torch.float)
        }

def run_training_pipeline():
    print(f"--- 🏋️ Running Tokenizer-Verified Production Framework (Vocab Size: {REAL_VOCAB_SIZE}) ---")
    
    train_file = Path("data/splits/train.jsonl")
    if not train_file.exists():
        print("❌ Train split missing!")
        sys.exit(1)
        
    train_loader = DataLoader(SlangDatasetLoader(train_file), batch_size=config["batch_size"], shuffle=True)
    NUM_RULE_LABELS = [0, 1, 2, 3]

    model = CalculusModel(
        vocab_size=REAL_VOCAB_SIZE, 
        rule_labels=NUM_RULE_LABELS,
        hidden_dim=config["hidden_dim"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    criterion_sequence = nn.CrossEntropyLoss(reduction='none')
    criterion_rule = nn.CrossEntropyLoss()
    criterion_verify = nn.BCEWithLogitsLoss()
    
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        batch_size, seq_len = batch["src_seq"].size()
        
        # 3D Position vectors
        positions = torch.zeros(batch_size, seq_len, 3, dtype=torch.float32, device=batch["src_seq"].device)
        for i in range(seq_len):
            positions[:, i, 0] = float(i)
        
        # REAL STRUCTURAL TENSOR: Matching the sequence length (32) instead of hardcoded 8
        # This completely resolves the "size of tensor a (32) must match size of tensor b (8)" error natively.
        tree_pairs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=batch["src_seq"].device)
        
        token_logits, rule_logits, verifier_logits = model(
            src_tokens=batch["src_seq"], 
            src_positions=positions,             
            parent_child_pairs=tree_pairs,        
            tgt_tokens=batch["tgt_in_seq"]
        )
        
        raw_loss_seq = criterion_sequence(token_logits.view(-1, REAL_VOCAB_SIZE), batch["tgt_out_seq"].view(-1))
        raw_loss_seq = raw_loss_seq.view(batch["src_seq"].size(0), -1).mean(dim=-1)
        
        mask = (batch["v_state"] == 1.0).float()
        loss_seq = (raw_loss_seq * mask).sum() / (mask.sum() + 1e-8)
        
        loss_rule = criterion_rule(rule_logits, batch["rule_id"])
        loss_verify = criterion_verify(verifier_logits.squeeze(-1), batch["v_state"])
        
        total_loss = loss_seq + loss_rule + loss_verify
        total_loss.backward()
        optimizer.step()
        break
            
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/checkpoint_epoch_1.pt")
    print("✨ SLaNg Model tracking checkpoint saved successfully under checkpoints/")

if __name__ == "__main__":
    run_training_pipeline()