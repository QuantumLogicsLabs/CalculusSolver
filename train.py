import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from solver_model import CalculusSolverModel
from tokenizer.slang_serializer import serialize_slang_math

# 🎯 FIX: Read configurations and real vocabulary scale sizes directly
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab_mapping = json.load(f)
REAL_VOCAB_SIZE = len(vocab_mapping)

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

class SlangTrainingDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def _tokenize_envelope_to_ids(self, envelope_dict, max_len=20):
        # Safely invoke serialize_slang_math on canonical fraction dict structures
        token_strings = serialize_slang_math(envelope_dict)
        if isinstance(token_strings, str):
            token_list = token_strings.split()
        else:
            token_list = token_strings
            
        encoded = [vocab_mapping.get(t, vocab_mapping.get("<unk>", 3)) for t in token_list]
        
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        return torch.tensor(encoded[:max_len], dtype=torch.long)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "src_seq": self._tokenize_envelope_to_ids(item["src_tokens"]),
            "tgt_in_seq": self._tokenize_envelope_to_ids(item["tgt_input_tokens"]),
            "tgt_out_seq": self._tokenize_envelope_to_ids(item["tgt_output_tokens"]),
            "rule_id": torch.tensor(item["rule_ids"], dtype=torch.long),
            "v_state": torch.tensor(item["verification_state"], dtype=torch.float)
        }

def main():
    print(f"--- 🏋️ Running Production Slang Architecture System (Vocab: {REAL_VOCAB_SIZE}) ---")
    
    # 🎯 FIX: Removed automatic problem_generator execution script step to decouple operations
    train_loader = DataLoader(
        SlangTrainingDataset("data/splits/train.jsonl"), 
        batch_size=config["batch_size"], 
        shuffle=True
    )
    
    model = CalculusSolverModel(
        vocab_size=REAL_VOCAB_SIZE,
        hidden_dim=config["hidden_dim"]
    )
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
        
        if batch_idx % 500 == 0:
            print(f"[Placeholder Log System] Step {batch_idx}/{config['max_steps']} | Loss: {total_loss.item():.4f}")
            
        if batch_idx >= config["max_steps"]:
            break
            
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/checkpoint_epoch_1.pt")
    print("✨ Checkpoint successfully synced.")

if __name__ == "__main__":
    main()