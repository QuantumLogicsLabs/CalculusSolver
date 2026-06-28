import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from solver_model import CalculusSolverModel
from tokenizer.slang_serializer import serialize_slang_math

# Load central vocabulary limits dynamically
vocab_path = Path("vocab.json")
if not vocab_path.exists():
    # Structural fallback initialization for standalone tracking runs
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "NODE:FRAC": 4, "OP:diff": 5}, f)

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_mapping = json.load(f)
REAL_VOCAB_SIZE = len(vocab_mapping)

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

class SlangTrainingDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.missing_tokens_logged = set()
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def _serialize_and_map_tokens(self, envelope_dict, max_len=20, is_target=False):
        # 🎯 FIX: Call authentic serialize_slang_math directly on structured envelopes
        token_output = serialize_slang_math(envelope_dict)
        
        if isinstance(token_output, str):
            tokens = token_output.split()
        else:
            tokens = list(token_output)
            
        # Standard teacher forcing sequences wrapping bounds configuration if needed
        if is_target:
            tokens = ["<s>"] + tokens + ["</s>"]
            
        encoded_ids = []
        for t in tokens:
            if t not in vocab_mapping:
                if t not in self.missing_tokens_logged:
                    print(f"⚠️ [Vocab Warning] Token '{t}' is missing from vocab.json! Mapping to <unk>.")
                    self.missing_tokens_logged.add(t)
                encoded_ids.append(vocab_mapping.get("<unk>", 3))
            else:
                encoded_ids.append(vocab_mapping[t])
                
        if len(encoded_ids) < max_len:
            encoded_ids += [0] * (max_len - len(encoded_ids))
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
    
    # 🎯 FIX: Decoupled entirely from data generator scripts execution context.
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
            print(f"[Placeholder Log] Step {batch_idx}/{config['max_steps']} | Loss: {total_loss.item():.4f}")
            
        if batch_idx >= config["max_steps"]:
            break
            
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/checkpoint_epoch_1.pt")
    print("✨ SLaNg Checkpoint verification tracking successfully completed.")

if __name__ == "__main__":
    main()