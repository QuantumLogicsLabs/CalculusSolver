import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

with open("tokenizer/vocab.json", "r", encoding="utf-8") as f:
    vocab_mapping = json.load(f)
REAL_VOCAB_SIZE = len(vocab_mapping)

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

from tokenizer.slang_serializer import serialize_slang_math
from solver_model import CalculusSolverModel

NUM_RULES = config.get("num_rules", 15)
MAX_LEN = config.get("max_len", 32)


class SlangDatasetLoader(Dataset):
    def __init__(self, file_path, max_len=MAX_LEN):
        self.data = []
        self.max_len = max_len
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def _tokenize(self, envelope, add_boundaries=False):
        tokens, parent_ids, child_ids = serialize_slang_math(envelope)
        if add_boundaries:
            tokens = ["<s>"] + tokens + ["</s>"]
            parent_ids = [-1] + parent_ids + [-1]
            child_ids = [-1] + child_ids + [-1]

        ids = []
        for t in tokens:
            if t in vocab_mapping:
                ids.append(vocab_mapping[t])
            else:
                raise KeyError(f"CRITICAL: Token '{t}' missing from vocab.json!")

        pad_idx = vocab_mapping["<pad>"]
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [pad_idx] * pad_len
            parent_ids += [-1] * pad_len
            child_ids += [-1] * pad_len

        return (
            torch.tensor(ids[: self.max_len], dtype=torch.long),
            torch.tensor(parent_ids[: self.max_len], dtype=torch.long),
            torch.tensor(child_ids[: self.max_len], dtype=torch.long),
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        src_ids, src_parents, src_children = self._tokenize(item["src_tokens"], add_boundaries=False)
        tgt_in_ids, _, _ = self._tokenize(item["tgt_input_tokens"], add_boundaries=True)
        tgt_out_ids, _, _ = self._tokenize(item["tgt_output_tokens"], add_boundaries=True)
        return {
            "src_seq": src_ids,
            "src_parent_ids": src_parents,
            "src_child_ids": src_children,
            "tgt_in_seq": tgt_in_ids,
            "tgt_out_seq": tgt_out_ids,
            "rule_id": torch.tensor(item["rule_ids"], dtype=torch.long),
            "v_state": torch.tensor(item["verification_state"], dtype=torch.float),
        }


def run_training_pipeline():
    print(f"--- Training (vocab size: {REAL_VOCAB_SIZE}) ---")

    train_file = Path("data/splits/train.jsonl")
    if not train_file.exists():
        print("Train split missing!")
        sys.exit(1)

    train_loader = DataLoader(SlangDatasetLoader(train_file), batch_size=config["batch_size"], shuffle=True)

    model = CalculusSolverModel(vocab_size=REAL_VOCAB_SIZE, num_rules=NUM_RULES, hidden_dim=config["hidden_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    criterion_sequence = nn.CrossEntropyLoss(reduction='none')
    criterion_rule = nn.CrossEntropyLoss()
    criterion_verify = nn.BCEWithLogitsLoss()

    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        token_logits, rule_logits, verifier_logits = model(
            input_ids=batch["src_seq"],
            tgt_ids=batch["tgt_in_seq"],
            parent_ids=batch["src_parent_ids"],
            child_ids=batch["src_child_ids"],
            gold_rule_labels=batch["rule_id"],
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
    print("Checkpoint saved to checkpoints/")


if __name__ == "__main__":
    run_training_pipeline()