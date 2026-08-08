import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tokenizer.slang_serializer import serialize_slang_math
from model.simple_transformer import SimpleCalculusModel

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)


def flatten_vocab(raw_vocab):
    flat = {}
    for key, value in raw_vocab.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            flat.update(value)
    return flat


with open("tokenizer/vocab.json", "r", encoding="utf-8") as f:
    _raw_vocab = json.load(f)

vocab_mapping = flatten_vocab(_raw_vocab)
REAL_VOCAB_SIZE = max(vocab_mapping.values()) + 1

# Rule labels/tokens, derived from vocab's rule_tokens, ordered by ID.
# Used only to translate a dataset row's existing rule_ids INDEX (0-12)
# back into its real RULE:xxx vocab token string, so it can be prepended
# to the target sequence. problem_generator.py / rule_ids format is
# unchanged -- this translation happens here in train.py only.
_rule_items = sorted(_raw_vocab.get("rule_tokens", {}).items(), key=lambda kv: kv[1])
RULE_TOKEN_STRINGS = [name for name, _ in _rule_items]  # e.g. "RULE:power_rule"

MAX_LEN = config.get("max_len", 32)
PAD_ID = vocab_mapping["[PAD]"]

CHECKPOINT_DIR = Path("checkpoints/final")
FINAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"


class SlangDatasetLoader(Dataset):
    def __init__(self, file_path, max_len=MAX_LEN):
        self.data = []
        self.max_len = max_len
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def _tokenize(self, envelope, extra_prefix_tokens=None, add_boundaries=False):
        tokens = serialize_slang_math(envelope)
        if extra_prefix_tokens:
            tokens = list(extra_prefix_tokens) + tokens
        if add_boundaries:
            tokens = ["[BOS]"] + tokens + ["[EOS]"]

        ids = []
        for t in tokens:
            if t in vocab_mapping:
                ids.append(vocab_mapping[t])
            else:
                raise KeyError(f"CRITICAL: Token '{t}' missing from vocab.json!")

        pad_idx = vocab_mapping["[PAD]"]
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [pad_idx] * pad_len

        return torch.tensor(ids[: self.max_len], dtype=torch.long)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Translate this row's rule_ids index into its real RULE:xxx token
        # string, so it becomes part of the sequence the decoder learns to
        # generate (as the very first token after [BOS]), instead of a
        # separate classifier target.
        rule_idx = item["rule_ids"]
        rule_token = (
            RULE_TOKEN_STRINGS[rule_idx]
            if 0 <= rule_idx < len(RULE_TOKEN_STRINGS)
            else None
        )
        prefix = [rule_token] if rule_token else []

        src_ids = self._tokenize(item["src_tokens"], add_boundaries=False)
        tgt_in_ids = self._tokenize(
            item["tgt_input_tokens"], extra_prefix_tokens=prefix, add_boundaries=True
        )
        tgt_out_ids = self._tokenize(
            item["tgt_output_tokens"], extra_prefix_tokens=prefix, add_boundaries=True
        )
        return {
            "src_seq": src_ids,
            "tgt_in_seq": tgt_in_ids,
            "tgt_out_seq": tgt_out_ids,
            "v_state": torch.tensor(item["verification_state"], dtype=torch.float),
        }


def evaluate_validation(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct_seq = 0
    total_seq = 0
    steps = 0
    with torch.no_grad():
        for batch in val_loader:
            src_seq = batch["src_seq"]
            # Standard teacher-forced shift: tgt_in_seq is tgt_out_seq minus
            # the last token; loss is computed against tgt_out_seq minus the
            # first token ([BOS]). Both already have [BOS]/[EOS] baked in
            # from _tokenize's add_boundaries=True.
            tgt_in = batch["tgt_in_seq"][:, :-1]
            tgt_out = batch["tgt_out_seq"][:, 1:]

            logits = model(src_seq, tgt_in)
            loss = criterion(
                logits.reshape(-1, REAL_VOCAB_SIZE), tgt_out.reshape(-1)
            )
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            mask = tgt_out != PAD_ID
            correct = ((preds == tgt_out) | ~mask).all(dim=1)
            total_correct_seq += correct.sum().item()
            total_seq += tgt_out.size(0)
            steps += 1

    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_correct_seq / max(total_seq, 1)


def write_training_results(metrics_log, best_val_loss):
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    lines = [
        "# Training Results",
        "",
        f"**Best Validation Loss:** {best_val_loss:.4f}" if best_val_loss < float("inf") else "**Best Validation Loss:** N/A",
        f"**Total Epochs Run:** {len(metrics_log)}",
        "",
        "## Per-Epoch Metrics",
        "",
        "| Epoch | Train Loss | Val Loss | Val Seq Accuracy | Checkpoint Saved |",
        "|-------|-----------|----------|-------------------|-----------------|",
    ]
    for m in metrics_log:
        val_loss = f"{m['val_loss']:.4f}" if m['val_loss'] is not None else "N/A"
        val_acc = f"{m['val_seq_acc']:.4f}" if m['val_seq_acc'] is not None else "N/A"
        saved = "Yes" if m['saved'] else "No"
        lines.append(
            f"| {m['epoch']} | {m['train_loss']:.4f} | {val_loss} | {val_acc} | {saved} |"
        )

    lines.extend([
        "",
        "## Configuration",
        "",
        f"- **Architecture:** SimpleCalculusModel (standard nn.Transformer encoder-decoder)",
        f"- **Learning Rate:** {config.get('learning_rate')}",
        f"- **Batch Size:** {config.get('batch_size')}",
        f"- **Hidden Dim:** {config.get('hidden_dim')}",
        f"- **Max Steps/Epoch:** {config.get('max_steps')}",
        f"- **Early Stopping:** patience={config.get('early_stopping', {}).get('patience', 'N/A')}, min_delta={config.get('early_stopping', {}).get('min_delta', 'N/A')}",
        f"- **Vocab Size:** {REAL_VOCAB_SIZE}",
        f"- **Gradient Clipping:** max_norm={config.get('grad_clip_max_norm', 1.0)}",
        f"- **Rule prediction:** folded into output sequence as leading RULE:xxx token (see docs/KNOWN_ISSUES.md)",
        "",
    ])

    with open(docs_dir / "TRAINING_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Training results written to docs/TRAINING_RESULTS.md")


def run_training_pipeline():
    print(f"--- Training SimpleCalculusModel (vocab size: {REAL_VOCAB_SIZE}) ---")

    train_file = Path("data/splits/train.jsonl")
    if not train_file.exists():
        print("Train split missing!")
        sys.exit(1)

    train_loader = DataLoader(SlangDatasetLoader(train_file), batch_size=config["batch_size"], shuffle=True)

    val_file = Path("data/splits/val.jsonl")
    val_loader = None
    if val_file.exists() and config.get("validation_logging", True):
        val_loader = DataLoader(SlangDatasetLoader(val_file), batch_size=config["batch_size"], shuffle=False)

    model = SimpleCalculusModel(
        vocab_size=REAL_VOCAB_SIZE,
        hidden_dim=config["hidden_dim"],
        pad_id=PAD_ID,
        max_len=MAX_LEN,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    grad_clip_max_norm = config.get("grad_clip_max_norm", 1.0)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_val_loss = float("inf")
    patience_counter = 0
    metrics_log = []

    early_stopping_cfg = config.get("early_stopping", False)
    if isinstance(early_stopping_cfg, dict):
        patience = early_stopping_cfg.get("patience", 3)
        min_delta = early_stopping_cfg.get("min_delta", 1e-4)
        use_early_stopping = True
    elif isinstance(early_stopping_cfg, int):
        patience = early_stopping_cfg
        min_delta = 1e-4
        use_early_stopping = True
    elif isinstance(early_stopping_cfg, bool) and early_stopping_cfg:
        patience = 3
        min_delta = 1e-4
        use_early_stopping = True
    else:
        use_early_stopping = False

    epochs = config.get("epochs", 1)

    if FINAL_CHECKPOINT_PATH.exists():
        try:
            model.load_state_dict(torch.load(str(FINAL_CHECKPOINT_PATH), map_location="cpu"))
            print(f"Loaded existing checkpoint from {FINAL_CHECKPOINT_PATH} to resume training.")
            if val_loader is not None:
                val_loss, val_acc = evaluate_validation(model, val_loader, criterion)
                best_val_loss = val_loss
                print(f"Initial val loss from resumed checkpoint: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Could not load checkpoint to resume: {e}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps_run = 0

        for step, batch in enumerate(train_loader):
            if step >= config.get("max_steps", 1500):
                break
            optimizer.zero_grad()

            src_seq = batch["src_seq"]
            tgt_in = batch["tgt_in_seq"][:, :-1]
            tgt_out = batch["tgt_out_seq"][:, 1:]

            logits = model(src_seq, tgt_in)
            loss = criterion(logits.reshape(-1, REAL_VOCAB_SIZE), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()

            epoch_loss += loss.item()
            steps_run += 1

        avg_train_loss = epoch_loss / max(steps_run, 1)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": None,
            "val_seq_acc": None,
            "saved": False,
        }

        if val_loader is not None:
            val_loss, val_acc = evaluate_validation(model, val_loader, criterion)
            print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}  Val Seq Accuracy: {val_acc:.4f}")

            epoch_metrics["val_loss"] = val_loss
            epoch_metrics["val_seq_acc"] = val_acc

            if val_loss < best_val_loss - (min_delta if use_early_stopping else 0):
                best_val_loss = val_loss
                patience_counter = 0
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(FINAL_CHECKPOINT_PATH))
                print(f"  New best validation loss! Saved checkpoint to {FINAL_CHECKPOINT_PATH}")
                epoch_metrics["saved"] = True
            else:
                patience_counter += 1
                print(f"  Epoch {epoch}: val loss {val_loss:.4f} did not improve from {best_val_loss:.4f}, skipping checkpoint save.")
                if use_early_stopping and patience_counter >= patience:
                    print("Early stopping triggered. Training stopped.")
                    metrics_log.append(epoch_metrics)
                    break
        else:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(FINAL_CHECKPOINT_PATH))
            print(f"Checkpoint saved to {FINAL_CHECKPOINT_PATH}")
            epoch_metrics["saved"] = True

        metrics_log.append(epoch_metrics)

    write_training_results(metrics_log, best_val_loss)
    print("--- Training complete ---")


if __name__ == "__main__":
    run_training_pipeline()