import sys
import os
import json
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tokenizer.slang_serializer import serialize_slang_math
from solver_model import CalculusSolverModel

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)


def get_git_commit_hash():
    """Returns the exact current git commit hash for provenance tracking."""
    try:
        hash_str = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return hash_str
    except Exception:
        return "UNKNOWN_COMMIT"


def flatten_vocab(raw_vocab):
    """
    Same flattening rule as inference/beam_search.flatten_vocab on org main:
    merge every sub-dict, skip keys starting with '_' (e.g. _comment, _version).
    """
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

# IDs are NOT contiguous (gaps by design — see docs/KNOWN_ISSUES.md, STRUCT:OPEN @ 23).
REAL_VOCAB_SIZE = max(vocab_mapping.values()) + 1

# Rule labels/tokens, derived from vocab's rule_tokens, ordered by ID.
_rule_items = sorted(_raw_vocab.get("rule_tokens", {}).items(), key=lambda kv: kv[1])
RULE_LABELS = [name.split("RULE:", 1)[1] for name, _ in _rule_items]
RULE_TOKEN_STRINGS = [name for name, _ in _rule_items]

MAX_LEN = config.get("max_len", 48)
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

        rule_idx = item["rule_ids"]
        rule_token = (
            RULE_TOKEN_STRINGS[rule_idx]
            if 0 <= rule_idx < len(RULE_TOKEN_STRINGS)
            else None
        )
        prefix = [rule_token] if rule_token else []

        src_ids = self._tokenize(item["src_tokens"], add_boundaries=False)
        tgt_in_ids = self._tokenize(item["tgt_input_tokens"], extra_prefix_tokens=prefix, add_boundaries=True)
        tgt_out_ids = self._tokenize(item["tgt_output_tokens"], extra_prefix_tokens=prefix, add_boundaries=True)
        return {
            "src_seq": src_ids,
            "tgt_in_seq": tgt_in_ids,
            "tgt_out_seq": tgt_out_ids,
            "rule_id": torch.tensor(item["rule_ids"], dtype=torch.long),
            "v_state": torch.tensor(item["verification_state"], dtype=torch.float),
        }


def preflight_check_max_len(dataset_path, max_len):
    worst = 0
    worst_field = None
    seen = 0
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for field in ("src_tokens", "tgt_input_tokens", "tgt_output_tokens"):
                n = len(serialize_slang_math(row[field])) + 2
                if n > worst:
                    worst = n
                    worst_field = field
            seen += 1
    print(f"[Pre-flight] Scanned {seen} rows in {dataset_path}. "
          f"Max token length observed: {worst} (field: {worst_field}). "
          f"Configured max_len: {max_len}.")
    if worst > max_len:
        print(f"[Pre-flight] WARNING: real max length ({worst}) exceeds "
              f"max_len ({max_len}) -- sequences WILL be silently truncated. "
              f"Raise max_len in config.json before continuing.")
    else:
        print(f"[Pre-flight] OK: max_len has {max_len - worst} tokens of headroom.")


def evaluate_validation(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct_seq = 0
    total_correct_tokens = 0
    total_valid_tokens = 0
    total_seq = 0
    steps = 0

    with torch.no_grad():
        for batch in val_loader:
            src_seq = batch["src_seq"]
            tgt_in = batch["tgt_in_seq"][:, :-1]
            tgt_out = batch["tgt_out_seq"][:, 1:]
            rule_id = batch["rule_id"]

            # REPLACED CODE: updated to multi-output model forward pass
            decoder_logits, rule_logits, verifier_logits = model(src_seq, tgt_in, true_rule_ids=rule_id)
            loss = criterion(decoder_logits.reshape(-1, REAL_VOCAB_SIZE), tgt_out.reshape(-1))
            total_loss += loss.item()

            preds = decoder_logits.argmax(dim=-1)
            mask = tgt_out != PAD_ID

            correct_token_mask = (preds == tgt_out) & mask
            total_correct_tokens += correct_token_mask.sum().item()
            total_valid_tokens += mask.sum().item()

            correct_seq = ((preds == tgt_out) | ~mask).all(dim=1)
            total_correct_seq += correct_seq.sum().item()
            total_seq += tgt_out.size(0)
            steps += 1

    if steps == 0:
        return 0.0, 0.0, 0.0

    avg_loss = total_loss / steps
    seq_acc = total_correct_seq / max(total_seq, 1)
    token_acc = (
        total_correct_tokens / max(total_valid_tokens, 1)
        if total_valid_tokens > 0
        else 0.0
    )

    return avg_loss, seq_acc, token_acc


def write_training_results(metrics_log, best_val_loss, git_commit_hash):
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    lines = [
        "# Training Results",
        "",
        f"**Git Commit Hash:** `{git_commit_hash}`",
        f"**Best Validation Loss:** {best_val_loss:.4f}" if best_val_loss < float("inf") else "**Best Validation Loss:** N/A",
        f"**Total Epochs Run:** {len(metrics_log)}",
        "",
        "## Per-Epoch Metrics",
        "",
        "| Epoch | Train Loss | Val Loss | Per-Token Acc | Val Seq Acc | Saved |",
        "|-------|-----------|----------|---------------|-------------|-------|",
    ]
    for m in metrics_log:
        val_loss = f"{m['val_loss']:.4f}" if m['val_loss'] is not None else "N/A"
        token_acc = f"{m['val_token_acc']:.4f}" if m['val_token_acc'] is not None else "N/A"
        val_acc = f"{m['val_seq_acc']:.4f}" if m['val_seq_acc'] is not None else "N/A"
        saved = "Yes" if m['saved'] else "No"
        lines.append(
            f"| {m['epoch']} | {m['train_loss']:.4f} | {val_loss} | {token_acc} | {val_acc} | {saved} |"
        )

    lines.extend([
        "",
        "## Configuration Snapshot",
        "",
        f"- **Learning Rate:** {config.get('learning_rate')}",
        f"- **Warmup Steps:** {config.get('warmup_steps', 1000)}",
        f"- **Batch Size:** {config.get('batch_size')}",
        f"- **Hidden Dim:** {config.get('hidden_dim')}",
        f"- **Max Len:** {MAX_LEN}",
        f"- **Max Steps/Epoch:** {config.get('max_steps')}",
        f"- **Early Stopping:** patience={config.get('early_stopping', {}).get('patience', 'N/A')}, min_delta={config.get('early_stopping', {}).get('min_delta', 'N/A')}",
        f"- **Vocab Size:** {REAL_VOCAB_SIZE}",
        f"- **Gradient Clipping:** max_norm={config.get('grad_clip_max_norm', 1.0)}",
        f"- **Rule Prediction:** Multi-head prediction output (decoder_logits, rule_logits, verifier_logits)",
        "",
    ])

    with open(docs_dir / "TRAINING_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Training results written to docs/TRAINING_RESULTS.md")


def run_training_pipeline():
    commit_hash = get_git_commit_hash()
    print(f"--- Training CalculusSolverModel (commit: {commit_hash}, vocab: {REAL_VOCAB_SIZE}) ---")

    train_file = Path("data/splits/train.jsonl")
    if not train_file.exists():
        print("CRITICAL: Train split missing! Run problem_generator.py first.")
        sys.exit(1)

    preflight_check_max_len(train_file, MAX_LEN)

    print("[DEBUG] Loading train dataset into memory...", flush=True)
    train_dataset = SlangDatasetLoader(train_file)
    print(f"[DEBUG] Train dataset loaded: {len(train_dataset)} examples", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    val_file = Path("data/splits/val.jsonl")
    val_loader = None
    if val_file.exists() and config.get("validation_logging", True):
        print("[DEBUG] Loading val dataset into memory...", flush=True)
        val_dataset = SlangDatasetLoader(val_file)
        print(f"[DEBUG] Val dataset loaded: {len(val_dataset)} examples", flush=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    print("[DEBUG] Building model...", flush=True)
    model = CalculusSolverModel(
        vocab_size=REAL_VOCAB_SIZE,
        num_rules=len(RULE_LABELS),
        hidden_dim=config["hidden_dim"],
        rule_labels=RULE_LABELS,
    )

    base_lr = config["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    warmup_steps = config.get("warmup_steps", 1000)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

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
    global_step = 0

    print("[DEBUG] Entering training loop...", flush=True)
    for epoch in range(1, epochs + 1):
        print(f"[DEBUG] Starting epoch {epoch}, waiting for first batch from DataLoader...", flush=True)
        model.train()
        epoch_loss = 0.0
        steps_run = 0

        for step, batch in enumerate(train_loader):
            if step >= config.get("max_steps", 1500):
                break
            if step < 3 or step % 10 == 0:
                print(f"[DEBUG] epoch {epoch} step {step} - batch received, running forward/backward...", flush=True)
            optimizer.zero_grad()

            src_seq = batch["src_seq"]
            tgt_in = batch["tgt_in_seq"][:, :-1]
            tgt_out = batch["tgt_out_seq"][:, 1:]
            rule_id = batch["rule_id"]

            # REPLACED CODE: updated to multi-output model forward pass
            decoder_logits, rule_logits, verifier_logits = model(src_seq, tgt_in, true_rule_ids=rule_id)
            loss = criterion(decoder_logits.reshape(-1, REAL_VOCAB_SIZE), tgt_out.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            steps_run += 1

        avg_train_loss = epoch_loss / max(steps_run, 1)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} (LR: {current_lr:.6f})")

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": None,
            "val_token_acc": None,
            "val_seq_acc": None,
            "saved": False,
        }

        if val_loader is not None:
            val_loss, val_seq_acc, val_token_acc = evaluate_validation(model, val_loader, criterion)
            print(
                f"Epoch {epoch} - Val Loss: {val_loss:.4f} | "
                f"Token Acc: {val_token_acc:.4f} | Seq Acc: {val_seq_acc:.4f}"
            )

            epoch_metrics["val_loss"] = val_loss
            epoch_metrics["val_token_acc"] = val_token_acc
            epoch_metrics["val_seq_acc"] = val_seq_acc

            if val_loss < best_val_loss - (min_delta if use_early_stopping else 0):
                best_val_loss = val_loss
                patience_counter = 0
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(FINAL_CHECKPOINT_PATH))
                print(f"  New best validation loss! Saved checkpoint to {FINAL_CHECKPOINT_PATH}")
                epoch_metrics["saved"] = True
            else:
                patience_counter += 1
                print(f"  Epoch {epoch}: val loss {val_loss:.4f} did not improve from {best_val_loss:.4f}.")
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

    write_training_results(metrics_log, best_val_loss, commit_hash)
    print("--- Training complete ---")


if __name__ == "__main__":
    run_training_pipeline()
