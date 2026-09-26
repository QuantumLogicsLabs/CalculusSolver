import sys
import os
import json
import subprocess
import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tokenizer.slang_serializer import serialize_slang_math
from solver_model import CalculusSolverModel, check_forward_contract

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)


def get_git_commit_hash():
    try:
        hash_str = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return hash_str
    except Exception:
        return "UNKNOWN_COMMIT"


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
        op = (
            item["src_tokens"].get("op", "unknown")
            if isinstance(item.get("src_tokens"), dict)
            else "unknown"
        )
        return {
            "src_seq": src_ids,
            "tgt_in_seq": tgt_in_ids,
            "tgt_out_seq": tgt_out_ids,
            "rule_id": torch.tensor(item["rule_ids"], dtype=torch.long),
            "v_state": torch.tensor(item["verification_state"], dtype=torch.float),
            "op": op,
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


def evaluate_validation(model, val_loader, criterion, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_correct_seq = 0
    total_correct_tokens = 0
    total_valid_tokens = 0
    total_seq = 0
    steps = 0

    category_correct = {}
    category_total = {}

    with torch.no_grad():
        for batch in val_loader:
            src_seq = batch["src_seq"].to(device)
            tgt_in = batch["tgt_in_seq"][:, :-1].to(device)
            tgt_out = batch["tgt_out_seq"][:, 1:].to(device)
            rule_id = batch["rule_id"].to(device)
            ops = batch.get("op", None)

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

            if ops is not None:
                for i, op in enumerate(ops):
                    category_total[op] = category_total.get(op, 0) + 1
                    if correct_seq[i].item():
                        category_correct[op] = category_correct.get(op, 0) + 1

            steps += 1

    if steps == 0:
        return 0.0, 0.0, 0.0, {}

    avg_loss = total_loss / steps
    seq_acc = total_correct_seq / max(total_seq, 1)
    token_acc = (
        total_correct_tokens / max(total_valid_tokens, 1)
        if total_valid_tokens > 0
        else 0.0
    )
    category_acc = {
        op: category_correct.get(op, 0) / category_total[op]
        for op in sorted(category_total.keys())
        if category_total[op] > 0
    }

    return avg_loss, seq_acc, token_acc, category_acc


def evaluate_free_running(model, val_dataset, num_examples=15, max_gen_len=None, device="cpu"):
    if max_gen_len is None:
        max_gen_len = MAX_LEN
    """
    Free-running generation check: greedy decode (no beam search, no teacher forcing).
    """
    model.eval()
    
    total = len(val_dataset)
    if total == 0:
        return 0.0, 0.0, 0.0
    
    indices = [int(i * total / num_examples) for i in range(min(num_examples, total))]
    
    bos_id = vocab_mapping["[BOS]"]
    eos_id = vocab_mapping["[EOS]"]
    pad_id = vocab_mapping["[PAD]"]
    
    correct_seqs = 0
    total_tokens = 0
    correct_tokens = 0
    total_gen_len = 0
    
    with torch.no_grad():
        for idx in indices:
            item = val_dataset[idx]
            src_seq = item["src_seq"].unsqueeze(0).to(device)
            
            tgt_out_full = item["tgt_out_seq"]
            tgt_out = tgt_out_full[1:]
            tgt_mask = tgt_out != pad_id
            tgt_len = int(tgt_mask.sum().item())
            tgt_out_clean = tgt_out[:tgt_len]
            if len(tgt_out_clean) > 0 and tgt_out_clean[-1] == eos_id:
                tgt_out_clean = tgt_out_clean[:-1]
                tgt_len = len(tgt_out_clean)
            
            generated = [bos_id]
            for _ in range(max_gen_len):
                tgt_in = torch.tensor([generated], dtype=torch.long, device=device)
                logits, _, _ = model(src_seq, tgt_in, true_rule_ids=None)
                next_token = int(logits[0, -1, :].argmax().item())
                generated.append(next_token)
                if next_token == eos_id:
                    break
            
            gen_tokens = generated[1:]
            if gen_tokens and gen_tokens[-1] == eos_id:
                gen_tokens = gen_tokens[:-1]
            
            total_gen_len += len(gen_tokens)
            
            gen_tensor = torch.tensor(gen_tokens)
            if len(gen_tokens) == tgt_len and torch.equal(gen_tensor, tgt_out_clean):
                correct_seqs += 1
            
            compare_len = min(len(gen_tokens), tgt_len)
            if compare_len > 0:
                gen_cmp = gen_tensor[:compare_len]
                tgt_cmp = tgt_out_clean[:compare_len]
                correct_tokens += int((gen_cmp == tgt_cmp).sum().item())
                total_tokens += compare_len
    
    n = len(indices)
    free_run_seq_acc = correct_seqs / max(n, 1)
    free_run_token_acc = correct_tokens / max(total_tokens, 1)
    avg_gen_len = total_gen_len / max(n, 1)
    
    return free_run_seq_acc, free_run_token_acc, avg_gen_len


def get_scheduled_sampling_prob(epoch, total_epochs, ss_config):
    """Return probability of using model's own prediction for this epoch."""
    if not ss_config.get("enabled", False):
        return 0.0
    
    start = ss_config.get("start_prob", 0.0)
    end = ss_config.get("end_prob", 0.3)
    schedule = ss_config.get("schedule", "linear")
    
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    
    if schedule == "linear":
        return start + (end - start) * progress
    elif schedule == "exponential":
        ratio = end / max(start, 1e-8)
        return start * (ratio ** progress)
    elif schedule == "sigmoid":
        midpoint = 0.5
        steepness = 10
        sigmoid = 1 / (1 + math.exp(-steepness * (progress - midpoint)))
        sigmoid_min = 1 / (1 + math.exp(steepness * midpoint))
        sigmoid_max = 1 / (1 + math.exp(-steepness * (1 - midpoint)))
        normalized = (sigmoid - sigmoid_min) / (sigmoid_max - sigmoid_min)
        return start + (end - start) * normalized
    else:
        return start + (end - start) * progress


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
        "| Epoch | Train Loss | Val Loss | Per-Token Acc | Val Seq Acc | Free-Run Seq Acc | Free-Run Token Acc | Avg Gen Len | Saved |",
        "|-------|-----------|----------|---------------|-------------|------------------|--------------------|-------------|-------|",
    ]
    for m in metrics_log:
        val_loss = f"{m['val_loss']:.4f}" if m['val_loss'] is not None else "N/A"
        token_acc = f"{m['val_token_acc']:.4f}" if m['val_token_acc'] is not None else "N/A"
        val_acc = f"{m['val_seq_acc']:.4f}" if m['val_seq_acc'] is not None else "N/A"
        fr_seq = f"{m.get('free_run_seq_acc', 0):.4f}"
        fr_tok = f"{m.get('free_run_token_acc', 0):.4f}"
        fr_len = f"{m.get('free_run_avg_len', 0):.1f}"
        saved = "Yes" if m['saved'] else "No"
        lines.append(
            f"| {m['epoch']} | {m['train_loss']:.4f} | {val_loss} | {token_acc} | {val_acc} | {fr_seq} | {fr_tok} | {fr_len} | {saved} |"
        )

    has_cat = any(m.get("val_category_acc") for m in metrics_log)
    if has_cat:
        all_cats = sorted({cat for m in metrics_log for cat in m.get("val_category_acc", {}).keys()})
        cat_header = " | ".join(c.capitalize() for c in all_cats)
        cat_sep = " | ".join(["---"] * len(all_cats))
        lines.extend([
            "",
            "## Per-Category Validation Sequence Accuracy",
            "",
            f"| Epoch | {cat_header} |",
            f"|-------|{cat_sep}|",
        ])
        for m in metrics_log:
            cats = m.get("val_category_acc", {})
            row_vals = " | ".join(f"{cats.get(c, 0.0):.4f}" if c in cats else "N/A" for c in all_cats)
            lines.append(f"| {m['epoch']} | {row_vals} |")

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
        f"- **Early Stopping:** {'patience=' + str(config['early_stopping'].get('patience', 'N/A')) + ', min_delta=' + str(config['early_stopping'].get('min_delta', 'N/A')) if isinstance(config.get('early_stopping'), dict) else str(config.get('early_stopping', 'Disabled'))}",
        f"- **Vocab Size:** {REAL_VOCAB_SIZE}",
        f"- **Gradient Clipping:** max_norm={config.get('grad_clip_max_norm', 1.0)}",
        f"- **Rule Prediction:** Multi-head prediction output (decoder_logits, rule_logits, verifier_logits)",
        "",
    ])

    with open(docs_dir / "TRAINING_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Training results written to docs/TRAINING_RESULTS.md")


def run_training_pipeline(from_scratch=False, override_epochs=None, override_max_steps=None):
    commit_hash = get_git_commit_hash()
    print(f"--- Training CalculusSolverModel (commit: {commit_hash}, vocab: {REAL_VOCAB_SIZE}) ---")

    if from_scratch and FINAL_CHECKPOINT_PATH.exists():
        FINAL_CHECKPOINT_PATH.unlink()
        print(f"Removed prior checkpoint {FINAL_CHECKPOINT_PATH} to retrain from scratch.")

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
    val_dataset = None
    if val_file.exists() and config.get("validation_logging", True):
        print("[DEBUG] Loading val dataset into memory...", flush=True)
        val_dataset = SlangDatasetLoader(val_file)
        print(f"[DEBUG] Val dataset loaded: {len(val_dataset)} examples", flush=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Hardware] Training device: {device} ({device_name})", flush=True)

    print("[DEBUG] Building model...", flush=True)
    model = CalculusSolverModel(
        vocab_size=REAL_VOCAB_SIZE,
        num_rules=len(RULE_LABELS),
        hidden_dim=config["hidden_dim"],
        rule_labels=RULE_LABELS,
    ).to(device)

    # Preflight: one dummy forward pass to confirm the (decoder_logits,
    # rule_logits, verifier_logits) return contract before any real training.
    # Interface mismatches used to surface only after multi-hour runs.
    check_forward_contract(model, vocab_size=REAL_VOCAB_SIZE, num_rules=len(RULE_LABELS))
    print("[Preflight] Model forward contract OK", flush=True)

    epochs = override_epochs if override_epochs is not None else config.get("epochs", 15)
    max_steps_cfg = override_max_steps if override_max_steps is not None else config.get("max_steps", 3500)

    base_lr = config["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    warmup_steps = config.get("warmup_steps", 1000)
    total_training_steps = epochs * min(max_steps_cfg, len(train_loader))
    lr_decay_cfg = config.get("lr_decay", {})
    min_lr_ratio = lr_decay_cfg.get("min_lr_ratio", 0.1)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = (current_step - warmup_steps) / max(total_training_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    grad_clip_max_norm = config.get("grad_clip_max_norm", 1.0)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_val_loss = float("inf")
    patience_counter = 0
    metrics_log = []

    early_stopping_cfg = config.get("early_stopping", False)
    if isinstance(early_stopping_cfg, dict):
        use_early_stopping = early_stopping_cfg.get("enabled", True)
        patience = early_stopping_cfg.get("patience", 3)
        min_delta = early_stopping_cfg.get("min_delta", 1e-4)
    elif isinstance(early_stopping_cfg, bool):
        use_early_stopping = early_stopping_cfg
        patience = 3 if early_stopping_cfg else None
        min_delta = 1e-4 if early_stopping_cfg else 0.0
    elif isinstance(early_stopping_cfg, int):
        patience = early_stopping_cfg
        min_delta = 1e-4
        use_early_stopping = True
    else:
        use_early_stopping = False
        patience = None
        min_delta = 0.0

    global_step = 0

    print("[DEBUG] Entering training loop...", flush=True)
    for epoch in range(1, epochs + 1):
        print(f"[DEBUG] Starting epoch {epoch}, waiting for first batch from DataLoader...", flush=True)
        model.train()
        epoch_loss = 0.0
        steps_run = 0

        ss_config = config.get("scheduled_sampling", {})
        ss_prob = get_scheduled_sampling_prob(epoch, epochs, ss_config)
        if ss_config.get("enabled"):
            print(f"  [Scheduled Sampling] Epoch {epoch}: model-token prob = {ss_prob:.3f}")

        for step, batch in enumerate(train_loader):
            if step >= max_steps_cfg:
                break
            if step < 3 or step % 10 == 0:
                print(f"[DEBUG] epoch {epoch} step {step} - batch received, running forward/backward...", flush=True)
            optimizer.zero_grad()

            src_seq = batch["src_seq"].to(device)
            tgt_in_full = batch["tgt_in_seq"].to(device)
            tgt_out = batch["tgt_out_seq"][:, 1:].to(device)
            rule_id = batch["rule_id"].to(device)

            tgt_in = tgt_in_full[:, :-1]
            
            if ss_prob > 0:
                with torch.no_grad():
                    pred_logits, _, _ = model(src_seq, tgt_in, true_rule_ids=rule_id)
                    pred_tokens = pred_logits.argmax(dim=-1)
                
                ss_mask = torch.rand_like(tgt_in, dtype=torch.float) < ss_prob
                bos_id = vocab_mapping["[BOS]"]
                eos_id = vocab_mapping["[EOS]"]
                pad_id = vocab_mapping["[PAD]"]
                special_mask = (tgt_in == bos_id) | (tgt_in == eos_id) | (tgt_in == pad_id)
                ss_mask = ss_mask & ~special_mask
                
                tgt_in = torch.where(ss_mask, pred_tokens, tgt_in)
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
            val_loss, val_seq_acc, val_token_acc, val_cat_acc = evaluate_validation(
                model, val_loader, criterion, device=device
            )
            
            num_proxy_examples = config.get("proxy_eval_examples", 15)
            fr_seq_acc, fr_token_acc, fr_avg_len = evaluate_free_running(
                model, val_dataset, num_examples=num_proxy_examples, max_gen_len=MAX_LEN, device=device
            )
            print(
                f"Epoch {epoch} - Val Loss: {val_loss:.4f} | "
                f"Token Acc: {val_token_acc:.4f} | Seq Acc: {val_seq_acc:.4f} | "
                f"Free-Run Seq Acc: {fr_seq_acc:.4f} | Free-Run Token Acc: {fr_token_acc:.4f} | "
                f"Avg Gen Len: {fr_avg_len:.1f}"
            )
            if val_cat_acc:
                cat_summary = " | ".join(f"{op}: {val_cat_acc[op]:.2%}" for op in sorted(val_cat_acc.keys()))
                print(f"  [Per-Category Val Seq Acc] {cat_summary}")
            
            train_val_gap = avg_train_loss - val_loss
            print(
                f"  [Diag] Train-Val Gap: {train_val_gap:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Patience Counter: {patience_counter}/{patience if use_early_stopping else 'Disabled'}"
            )

            epoch_metrics["val_loss"] = val_loss
            epoch_metrics["val_token_acc"] = val_token_acc
            epoch_metrics["val_seq_acc"] = val_seq_acc
            epoch_metrics["val_category_acc"] = val_cat_acc
            epoch_metrics["free_run_seq_acc"] = fr_seq_acc
            epoch_metrics["free_run_token_acc"] = fr_token_acc
            epoch_metrics["free_run_avg_len"] = fr_avg_len

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
                if use_early_stopping and patience is not None and patience_counter >= patience:
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
    import argparse
    parser = argparse.ArgumentParser(description="Train CalculusSolverModel")
    parser.add_argument("--from-scratch", action="store_true", help="Delete existing checkpoint before training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config.json")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps per epoch")
    args = parser.parse_args()
    run_training_pipeline(
        from_scratch=args.from_scratch,
        override_epochs=args.epochs,
        override_max_steps=args.max_steps,
    )