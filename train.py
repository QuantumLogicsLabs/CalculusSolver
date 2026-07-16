import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tokenizer.slang_serializer import serialize_slang_math
from solver_model import CalculusSolverModel

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)


def flatten_vocab(raw_vocab):
    """
    Same flattening rule as inference/beam_search.flatten_vocab on org main:
    merge every sub-dict, skip keys starting with '_' (e.g. _comment, _version).
    Keeping this identical to beam_search's version on purpose, so training-time
    token IDs and inference-time token IDs can never drift apart again.
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
# len(vocab_mapping) undercounts; embedding table must cover the highest real ID.
REAL_VOCAB_SIZE = max(vocab_mapping.values()) + 1

# Rule labels for RuleHead, derived from vocab's rule_tokens, ordered by ID.
_rule_items = sorted(_raw_vocab.get("rule_tokens", {}).items(), key=lambda kv: kv[1])
RULE_LABELS = [name.split("RULE:", 1)[1] for name, _ in _rule_items]

MAX_LEN = config.get("max_len", 32)

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

    def _tokenize_ids(self, envelope, add_boundaries=False):
        """Tokenize into a raw list of vocab ids (no padding)."""
        # serialize_slang_math returns a single List[str] — no parent/child tuple.
        tokens = serialize_slang_math(envelope)
        if add_boundaries:
            tokens = ["[BOS]"] + tokens + ["[EOS]"]

        ids = []
        for t in tokens:
            if t in vocab_mapping:
                ids.append(vocab_mapping[t])
            else:
                raise KeyError(f"CRITICAL: Token '{t}' missing from vocab.json!")
        return ids

    def _pad(self, ids, length):
        pad_idx = vocab_mapping["[PAD]"]
        ids = ids[:length]
        ids = ids + [pad_idx] * (length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_ids = self._pad(
            self._tokenize_ids(item["src_tokens"], add_boundaries=False), self.max_len
        )

        # FIX (docs/KNOWN_ISSUES.md — "Decoder trained without target shift"):
        # tgt_input_tokens and tgt_output_tokens in data/slang_dataset.jsonl are
        # the SAME tree for every row, so tokenizing them separately (as before)
        # produced two identical sequences. Combined with the decoder's causal
        # mask (which still lets position i see its own position-i input), the
        # model could trivially copy tgt_in[i] -> tgt_out[i] without learning to
        # predict the next token from previous ones. Train/val loss dropped to
        # near zero, but real autoregressive eval (eval/run_eval.py) scored 0%.
        #
        # Fix: tokenize the target ONCE with boundaries, then shift by one:
        #   decoder input  = [BOS, t1, ..., t_{n-1}]  (drop the last token)
        #   decoder target = [t1, ..., t_n, EOS]       (drop the first token)
        # so at position i the decoder must predict the token that comes AFTER
        # what it has seen so far, matching how eval/run_eval.py actually
        # generates (no ground truth available at inference time).
        full_tgt = self._tokenize_ids(item["tgt_output_tokens"], add_boundaries=True)
        tgt_in_ids = self._pad(full_tgt[:-1], self.max_len)
        tgt_out_ids = self._pad(full_tgt[1:], self.max_len)

        return {
            "src_seq": src_ids,
            "tgt_in_seq": tgt_in_ids,
            "tgt_out_seq": tgt_out_ids,
            "rule_id": torch.tensor(item["rule_ids"], dtype=torch.long),
            "v_state": torch.tensor(item["verification_state"], dtype=torch.float),
        }


def evaluate_validation(model, val_loader, criterion_sequence, criterion_rule, criterion_verify):
    model.eval()
    total_val_loss = 0.0
    total_seq_loss = 0.0
    total_rule_loss = 0.0
    total_verify_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_size, seq_len = batch["src_seq"].shape
            decoder_logits, rule_logits, verifier_logits = model(
                batch["src_seq"],
                batch["tgt_in_seq"],
            )

            raw_loss_seq = criterion_sequence(
                decoder_logits.reshape(-1, REAL_VOCAB_SIZE), batch["tgt_out_seq"].reshape(-1)
            )
            raw_loss_seq = raw_loss_seq.view(batch_size, -1).mean(dim=-1)

            mask = (batch["v_state"] == 1.0).float()
            loss_seq = (raw_loss_seq * mask).sum() / (mask.sum() + 1e-8)

            loss_rule = criterion_rule(rule_logits, batch["rule_id"])
            loss_verify = criterion_verify(verifier_logits.squeeze(-1), batch["v_state"])

            total_loss = loss_seq + loss_rule + loss_verify
            
            total_val_loss += total_loss.item()
            total_seq_loss += loss_seq.item()
            total_rule_loss += loss_rule.item()
            total_verify_loss += loss_verify.item()
            steps += 1
            
    if steps == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_val_loss / steps,
        total_seq_loss / steps,
        total_rule_loss / steps,
        total_verify_loss / steps,
    )


def write_training_results(metrics_log, best_val_loss):
    """Write per-epoch metrics to docs/TRAINING_RESULTS.md."""
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
        "| Epoch | Train Loss | Val Loss | Val Seq | Val Rule | Val Verify | Checkpoint Saved |",
        "|-------|-----------|----------|---------|----------|------------|-----------------|",
    ]
    for m in metrics_log:
        val_loss = f"{m['val_loss']:.4f}" if m['val_loss'] is not None else "N/A"
        val_seq = f"{m['val_seq']:.4f}" if m['val_seq'] is not None else "N/A"
        val_rule = f"{m['val_rule']:.4f}" if m['val_rule'] is not None else "N/A"
        val_verify = f"{m['val_verify']:.4f}" if m['val_verify'] is not None else "N/A"
        saved = "Yes" if m['saved'] else "No"
        lines.append(
            f"| {m['epoch']} | {m['train_loss']:.4f} | {val_loss} | {val_seq} | {val_rule} | {val_verify} | {saved} |"
        )

    lines.extend([
        "",
        "## Configuration",
        "",
        f"- **Learning Rate:** {config.get('learning_rate')}",
        f"- **Batch Size:** {config.get('batch_size')}",
        f"- **Hidden Dim:** {config.get('hidden_dim')}",
        f"- **Max Steps/Epoch:** {config.get('max_steps')}",
        f"- **Early Stopping:** patience={config.get('early_stopping', {}).get('patience', 'N/A')}, min_delta={config.get('early_stopping', {}).get('min_delta', 'N/A')}",
        f"- **Vocab Size:** {REAL_VOCAB_SIZE}",
        f"- **Num Rules:** {len(RULE_LABELS)}",
        "",
    ])

    with open(docs_dir / "TRAINING_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Training results written to docs/TRAINING_RESULTS.md")


def run_training_pipeline():
    print(f"--- Training (vocab size: {REAL_VOCAB_SIZE}, {len(RULE_LABELS)} rules) ---")

    train_file = Path("data/splits/train.jsonl")
    if not train_file.exists():
        print("Train split missing!")
        sys.exit(1)

    train_loader = DataLoader(SlangDatasetLoader(train_file), batch_size=config["batch_size"], shuffle=True)

    val_file = Path("data/splits/val.jsonl")
    val_loader = None
    if val_file.exists() and config.get("validation_logging", True):
        val_loader = DataLoader(SlangDatasetLoader(val_file), batch_size=config["batch_size"], shuffle=False)

    model = CalculusSolverModel(
        vocab_size=REAL_VOCAB_SIZE,
        num_rules=len(RULE_LABELS),
        hidden_dim=config["hidden_dim"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    criterion_sequence = nn.CrossEntropyLoss(reduction='none')
    criterion_rule = nn.CrossEntropyLoss()
    criterion_verify = nn.BCEWithLogitsLoss()

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

    # Resume logic
    if FINAL_CHECKPOINT_PATH.exists():
        try:
            model.load_state_dict(torch.load(str(FINAL_CHECKPOINT_PATH), map_location="cpu"))
            print(f"Loaded existing checkpoint from {FINAL_CHECKPOINT_PATH} to resume training.")
            if val_loader is not None:
                val_loss, val_seq, val_rule, val_verify = evaluate_validation(
                    model, val_loader, criterion_sequence, criterion_rule, criterion_verify
                )
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

            batch_size, seq_len = batch["src_seq"].shape

            decoder_logits, rule_logits, verifier_logits = model(
                batch["src_seq"],
                batch["tgt_in_seq"],
            )

            raw_loss_seq = criterion_sequence(
                decoder_logits.reshape(-1, REAL_VOCAB_SIZE), batch["tgt_out_seq"].reshape(-1)
            )
            raw_loss_seq = raw_loss_seq.view(batch_size, -1).mean(dim=-1)

            mask = (batch["v_state"] == 1.0).float()
            loss_seq = (raw_loss_seq * mask).sum() / (mask.sum() + 1e-8)

            loss_rule = criterion_rule(rule_logits, batch["rule_id"])
            loss_verify = criterion_verify(verifier_logits.squeeze(-1), batch["v_state"])

            total_loss = loss_seq + loss_rule + loss_verify
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            steps_run += 1

        avg_train_loss = epoch_loss / max(steps_run, 1)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # ── Validation + best-checkpoint logic ────────────────────────────────
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": None,
            "val_seq": None,
            "val_rule": None,
            "val_verify": None,
            "saved": False,
        }

        if val_loader is not None:
            val_loss, val_seq, val_rule, val_verify = evaluate_validation(
                model, val_loader, criterion_sequence, criterion_rule, criterion_verify
            )
            print(f"Epoch {epoch} - Val Loss: {val_loss:.4f} (Seq: {val_seq:.4f}, Rule: {val_rule:.4f}, Verify: {val_verify:.4f})")
            
            epoch_metrics["val_loss"] = val_loss
            epoch_metrics["val_seq"] = val_seq
            epoch_metrics["val_rule"] = val_rule
            epoch_metrics["val_verify"] = val_verify

            # Best-checkpoint logic: only save when val loss improves
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
            # No validation set — save every epoch
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(FINAL_CHECKPOINT_PATH))
            print(f"Checkpoint saved to {FINAL_CHECKPOINT_PATH}")
            epoch_metrics["saved"] = True

        metrics_log.append(epoch_metrics)

    # ── Write training results ────────────────────────────────────────────────
    write_training_results(metrics_log, best_val_loss)
    print("--- Training complete ---")


if __name__ == "__main__":
    run_training_pipeline()