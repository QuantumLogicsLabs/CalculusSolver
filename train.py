import sys
import os
import copy
import json
import subprocess
import torch
import math
import torch.nn as nn
from datetime import datetime, timezone
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tokenizer.slang_serializer import serialize_slang_math
from solver_model import CalculusSolverModel, check_forward_contract
from checkpoint_provenance import stamp_checkpoint, append_provenance_to_training_results
from training.contrast import ContrastBatchSource, merge_batches
from training.metrics import contrast_pair_accuracy, evaluate_exact_match
from training.reproducibility import build_run_manifest, set_global_seed, write_json
from training.schedule import (
    build_lr_lambda,
    resolve_schedule,
    scheduler_matches_expected,
    verify_schedule,
)
from training.token_weights import (
    TokenWeightedCrossEntropy,
    build_token_weight_vector,
    per_token_nll,
)

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
        self.gradient_var_counts = []
        self.ops = []
        self.contrast_groups = []
        self.max_len = max_len
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
                op = (
                    item["src_tokens"].get("op", "unknown")
                    if isinstance(item.get("src_tokens"), dict)
                    else "unknown"
                )
                self.ops.append(op)
                self.contrast_groups.append(item.get("contrast_group"))
                if op == "gradient":
                    tgt = item.get("tgt_output_tokens", {})
                    grad = tgt.get("gradient", {}) if isinstance(tgt, dict) else {}
                    n_vars = len(grad.keys()) if isinstance(grad, dict) else 1
                else:
                    n_vars = 0
                self.gradient_var_counts.append(n_vars)

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


def evaluate_validation(model, val_loader, criterion, device="cpu", return_details=False, weighted_criterion=None):
    """Teacher-forced validation pass.

    `criterion` should be the UNWEIGHTED loss so val loss stays comparable
    across runs that train with different objectives. With return_details=True
    a 5th element is returned: per-operation token-mean loss and (if
    `weighted_criterion` is given) the token-weighted validation loss.
    """
    model.eval()
    total_loss = 0.0
    total_weighted_loss = 0.0
    total_correct_seq = 0
    total_correct_tokens = 0
    total_valid_tokens = 0
    total_seq = 0
    steps = 0

    category_correct = {}
    category_total = {}
    op_nll_sum = {}
    op_token_count = {}

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
            if weighted_criterion is not None:
                total_weighted_loss += weighted_criterion(decoder_logits, tgt_out).item()

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

                if return_details:
                    nll, valid = per_token_nll(decoder_logits, tgt_out, PAD_ID)
                    per_example_nll = nll.view(tgt_out.shape).sum(dim=1).tolist()
                    per_example_tokens = valid.view(tgt_out.shape).sum(dim=1).tolist()
                    for i, op in enumerate(ops):
                        op_nll_sum[op] = op_nll_sum.get(op, 0.0) + per_example_nll[i]
                        op_token_count[op] = op_token_count.get(op, 0) + per_example_tokens[i]

            steps += 1

    if steps == 0:
        empty = (0.0, 0.0, 0.0, {})
        return empty + ({},) if return_details else empty

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

    if not return_details:
        return avg_loss, seq_acc, token_acc, category_acc

    details = {
        "loss_by_op": {
            op: op_nll_sum[op] / op_token_count[op]
            for op in sorted(op_nll_sum)
            if op_token_count.get(op, 0) > 0
        },
        "weighted_loss": (total_weighted_loss / steps) if weighted_criterion is not None else None,
    }
    return avg_loss, seq_acc, token_acc, category_acc, details


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
    if not ss_config.get("enabled", False):
        return 0.0
    
    start = ss_config.get("start_prob", 0.0)
    end = ss_config.get("end_prob", 0.7)
    schedule = ss_config.get("schedule", "linear")
    ramp_epochs = ss_config.get("ramp_epochs", min(11, total_epochs))
    
    progress = min(1.0, max(0.0, (epoch - 1) / max(ramp_epochs - 1, 1)))
    
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


def get_curriculum_weights(dataset, epoch, curriculum_cfg):
    if not curriculum_cfg.get("enabled", False):
        return None
    start_w = curriculum_cfg.get("gradient_3var_start_weight", 0.1)
    ramp_ep = curriculum_cfg.get("gradient_3var_ramp_epochs", 5)
    progress = min(1.0, max(0.0, (epoch - 1) / max(ramp_ep - 1, 1)))
    if progress >= 1.0:
        return None
    w_3var = start_w + (1.0 - start_w) * progress
    weights = [
        w_3var if n_vars >= 3 else 1.0
        for n_vars in dataset.gradient_var_counts
    ]
    return torch.tensor(weights, dtype=torch.double)


def apply_overrides(cfg, overrides):
    """Apply ``{"a.b.c": value}`` overrides onto a nested config dict in place."""
    for dotted, value in (overrides or {}).items():
        parts = dotted.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override '{dotted}': '{part}' is not a section")
        node[parts[-1]] = value
    return cfg


def parse_set_args(items):
    """Parse repeated ``--set key.path=value`` CLI items (values are JSON when possible)."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        key, raw = item.split("=", 1)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def write_training_results(
    metrics_log,
    best_val_loss,
    git_commit_hash,
    cfg=None,
    results_path=None,
    final_eval=None,
):
    cfg = cfg if cfg is not None else config
    results_path = Path(results_path) if results_path else Path("docs") / "TRAINING_RESULTS.md"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Training Results",
        "",
        f"**Git Commit Hash:** `{git_commit_hash}`",
        f"**Best Validation Loss:** {best_val_loss:.4f}" if best_val_loss < float("inf") else "**Best Validation Loss:** N/A",
        f"**Total Epochs Run:** {len(metrics_log)}",
        "",
        "Validation loss is always the *unweighted* token cross-entropy so it stays comparable between "
        "runs. Train loss is the training objective (token-weighted when enabled). Free-Run columns are "
        "greedy whole-sequence exact match, not teacher-forced token accuracy.",
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

    has_em = any(m.get("val_exact_match_by_op") for m in metrics_log)
    if has_em:
        em_ops = sorted({op for m in metrics_log for op in m.get("val_exact_match_by_op", {})})
        lines.extend([
            "",
            "## Validation Exact Match by Operation (greedy, whole sequence)",
            "",
            "| Epoch | Overall | " + " | ".join(o.capitalize() for o in em_ops) + " | Contrast Pair Acc |",
            "|-------|---------|" + "|".join(["---"] * len(em_ops)) + "|-------------------|",
        ])
        for m in metrics_log:
            by_op = m.get("val_exact_match_by_op", {})
            cells = " | ".join(f"{by_op[o]:.4f}" if o in by_op else "N/A" for o in em_ops)
            cp = m.get("contrast_pair_acc")
            cp_txt = f"{cp:.4f}" if cp is not None else "N/A"
            lines.append(f"| {m['epoch']} | {m.get('val_exact_match', 0.0):.4f} | {cells} | {cp_txt} |")

    has_loss_op = any(m.get("val_loss_by_op") for m in metrics_log)
    if has_loss_op:
        lo_ops = sorted({op for m in metrics_log for op in m.get("val_loss_by_op", {})})
        lines.extend([
            "",
            "## Validation Loss by Operation (token-mean, unweighted)",
            "",
            "| Epoch | " + " | ".join(o.capitalize() for o in lo_ops) + " |",
            "|-------|" + "|".join(["---"] * len(lo_ops)) + "|",
        ])
        for m in metrics_log:
            by_op = m.get("val_loss_by_op", {})
            cells = " | ".join(f"{by_op[o]:.4f}" if o in by_op else "N/A" for o in lo_ops)
            lines.append(f"| {m['epoch']} | {cells} |")

    if final_eval:
        em = final_eval["exact_match"]
        lines.extend([
            "",
            "## Final Evaluation of the Saved Checkpoint",
            "",
            f"Best epoch: {final_eval.get('best_epoch')} | Validation rows scored: {em['overall']['total']} "
            f"| Val loss: {final_eval['val_loss']:.4f}",
            "",
            "| Operation | Rows | Exact match | Val loss (token-mean) |",
            "|---|---:|---:|---:|",
        ])
        for op, row in em["by_op"].items():
            op_loss = final_eval.get("val_loss_by_op", {}).get(op)
            op_loss_txt = f"{op_loss:.4f}" if op_loss is not None else "N/A"
            lines.append(f"| {op} | {row['total']} | {row['correct']}/{row['total']} ({row['accuracy']:.1%}) | {op_loss_txt} |")
        lines.append(
            f"| **Overall** | **{em['overall']['total']}** | "
            f"**{em['overall']['correct']}/{em['overall']['total']} ({em['overall']['accuracy']:.1%})** | "
            f"**{final_eval['val_loss']:.4f}** |"
        )
        cp = final_eval.get("contrast")
        if cp:
            lines.extend([
                "",
                f"Contrast validation: {cp['pairs_both_correct']}/{cp['pairs']} pairs with both members exact "
                f"({cp['pair_accuracy']:.1%}); row accuracy {cp['row_accuracy']:.1%}.",
            ])

    tw = cfg.get("token_weighted_loss", {}) or {}
    contrast = cfg.get("contrast", {}) or {}
    lr_decay = cfg.get("lr_decay", {}) or {}
    lines.extend([
        "",
        "## Configuration Snapshot",
        "",
        f"- **Learning Rate:** {cfg.get('learning_rate')}",
        f"- **Epochs:** {cfg.get('epochs')}",
        f"- **Warmup Steps:** {cfg.get('warmup_steps', 1000)}",
        f"- **LR Decay:** {lr_decay.get('type', 'cosine')} (min_lr_ratio={lr_decay.get('min_lr_ratio')})",
        f"- **Seed:** {cfg.get('seed')}",
        f"- **Batch Size:** {cfg.get('batch_size')}",
        f"- **Hidden Dim:** {cfg.get('hidden_dim')}",
        f"- **Max Len:** {cfg.get('max_len', MAX_LEN)}",
        f"- **Max Steps/Epoch:** {cfg.get('max_steps')}",
        f"- **Early Stopping:** {'patience=' + str(cfg['early_stopping'].get('patience', 'N/A')) + ', min_delta=' + str(cfg['early_stopping'].get('min_delta', 'N/A')) if isinstance(cfg.get('early_stopping'), dict) else str(cfg.get('early_stopping', 'Disabled'))}",
        f"- **Vocab Size:** {REAL_VOCAB_SIZE}",
        f"- **Gradient Clipping:** max_norm={cfg.get('grad_clip_max_norm', 1.0)}",
        f"- **Token-Weighted Loss:** {'enabled ' + json.dumps(tw.get('weights')) if tw.get('enabled') else 'disabled'}",
        f"- **Contrast Minibatches:** {'enabled (pairs_per_batch=' + str(contrast.get('pairs_per_batch')) + ')' if contrast.get('enabled') else 'disabled'}",
        f"- **Rule Prediction:** Multi-head prediction output (decoder_logits, rule_logits, verifier_logits)",
        "",
    ])

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Training results written to {results_path.as_posix()}")


def run_training_pipeline(
    from_scratch=False,
    override_epochs=None,
    override_max_steps=None,
    run_name=None,
    output_dir=None,
    config_overrides=None,
    seed=None,
):
    """Train and save the best checkpoint.

    Without `output_dir` this behaves as before (checkpoint in
    checkpoints/final/, report in docs/TRAINING_RESULTS.md). With `output_dir`
    everything the run produces -- best.pt, TRAINING_RESULTS.md, metrics.json,
    run_manifest.json, config_used.json -- stays inside that directory, so
    controlled experiments never touch the published checkpoint.
    """
    cfg = copy.deepcopy(config)
    apply_overrides(cfg, config_overrides)
    run_name = run_name or "default"
    seed = int(seed if seed is not None else cfg.get("seed", 42))
    seed_info = set_global_seed(seed, deterministic=cfg.get("deterministic", True))
    cfg["seed"] = seed

    commit_hash = get_git_commit_hash()
    print(f"--- Training CalculusSolverModel (run: {run_name}, seed: {seed}, commit: {commit_hash}, vocab: {REAL_VOCAB_SIZE}) ---")

    out_dir = Path(output_dir) if output_dir else CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    results_path = (out_dir / "TRAINING_RESULTS.md") if output_dir else Path("docs") / "TRAINING_RESULTS.md"
    metrics_path = out_dir / "metrics.json"
    manifest_path = out_dir / "run_manifest.json"
    config_used_path = out_dir / "config_used.json"

    if from_scratch:
        print("Retraining from scratch: initializing fresh model weights.")

    train_file = Path("data/splits/train.jsonl")
    if not train_file.exists():
        print("CRITICAL: Train split missing! Run problem_generator.py first.")
        sys.exit(1)

    preflight_check_max_len(train_file, MAX_LEN)

    print("[DEBUG] Loading train dataset into memory...", flush=True)
    train_dataset = SlangDatasetLoader(train_file)
    print(f"[DEBUG] Train dataset loaded: {len(train_dataset)} examples", flush=True)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True, generator=loader_generator
    )

    val_file = Path("data/splits/val.jsonl")
    val_loader = None
    val_dataset = None
    if val_file.exists() and cfg.get("validation_logging", True):
        print("[DEBUG] Loading val dataset into memory...", flush=True)
        val_dataset = SlangDatasetLoader(val_file)
        print(f"[DEBUG] Val dataset loaded: {len(val_dataset)} examples", flush=True)
        val_limit = cfg.get("validation_max_examples")
        val_subset = (
            torch.utils.data.Subset(val_dataset, range(min(val_limit, len(val_dataset))))
            if val_limit
            else val_dataset
        )
        val_loader = DataLoader(val_subset, batch_size=cfg["batch_size"], shuffle=False)

    contrast_cfg = cfg.get("contrast", {}) or {}
    contrast_train_file = Path("data/splits/contrast_train.jsonl")
    contrast_val_file = Path("data/splits/contrast_val.jsonl")
    contrast_source = None
    if contrast_cfg.get("enabled", False):
        if not contrast_train_file.exists():
            print("CRITICAL: contrast.enabled but data/splits/contrast_train.jsonl is missing. "
                  "Run: python problem_generator.py --contrast-only")
            sys.exit(1)
        contrast_train_dataset = SlangDatasetLoader(contrast_train_file)
        contrast_source = ContrastBatchSource(
            contrast_train_dataset, int(contrast_cfg.get("pairs_per_batch", 4)), seed
        )
        print(f"[Contrast] {len(contrast_source.pairs)} train pairs; "
              f"{contrast_source.pairs_per_batch} pairs added to every minibatch", flush=True)
    contrast_val_dataset = SlangDatasetLoader(contrast_val_file) if contrast_val_file.exists() else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Hardware] Training device: {device} ({device_name})", flush=True)

    print("[DEBUG] Building model...", flush=True)
    torch.manual_seed(seed)  # identical initial weights for every experiment arm
    model = CalculusSolverModel(
        vocab_size=REAL_VOCAB_SIZE,
        num_rules=len(RULE_LABELS),
        hidden_dim=cfg["hidden_dim"],
        rule_labels=RULE_LABELS,
    ).to(device)

    # Preflight: one dummy forward pass to confirm the (decoder_logits,
    # rule_logits, verifier_logits) return contract before any real training.
    # Interface mismatches used to surface only after multi-hour runs.
    check_forward_contract(model, vocab_size=REAL_VOCAB_SIZE, num_rules=len(RULE_LABELS))
    print("[Preflight] Model forward contract OK", flush=True)

    epochs = override_epochs if override_epochs is not None else cfg.get("epochs", 15)
    max_steps_cfg = override_max_steps if override_max_steps is not None else cfg.get("max_steps", 3500)
    cfg["epochs"] = epochs
    cfg["max_steps"] = max_steps_cfg

    base_lr = cfg["learning_rate"]
    weight_decay = cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    steps_per_epoch = min(max_steps_cfg, len(train_loader))
    schedule = resolve_schedule(cfg, steps_per_epoch, epochs)
    schedule_check = verify_schedule(schedule)
    for warning in schedule["warnings"]:
        print(f"[Preflight] WARNING: {warning}", flush=True)
    print(
        f"[Preflight] LR schedule verified: {schedule['type']} | warmup={schedule['warmup_steps']} steps | "
        f"total={schedule['total_steps']} steps ({epochs} epochs x {steps_per_epoch}) | "
        f"peak={base_lr:g} -> floor={base_lr * schedule['min_lr_ratio']:g} "
        f"(min_lr_ratio={schedule['min_lr_ratio']})",
        flush=True,
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(
            schedule["warmup_steps"], schedule["total_steps"], schedule["min_lr_ratio"]
        ),
    )

    grad_clip_max_norm = cfg.get("grad_clip_max_norm", 1.0)

    tw_cfg = cfg.get("token_weighted_loss", {}) or {}
    weight_vector, weight_report = build_token_weight_vector(
        vocab_mapping, REAL_VOCAB_SIZE, tw_cfg.get("weights")
    )
    use_weighted_loss = bool(tw_cfg.get("enabled", False))
    train_criterion = TokenWeightedCrossEntropy(
        REAL_VOCAB_SIZE, PAD_ID, weight_vector if use_weighted_loss else None
    ).to(device)
    val_criterion = TokenWeightedCrossEntropy(REAL_VOCAB_SIZE, PAD_ID).to(device)
    val_weighted_criterion = TokenWeightedCrossEntropy(REAL_VOCAB_SIZE, PAD_ID, weight_vector).to(device)
    print(
        f"[Loss] token-weighted={'ON' if use_weighted_loss else 'off'} "
        f"class weights={weight_report['class_weights']} "
        f"tokens per class={weight_report['token_counts']}",
        flush=True,
    )

    write_json(config_used_path, cfg)
    manifest = build_run_manifest(
        run_name=run_name,
        cfg=cfg,
        seed_info=seed_info,
        dataset_files=[train_file, val_file, contrast_train_file, contrast_val_file],
        vocab_path="tokenizer/vocab.json",
        serializer_path="tokenizer/slang_serializer.py",
        vocab_size=REAL_VOCAB_SIZE,
        checkpoint_path=str(ckpt_path),
        optimizer_info={
            "name": "AdamW",
            "learning_rate": base_lr,
            "weight_decay": weight_decay,
            "betas": list(optimizer.defaults["betas"]),
            "eps": optimizer.defaults["eps"],
            "grad_clip_max_norm": grad_clip_max_norm,
        },
        scheduler_info={**schedule, "preflight": schedule_check},
        loss_info={
            "token_weighted": use_weighted_loss,
            "token_weight_report": weight_report,
            "contrast_minibatches": bool(contrast_source),
            "contrast_pairs_per_batch": contrast_cfg.get("pairs_per_batch") if contrast_source else None,
        },
        extra={
            "loader_seed": seed,
            "max_len": MAX_LEN,
            "device": str(device),
            "config_used_path": config_used_path.as_posix(),
        },
    )
    write_json(manifest_path, manifest)

    best_val_loss = float("inf")
    best_epoch = None
    patience_counter = 0
    metrics_log = []
    nonfinite_steps = 0
    max_nonfinite_steps = int(cfg.get("max_nonfinite_loss_steps", 5))

    early_stopping_cfg = cfg.get("early_stopping", False)
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

    curriculum_cfg = cfg.get("curriculum", {})
    em_cfg = cfg.get("exact_match_eval", {}) or {}
    bos_id = vocab_mapping["[BOS]"]
    eos_id = vocab_mapping["[EOS]"]

    print("[DEBUG] Entering training loop...", flush=True)
    for epoch in range(1, epochs + 1):
        print(f"[DEBUG] Starting epoch {epoch}, waiting for first batch from DataLoader...", flush=True)
        model.train()
        epoch_loss = 0.0
        steps_run = 0

        curr_weights = get_curriculum_weights(train_dataset, epoch, curriculum_cfg)
        if curr_weights is not None:
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(seed * 1_000 + epoch)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=curr_weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=sampler_generator,
            )
            epoch_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler)
        else:
            epoch_loader = train_loader

        if contrast_source is not None:
            contrast_source.start_epoch(epoch)

        ss_config = cfg.get("scheduled_sampling", {})
        ss_prob = get_scheduled_sampling_prob(epoch, epochs, ss_config)
        if ss_config.get("enabled"):
            print(f"  [Scheduled Sampling] Epoch {epoch}: model-token prob = {ss_prob:.3f}")

        for step, batch in enumerate(epoch_loader):
            if step >= max_steps_cfg:
                break
            if step < 3 or step % 10 == 0:
                print(f"[DEBUG] epoch {epoch} step {step} - batch received, running forward/backward...", flush=True)
            optimizer.zero_grad()

            if contrast_source is not None:
                batch = merge_batches(batch, contrast_source.next_batch())

            src_seq = batch["src_seq"].to(device)
            tgt_in_full = batch["tgt_in_seq"].to(device)
            tgt_out = batch["tgt_out_seq"][:, 1:].to(device)
            rule_id = batch["rule_id"].to(device)

            tgt_in = tgt_in_full[:, :-1]
            
            if ss_prob > 0:
                with torch.no_grad():
                    pred_logits, _, _ = model(src_seq, tgt_in, true_rule_ids=rule_id)
                    pred_tokens = pred_logits.argmax(dim=-1)
                    shifted_pred = torch.cat([tgt_in[:, :1], pred_tokens[:, :-1]], dim=1)
                
                ss_mask = torch.rand_like(tgt_in, dtype=torch.float) < ss_prob
                bos_id = vocab_mapping["[BOS]"]
                eos_id = vocab_mapping["[EOS]"]
                pad_id = vocab_mapping["[PAD]"]
                special_mask = (tgt_in == bos_id) | (tgt_in == eos_id) | (tgt_in == pad_id)
                ss_mask = ss_mask & ~special_mask
                
                tgt_in = torch.where(ss_mask, shifted_pred, tgt_in)
            decoder_logits, rule_logits, verifier_logits = model(src_seq, tgt_in, true_rule_ids=rule_id)
            loss = train_criterion(decoder_logits.reshape(-1, REAL_VOCAB_SIZE), tgt_out.reshape(-1))

            if not torch.isfinite(loss):
                nonfinite_steps += 1
                print(f"[WARN] Non-finite loss at epoch {epoch} step {step}; update skipped "
                      f"({nonfinite_steps}/{max_nonfinite_steps} allowed)", flush=True)
                if nonfinite_steps > max_nonfinite_steps:
                    raise FloatingPointError(
                        f"Aborting: {nonfinite_steps} non-finite loss values (limit {max_nonfinite_steps})"
                    )
                continue
            
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
            val_loss, val_seq_acc, val_token_acc, val_cat_acc, val_details = evaluate_validation(
                model, val_loader, val_criterion, device=device,
                return_details=True, weighted_criterion=val_weighted_criterion,
            )

            em = evaluate_exact_match(
                model, val_dataset,
                bos_id=bos_id, eos_id=eos_id, pad_id=PAD_ID, device=device,
                max_gen_len=MAX_LEN,
                batch_size=em_cfg.get("batch_size", 128),
                max_examples=em_cfg.get("per_epoch_examples", 600),
            )
            fr_seq_acc = em["overall"]["accuracy"]
            fr_token_acc = em["token_accuracy"]
            fr_avg_len = em["avg_gen_len"]

            contrast_metrics = None
            if contrast_val_dataset is not None:
                contrast_result = evaluate_exact_match(
                    model, contrast_val_dataset,
                    bos_id=bos_id, eos_id=eos_id, pad_id=PAD_ID, device=device,
                    max_gen_len=MAX_LEN, batch_size=em_cfg.get("batch_size", 128),
                    max_examples=cfg.get("contrast_val_max_examples"),
                )
                contrast_metrics = contrast_pair_accuracy(contrast_result, contrast_val_dataset.contrast_groups)

            print(
                f"Epoch {epoch} - Val Loss: {val_loss:.4f} | "
                f"Token Acc: {val_token_acc:.4f} | Seq Acc: {val_seq_acc:.4f} | "
                f"Free-Run Seq Acc: {fr_seq_acc:.4f} | Free-Run Token Acc: {fr_token_acc:.4f} | "
                f"Avg Gen Len: {fr_avg_len:.1f}"
            )
            if val_cat_acc:
                cat_summary = " | ".join(f"{op}: {val_cat_acc[op]:.2%}" for op in sorted(val_cat_acc.keys()))
                print(f"  [Per-Category Val Seq Acc] {cat_summary}")
            em_summary = " | ".join(f"{op}: {row['accuracy']:.2%}" for op, row in em["by_op"].items())
            print(f"  [Exact Match by Operation, {em['overall']['total']} rows] {em_summary}")
            if contrast_metrics is not None:
                print(f"  [Contrast Pairs] both-correct {contrast_metrics['pairs_both_correct']}/"
                      f"{contrast_metrics['pairs']} ({contrast_metrics['pair_accuracy']:.2%})")
            
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
            epoch_metrics["val_loss_by_op"] = val_details["loss_by_op"]
            epoch_metrics["val_weighted_loss"] = val_details["weighted_loss"]
            epoch_metrics["free_run_seq_acc"] = fr_seq_acc
            epoch_metrics["free_run_token_acc"] = fr_token_acc
            epoch_metrics["free_run_avg_len"] = fr_avg_len
            epoch_metrics["val_exact_match"] = fr_seq_acc
            epoch_metrics["val_exact_match_rows"] = em["overall"]["total"]
            epoch_metrics["val_exact_match_by_op"] = {op: row["accuracy"] for op, row in em["by_op"].items()}
            epoch_metrics["contrast_pair_acc"] = contrast_metrics["pair_accuracy"] if contrast_metrics else None
            epoch_metrics["contrast_row_acc"] = contrast_metrics["row_accuracy"] if contrast_metrics else None
            epoch_metrics["learning_rate"] = current_lr

            if val_loss < best_val_loss - (min_delta if use_early_stopping else 0):
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(ckpt_path))
                print(f"  New best validation loss! Saved checkpoint to {ckpt_path}")
                epoch_metrics["saved"] = True
            else:
                patience_counter += 1
                print(f"  Epoch {epoch}: val loss {val_loss:.4f} did not improve from {best_val_loss:.4f}.")
                if use_early_stopping and patience is not None and patience_counter >= patience:
                    print("Early stopping triggered. Training stopped.")
                    metrics_log.append(epoch_metrics)
                    break
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(ckpt_path))
            print(f"Checkpoint saved to {ckpt_path}")
            epoch_metrics["saved"] = True
            best_epoch = epoch

        metrics_log.append(epoch_metrics)

    final_eval = None
    if val_loader is not None and ckpt_path.exists() and em_cfg.get("final_full_eval", True):
        print("[Final Eval] Scoring the saved best checkpoint on the validation split...", flush=True)
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        f_loss, f_seq, f_tok, _, f_details = evaluate_validation(
            model, val_loader, val_criterion, device=device,
            return_details=True, weighted_criterion=val_weighted_criterion,
        )
        f_em = evaluate_exact_match(
            model, val_dataset,
            bos_id=bos_id, eos_id=eos_id, pad_id=PAD_ID, device=device,
            max_gen_len=MAX_LEN, batch_size=em_cfg.get("batch_size", 128),
            max_examples=em_cfg.get("final_max_examples"),
        )
        f_contrast = None
        if contrast_val_dataset is not None:
            f_cres = evaluate_exact_match(
                model, contrast_val_dataset,
                bos_id=bos_id, eos_id=eos_id, pad_id=PAD_ID, device=device,
                max_gen_len=MAX_LEN, batch_size=em_cfg.get("batch_size", 128),
            )
            f_contrast = contrast_pair_accuracy(f_cres, contrast_val_dataset.contrast_groups)
        final_eval = {
            "best_epoch": best_epoch,
            "val_loss": f_loss,
            "val_weighted_loss": f_details["weighted_loss"],
            "val_loss_by_op": f_details["loss_by_op"],
            "teacher_forced_seq_acc": f_seq,
            "token_acc": f_tok,
            "exact_match": {k: v for k, v in f_em.items() if k != "correct_flags"},
            "contrast": f_contrast,
        }
        print(f"[Final Eval] exact match {f_em['overall']['correct']}/{f_em['overall']['total']} "
              f"({f_em['overall']['accuracy']:.2%}), val loss {f_loss:.4f}", flush=True)

    write_training_results(
        metrics_log, best_val_loss, commit_hash,
        cfg=cfg, results_path=results_path, final_eval=final_eval,
    )

    write_json(metrics_path, {
        "run_name": run_name,
        "seed": seed,
        "epochs_configured": epochs,
        "epochs_run": len(metrics_log),
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "best_epoch": best_epoch,
        "nonfinite_loss_steps": nonfinite_steps,
        "per_epoch": metrics_log,
        "final_eval": final_eval,
    })

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["scheduler"]["final_lr_matches_formula"] = scheduler_matches_expected(
        scheduler, schedule, global_step
    )
    manifest["scheduler"]["optimizer_steps_taken"] = global_step
    manifest["results"] = {
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "best_epoch": best_epoch,
        "epochs_run": len(metrics_log),
        "nonfinite_loss_steps": nonfinite_steps,
        "final_val_exact_match": final_eval["exact_match"]["overall"]["accuracy"] if final_eval else None,
    }
    if ckpt_path.exists():
        provenance = stamp_checkpoint(str(ckpt_path), config_path=str(config_used_path))
        manifest["checkpoint"]["sha256"] = provenance["checkpoint_sha256"]
        manifest["checkpoint"]["provenance"] = provenance
        append_provenance_to_training_results(provenance, str(results_path))
    write_json(manifest_path, manifest)
    print(f"Run manifest written to {manifest_path.as_posix()}")
    print("--- Training complete ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CalculusSolverModel")
    parser.add_argument("--from-scratch", action="store_true", help="Delete existing checkpoint before training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config.json")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps per epoch")
    parser.add_argument("--run-name", default=None, help="Label recorded in the run manifest")
    parser.add_argument("--output-dir", default=None,
                        help="Write checkpoint/report/manifest here instead of checkpoints/final and docs/")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set token_weighted_loss.enabled=true (repeatable)")
    args = parser.parse_args()
    run_training_pipeline(
        from_scratch=args.from_scratch,
        override_epochs=args.epochs,
        override_max_steps=args.max_steps,
        run_name=args.run_name,
        output_dir=args.output_dir,
        config_overrides=parse_set_args(args.set),
        seed=args.seed,
    )