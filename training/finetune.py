import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.architecture import CalculusModel
from training.training_utils import SlangJsonlDataset, load_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 supervised fine-tuning for SLaNg model."
    )
    parser.add_argument(
        "--data", required=True, help="Path to train JSONL file or directory."
    )
    parser.add_argument(
        "--val", required=True, help="Path to validation JSONL file or directory."
    )
    parser.add_argument(
        "--vocab", default="tokenizer/vocab.json", help="Path to tokenizer vocab JSON."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Stage 1 checkpoint file to initialize the model.",
    )
    parser.add_argument(
        "--config",
        default="training/config/finetune.yaml",
        help="Fine-tuning config YAML.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/sft",
        help="Directory where stage 2 checkpoints are stored.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def save_checkpoint(
    model, optimizer, step: int, output_dir: Path, config: Dict[str, Any]
):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "best.pt"
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )
    return path


def evaluate(model, dataloader, device, criterion, pad_id):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_rule_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits, rule_logits, _ = model(
                src_tokens=batch["src_tokens"],
                src_positions=batch["src_positions"],
                parent_child_pairs=None,
                tgt_tokens=batch["tgt_input_tokens"],
                root_mask=None,
                rule_ids=batch["rule_ids"],
                src_padding_mask=batch["src_padding_mask"],
                tgt_padding_mask=batch["tgt_padding_mask"],
                memory_key_padding_mask=batch["src_padding_mask"],
            )
            token_loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch["tgt_output_tokens"].view(-1),
            )
            rule_loss = nn.CrossEntropyLoss()(rule_logits, batch["rule_ids"])
            total_loss += token_loss.item() * batch["tgt_output_tokens"].numel()
            total_rule_loss += rule_loss.item() * batch["rule_ids"].size(0)
            total_tokens += batch["tgt_output_tokens"].numel()
            total_steps += batch["rule_ids"].size(0)
    return total_loss / max(total_tokens, 1), total_rule_loss / max(total_steps, 1)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / args.config)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    train_dataset = SlangJsonlDataset(
        data_path=Path(args.data),
        vocab_path=Path(root / args.vocab),
        serializer_path=Path(root / "inference" / "serialize_input.js"),
        mode="finetune",
        max_len=model_cfg.get("max_len", 256),
    )
    val_dataset = SlangJsonlDataset(
        data_path=Path(args.val),
        vocab_path=Path(root / args.vocab),
        serializer_path=Path(root / "inference" / "serialize_input.js"),
        mode="finetune",
        max_len=model_cfg.get("max_len", 256),
    )

    device = torch.device(args.device)
    model = CalculusModel(
        vocab_size=len(train_dataset.token_to_id),
        rule_labels=train_dataset.rule_labels,
        hidden_dim=model_cfg.get("hidden_dim", 512),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 8),
        ffn_dim=model_cfg.get("ffn_dim", 2048),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    checkpoint_data = torch.load(Path(args.checkpoint), map_location="cpu")
    model.load_state_dict(checkpoint_data["model_state"])

    batch_size = training_cfg.get("batch_size", 64)
    learning_rate = training_cfg.get("lr", 1e-4)
    max_epochs = training_cfg.get("max_epochs", 5)
    rule_loss_weight = training_cfg.get("rule_loss_weight", 1.0)
    step_loss_weight = training_cfg.get("step_loss_weight", 0.5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_id)
    rule_criterion = nn.CrossEntropyLoss()

    step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            step += 1
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            logits, rule_logits, step_logits = model(
                src_tokens=batch["src_tokens"],
                src_positions=batch["src_positions"],
                parent_child_pairs=None,
                tgt_tokens=batch["tgt_input_tokens"],
                root_mask=None,
                rule_ids=batch["rule_ids"],
                src_padding_mask=batch["src_padding_mask"],
                tgt_padding_mask=batch["tgt_padding_mask"],
                memory_key_padding_mask=batch["src_padding_mask"],
            )
            token_loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch["tgt_output_tokens"].view(-1),
            )
            rule_loss = rule_criterion(rule_logits, batch["rule_ids"])
            step_loss = rule_criterion(step_logits, batch["rule_ids"])
            loss = (
                token_loss + rule_loss_weight * rule_loss + step_loss_weight * step_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % training_cfg.get("log_every_steps", 100) == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch}] step={step} loss={loss.item():.4f} "
                    f"token={token_loss.item():.4f} rule={rule_loss.item():.4f} "
                    f"step={step_loss.item():.4f} time={elapsed:.1f}s"
                )
                start_time = time.time()

        val_token_loss, val_rule_loss = evaluate(
            model, val_loader, device, criterion, train_dataset.pad_id
        )
        val_score = val_token_loss + val_rule_loss
        print(
            f"Validation after epoch {epoch}: token_loss={val_token_loss:.4f} "
            f"rule_loss={val_rule_loss:.4f} total={val_score:.4f}"
        )

        if val_score < best_val_loss:
            best_val_loss = val_score
            save_checkpoint(
                model,
                optimizer,
                step,
                Path(args.checkpoint_dir),
                {
                    "vocab_size": len(train_dataset.token_to_id),
                    "hidden_dim": model_cfg.get("hidden_dim", 512),
                    "num_heads": model_cfg.get("num_heads", 8),
                    "num_layers": model_cfg.get("num_layers", 8),
                    "ffn_dim": model_cfg.get("ffn_dim", 2048),
                    "dropout": model_cfg.get("dropout", 0.1),
                    "position_dim": model_cfg.get("position_dim", 3),
                },
            )
            print(f"Saved best finetune checkpoint to {args.checkpoint_dir}")

    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
import json, pathlib
metrics = {
    'val/numerical_equiv': float(best_num_equiv),
    'val/rule_accuracy': float(best_rule_acc),
    'val/step_accuracy': float(best_step_acc),
    'training_steps': int(global_step)
}
(pathlib.Path(output_dir) / 'metrics.json').write_text(json.dumps(metrics, indent=2))