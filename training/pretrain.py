import argparse
import math
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
        description="Stage 1 pretraining for SLaNg reconstruction."
    )
    parser.add_argument(
        "--data", required=True, help="Path to train JSONL file or directory."
    )
    parser.add_argument(
        "--vocab", default="tokenizer/vocab.json", help="Path to tokenizer vocab JSON."
    )
    parser.add_argument(
        "--config",
        default="training/config/pretrain.yaml",
        help="Training config YAML.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/pretrain",
        help="Directory where stage 1 checkpoints are stored.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def make_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, float(step) / float(max(1, warmup_steps)))
        return max(
            1e-8, float(max_steps - step) / float(max(1, max_steps - warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / args.config)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    dataset = SlangJsonlDataset(
        data_path=Path(args.data),
        vocab_path=Path(root / args.vocab),
        serializer_path=Path(root / "inference" / "serialize_input.js"),
        mode="pretrain",
        max_len=model_cfg.get("max_len", 256),
        mask_ratio=training_cfg.get("mask_ratio", 0.2),
    )

    device = torch.device(args.device)
    model = CalculusModel(
        vocab_size=len(dataset.token_to_id),
        rule_labels=dataset.rule_labels,
        hidden_dim=model_cfg.get("hidden_dim", 512),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 8),
        ffn_dim=model_cfg.get("ffn_dim", 2048),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    batch_size = training_cfg.get("batch_size", 128)
    max_steps = training_cfg.get("max_steps", 50000)
    learning_rate = training_cfg.get("lr", 2e-4)
    warmup_steps = training_cfg.get("warmup_steps", 5000)
    save_every = training_cfg.get("save_every_steps", 10000)
    validate_every = training_cfg.get("validate_every_steps", None)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = make_lr_scheduler(optimizer, warmup_steps, max_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    step = 0
    epoch = 0
    start = time.time()

    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            step += 1
            model.train()
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _, _ = model(
                    src_tokens=batch["src_tokens"],
                    src_positions=batch["src_positions"],
                    parent_child_pairs=None,
                    tgt_tokens=batch["tgt_input_tokens"],
                    src_padding_mask=batch["src_padding_mask"],
                    tgt_padding_mask=batch["tgt_padding_mask"],
                    memory_key_padding_mask=batch["src_padding_mask"],
                )
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    batch["tgt_output_tokens"].view(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 100 == 0 or step == 1:
                elapsed = time.time() - start
                print(
                    f"[Epoch {epoch}] step={step}/{max_steps} loss={loss.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.1f}s"
                )
                start = time.time()

            if save_every and step % save_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    Path(args.checkpoint_dir),
                    {
                        "vocab_size": len(dataset.token_to_id),
                        "hidden_dim": model_cfg.get("hidden_dim", 512),
                        "num_heads": model_cfg.get("num_heads", 8),
                        "num_layers": model_cfg.get("num_layers", 8),
                        "ffn_dim": model_cfg.get("ffn_dim", 2048),
                        "dropout": model_cfg.get("dropout", 0.1),
                        "position_dim": model_cfg.get("position_dim", 3),
                    },
                )

            if (
                validate_every
                and step % validate_every == 0
                and config.get("validation")
            ):
                pass

            if step >= max_steps:
                break

    save_checkpoint(
        model,
        optimizer,
        step,
        Path(args.checkpoint_dir),
        {
            "vocab_size": len(dataset.token_to_id),
            "hidden_dim": model_cfg.get("hidden_dim", 512),
            "num_heads": model_cfg.get("num_heads", 8),
            "num_layers": model_cfg.get("num_layers", 8),
            "ffn_dim": model_cfg.get("ffn_dim", 2048),
            "dropout": model_cfg.get("dropout", 0.1),
            "position_dim": model_cfg.get("position_dim", 3),
        },
    )
    print(f"Pretraining complete. Saved checkpoint to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
import json, pathlib
metrics = {
    'val/reconstruction_loss': float(best_loss),
    'training_steps': int(global_step)
}
(pathlib.Path(output_dir) / 'metrics.json').write_text(json.dumps(metrics, indent=2))