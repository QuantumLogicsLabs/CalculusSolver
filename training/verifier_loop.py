import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from inference.beam_search import NodeValidityPool, beam_search, load_vocab
from model.architecture import CalculusModel
from training.training_utils import (
    SlangJsonlDataset,
    load_yaml,
    serialize_slang_object,
    resolve_jsonl_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3 hard example mining and final fine-tuning."
    )
    parser.add_argument(
        "--data", required=True, help="Path to train JSONL file or directory."
    )
    parser.add_argument("--checkpoint", required=True, help="Stage 2 checkpoint file.")
    parser.add_argument(
        "--vocab", default="tokenizer/vocab.json", help="Path to tokenizer vocab JSON."
    )
    parser.add_argument(
        "--config",
        default="training/config/finetune.yaml",
        help="Training config YAML for verifier loop.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/final",
        help="Directory where final checkpoint is stored.",
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam width for inference."
    )
    parser.add_argument(
        "--max-len", type=int, default=256, help="Max token length for inference."
    )
    parser.add_argument(
        "--hard-ratio",
        type=float,
        default=0.4,
        help="Fraction of hard examples to oversample.",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of verifier loop epochs."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
    config: Dict[str, Any],
) -> Path:
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


def build_model_from_checkpoint(
    checkpoint_path: Path, vocab_path: Path, device: torch.device
) -> tuple:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    token_to_id, _, rule_labels = load_vocab(str(vocab_path))
    config = checkpoint.get("config", {})
    model = CalculusModel(
        vocab_size=config.get("vocab_size", len(token_to_id)),
        rule_labels=rule_labels,
        hidden_dim=config.get("hidden_dim", 512),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 8),
        ffn_dim=config.get("ffn_dim", 2048),
        dropout=config.get("dropout", 0.1),
        position_dim=config.get("position_dim", 3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.train()
    return model, checkpoint


def infer_example(
    model: torch.nn.Module,
    vocab_map: Dict[str, Any],
    input_obj: Dict[str, Any],
    beam_size: int,
    max_len: int,
    node_pool: NodeValidityPool,
    serializer_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    token_strings = serialize_slang_object(input_obj, serializer_path)
    token_ids = [
        vocab_map["token_to_id"].get(tok, vocab_map["token_to_id"].get("[PAD]", 0))
        for tok in token_strings
    ]
    token_ids = token_ids[:max_len]
    padded = token_ids + [vocab_map["token_to_id"]["[PAD]"]] * (
        max_len - len(token_ids)
    )
    src_tokens = torch.tensor([padded], dtype=torch.long, device=device)
    src_positions = torch.zeros((1, max_len, 3), dtype=torch.float32, device=device)
    parent_child_pairs = torch.zeros(
        (1, max_len, max_len), dtype=torch.float32, device=device
    )
    result = beam_search(
        model=model,
        src_tokens=src_tokens,
        src_positions=src_positions,
        parent_child_pairs=parent_child_pairs,
        vocab_map=vocab_map,
        beam_size=beam_size,
        max_len=max_len,
        node_pool=node_pool,
    )
    return result


def verify_output(
    input_obj: Dict[str, Any], output_tokens: List[str], script_path: Path
) -> Dict[str, Any]:
    payload = {"input": input_obj, "output_tokens": output_tokens}
    proc = subprocess.run(
        ["node", "--input-type=module", str(script_path)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"Verification failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return json.loads(proc.stdout)


def build_training_dataset(
    data_path: Path,
    vocab_path: Path,
    serializer_path: Path,
    max_len: int,
) -> SlangJsonlDataset:
    return SlangJsonlDataset(
        data_path=data_path,
        vocab_path=vocab_path,
        serializer_path=serializer_path,
        mode="finetune",
        max_len=max_len,
    )


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
    rule_weight: float,
    step_weight: float,
):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    rule_criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_batches = 0
    model.train()
    for batch in dataloader:
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
        loss = token_loss + rule_weight * rule_loss + step_weight * step_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / args.config)
    training_cfg = config.get("training", {})

    device = torch.device(args.device)
    vocab_path = root / args.vocab
    serializer_path = root / "inference" / "serialize_input.js"
    verifier_script = root / "inference" / "verifier.js"
    vocab_map = load_vocab(str(vocab_path))

    model, checkpoint = build_model_from_checkpoint(
        Path(args.checkpoint), vocab_path, device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.get("lr", 1e-5))

    data_path = Path(args.data)
    dataset = build_training_dataset(
        data_path=data_path,
        vocab_path=vocab_path,
        serializer_path=serializer_path,
        max_len=args.max_len,
    )

    hard_examples: List[Dict[str, Any]] = []
    hard_example_keys = set()
    if args.hard_ratio <= 0.0:
        args.hard_ratio = 0.0

    node_pool = NodeValidityPool(
        str(root / "inference" / "validity_worker.js"),
        num_workers=max(2, args.beam_size),
    )
    step = checkpoint.get("step", 0)

    for epoch in range(1, args.epochs + 1):
        print(f"Verifier loop epoch {epoch}/{args.epochs}")
        oversample = int(max(1, args.hard_ratio * len(dataset)))
        sampled_hard = (
            random.choices(hard_examples, k=oversample) if hard_examples else []
        )
        mixed_examples = dataset.examples + sampled_hard
        dataset.examples = mixed_examples
        dataset._cache = [{} for _ in dataset.examples]

        dataloader = DataLoader(
            dataset,
            batch_size=training_cfg.get("batch_size", 32),
            shuffle=True,
            collate_fn=dataset.collate,
            num_workers=0,
        )

        train_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            pad_id=dataset.pad_id,
            rule_weight=training_cfg.get("rule_loss_weight", 1.0),
            step_weight=training_cfg.get("step_loss_weight", 0.5),
        )
        step += 1
        print(f"Epoch {epoch} training loss: {train_loss:.4f}")

        new_hard_examples: List[Dict[str, Any]] = []
        raw_examples = []
        for file_path in resolve_jsonl_path(data_path):
            with file_path.open("r", encoding="utf-8") as handle:
                raw_examples.extend(
                    [json.loads(line) for line in handle if line.strip()]
                )

        sample_count = min(
            len(raw_examples), int(training_cfg.get("hard_sample_size", 200))
        )
        sample_examples = random.sample(raw_examples, sample_count)

        for idx, example in enumerate(sample_examples, start=1):
            result = infer_example(
                model=model,
                vocab_map=vocab_map,
                input_obj=example["input"],
                beam_size=args.beam_size,
                max_len=args.max_len,
                node_pool=node_pool,
                serializer_path=serializer_path,
                device=device,
            )
            bos_id = vocab_map["token_to_id"].get("[BOS]")
            eos_id = vocab_map["token_to_id"].get("[EOS]")
            token_ids = [
                token_id
                for token_id in result["tokens"]
                if token_id != bos_id and token_id != eos_id
            ]
            output_tokens = [
                vocab_map["id_to_token"].get(i, "[UNK]") for i in token_ids
            ]
            verifier = verify_output(example["input"], output_tokens, verifier_script)
            if (
                not verifier.get("verified", False)
                or verifier.get("status") != "solved"
            ):
                example_key = json.dumps(example, sort_keys=True)
                if example_key not in hard_example_keys:
                    hard_example_keys.add(example_key)
                    new_hard_examples.append(example)

            if idx % 50 == 0:
                print(
                    f"Verified {idx}/{sample_count} examples, hard candidates={len(new_hard_examples)}"
                )

        if new_hard_examples:
            hard_examples.extend(new_hard_examples)
            print(f"Found {len(new_hard_examples)} new hard examples.")
        else:
            print("No new hard examples found this epoch.")

        save_checkpoint(model, optimizer, step, Path(args.checkpoint_dir))
        print(f"Saved final checkpoint to {args.checkpoint_dir}")

    node_pool.close()
    print("Verifier loop complete.")


if __name__ == "__main__":
    main()
import json, pathlib
metrics = {
    'val/numerical_equiv': float(final_num_equiv),
    'val/rule_accuracy': float(final_rule_acc),
    'train/hard_pool_size': int(hard_pool_size),
    'training_steps': int(global_step)
}
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
(pathlib.Path(output_dir) / 'metrics.json').write_text(json.dumps(metrics, indent=2))
