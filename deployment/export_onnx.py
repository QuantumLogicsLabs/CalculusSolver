"""
Developer 4 — Option (A): export the current checkpoint to ONNX and see if
that alone gets the deployment under Vercel's ~250MB serverless limit.

Why ONNX and not plain TorchScript: torch.jit.load() still requires
importing the full `torch` package at runtime, which is the actual thing
blowing the budget (500-800MB installed) — the checkpoint itself is tiny.
ONNX Runtime is a separate, much smaller package (tens of MB for CPU
inference), so it's the only variant of "Option A" that can plausibly work.

This script exports the two neural sub-modules used during generation:
  - encoder.onnx   (TreeEncoder — runs once per request)
  - decoder.onnx   (TreeDecoder — runs once per beam-search step)

The RuleHead is NOT exported to ONNX: it's just one nn.Linear (classifier)
and one nn.Embedding (rule_embeddings). Its weights are dumped straight to
.npy files and applied with plain numpy in deployment/onnx_beam_search.py —
simpler and more robust than round-tripping something this small through ONNX.

Run from the repo root:
    python -u deployment/export_onnx.py [checkpoint_path]

Requires requirements-neural.txt (torch) to be installed locally — this
script itself is a build step, not something that runs in production.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.transformer import CalculusSolverModel  # noqa: E402


def load_rule_labels(vocab_path: Path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    labels = []
    for token in vocab_json.get("rule_tokens", {}).keys():
        labels.append(token.split("RULE:", 1)[-1] if token.startswith("RULE:") else token)
    return labels


def main():
    checkpoint_arg = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final/best.pt"
    checkpoint_path = ROOT / checkpoint_arg
    vocab_path = ROOT / "tokenizer" / "vocab.json"
    config_path = ROOT / "config.json"
    out_dir = ROOT / "deployment" / "onnx_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    flat_vocab = {}
    for k, v in vocab_json.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            flat_vocab.update(v)
    vocab_size = max(flat_vocab.values()) + 1
    rule_labels = load_rule_labels(vocab_path)

    hidden_dim = 128
    if config_path.exists():
        with open(config_path, "r") as f:
            hidden_dim = json.load(f).get("hidden_dim", 128)

    print(f"Loading checkpoint {checkpoint_path} (vocab_size={vocab_size}, "
          f"num_rules={len(rule_labels)}, hidden_dim={hidden_dim}) ...")
    model = CalculusSolverModel(vocab_size=vocab_size, num_rules=len(rule_labels), hidden_dim=hidden_dim)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    max_len = 32
    if config_path.exists():
        with open(config_path, "r") as f:
            max_len = json.load(f).get("max_len", 32)

    # ---- 1. Export encoder ----
    dummy_tokens = torch.zeros((1, max_len), dtype=torch.long)
    dummy_positions = torch.zeros((1, max_len, 3), dtype=torch.float32)
    dummy_pairs = torch.zeros((1, max_len, max_len), dtype=torch.float32)

    encoder_path = out_dir / "encoder.onnx"
    torch.onnx.export(
        model.encoder,
        (dummy_tokens, dummy_positions, dummy_pairs),
        str(encoder_path),
        input_names=["tokens", "positions", "parent_child_pairs"],
        output_names=["encoder_output"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "seq"},
            "positions": {0: "batch", 1: "seq"},
            "parent_child_pairs": {0: "batch", 1: "seq", 2: "seq"},
            "encoder_output": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    print(f"Wrote {encoder_path} ({encoder_path.stat().st_size / 1e6:.2f} MB)")

    # ---- 2. Export decoder ----
    # Wrap so tracing only sees the (target_tokens, encoder_output, rule_embeddings)
    # call shape actually used by beam search (validity_mask / padding masks
    # are always None at inference time — see inference/beam_search.py).
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, target_tokens, encoder_output, rule_embeddings):
            logits, hidden = self.decoder(
                target_tokens, encoder_output, rule_embeddings=rule_embeddings
            )
            return logits, hidden

    decoder_wrapper = DecoderWrapper(model.decoder)
    dummy_tgt = torch.zeros((1, 1), dtype=torch.long)  # grows by 1 each beam step
    dummy_enc_out = torch.zeros((1, max_len, hidden_dim), dtype=torch.float32)
    dummy_rule_emb = torch.zeros((1, hidden_dim), dtype=torch.float32)

    decoder_path = out_dir / "decoder.onnx"
    torch.onnx.export(
        decoder_wrapper,
        (dummy_tgt, dummy_enc_out, dummy_rule_emb),
        str(decoder_path),
        input_names=["target_tokens", "encoder_output", "rule_embeddings"],
        output_names=["logits", "hidden"],
        dynamic_axes={
            "target_tokens": {0: "batch", 1: "tgt_seq"},
            "encoder_output": {0: "batch", 1: "src_seq"},
            "rule_embeddings": {0: "batch"},
            "logits": {0: "batch", 1: "tgt_seq"},
            "hidden": {0: "batch", 1: "tgt_seq"},
        },
        opset_version=17,
    )
    print(f"Wrote {decoder_path} ({decoder_path.stat().st_size / 1e6:.2f} MB)")

    # ---- 3. Dump RuleHead weights as plain numpy (no ONNX needed) ----
    rule_head_dir = out_dir / "rule_head"
    rule_head_dir.mkdir(exist_ok=True)
    np.save(rule_head_dir / "classifier_weight.npy", model.rule_head.classifier.weight.detach().numpy())
    np.save(rule_head_dir / "classifier_bias.npy", model.rule_head.classifier.bias.detach().numpy())
    np.save(rule_head_dir / "rule_embeddings.npy", model.rule_head.rule_embeddings.weight.detach().numpy())
    with open(rule_head_dir / "labels.json", "w") as f:
        json.dump(rule_labels, f)
    print(f"Wrote rule_head weights to {rule_head_dir}")

    # ---- 4. Manifest ----
    manifest = {
        "source_checkpoint": str(checkpoint_arg),
        "vocab_size": vocab_size,
        "num_rules": len(rule_labels),
        "hidden_dim": hidden_dim,
        "max_len": max_len,
        "opset_version": 17,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"\nTotal ONNX artifact size: {total_bytes / 1e6:.2f} MB")
    print("Next: run deployment/verify_export.py to check correctness and total bundle size.")


if __name__ == "__main__":
    main()
