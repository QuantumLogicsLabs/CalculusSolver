"""
Exports the trained SimpleCalculusModel to ONNX for production deployment.
"""

import os
import sys
import torch  # Fixes NameError: name 'torch' is not defined

# Ensure project root is in path so local modules like 'model' and 'inference' are found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.simple_transformer import SimpleCalculusModel
from inference.grammar import load_vocab


def export_to_onnx(
    checkpoint_path: str = os.path.join("checkpoints", "final", "best.pt"),
    output_path: str = os.path.join("deployment", "artifacts", "best.onnx"),
    vocab_path: str = os.path.join("tokenizer", "vocab.json"),
) -> str:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    vocab_map = load_vocab(vocab_path)
    pad_id = vocab_map["token_to_id"]["[PAD]"]

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    vocab_size = state_dict["token_embedding.weight"].shape[0]
    hidden_dim = state_dict["token_embedding.weight"].shape[1]
    ckpt_max_len = state_dict["pos_encoding.pe"].shape[1]

    enc_layers = sum(1 for k in state_dict.keys() if k.startswith("transformer.encoder.layers.") and k.endswith(".norm1.weight"))
    dec_layers = sum(1 for k in state_dict.keys() if k.startswith("transformer.decoder.layers.") and k.endswith(".norm1.weight"))

    print(f"[export_onnx] Detected dimensions -> vocab_size: {vocab_size}, hidden_dim: {hidden_dim}, max_len: {ckpt_max_len}, enc_layers: {enc_layers}, dec_layers: {dec_layers}")

    model = SimpleCalculusModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        pad_id=pad_id,
        max_len=ckpt_max_len,
    )

    if hasattr(model, "pos_encoding") and hasattr(model.pos_encoding, "pe"):
        model.pos_encoding.pe = state_dict["pos_encoding.pe"].clone()

    model.load_state_dict(state_dict)
    model.eval()

    dummy_src = torch.randint(1, vocab_size, (1, 32), dtype=torch.long)
    dummy_tgt_in = torch.randint(1, vocab_size, (1, 32), dtype=torch.long)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export using TorchScript engine to prevent PyTorch 2.x item() symbol guards
    torch.onnx.export(
        model,
        (dummy_src, dummy_tgt_in),
        output_path,
        input_names=["src_seq", "tgt_in_seq"],
        output_names=["logits"],
        dynamic_axes={
            "src_seq": {0: "batch_size", 1: "seq_len"},
            "tgt_in_seq": {0: "batch_size", 1: "tgt_len"},
            "logits": {0: "batch_size", 1: "tgt_len"},
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[export_onnx] Successfully exported {checkpoint_path} -> {output_path} ({size_mb:.2f} MB)")
    return output_path


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join("checkpoints", "final", "best.pt")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("deployment", "artifacts", "best.onnx")
    export_to_onnx(ckpt, out)