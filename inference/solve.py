import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import joblib
import torch

from model.architecture import CalculusModel
from inference.beam_search import NodeValidityPool, beam_search, load_vocab


class _LegacySLaNgTokenizer:
    pass


class PklTransformerModel(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 128):
        super().__init__()
        self.pos_encoder = torch.nn.Parameter(torch.zeros(1, 100, hidden_dim))
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.transformer = torch.nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            batch_first=True,
        )
        self.fc_out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, src_seq, tgt_in_seq):
        max_id = self.embedding.num_embeddings - 1
        src_seq = torch.clamp(src_seq, 0, max_id)
        tgt_in_seq = torch.clamp(tgt_in_seq, 0, max_id)
        max_pos = self.pos_encoder.size(1)
        src_seq = src_seq[:, :max_pos]
        tgt_in_seq = tgt_in_seq[:, :max_pos]

        src_emb = self.embedding(src_seq) + self.pos_encoder[:, : src_seq.size(1), :]
        tgt_emb = self.embedding(tgt_in_seq) + self.pos_encoder[:, : tgt_in_seq.size(1), :]
        device = src_seq.device
        tgt_mask = torch.triu(
            torch.ones(tgt_in_seq.size(1), tgt_in_seq.size(1), dtype=torch.bool, device=device),
            diagonal=1,
        )
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits


class CalculusSolverInference:
    def __init__(
        self,
        model_path: str = os.path.join("model", "model.pkl"),
        vocab_path: str = os.path.join("tokenizer", "vocab.json"),
        beam_size: int = 5,
        max_len: int = 256,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        self.model_data = self._load_checkpoint(model_path)
        self.vocab_map = load_vocab(vocab_path)
        config = (
            self.model_data.get("config", {})
            if isinstance(self.model_data, dict) and "config" in self.model_data
            else {}
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rule_labels = self._load_rule_labels(vocab_path)

        if isinstance(self.model_data, torch.nn.Module):
            self.model = self.model_data.to(self.device)
        else:
            state_dict = self._resolve_state_dict(self.model_data)

            if "fc_out.weight" in state_dict and "pos_encoder" in state_dict:
                vocab_size = state_dict["embedding.weight"].shape[0]
                hidden_dim = state_dict["embedding.weight"].shape[1]
                self.model = PklTransformerModel(vocab_size=vocab_size, hidden_dim=hidden_dim).to(self.device)
            elif any(k.startswith(("encoder.", "decoder.", "rule_head.")) for k in state_dict.keys()) and not model_path.endswith((".pt", ".pth")):
                self.model = CalculusModel(
                    vocab_size=config.get("vocab_size", len(self.vocab_map["token_to_id"])),
                    rule_labels=rule_labels,
                    hidden_dim=config.get("hidden_dim", 512),
                    num_heads=config.get("num_heads", 8),
                    num_layers=config.get("num_layers", 8),
                    ffn_dim=config.get("ffn_dim", 2048),
                    dropout=config.get("dropout", 0.1),
                    position_dim=config.get("position_dim", 3),
                ).to(self.device)
            else:
                from model.transformer import CalculusSolverModel
                hidden_dim = 128
                try:
                    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    config_path = os.path.join(root_dir, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                            cfg = json.load(f)
                            hidden_dim = cfg.get("hidden_dim", 128)
                except Exception:
                    pass
                vocab_size = max(self.vocab_map["token_to_id"].values()) + 1
                self.model = CalculusSolverModel(
                    vocab_size=vocab_size,
                    num_rules=len(rule_labels),
                    hidden_dim=hidden_dim,
                    rule_labels=rule_labels,
                ).to(self.device)

            self.model.load_state_dict(state_dict)

        self.model.eval()

        self.beam_size = beam_size
        self.max_len = max_len
        self.node_pool = NodeValidityPool(
            os.path.join(os.path.dirname(__file__), "validity_worker.js"),
            num_workers=max(2, beam_size),
        )
        self.bos_id = self.vocab_map["token_to_id"]["[BOS]"]
        self.eos_id = self.vocab_map["token_to_id"]["[EOS]"]
        self.pad_id = self.vocab_map["token_to_id"]["[PAD]"]

    def close(self) -> None:
        self.node_pool.close()

    def _load_rule_labels(self, vocab_path: str) -> List[str]:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        rule_labels = []
        for token in vocab_json.get("rule_tokens", {}).keys():
            if token.startswith("RULE:"):
                rule_labels.append(token.split("RULE:", 1)[-1])
            else:
                rule_labels.append(token)
        return rule_labels

    def _load_checkpoint(self, model_path: str) -> Any:
        if model_path.endswith((".pt", ".pth")):
            return torch.load(model_path, map_location="cpu")
        try:
            if not hasattr(sys.modules["__main__"], "SLaNgTokenizer"):
                setattr(sys.modules["__main__"], "SLaNgTokenizer", _LegacySLaNgTokenizer)
            return joblib.load(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint {model_path}: {exc}"
            ) from exc

    def _resolve_state_dict(self, checkpoint: Any) -> Dict[str, Any]:
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                return checkpoint["model_state"]
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            return checkpoint
        raise ValueError("Unsupported checkpoint format for model state.")

    def _serialize_input(self, input_env: Dict[str, Any]) -> List[str]:
        from tokenizer.slang_serializer import serialize_slang_math
        return serialize_slang_math(input_env)

    def _verify_output(
        self, input_env: Dict[str, Any], output_tokens: List[str]
    ) -> Dict[str, Any]:
        from inference.verifier import verify
        return verify(input_env, output_tokens)

    def solve(self, input_env: Dict[str, Any]) -> Dict[str, Any]:
        token_strings = self._serialize_input(input_env)
        token_ids = [
            self.vocab_map["token_to_id"].get(token, self.pad_id)
            for token in token_strings
        ]
        token_ids = token_ids[: self.max_len]

        padded_tokens = token_ids + [self.pad_id] * (self.max_len - len(token_ids))
        src_tokens = torch.tensor([padded_tokens], dtype=torch.long, device=self.device)

        result = beam_search(
            model=self.model,
            src_tokens=src_tokens,
            vocab_map=self.vocab_map,
            beam_size=self.beam_size,
            max_len=self.max_len,
            node_pool=self.node_pool,
        )

        output_token_strings = [
            self.vocab_map["id_to_token"][token_id]
            for token_id in result["tokens"]
            if token_id in self.vocab_map["id_to_token"]
        ]

        if output_token_strings and output_token_strings[0] == "[BOS]":
            output_token_strings = output_token_strings[1:]

        predicted_rule = None
        if output_token_strings and output_token_strings[0].startswith("RULE:"):
            predicted_rule = output_token_strings[0]
            output_token_strings = output_token_strings[1:]

        verifier_result = self._verify_output(input_env, output_token_strings)
        if verifier_result.get("status") in ("solved", "unverified", "unsolvable"):
            result["status"] = verifier_result["status"]
        result["verified"] = verifier_result.get("verified", False)
        result["confidence"] = verifier_result.get("confidence", 0)
        result["output"] = verifier_result.get("output")
        if verifier_result.get("error"):
            result["warning"] = verifier_result["error"]

        return {
            "input": input_env,
            "output_tokens": output_token_strings,
            "status": result["status"],
            "verified": result["verified"],
            "confidence": result["confidence"],
            "rule": predicted_rule,
            "output": result["output"],
            "warning": result.get("warning"),
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python inference/solve.py input.json")

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)

    solver = CalculusSolverInference()
    try:
        print(json.dumps(solver.solve(payload), indent=2))
    finally:
        solver.close()