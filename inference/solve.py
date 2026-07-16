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

        # PR fix: load based on what the checkpoint says it is, not the file
        # extension. A file extension is not a reliable guarantee of which
        # model class produced it (that's exactly how the old model.pkl /
        # architecture mismatch went unnoticed for a while).
        architecture = (
            self.model_data.get("architecture")
            if isinstance(self.model_data, dict)
            else None
        )

        if architecture == "CalculusSolverModel":
            from model.transformer import CalculusSolverModel
            self.model = CalculusSolverModel(
                vocab_size=config.get(
                    "vocab_size", max(self.vocab_map["token_to_id"].values()) + 1
                ),
                num_rules=config.get("num_rules", len(rule_labels)),
                hidden_dim=config.get("hidden_dim", 128),
            ).to(self.device)
        elif architecture == "CalculusModel":
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
            # Legacy checkpoint saved before the "architecture" field existed.
            # Fall back to the old file-extension guess, with a visible
            # warning so a stale/ambiguous checkpoint is never silently
            # assumed to be the right one.
            print(
                f"WARNING: checkpoint at {model_path} has no 'architecture' "
                "field (pre-dates this fix). Guessing model class from the "
                "file extension — re-save this checkpoint with train.py to "
                "remove this warning."
            )
            if model_path.endswith((".pt", ".pth")):
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
                ).to(self.device)
            else:
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
        self.model.load_state_dict(self._resolve_state_dict(self.model_data))
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
        # PR fix: try torch.load first regardless of extension — new
        # checkpoints (from train.py's save_checkpoint) are always
        # torch-saved dicts with an "architecture" field, whether the file
        # is named .pt or .pkl. Only fall back to joblib for genuinely
        # legacy checkpoints that torch.load can't parse.
        try:
            return torch.load(model_path, map_location="cpu")
        except Exception:
            pass
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
        src_positions = torch.zeros(
            (1, self.max_len, 3), dtype=torch.float32, device=self.device
        )
        parent_child_pairs = torch.zeros(
            (1, self.max_len, self.max_len), dtype=torch.float32, device=self.device
        )

        result = beam_search(
            model=self.model,
            src_tokens=src_tokens,
            src_positions=src_positions,
            parent_child_pairs=parent_child_pairs,
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
            "rule": result.get("root_rule_label"),
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