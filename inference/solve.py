import json
import os
from typing import Any, Dict, List

import torch

from inference.beam_search import NodeValidityPool, beam_search, load_vocab


class CalculusSolverInference:
    def __init__(
        self,
        model_path: str = os.path.join("checkpoints", "final", "best.pt"),
        vocab_path: str = os.path.join("tokenizer", "vocab.json"),
        beam_size: int = 5,
        max_len: int = 32,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        self.vocab_map = load_vocab(vocab_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pad_id = self.vocab_map["token_to_id"]["[PAD]"]

        from model.simple_transformer import SimpleCalculusModel

        hidden_dim = 128
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(root_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
                hidden_dim = cfg.get("hidden_dim", 128)
                max_len = cfg.get("max_len", max_len)

        vocab_size = max(self.vocab_map["token_to_id"].values()) + 1
        self.model = SimpleCalculusModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            pad_id=pad_id,
            max_len=max_len,
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.beam_size = beam_size
        self.max_len = max_len
        self.node_pool = NodeValidityPool()
        self.bos_id = self.vocab_map["token_to_id"]["[BOS]"]
        self.eos_id = self.vocab_map["token_to_id"]["[EOS]"]
        self.pad_id = pad_id

    def close(self) -> None:
        self.node_pool.close()

    def _serialize_input(self, input_env: Dict[str, Any]) -> List[str]:
        from tokenizer.slang_serializer import serialize_slang_math
        return serialize_slang_math(input_env)

    def _verify_output(self, input_env: Dict[str, Any], output_tokens: List[str]) -> Dict[str, Any]:
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
            self.vocab_map["id_to_token"][t]
            for t in result["tokens"]
            if t in self.vocab_map["id_to_token"]
        ]

        # Strip [BOS]
        if output_token_strings and output_token_strings[0] == "[BOS]":
            output_token_strings = output_token_strings[1:]

        # Extract and strip the leading RULE:xxx token, if present -- it's
        # not part of the SLaNg AST grammar the verifier deserializes.
        predicted_rule = None
        if output_token_strings and output_token_strings[0].startswith("RULE:"):
            predicted_rule = output_token_strings[0]
            output_token_strings = output_token_strings[1:]

        verifier_result = self._verify_output(input_env, output_token_strings)
        result["status"] = verifier_result.get("status", result.get("status"))
        result["verified"] = verifier_result.get("verified", False)
        result["confidence"] = verifier_result.get("confidence", 0)
        result["output"] = verifier_result.get("output")
        warning = verifier_result.get("error")

        return {
            "input": input_env,
            "output_tokens": output_token_strings,
            "status": result["status"],
            "verified": result["verified"],
            "confidence": result["confidence"],
            "rule": predicted_rule,
            "output": result["output"],
            "warning": warning,
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