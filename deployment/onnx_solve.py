"""
Torch-free mirror of inference/solve.py::CalculusSolverInference, using
onnxruntime instead of a loaded PyTorch model. Mirrors that file's solve()
line-for-line in logic. This is what the production Vercel API imports
under Option A -- inference/solve.py (and torch) never gets imported in
that deployment.
"""

import json
import os
from typing import Any, Dict, List

import onnxruntime as ort

from inference.grammar import NodeValidityPool, load_vocab
from deployment.onnx_beam_search import onnx_beam_search


class ONNXCalculusSolverInference:
    def __init__(
        self,
        model_path: str = os.path.join("deployment", "artifacts", "best.onnx"),
        vocab_path: str = os.path.join("tokenizer", "vocab.json"),
        beam_size: int = 5,
        max_len: int = 32,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        self.vocab_map = load_vocab(vocab_path)
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
                max_len = cfg.get("max_len", max_len)

        self.beam_size = beam_size
        self.max_len = max_len
        self.node_pool = NodeValidityPool()
        self.bos_id = self.vocab_map["token_to_id"]["[BOS]"]
        self.eos_id = self.vocab_map["token_to_id"]["[EOS]"]
        self.pad_id = self.vocab_map["token_to_id"]["[PAD]"]

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

        allowed_opvars = None
        if input_env.get("op") == "gradient":
            from inference.verifier import get_variables
            input_vars = get_variables(input_env)
            if input_vars:
                allowed_opvars = {f"OPVAR:{v}" for v in input_vars}

        result = onnx_beam_search(
            session=self.session,
            src_tokens=padded_tokens,
            vocab_map=self.vocab_map,
            beam_size=self.beam_size,
            max_len=self.max_len,
            node_pool=self.node_pool,
            allowed_opvars=allowed_opvars,
        )

        output_token_strings = [
            self.vocab_map["id_to_token"][t]
            for t in result["tokens"]
            if t in self.vocab_map["id_to_token"]
        ]

        if output_token_strings and output_token_strings[0] == "[BOS]":
            output_token_strings = output_token_strings[1:]

        predicted_rule = None
        if output_token_strings and output_token_strings[0].startswith("RULE:"):
            predicted_rule = output_token_strings[0]
            output_token_strings = output_token_strings[1:]

        verifier_result = self._verify_output(input_env, output_token_strings)
        status = verifier_result.get("status", result.get("status"))
        warning = verifier_result.get("error")

        return {
            "input": input_env,
            "output_tokens": output_token_strings,
            "status": status,
            "verified": verifier_result.get("verified", False),
            "confidence": verifier_result.get("confidence", 0),
            "rule": predicted_rule,
            "output": verifier_result.get("output"),
            "warning": warning,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python deployment/onnx_solve.py input.json")
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)
    solver = ONNXCalculusSolverInference()
    try:
        print(json.dumps(solver.solve(payload), indent=2))
    finally:
        solver.close()