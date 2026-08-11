"""
Torch-free, ONNX Runtime-backed equivalent of inference/solve.py.

This is what a Vercel function would import under Option A. It never
imports torch — only onnxruntime + numpy, plus the existing pure-Python
tokenizer/verifier modules (already torch-free, checked before reuse here).

Usage:
    solver = OnnxCalculusSolverInference("deployment/onnx_artifacts")
    result = solver.solve(input_env)
"""
import json
import os
from typing import Any, Dict, List

import numpy as np

from deployment.onnx_beam_search import OnnxCalculusModel, beam_search_onnx
from inference.grammar import NodeValidityPool, load_vocab
from tokenizer.slang_serializer import serialize_slang_math
from inference.verifier import verify


class OnnxCalculusSolverInference:
    def __init__(
        self,
        artifacts_dir: str = os.path.join("deployment", "onnx_artifacts"),
        vocab_path: str = os.path.join("tokenizer", "vocab.json"),
        beam_size: int = 5,
    ):
        if not os.path.isdir(artifacts_dir):
            raise FileNotFoundError(f"ONNX artifacts not found: {artifacts_dir}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        self.model = OnnxCalculusModel(artifacts_dir)
        self.vocab_map = load_vocab(vocab_path)
        self.beam_size = beam_size
        self.max_len = self.model.manifest.get("max_len", 32)
        self.node_pool = NodeValidityPool()

        self.bos_id = self.vocab_map["token_to_id"]["[BOS]"]
        self.eos_id = self.vocab_map["token_to_id"]["[EOS]"]
        self.pad_id = self.vocab_map["token_to_id"]["[PAD]"]

    def close(self) -> None:
        self.node_pool.close()

    def _serialize_input(self, input_env: Dict[str, Any]) -> List[str]:
        return serialize_slang_math(input_env)

    def solve(self, input_env: Dict[str, Any]) -> Dict[str, Any]:
        token_strings = self._serialize_input(input_env)
        token_ids = [
            self.vocab_map["token_to_id"].get(token, self.pad_id)
            for token in token_strings
        ]
        token_ids = token_ids[: self.max_len]
        padded_tokens = token_ids + [self.pad_id] * (self.max_len - len(token_ids))

        src_tokens = np.array([padded_tokens], dtype=np.int64)
        src_positions = np.zeros((1, self.max_len, 3), dtype=np.float32)
        parent_child_pairs = np.zeros((1, self.max_len, self.max_len), dtype=np.float32)

        result = beam_search_onnx(
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
            self.vocab_map["id_to_token"][t]
            for t in result["tokens"]
            if t in self.vocab_map["id_to_token"]
        ]
        # Same [BOS]-strip fix documented in inference/solve.py — beam search
        # always seeds with a leading [BOS], which isn't part of the AST
        # grammar the verifier's deserializer expects.
        if output_token_strings and output_token_strings[0] == "[BOS]":
            output_token_strings = output_token_strings[1:]

        verifier_result = verify(input_env, output_token_strings)
        if verifier_result.get("status") in ("solved", "unverified", "unsolvable"):
            result["status"] = verifier_result["status"]

        return {
            "input": input_env,
            "output_tokens": output_token_strings,
            "status": result["status"],
            "verified": verifier_result.get("verified", False),
            "confidence": verifier_result.get("confidence", 0),
            "rule": result.get("root_rule_label"),
            "output": verifier_result.get("output"),
            "warning": verifier_result.get("error"),
        }
