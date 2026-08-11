"""
ONNX Runtime + numpy reimplementation of inference/beam_search.py's
beam_search() function. Same algorithm, same masking rules, same grammar
checker (imported from inference/grammar.py, which is already torch-free) —
only the tensor backend changes: onnxruntime.InferenceSession + numpy
instead of a torch nn.Module.

This file must NOT import torch, directly or indirectly — that's the whole
point of Option A. inference/grammar.py is safe to import (verified
torch-free). inference/beam_search.py and inference/solve.py are NOT safe to
import here, since both import torch at module level.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from inference.grammar import NodeValidityPool, is_valid_prefix  # torch-free


def _softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class OnnxCalculusModel:
    """Loads the exported ONNX encoder/decoder + numpy RuleHead weights and
    exposes the same call shapes beam_search_onnx() needs."""

    def __init__(self, artifacts_dir: str):
        import json

        with open(os.path.join(artifacts_dir, "manifest.json")) as f:
            self.manifest = json.load(f)

        self.encoder_session = ort.InferenceSession(
            os.path.join(artifacts_dir, "encoder.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.decoder_session = ort.InferenceSession(
            os.path.join(artifacts_dir, "decoder.onnx"),
            providers=["CPUExecutionProvider"],
        )

        rh_dir = os.path.join(artifacts_dir, "rule_head")
        self.classifier_weight = np.load(os.path.join(rh_dir, "classifier_weight.npy"))  # [num_rules, hidden]
        self.classifier_bias = np.load(os.path.join(rh_dir, "classifier_bias.npy"))      # [num_rules]
        self.rule_embeddings = np.load(os.path.join(rh_dir, "rule_embeddings.npy"))      # [num_rules, hidden]
        with open(os.path.join(rh_dir, "labels.json")) as f:
            self.rule_labels = json.load(f)

    def encode(self, tokens: np.ndarray, positions: np.ndarray, parent_child_pairs: np.ndarray) -> np.ndarray:
        (out,) = self.encoder_session.run(
            ["encoder_output"],
            {
                "tokens": tokens.astype(np.int64),
                "positions": positions.astype(np.float32),
                "parent_child_pairs": parent_child_pairs.astype(np.float32),
            },
        )
        return out

    def rule_logits(self, encoder_output: np.ndarray) -> np.ndarray:
        # Matches RuleHead.forward()'s root_mask branch as used by
        # beam_search(): root_mask is True only at position 0, so pooled
        # output is exactly encoder_output[:, 0, :].
        pooled = encoder_output[:, 0, :]
        return pooled @ self.classifier_weight.T + self.classifier_bias

    def embed_rule(self, rule_id: int) -> np.ndarray:
        return self.rule_embeddings[rule_id : rule_id + 1]  # [1, hidden]

    def decode_step(self, target_tokens: np.ndarray, encoder_output: np.ndarray, rule_embeddings: np.ndarray) -> np.ndarray:
        (logits, _hidden) = self.decoder_session.run(
            ["logits", "hidden"],
            {
                "target_tokens": target_tokens.astype(np.int64),
                "encoder_output": encoder_output.astype(np.float32),
                "rule_embeddings": rule_embeddings.astype(np.float32),
            },
        )
        return logits


def beam_search_onnx(
    model: OnnxCalculusModel,
    src_tokens: np.ndarray,          # [1, L] int64
    src_positions: np.ndarray,       # [1, L, 3] float32
    parent_child_pairs: np.ndarray,  # [1, L, L] float32
    vocab_map: Dict[str, Any],
    beam_size: int = 5,
    max_len: int = 128,
    node_pool: Optional[NodeValidityPool] = None,
) -> Dict[str, Any]:
    vocab = vocab_map["token_to_id"]
    id_to_token = vocab_map["id_to_token"]
    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]

    if node_pool is None:
        node_pool = NodeValidityPool()

    encoder_output = model.encode(src_tokens, src_positions, parent_child_pairs)
    rlogits = model.rule_logits(encoder_output)
    root_rule_id = int(np.argmax(rlogits, axis=-1)[0])
    rule_embeddings = model.embed_rule(root_rule_id)

    vocab_size = max(id_to_token.keys()) + 1
    all_candidate_tokens = [id_to_token.get(idx, "[PAD]") for idx in range(vocab_size)]

    beams = [{"tokens": [bos_id], "score": 0.0, "finished": False}]
    completed = []

    for _ in range(max_len):
        candidates = []
        for beam in beams:
            if beam["finished"]:
                candidates.append(beam)
                continue

            current_tokens = beam["tokens"]
            token_strings = [id_to_token[t] for t in current_tokens]
            validity_tokens = token_strings[1:] if token_strings and token_strings[0] == "[BOS]" else token_strings

            tgt = np.array([current_tokens], dtype=np.int64)
            decoder_logits = model.decode_step(tgt, encoder_output, rule_embeddings)
            next_logits = decoder_logits[0, -1, :]  # [vocab]

            mask = node_pool.mask(validity_tokens, all_candidate_tokens)
            invalid = np.array([not v for v in mask], dtype=bool)
            safe_logits = np.where(invalid, -np.inf, next_logits)

            if np.all(np.isinf(safe_logits)):
                continue

            log_probs = safe_logits - np.log(np.sum(np.exp(safe_logits - np.max(safe_logits))))
            k = min(beam_size, safe_logits.shape[0])
            top_idx = np.argpartition(-log_probs, k - 1)[:k]
            top_idx = top_idx[np.argsort(-log_probs[top_idx])]

            for token_id in top_idx.tolist():
                score = float(log_probs[token_id])
                new_tokens = current_tokens + [int(token_id)]
                finished = token_id == eos_id
                candidates.append({
                    "tokens": new_tokens,
                    "score": beam["score"] + score,
                    "finished": finished,
                })

        if not candidates:
            break

        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:beam_size]
        if all(b["finished"] for b in beams):
            completed.extend(beams)
            break

    best = None
    if completed:
        best = sorted(completed, key=lambda x: x["score"], reverse=True)[0]
    else:
        best = beams[0] if beams else {"tokens": [bos_id], "score": 0.0, "finished": False}

    status = "solved"
    root_rule_label = None
    if root_rule_id < len(model.rule_labels):
        root_rule_label = model.rule_labels[root_rule_id]
        if root_rule_label == "undefined":
            status = "unsolvable"
    if not best["finished"]:
        status = "partial"

    return {
        "tokens": best["tokens"],
        "score": best["score"],
        "status": status,
        "root_rule_id": root_rule_id,
        "root_rule_label": root_rule_label,
    }
