"""
numpy/onnxruntime-only mirror of inference/beam_search.py -- mirrors that
file in logic, but never imports torch. This is the entire point of the
Option A (ONNX) deployment path: the production Vercel bundle only needs
onnxruntime + numpy, not the full PyTorch package, which is what pushed the
old bundle over the ~250MB serverless size limit.

The two files had drifted badly -- every termination and repetition fix
landed on the torch side only. The shared policy now lives in
inference/decoding.py (pure stdlib) so they cannot silently diverge again;
only the array arithmetic differs between them.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from inference.decoding import (
    content_token_ids,
    escape_token_ids,
    longest_complete_prefix,
    penalised_logit,
    repeat_blocked_tokens,
    repetition_penalty_targets,
    strip_terminator,
)
from inference.grammar import NodeValidityPool, is_complete


def _log_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax that PRESERVES -inf for masked entries.

    The previous implementation was np.log(softmax(x) + 1e-12), which mapped
    every grammatically illegal token to log(1e-12) ~= -27.6 -- a FINITE
    score. argpartition would then happily select masked tokens, and since
    [EOS] is one of them, that manufactured "finished" beams holding a
    truncated AST. A numpy-only hazard: the torch path gets true -inf from
    torch.log_softmax and never had it.
    """
    finite = np.isfinite(x)
    out = np.full(x.shape, -np.inf, dtype=np.float64)
    if not finite.any():
        return out
    shifted = x - np.max(x[finite])
    exponentiated = np.where(finite, np.exp(shifted), 0.0)
    out[finite] = shifted[finite] - np.log(exponentiated.sum())
    return out


def onnx_beam_search(
    session: ort.InferenceSession,
    src_tokens: List[int],
    vocab_map: Dict[str, Any],
    beam_size: int = 5,
    max_len: int = 32,
    node_pool: Optional[NodeValidityPool] = None,
    max_token_run: int = 4,
    no_repeat_ngram_size: int = 2,
    repetition_penalty: float = 1.2,
    repetition_min_count: int = 4,
) -> Dict[str, Any]:
    """Mirrors inference/beam_search.py::beam_search(), but drives the
    exported ONNX graph via onnxruntime instead of a torch.nn.Module."""
    vocab = vocab_map["token_to_id"]
    id_to_token = vocab_map["id_to_token"]
    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]

    if node_pool is None:
        node_pool = NodeValidityPool()

    exempt_ids = escape_token_ids(vocab)
    content_ids = content_token_ids(id_to_token)

    vocab_size = max(id_to_token.keys()) + 1
    all_candidate_tokens = [id_to_token.get(idx, "[PAD]") for idx in range(vocab_size)]

    src_arr = np.array([src_tokens], dtype=np.int64)

    beams = [{"tokens": [bos_id], "score": 0.0, "finished": False}]
    completed: List[Dict[str, Any]] = []

    for _ in range(max_len):
        candidates: List[Dict[str, Any]] = []
        for beam in beams:
            current_tokens = beam["tokens"]
            token_strings = [id_to_token[t] for t in current_tokens]
            validity_tokens = (
                token_strings[1:]
                if token_strings and token_strings[0] == "[BOS]"
                else token_strings
            )

            tgt_arr = np.array([current_tokens], dtype=np.int64)
            logits = session.run(
                ["logits"],
                {"src_seq": src_arr, "tgt_in_seq": tgt_arr},
            )[0]
            next_logits = np.asarray(logits[0, -1, :], dtype=np.float64).copy()

            if repetition_penalty != 1.0:
                for token_id in repetition_penalty_targets(
                    current_tokens, repetition_min_count, exempt_ids, content_ids
                ):
                    if 0 <= token_id < next_logits.shape[0]:
                        next_logits[token_id] = penalised_logit(
                            float(next_logits[token_id]), repetition_penalty
                        )

            mask = node_pool.mask(validity_tokens, all_candidate_tokens)
            safe_logits = next_logits.copy()
            safe_logits[[not v for v in mask]] = -np.inf

            blocked = repeat_blocked_tokens(
                current_tokens,
                max_token_run,
                no_repeat_ngram_size,
                exempt_ids,
                content_ids,
            )
            if blocked:
                guarded = safe_logits.copy()
                for blocked_id in blocked:
                    if 0 <= blocked_id < guarded.shape[0]:
                        guarded[blocked_id] = -np.inf
                if np.isfinite(guarded).any():
                    safe_logits = guarded

            valid_count = int(np.isfinite(safe_logits).sum())
            if valid_count == 0:
                if is_complete(validity_tokens):
                    completed.append({
                        "tokens": current_tokens + [eos_id],
                        "score": beam["score"],
                        "finished": True,
                    })
                continue

            log_probs = _log_softmax(safe_logits)

            k = min(beam_size, valid_count, log_probs.shape[0])
            top_idx = np.argpartition(-log_probs, k - 1)[:k]
            top_idx = top_idx[np.argsort(-log_probs[top_idx])]

            for raw_id in top_idx:
                token_id = int(raw_id)
                score = float(log_probs[token_id])
                if not np.isfinite(score):
                    continue
                new_beam = {
                    "tokens": current_tokens + [token_id],
                    "score": beam["score"] + score,
                    "finished": token_id == eos_id,
                }
                target = completed if new_beam["finished"] else candidates
                target.append(new_beam)

        if not candidates:
            break

        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:beam_size]

        if completed and beams[0]["score"] <= max(c["score"] for c in completed):
            break

    best = sorted(completed, key=lambda x: x["score"], reverse=True)[0] if completed else (
        beams[0] if beams else {"tokens": [bos_id], "score": 0.0, "finished": False}
    )

    status = "solved" if best["finished"] else "partial"
    tokens = strip_terminator(best["tokens"], eos_id)

    if not best["finished"]:
        salvaged = longest_complete_prefix(tokens, id_to_token)
        if salvaged:
            tokens = salvaged

    return {"tokens": tokens, "score": best["score"], "status": status}
