import json
import os
from typing import Any, Dict, List, Optional

import torch

from inference.grammar import NodeValidityPool, flatten_vocab, load_vocab, is_valid_prefix

# NOTE: NodeValidityPool, flatten_vocab, load_vocab, and is_valid_prefix now
# live in inference/grammar.py (torch-free) so the ONNX deployment path
# (deployment/onnx_beam_search.py) can reuse them without importing torch.
# This is a pure move -- no logic changed from the previous inline versions.


def beam_search(
    model,
    src_tokens: torch.Tensor,
    vocab_map: Dict[str, Any],
    beam_size: int = 5,
    max_len: int = 32,
    node_pool: Optional[NodeValidityPool] = None,
    src_positions: Optional[torch.Tensor] = None,
    parent_child_pairs: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Beam search for the tree-based CalculusSolverModel (model/transformer.py).

    NOTE: CalculusSolverModel.forward(src_seq, tgt_in_seq, true_rule_ids=None)
    computes src_positions/parent_child_pairs internally (as zero tensors) and
    does not take them as inputs. src_positions/parent_child_pairs are accepted
    here only so callers built for the older tree-kwarg interface (e.g.
    inference/solve.py) don't break -- they are unused.

    forward() returns (decoder_logits, rule_logits, verifier_logits); only
    decoder_logits is used for next-token scoring here. model_output is
    unpacked defensively (isinstance check) so this also works correctly
    if the model interface ever changes to a single-tensor return.

    beam_size default raised from 1/2 to 4: beam_size=1 (greedy) and
    beam_size=2 both gave far lower real accuracy (0% and 5.7% overall
    respectively on eval/run_eval.py) than the model's teacher-forced
    training accuracy (93.9%) suggested was achievable -- consistent with
    exposure bias, where free-running generation needs real search width
    to recover from an early wrong token. beam_size=4 trades more CPU time
    for meaningfully more error-recovery capacity. See docs/KNOWN_ISSUES.md.
    """
    device = src_tokens.device
    vocab = vocab_map["token_to_id"]
    id_to_token = vocab_map["id_to_token"]
    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]

    if node_pool is None:
        node_pool = NodeValidityPool()

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
            validity_tokens = (
                token_strings[1:]
                if token_strings and token_strings[0] == "[BOS]"
                else token_strings
            )

            tgt = torch.tensor([current_tokens], device=device)

            # FIX: single model() call, unpacked defensively. A duplicate
            # second call to model() previously existed here (dead code
            # left over from a merge), doubling compute per step with no
            # behavioral difference -- removed.
            model_output = model(src_tokens, tgt)
            decoder_logits = model_output[0] if isinstance(model_output, tuple) else model_output
            next_logits = decoder_logits[0, -1, :]

            mask = node_pool.mask(validity_tokens, all_candidate_tokens)
            invalid_mask = torch.tensor([not v for v in mask], device=device)
            safe_logits = next_logits.masked_fill(invalid_mask, float("-inf"))

            if torch.isinf(safe_logits).all():
                continue

            log_probs = torch.log_softmax(safe_logits, dim=-1)
            topk = torch.topk(log_probs, min(beam_size, safe_logits.size(0)))
            for score, token_id in zip(topk.values.tolist(), topk.indices.tolist()):
                new_tokens = current_tokens + [int(token_id)]
                finished = token_id == eos_id
                candidates.append({
                    "tokens": new_tokens,
                    "score": beam["score"] + float(score),
                    "finished": finished,
                })

        if not candidates:
            break

        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:beam_size]
        if all(b["finished"] for b in beams):
            completed.extend(beams)
            break

    best = sorted(completed, key=lambda x: x["score"], reverse=True)[0] if completed else (
        beams[0] if beams else {"tokens": [bos_id], "score": 0.0, "finished": False}
    )

    status = "solved" if best["finished"] else "partial"
    return {"tokens": best["tokens"], "score": best["score"], "status": status}
