import json
import os
from typing import Any, Dict, List, Optional

import torch

from inference.grammar import NodeValidityPool, flatten_vocab, load_vocab, is_valid_prefix


def _call_model(
    model,
    src_tokens: torch.Tensor,
    tgt_tokens: torch.Tensor,
    src_positions: Optional[torch.Tensor] = None,
    parent_child_pairs: Optional[torch.Tensor] = None,
) -> Any:
    try:
        return model(src_tokens, tgt_tokens)
    except TypeError:
        device = src_tokens.device
        batch_size, seq_len = src_tokens.size()
        if src_positions is None:
            src_positions = torch.zeros(
                (batch_size, seq_len, 3), dtype=torch.float32, device=device
            )
        if parent_child_pairs is None:
            parent_child_pairs = torch.zeros(
                (batch_size, seq_len, seq_len), dtype=torch.float32, device=device
            )
        return model(src_tokens, src_positions, parent_child_pairs, tgt_tokens)


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
    device = src_tokens.device
    vocab = vocab_map["token_to_id"]
    id_to_token = vocab_map["id_to_token"]
    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]

    if node_pool is None:
        node_pool = NodeValidityPool()

    vocab_size = max(id_to_token.keys()) + 1
    all_candidate_tokens = [id_to_token.get(idx, "[PAD]") for idx in range(vocab_size)]

    rule_token_entries = [
        tok for tok in vocab.keys() if tok.startswith("RULE:")
    ]

    seed_tokens = [bos_id]

    if rule_token_entries:
        init_tgt = torch.tensor([[bos_id]], device=device)
        with torch.no_grad():
            init_output = _call_model(
                model,
                src_tokens,
                init_tgt,
                src_positions=src_positions,
                parent_child_pairs=parent_child_pairs,
            )
        init_rule_logits = init_output[1] if isinstance(init_output, tuple) else None

        if init_rule_logits is not None:
            pred_rule_idx = torch.argmax(init_rule_logits, dim=-1).item()
            pred_rule_idx = min(pred_rule_idx, len(rule_token_entries) - 1)
            rule_token_str = rule_token_entries[pred_rule_idx]
            rule_token_id = vocab.get(rule_token_str)
            if rule_token_id is not None:
                seed_tokens = [bos_id, rule_token_id]

    beams = [{"tokens": seed_tokens, "score": 0.0, "finished": False}]
    completed = []

    for _ in range(max_len):
        candidates = []
        for beam in beams:
            if beam["finished"]:
                candidates.append(beam)
                continue

            current_tokens = beam["tokens"]
            token_strings = [id_to_token[t] for t in current_tokens if t in id_to_token]
            validity_tokens = token_strings[:]
            if validity_tokens and validity_tokens[0] == "[BOS]":
                validity_tokens = validity_tokens[1:]
            if validity_tokens and validity_tokens[0].startswith("RULE:"):
                validity_tokens = validity_tokens[1:]

            tgt = torch.tensor([current_tokens], device=device)

            model_output = _call_model(
                model,
                src_tokens,
                tgt,
                src_positions=src_positions,
                parent_child_pairs=parent_child_pairs,
            )
            decoder_logits = model_output[0] if isinstance(model_output, tuple) else model_output
            next_logits = decoder_logits[0, -1, :]

            mask = node_pool.mask(validity_tokens, all_candidate_tokens)
            vocab_len = next_logits.size(0)
            padded_mask = mask + [True] * max(0, vocab_len - len(mask))
            invalid_mask = torch.tensor(
                [not v for v in padded_mask[:vocab_len]], device=device
            )
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
