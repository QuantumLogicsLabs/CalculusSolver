import json
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional

import torch

from model.architecture import CalculusModel
from inference.grammar import is_valid_prefix, NodeValidityPool, flatten_vocab, load_vocab


def beam_search(
    model: CalculusModel,
    src_tokens: torch.Tensor,
    src_positions: torch.Tensor,
    parent_child_pairs: torch.Tensor,
    vocab_map: Dict[str, Any],
    beam_size: int = 5,
    max_len: int = 128,
    node_pool: Optional[NodeValidityPool] = None,
) -> Dict[str, Any]:
    device = src_tokens.device
    vocab = vocab_map["token_to_id"]
    id_to_token = vocab_map["id_to_token"]
    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]

    if node_pool is None:
        script_path = os.path.join(os.path.dirname(__file__), "validity_worker.js")
        node_pool = NodeValidityPool(script_path, num_workers=max(2, beam_size))

    root_mask = torch.zeros(
        src_tokens.size(0), src_tokens.size(1), dtype=torch.bool, device=device
    )
    root_mask[:, 0] = True

    encoder_output = model.encoder(
        src_tokens,
        src_positions,
        parent_child_pairs,
        padding_mask=None,
    )
    rule_logits = model.rule_head(encoder_output, root_mask=root_mask)
    root_rule_ids = torch.argmax(rule_logits, dim=-1)
    root_rule_id = int(root_rule_ids[0].item())
    rule_embeddings = model.rule_head.embed_rules(root_rule_ids)

    vocab_size = max(id_to_token.keys()) + 1
    all_candidate_tokens = [id_to_token.get(idx, "[PAD]") for idx in range(vocab_size)]
    beams = [
        {
            "tokens": [bos_id],
            "score": 0.0,
            "finished": False,
        }
    ]
    completed = []

    for _ in range(max_len):
        candidates = []
        for beam in beams:
            if beam["finished"]:
                candidates.append(beam)
                continue

            current_tokens = beam["tokens"]
            token_strings = [id_to_token[token_id] for token_id in current_tokens]

            # FIX (docs/KNOWN_ISSUES.md): is_valid_prefix()'s grammar parses SLaNg
            # AST structure only — it has no rule for a leading [BOS] token, so
            # passing token_strings as-is caused every candidate to be marked
            # invalid on the very first decoding step, producing empty output
            # ([BOS] only) on every solve() call regardless of training quality.
            # Strip the seed [BOS] before validity checking; it is not part of
            # the AST grammar being validated.
            validity_tokens = (
                token_strings[1:]
                if token_strings and token_strings[0] == "[BOS]"
                else token_strings
            )

            tgt = torch.tensor([current_tokens], device=device)
            decoder_logits, _ = model.decoder(
                tgt,
                encoder_output,
                rule_embeddings=rule_embeddings,
                validity_mask=None,
                tgt_padding_mask=None,
                memory_key_padding_mask=None,
            )
            next_logits = decoder_logits[0, -1, :]
            mask = node_pool.mask(validity_tokens, all_candidate_tokens)
            invalid_mask = torch.tensor([not valid for valid in mask], device=device)
            safe_logits = next_logits.masked_fill(invalid_mask, float("-inf"))

            if torch.isinf(safe_logits).all():
                continue

            log_probs = torch.log_softmax(safe_logits, dim=-1)
            topk = torch.topk(log_probs, min(beam_size, safe_logits.size(0)))
            for score, token_id in zip(topk.values.tolist(), topk.indices.tolist()):
                new_tokens = current_tokens + [int(token_id)]
                finished = token_id == eos_id
                candidates.append(
                    {
                        "tokens": new_tokens,
                        "score": beam["score"] + float(score),
                        "finished": finished,
                    }
                )

        if not candidates:
            break

        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:beam_size]
        if all(beam["finished"] for beam in beams):
            completed.extend(beams)
            break

    best = None
    if completed:
        best = sorted(completed, key=lambda x: x["score"], reverse=True)[0]
    else:
        best = (
            beams[0] if beams else {"tokens": [bos_id], "score": 0.0, "finished": False}
        )

    status = "solved"
    root_rule_label = None
    rule_labels = getattr(model.rule_head, "labels", lambda: [])()
    if root_rule_id < len(rule_labels):
        root_rule_label = rule_labels[root_rule_id]
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