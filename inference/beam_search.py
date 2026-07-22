import json
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional

import torch

from model.architecture import CalculusModel


def is_valid_prefix(tokens: List[str]) -> bool:
    """Check if the given tokens form a valid prefix of a SLaNg AST."""
    if not tokens:
        return True
        
    def parse_term(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "NODE:TERM":
            return {"status": "invalid"}
        index += 1

        if index >= len(tokens):
            return {"status": "incomplete"}
        if not tokens[index].startswith("COEF:"):
            return {"status": "invalid"}
        index += 1

        while index < len(tokens):
            token = tokens[index]
            if token.startswith("VAR:"):
                index += 1
                if index >= len(tokens):
                    return {"status": "incomplete"}
                if not tokens[index].startswith("EXP:"):
                    return {"status": "invalid"}
                index += 1
                continue
            break

        return {"status": "complete", "next": index}

    def parse_term_list(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] == "STRUCT:CLOSE":
            return {"status": "complete", "next": index}

        current = index
        while True:
            node = parse_node(current)
            if node["status"] == "invalid":
                return {"status": "invalid"}
            if node["status"] == "incomplete":
                return {"status": "incomplete"}
            current = node["next"]
            if current >= len(tokens):
                return {"status": "incomplete"}
            if tokens[current] == "STRUCT:SEP":
                current += 1
                continue
            if tokens[current] == "STRUCT:CLOSE":
                return {"status": "complete", "next": current}
            return {"status": "invalid"}

    def parse_fraction(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "NODE:FRAC":
            return {"status": "invalid"}
        index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:OPEN":
            return {"status": "invalid"}
        index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:NUMI":
            return {"status": "invalid"}
        index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:OPEN":
            return {"status": "invalid"}
        index += 1

        numerator = parse_term_list(index)
        if numerator["status"] != "complete":
            return numerator
        index = numerator["next"]
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:CLOSE":
            return {"status": "invalid"}
        index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:SEP":
            return {"status": "invalid"}
        index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:DENO":
            return {"status": "invalid"}
        index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:OPEN":
            return {"status": "invalid"}
        index += 1

        denominator = parse_term_list(index)
        if denominator["status"] != "complete":
            return denominator
        index = denominator["next"]
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:CLOSE":
            return {"status": "invalid"}
        index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:CLOSE":
            return {"status": "invalid"}
        index += 1

        return {"status": "complete", "next": index}

    def parse_op_node(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        token = tokens[index]
        if not isinstance(token, str) or not token.startswith("OP:"):
            return {"status": "invalid"}
        index += 1

        while (
            index < len(tokens)
            and isinstance(tokens[index], str)
            and tokens[index].startswith("OPVAR:")
        ):
            index += 1

        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:OPEN":
            return {"status": "invalid"}
        index += 1

        seen_child = False
        while True:
            node = parse_node(index)
            if node["status"] == "invalid":
                return {"status": "invalid"}
            if node["status"] == "incomplete":
                return {"status": "incomplete"}
            seen_child = True
            index = node["next"]
            if index >= len(tokens):
                return {"status": "incomplete"}
            if tokens[index] == "STRUCT:SEP":
                index += 1
                continue
            if tokens[index] == "STRUCT:CLOSE":
                if not seen_child:
                    return {"status": "invalid"}
                index += 1
                return {"status": "complete", "next": index}
            return {"status": "invalid"}

    def parse_node(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        token = tokens[index]
        if token == "NODE:TERM":
            return parse_term(index)
        if token == "NODE:FRAC":
            return parse_fraction(index)
        if isinstance(token, str) and token.startswith("OP:"):
            return parse_op_node(index)
        return {"status": "invalid"}

    result = parse_node(0)
    if result["status"] == "invalid":
        return False
    if result["status"] == "incomplete":
        return True
    return result["status"] == "complete" and result["next"] == len(tokens)


class NodeValidityPool:
    """Pure-Python replacement for NodeValidityPool that runs completely in-memory."""
    def __init__(self, script_path: str = "", num_workers: int = 1):
        pass

    def mask(self, tokens: List[str], candidate_tokens: List[str]) -> List[bool]:
        return [is_valid_prefix(tokens + [candidate]) for candidate in candidate_tokens]

    def close(self) -> None:
        pass



def flatten_vocab(vocab: Dict[str, Any]) -> Dict[str, int]:
    token_to_id = {}
    for key, value in vocab.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            token_to_id.update(value)
    return token_to_id


def load_vocab(vocab_path: str) -> Dict[str, Any]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    flat = flatten_vocab(raw)
    id_to_token = {idx: token for token, idx in flat.items()}
    return {
        "token_to_id": flat,
        "id_to_token": id_to_token,
        "special": raw.get("special_tokens", {}),
    }


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