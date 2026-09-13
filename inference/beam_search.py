import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import torch

from inference.decoding import (
    content_token_ids,
    escape_token_ids,
    longest_complete_prefix,
    penalised_logit,
    repeat_blocked_tokens,
    repetition_penalty_targets,
    strip_terminator,
)
from inference.grammar import (
    NodeValidityPool,
    flatten_vocab,
    is_complete,
    is_valid_prefix,
    load_vocab,
)

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


def _apply_repetition_penalty(
    logits: "torch.Tensor",
    tokens: List[int],
    penalty: float,
    min_count: int,
    exempt: frozenset,
    content_ids: frozenset,
) -> "torch.Tensor":
    """Apply the CTRL-style penalty to raw logits, BEFORE the grammar mask
    and before log_softmax, so the softmax renormalises with it in place.

    Policy (which tokens, at what count) lives in inference/decoding.py so
    the ONNX path applies exactly the same rule; only the tensor arithmetic
    is here.
    """
    if penalty == 1.0:
        return logits
    targets = repetition_penalty_targets(tokens, min_count, exempt, content_ids)
    if not targets:
        return logits

    penalised = logits.clone()
    for token_id in targets:
        if 0 <= token_id < penalised.size(0):
            penalised[token_id] = penalised_logit(
                float(penalised[token_id]), penalty
            )
    return penalised


# Hard guards and terminator handling are pure policy -- shared verbatim with
# deployment/onnx_beam_search.py via inference/decoding.py. Aliased here so
# this module's existing call sites and tests keep working.
_repeat_blocked_tokens = repeat_blocked_tokens
_strip_terminator = strip_terminator


def beam_search(
    model,
    src_tokens: torch.Tensor,
    vocab_map: Dict[str, Any],
    beam_size: int = 5,
    max_len: int = 32,
    node_pool: Optional[NodeValidityPool] = None,
    src_positions: Optional[torch.Tensor] = None,
    parent_child_pairs: Optional[torch.Tensor] = None,
    max_token_run: int = 4,
    no_repeat_ngram_size: int = 2,
    repetition_penalty: float = 1.2,
    repetition_min_count: int = 4,
) -> Dict[str, Any]:
    device = src_tokens.device
    vocab = vocab_map["token_to_id"]
    id_to_token = vocab_map["id_to_token"]
    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]

    if node_pool is None:
        node_pool = NodeValidityPool()

    # Never repeat-guard the escape hatch. STRUCT:CLOSE legitimately runs to
    # 3 in real data (30,000 targets close three structures at once) and is,
    # with [EOS], the only way a beam terminates. Both are already bounded by
    # the grammar -- an unmatched close is ungrammatical -- so they cannot
    # run away on their own.
    # Never suppress the escape hatch, and never let repetition control see
    # structural scaffolding. Both rules live in inference/decoding.py so the
    # ONNX mirror cannot drift from them again.
    exempt_ids = escape_token_ids(vocab)
    content_ids = content_token_ids(id_to_token)

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

            # Soft repetition penalty (task 1) -- on RAW logits, before the
            # grammar mask and before log_softmax.
            next_logits = _apply_repetition_penalty(
                next_logits,
                current_tokens,
                repetition_penalty,
                repetition_min_count,
                exempt_ids,
                content_ids,
            )

            mask = node_pool.mask(validity_tokens, all_candidate_tokens)
            vocab_len = next_logits.size(0)
            padded_mask = mask + [True] * max(0, vocab_len - len(mask))
            invalid_mask = torch.tensor(
                [not v for v in padded_mask[:vocab_len]], device=device
            )
            safe_logits = next_logits.masked_fill(invalid_mask, float("-inf"))

            # Hard repeat guard (task 2b). Applied as a mask, not a penalty:
            # a degenerate beam must be unable to continue, not merely
            # expensive. Deliberately NOT "force the beam to terminate" --
            # terminating a grammatically incomplete beam emits exactly the
            # truncated, undeserializable sequence this is meant to prevent.
            # Masking instead forces the beam to make different progress,
            # which in practice means closing the structure it opened.
            blocked = _repeat_blocked_tokens(
                current_tokens,
                max_token_run,
                no_repeat_ngram_size,
                exempt_ids,
                content_ids,
            )
            if blocked:
                guarded = safe_logits.clone()
                for blocked_id in blocked:
                    if 0 <= blocked_id < guarded.size(0):
                        guarded[blocked_id] = float("-inf")
                # Liveness: the guard must never be what empties a beam's
                # candidate set. If it would, drop it for this step and let
                # the grammar mask stand alone.
                if not bool(torch.isinf(guarded).all()):
                    safe_logits = guarded

            valid_count = int(torch.isfinite(safe_logits).sum().item())

            if valid_count == 0 or bool(torch.isinf(safe_logits).all()):
                # No legal continuation exists. If this beam has already
                # closed a complete AST then it IS an answer -- retire it
                # rather than discarding it.
                #
                # Discarding it here is what made every input return
                # status="partial": is_valid_prefix rejected every
                # continuation of a closed AST, so the mask went all -inf
                # the moment a beam got the answer right, and that beam was
                # deleted while degenerate still-open beams survived to fill
                # the beam slots. grammar.EOS_TOKEN handling now makes [EOS]
                # reachable, so this is a safety net rather than the normal
                # path -- but it must not silently drop a finished beam.
                if is_complete(validity_tokens):
                    completed.append({
                        "tokens": current_tokens + [eos_id],
                        "score": beam["score"],
                        "finished": True,
                    })
                continue

            log_probs = torch.log_softmax(safe_logits, dim=-1)

            # topk must never reach past the legal candidates. Only 11-36 of
            # the 124 vocab slots are grammatically valid at a typical step,
            # so asking for beam_size entries unconditionally makes torch.topk
            # pad the selection with masked -inf tokens -- and ties there break
            # by index, so [EOS] (id 2) gets picked first. That silently
            # manufactured "finished" beams carrying a truncated AST and a
            # score of -inf, which then won the final sort because every
            # -inf compares equal.
            k = min(beam_size, valid_count, safe_logits.size(0))
            topk = torch.topk(log_probs, k)
            for score, token_id in zip(topk.values.tolist(), topk.indices.tolist()):
                if not math.isfinite(score):
                    continue  # belt and braces: never extend a beam illegally
                new_beam = {
                    "tokens": current_tokens + [int(token_id)],
                    "score": beam["score"] + float(score),
                    "finished": int(token_id) == eos_id,
                }
                # Retire finished beams immediately instead of leaving them
                # to compete in the top-k. A finished beam's score is frozen
                # while open beams keep accumulating log-probs, so a correct
                # finished beam could otherwise be evicted by a longer,
                # still-open one before it is ever collected.
                target = completed if new_beam["finished"] else candidates
                target.append(new_beam)

        if not candidates:
            break

        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:beam_size]

        # Scores are sums of log-probs, so an open beam's score can only get
        # worse. Once the best open beam is already no better than the best
        # completed one, every future completion derived from it is bounded
        # by that same score -- so nothing still open can overtake the answer
        # we already have. Exiting here is lossless, and it is what stops the
        # search grinding out the full max_len budget on every problem.
        # A count-based cap (stop once completed >= beam_size) was deliberately
        # not used: it can fire while a better answer is still forming.
        # NOTE: this relies on raw (un-normalised) scoring. If a GNMT-style
        # length penalty is added, revisit -- a longer beam can then improve
        # its normalised score and this exit would become unsound.
        if completed and beams[0]["score"] <= max(c["score"] for c in completed):
            break

    best = sorted(completed, key=lambda x: x["score"], reverse=True)[0] if completed else (
        beams[0] if beams else {"tokens": [bos_id], "score": 0.0, "finished": False}
    )

    status = "solved" if best["finished"] else "partial"
    tokens = _strip_terminator(best["tokens"], eos_id)

    if not best["finished"]:
        # No beam terminated. Hand back the longest closed sub-AST rather than
        # a truncated sequence the deserializer cannot parse. Status stays
        # "partial" -- this is a salvaged answer, not a solved one.
        salvaged = longest_complete_prefix(tokens, id_to_token)
        if salvaged:
            tokens = salvaged

    return {"tokens": tokens, "score": best["score"], "status": status}
