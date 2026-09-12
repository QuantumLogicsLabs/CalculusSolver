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

# NOTE: NodeValidityPool, flatten_vocab, load_vocab, is_valid_prefix and
# is_complete live in inference/grammar.py (torch-free) so the ONNX
# deployment path (deployment/onnx_beam_search.py) can reuse them without
# importing torch.
#
# WARNING: deployment/onnx_beam_search.py mirrors this file's search loop.
# The grammar-side termination fix reaches it automatically via the shared
# import, but the loop changes here do NOT -- mirror them, or the ONNX path
# keeps the never-terminating behaviour described below.


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

    CORRECTION: an earlier version of this docstring attributed the 0%
    eval accuracy to exposure bias, and raised beam_size to compensate.
    That diagnosis was wrong and widening the beam could not have fixed it.
    The decoder had no legal way to terminate at any beam width: the grammar
    rejected every continuation of a closed AST, [EOS] included, so a beam
    that produced the correct answer had its entire candidate row masked to
    -inf and was dropped from the search. Only degenerate, still-open beams
    survived, which is why every prediction came back truncated and failed
    to deserialize. See inference/grammar.is_valid_prefix and
    tests/unit/test_beam_search.py::test_oracle_reaches_solved -- an oracle
    model that is certain about every token now recovers gold exactly, at
    every beam width from 1 to 8.

    Beam width remains a real accuracy/CPU trade-off, but it should be
    re-tuned against a checkpoint now that termination works; the old
    0%/5.7% figures measured the termination bug, not the search width.
    """
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

    beams = [{"tokens": [bos_id], "score": 0.0, "finished": False}]
    completed = []

    for _ in range(max_len):
        candidates = []
        for beam in beams:
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
            invalid_mask = torch.tensor([not v for v in mask], device=device)
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
