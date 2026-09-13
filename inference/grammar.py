"""
Torch-free SLaNg grammar and vocab helpers, split out of inference/beam_search.py.

Purpose: the ONNX deployment path (deployment/onnx_beam_search.py,
deployment/onnx_solve.py) must never import torch -- that's the entire
point of Option A (ONNX export) as a fix for Vercel's ~250MB serverless
size limit. Everything in this file is pure Python / stdlib only, so both
the PyTorch path (inference/beam_search.py) and the ONNX path
(deployment/onnx_beam_search.py) can import it without pulling torch in.
"""

import json
from typing import Any, Dict, List

# Sequence terminator. Kept here (not imported from a vocab) so this module
# stays pure-stdlib for the ONNX path.
EOS_TOKEN = "[EOS]"

# Structural bounds. Without these the grammar admits infinitely long valid
# prefixes (unbounded OPVAR runs, unbounded node nesting), so a degenerate
# decoder can out-survive max_len and be hard-truncated into an
# undeserializable sequence -- exactly the failure the repeat guards exist
# to prevent, and which they cannot prevent on their own because a cycling
# model never repeats a token or bigram.
#
# Both limits are set from data, with large margin:
#   max contiguous OPVAR run in 155,000 targets : 1   -> cap 4
#   max node nesting depth in 155,000 targets   : 3   -> cap 8
MAX_OPVARS_PER_OP = 4
MAX_NESTING_DEPTH = 8
#   max NODE:TERM in one term-list                : 2   -> cap 8
MAX_SIBLINGS = 8


def _is_numeric_token(token: object, prefix: str) -> bool:
    """True if token is "<prefix><number>".

    The vocabulary carries OOV placeholders -- COEF:OTHER and EXP:OTHER --
    that match the prefix but are not numbers. deserialize_slang_math does
    float(value) on them and raises, so a decoder allowed to emit one
    produces an answer that cannot be parsed back. Found by fuzzing the
    grammar mask against the deserializer; verified zero occurrences in
    155,000 real targets, so rejecting them here loses nothing.
    """
    if not isinstance(token, str) or not token.startswith(prefix):
        return False
    try:
        float(token[len(prefix):])
    except ValueError:
        return False
    return True


def _parse_status(tokens: List[str]) -> str:
    """Parse a SLaNg token sequence and classify it.

    Returns exactly one of:
      "invalid"    -- violates the grammar, or closes an AST and then has
                      leftover tokens after it
      "incomplete" -- a well-formed prefix; more tokens are expected
      "complete"   -- exactly one fully-closed AST, nothing left over

    First token, if present, may be a RULE:xxx token (SimpleCalculusModel
    prepends one) -- skip it before running the AST grammar check.
    """
    if not tokens:
        return "incomplete"

    check_tokens = tokens
    if tokens[0].startswith("RULE:"):
        check_tokens = tokens[1:]
        if not check_tokens:
            return "incomplete"

    tokens = check_tokens

    def parse_term(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "NODE:TERM":
            return {"status": "invalid"}
        index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if not _is_numeric_token(tokens[index], "COEF:"):
            return {"status": "invalid"}
        index += 1
        seen_vars = set()
        while index < len(tokens):
            token = tokens[index]
            if token.startswith("VAR:"):
                # A term stores its variables as a JSON dict --
                # {"coeff": 3, "var": {"x": 2}} -- so a variable can appear
                # at most ONCE per term by construction (x * x is x^2, not
                # two VAR:x entries). A repeated VAR is therefore not merely
                # unlikely, it is un-representable SLaNg: verified zero
                # occurrences across all 155,000 targets in data/splits/.
                #
                # Rejecting it here makes the observed degenerate decode
                # loop -- VAR:u EXP:-3 VAR:u EXP:-3 ... -- structurally
                # impossible at the grammar-mask level, before any scoring
                # runs. That is strictly stronger than a soft repetition
                # penalty, which can only make the loop expensive.
                if token in seen_vars:
                    return {"status": "invalid"}
                seen_vars.add(token)
                index += 1
                if index >= len(tokens):
                    return {"status": "incomplete"}
                if not _is_numeric_token(tokens[index], "EXP:"):
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
        siblings = 0
        while True:
            node = parse_node(current)
            if node["status"] == "invalid":
                return {"status": "invalid"}
            if node["status"] == "incomplete":
                return {"status": "incomplete"}
            siblings += 1
            if siblings > MAX_SIBLINGS:
                return {"status": "invalid"}
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
        for expected in ["STRUCT:OPEN", "STRUCT:NUMI", "STRUCT:OPEN"]:
            if index >= len(tokens):
                return {"status": "incomplete"}
            if tokens[index] != expected:
                return {"status": "invalid"}
            index += 1
        numerator = parse_term_list(index)
        if numerator["status"] != "complete":
            return numerator
        index = numerator["next"]
        for expected in ["STRUCT:CLOSE", "STRUCT:SEP", "STRUCT:DENO", "STRUCT:OPEN"]:
            if index >= len(tokens):
                return {"status": "incomplete"}
            if tokens[index] != expected:
                return {"status": "invalid"}
            index += 1
        denominator = parse_term_list(index)
        if denominator["status"] != "complete":
            return denominator
        index = denominator["next"]
        for expected in ["STRUCT:CLOSE", "STRUCT:CLOSE"]:
            if index >= len(tokens):
                return {"status": "incomplete"}
            if tokens[index] != expected:
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

        # Optional decorators, in the FIXED order serialize_op_node emits
        # them (tokenizer/slang_serializer.py): COEF (scale/sign, e.g.
        # -sin(x)), then EXP (power, e.g. sec^2(x)), then POINT
        # (tangent_line's x0). At most one of each.
        #
        # These were missing entirely, which made every decorated op-node
        # ungrammatical -- roughly 13% of real targets (OP:cos COEF:-8 ...,
        # OP:sec COEF:-9 EXP:2 ...). Because this grammar drives the decode
        # mask, those answers were not merely scored badly, they were
        # unreachable. The order is fixed by design ("to keep parse_op_node's
        # fixed decorator order unambiguous"), so enforcing the order here
        # keeps the mask tight rather than admitting permutations that
        # deserialize_slang_math would then reject.
        for decorator_prefix in ("COEF:", "EXP:", "POINT:"):
            if _is_numeric_token(
                tokens[index] if index < len(tokens) else None, decorator_prefix
            ):
                index += 1

        # NOTE: an op-node's variables come from n["var"] plus the n["vars"]
        # LIST, so unlike a term's dict a repeated OPVAR is representable
        # here -- this loop is deliberately left unbounded. Bounding a
        # runaway OPVAR run is the hard repeat-guard's job (task 2b), not
        # the grammar's.
        opvar_count = 0
        while (
            index < len(tokens)
            and isinstance(tokens[index], str)
            and tokens[index].startswith("OPVAR:")
        ):
            opvar_count += 1
            if opvar_count > MAX_OPVARS_PER_OP:
                return {"status": "invalid"}
            index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:OPEN":
            return {"status": "invalid"}
        index += 1
        seen_child = False
        children = 0
        while True:
            node = parse_node(index)
            if node["status"] == "invalid":
                return {"status": "invalid"}
            if node["status"] == "incomplete":
                return {"status": "incomplete"}
            seen_child = True
            children += 1
            if children > MAX_SIBLINGS:
                return {"status": "invalid"}
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

    def parse_gradient_node(index: int) -> dict:
        """NODE:GRADIENT OPEN OPVAR:<v> node (SEP OPVAR:<v> node)* CLOSE

        Mirrors serialize_gradient_node / parse_gradient_node in
        tokenizer/slang_serializer.py. This node type had no branch at all,
        so every gradient answer was ungrammatical and therefore unreachable
        by the decoder -- which is why gradient scores 0/50 in
        docs/EVAL_RESULTS.md regardless of model quality.

        A gradient serializes a {var: expr, ...} DICT, so a variable can
        appear at most once -- the same argument that bounds VAR inside a
        term. Verified: zero duplicate OPVARs across the dataset.
        """
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "NODE:GRADIENT":
            return {"status": "invalid"}
        index += 1
        if index >= len(tokens):
            return {"status": "incomplete"}
        if tokens[index] != "STRUCT:OPEN":
            return {"status": "invalid"}
        index += 1

        seen_vars = set()
        seen_child = False
        while True:
            if index >= len(tokens):
                return {"status": "incomplete"}
            if tokens[index] == "STRUCT:CLOSE":
                if not seen_child:
                    return {"status": "invalid"}
                index += 1
                return {"status": "complete", "next": index}

            var_token = tokens[index]
            if not isinstance(var_token, str) or not var_token.startswith("OPVAR:"):
                return {"status": "invalid"}
            if var_token in seen_vars:
                return {"status": "invalid"}
            seen_vars.add(var_token)
            index += 1

            child = parse_node(index)
            if child["status"] != "complete":
                return child
            seen_child = True
            index = child["next"]

            if index >= len(tokens):
                return {"status": "incomplete"}
            if tokens[index] == "STRUCT:SEP":
                index += 1
                continue
            if tokens[index] == "STRUCT:CLOSE":
                index += 1
                return {"status": "complete", "next": index}
            return {"status": "invalid"}

    depth = [0]

    def parse_node(index: int) -> dict:
        if index >= len(tokens):
            return {"status": "incomplete"}
        # Depth cap: without it, NODE:FRAC (and OP:) nest without bound, so
        # a decoder can keep opening structures forever and still be emitting
        # a "valid prefix".
        if depth[0] >= MAX_NESTING_DEPTH:
            return {"status": "invalid"}
        depth[0] += 1
        try:
            token = tokens[index]
            if token == "NODE:TERM":
                return parse_term(index)
            if token == "NODE:FRAC":
                return parse_fraction(index)
            if token == "NODE:GRADIENT":
                return parse_gradient_node(index)
            if isinstance(token, str) and token.startswith("OP:"):
                return parse_op_node(index)
            return {"status": "invalid"}
        finally:
            depth[0] -= 1

    result = parse_node(0)
    if result["status"] == "invalid":
        return "invalid"
    if result["status"] == "incomplete":
        return "incomplete"
    # A closed AST followed by leftover tokens is not a valid sequence.
    return "complete" if result["next"] == len(tokens) else "invalid"


def is_complete(tokens: List[str]) -> bool:
    """True if tokens form exactly one fully-closed SLaNg AST.

    A trailing EOS_TOKEN is tolerated, so that is_complete(x) and
    is_valid_prefix(x + [EOS_TOKEN]) always agree.
    """
    if tokens and tokens[-1] == EOS_TOKEN:
        tokens = tokens[:-1]
    return bool(tokens) and _parse_status(tokens) == "complete"


def is_valid_prefix(tokens: List[str]) -> bool:
    """True if tokens are a valid prefix of (or exactly) one SLaNg AST.

    EOS_TOKEN is accepted only as the final token of an already-complete
    AST. That case is load-bearing: every continuation of a closed AST --
    EOS_TOKEN included -- parses as "invalid", so without this the grammar
    mask in beam_search goes all -inf the moment a beam gets the answer
    right, and the finished beam is discarded instead of returned. That is
    why every input returned status="partial" and a truncated,
    undeserializable sequence. See tests/unit/test_beam_search.py.
    """
    if not tokens:
        return True
    if tokens[-1] == EOS_TOKEN:
        body = tokens[:-1]
        return bool(body) and _parse_status(body) == "complete"
    return _parse_status(tokens) != "invalid"


class NodeValidityPool:
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