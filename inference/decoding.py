"""Torch-free decoding policy shared by the PyTorch and ONNX beam searches.

inference/beam_search.py (torch) and deployment/onnx_beam_search.py
(numpy/onnxruntime) are meant to be line-for-line equivalents. They drifted:
every termination and repetition fix landed in the torch path only, so the
ONNX path kept discarding finished beams and could never emit [EOS].

Everything here is pure stdlib and returns plain Python sets, so each caller
applies the policy in its own array idiom (torch.Tensor vs np.ndarray)
without this module ever importing either. That keeps the ONNX deployment
bundle free of torch, which is the entire point of that path.
"""

from collections import Counter
from typing import Any, Dict, FrozenSet, List, Set

from inference.grammar import EOS_TOKEN, is_complete

# Structural scaffolding and specials. These repeat constantly and
# legitimately -- a minimal fraction carries STRUCT:OPEN and STRUCT:CLOSE
# four times each, and NODE:TERM/COEF: recur once per term -- so no
# repetition control may ever apply to them.
_NON_CONTENT_PREFIXES = ("STRUCT:", "NODE:", "[")


def content_token_ids(id_to_token: Dict[int, str]) -> FrozenSet[int]:
    """Ids of tokens carrying content (OP:, OPVAR:, VAR:, EXP:, COEF:, ...)."""
    return frozenset(
        token_id
        for token_id, token in id_to_token.items()
        if not token.startswith(_NON_CONTENT_PREFIXES)
    )


def escape_token_ids(token_to_id: Dict[str, int]) -> FrozenSet[int]:
    """Ids that must never be suppressed: the ways a beam terminates.

    Suppressing either re-creates the original defect -- a beam with no legal
    way to finish, hard-truncated into an undeserializable sequence.
    """
    return frozenset(
        i for i in (token_to_id.get(EOS_TOKEN), token_to_id.get("STRUCT:CLOSE"))
        if i is not None
    )


def repeat_blocked_tokens(
    tokens: List[int],
    max_token_run: int,
    no_repeat_ngram_size: int,
    exempt: FrozenSet[int],
    content_ids: FrozenSet[int],
) -> Set[int]:
    """Token ids that must not be allowed to extend `tokens` (hard guards).

    * consecutive-run cap -- block the trailing token once it has already
      repeated `max_token_run` times. Real targets never exceed a run of 3,
      and only STRUCT:CLOSE reaches it, so 4 leaves margin.
    * no-repeat-ngram -- block any token recreating an n-gram already in this
      beam, restricted to n-grams made ENTIRELY of content tokens. That
      restriction is load-bearing: 77.4% of real targets repeat some bigram,
      but zero of the 155,000 repeat a content-only one.
    """
    blocked: Set[int] = set()

    if max_token_run > 0 and tokens:
        tail = tokens[-1]
        if tail not in exempt:
            run = 0
            for token in reversed(tokens):
                if token != tail:
                    break
                run += 1
            if run >= max_token_run:
                blocked.add(tail)

    n = no_repeat_ngram_size
    if n and n > 1 and len(tokens) >= n - 1:
        prefix = tuple(tokens[len(tokens) - (n - 1):])
        if all(token in content_ids for token in prefix):
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i + n])
                if (
                    gram[:-1] == prefix
                    and gram[-1] in content_ids
                    and gram[-1] not in exempt
                ):
                    blocked.add(gram[-1])

    return blocked


def repetition_penalty_targets(
    tokens: List[int],
    min_count: int,
    exempt: FrozenSet[int],
    content_ids: FrozenSet[int],
) -> Set[int]:
    """Token ids whose logit should be penalised (soft guard).

    min_count=4 is set from data: across all 155,000 targets the most any
    single content token occurs within one target is 4, and only in 4 of
    them -- so the penalty cannot fire on a correct answer.
    """
    if min_count <= 0 or not tokens:
        return set()
    counts = Counter(
        token for token in tokens if token in content_ids and token not in exempt
    )
    return {token for token, n in counts.items() if n >= min_count}


def penalised_logit(score: float, penalty: float) -> float:
    """CTRL-style repetition penalty (Keskar et al. 2019).

    Divide a positive logit, multiply a negative one. The sign branch is what
    keeps the transform discouraging for both signs -- dividing a negative
    logit would make an already-repeated token MORE attractive.
    """
    return score / penalty if score > 0 else score * penalty


def strip_terminator(tokens: List[int], eos_id: int) -> List[int]:
    """Drop the trailing [EOS] from a finished sequence.

    deserialize_slang_math raises "Extra tokens found after deserialization"
    on any token following a closed AST, [EOS] included. Neither solve.py nor
    onnx_solve.py strips it, so the terminator must not be handed on.
    """
    while tokens and tokens[-1] == eos_id:
        tokens = tokens[:-1]
    return tokens


def longest_complete_prefix(
    tokens: List[int], id_to_token: Dict[int, str]
) -> List[int]:
    """Longest prefix of `tokens` that forms a complete SLaNg AST, or [] if none.

    Safety net for the case where no beam terminated. Without it the caller
    receives the raw truncated sequence, which fails deserialization with
    "reached end of tokens" -- the degenerate beam has out-survived max_len
    and been hard-truncated into an undeserializable sequence, which is
    precisely what the repeat guards are meant to prevent and cannot on their
    own (a cycling model repeats neither a token nor a bigram).

    Returning a closed sub-AST instead means the caller always gets something
    parseable. The result is still reported as status="partial" -- it is a
    truncated answer, not a clean termination -- so nothing downstream treats
    it as a solved problem.
    """
    if not tokens:
        return []
    start = 1 if id_to_token.get(tokens[0]) == "[BOS]" else 0
    for end in range(len(tokens), start, -1):
        strings = [id_to_token[t] for t in tokens[start:end] if t in id_to_token]
        if is_complete(strings):
            return tokens[:end]
    return []
