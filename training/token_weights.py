"""Token-class weighted cross-entropy.

Weights are assigned by token CLASS, derived from the token strings in
vocab.json -- never from assumed token IDs -- so renumbering or extending the
vocabulary cannot silently move a weight onto the wrong token.

    structural  STRUCT:* / NODE:*     1.0
    numerical   COEF:*   / EXP:*      3.5
    variable    VAR:*    / OPVAR:*    4.0
    other       OP:* RULE:* POINT:* and special tokens ([BOS], [EOS], ...)  1.0
    padding     [PAD]                 0.0 (and excluded via ignore_index)

The loss is a weight-normalised mean over valid target positions, i.e. the
same semantics as ``nn.CrossEntropyLoss(weight=..., ignore_index=...)``, so
with uniform weights it is numerically identical to the previous loss. It
additionally never produces NaN when every label in a batch is ignored.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STRUCTURAL = "structural"
NUMERICAL = "numerical"
VARIABLE = "variable"
OTHER = "other"
PADDING = "padding"

CLASS_PREFIXES: Dict[str, Tuple[str, ...]] = {
    STRUCTURAL: ("STRUCT:", "NODE:"),
    NUMERICAL: ("COEF:", "EXP:"),
    VARIABLE: ("VAR:", "OPVAR:"),
}

DEFAULT_CLASS_WEIGHTS: Dict[str, float] = {
    STRUCTURAL: 1.0,
    NUMERICAL: 3.5,
    VARIABLE: 4.0,
    OTHER: 1.0,
}


def classify_token(token: str, pad_token: str = "[PAD]") -> str:
    """Return the token class for a vocab token string."""
    if token == pad_token:
        return PADDING
    for cls, prefixes in CLASS_PREFIXES.items():
        if token.startswith(prefixes):
            return cls
    return OTHER


def build_token_weight_vector(
    vocab_mapping: Mapping[str, int],
    vocab_size: int,
    class_weights: Optional[Mapping[str, float]] = None,
    pad_token: str = "[PAD]",
) -> Tuple[torch.Tensor, Dict]:
    """Build a ``(vocab_size,)`` weight vector from a flat token->id mapping.

    Returns ``(weights, report)``. ``report`` lists how many tokens landed in
    each class and which ids have no vocab token (those get the ``other``
    weight, since they can never be a valid label anyway).
    """
    resolved = dict(DEFAULT_CLASS_WEIGHTS)
    if class_weights:
        unknown = set(class_weights) - set(resolved)
        if unknown:
            raise ValueError(f"Unknown token classes in weights: {sorted(unknown)}")
        resolved.update({k: float(v) for k, v in class_weights.items()})
    for cls, w in resolved.items():
        if not w > 0:
            raise ValueError(f"Class weight for '{cls}' must be > 0, got {w}")

    weights = torch.full((vocab_size,), resolved[OTHER], dtype=torch.float32)
    assigned: Dict[int, str] = {}
    counts = {STRUCTURAL: 0, NUMERICAL: 0, VARIABLE: 0, OTHER: 0, PADDING: 0}

    for token, idx in vocab_mapping.items():
        if not 0 <= idx < vocab_size:
            raise ValueError(f"Token '{token}' has id {idx} outside vocab_size {vocab_size}")
        cls = classify_token(token, pad_token)
        if idx in assigned and assigned[idx] != cls:
            raise ValueError(
                f"Token id {idx} is shared by tokens of different classes "
                f"('{assigned[idx]}' and '{cls}' via '{token}'); fix vocab.json"
            )
        if idx in assigned:
            continue
        assigned[idx] = cls
        counts[cls] += 1
        weights[idx] = 0.0 if cls == PADDING else resolved[cls]

    report = {
        "class_weights": resolved,
        "token_counts": counts,
        "mapped_ids": len(assigned),
        "vocab_size": vocab_size,
        "unmapped_ids": sorted(set(range(vocab_size)) - set(assigned)),
    }
    return weights, report


def per_token_nll(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unreduced NLL per position plus a validity mask, both ``(N,)``.

    A position is valid when its label is neither ``ignore_index`` nor outside
    ``[0, vocab_size)`` (e.g. HuggingFace-style -100 labels).
    """
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    valid = (flat_targets != ignore_index) & (flat_targets >= 0) & (flat_targets < vocab_size)
    safe_targets = flat_targets.masked_fill(~valid, 0)
    nll = F.cross_entropy(flat_logits, safe_targets, reduction="none")
    return nll * valid.to(nll.dtype), valid


class TokenWeightedCrossEntropy(nn.Module):
    """Weight-normalised cross-entropy over valid target positions."""

    def __init__(
        self,
        vocab_size: int,
        ignore_index: int,
        token_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        if token_weights is None:
            token_weights = torch.ones(vocab_size, dtype=torch.float32)
        if token_weights.shape != (vocab_size,):
            raise ValueError(
                f"token_weights must have shape ({vocab_size},), got {tuple(token_weights.shape)}"
            )
        self.register_buffer("token_weights", token_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass
