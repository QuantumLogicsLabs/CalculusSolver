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
