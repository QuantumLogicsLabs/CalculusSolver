"""Binding-contrast minibatches.

Contrast rows come in pairs that share one multivariable expression and
differ only in the differentiation variable (and therefore the target).
Each training minibatch is topped up with a few complete pairs, so the model
sees both sides of the same expression in the same gradient step and has to
use the target variable -- not the expression alone -- to pick the answer.

Rows live in ``data/splits/contrast_{train,val}.jsonl`` (written by
``python problem_generator.py --contrast-only``) and carry a ``contrast_group``
id shared by the two members of a pair.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import default_collate


def build_pair_index(groups: Sequence) -> List[Tuple[int, int]]:
    """Map a per-row ``contrast_group`` list to ``(row_i, row_j)`` pairs.

    Rows with a missing group id are ignored. A group that does not contain
    exactly two rows is an error: a half-pair would silently defeat the point.
    """
    members: Dict = defaultdict(list)
    for idx, group in enumerate(groups):
        if group is not None:
            members[group].append(idx)
    pairs = []
    for group, rows in members.items():
        if len(rows) != 2:
            raise ValueError(f"contrast_group {group!r} has {len(rows)} rows, expected exactly 2")
        pairs.append((rows[0], rows[1]))
    return pairs


def merge_batches(main: Dict, extra: Dict) -> Dict:
    """Concatenate two collated batches along the batch dimension."""
    merged = {}
    for key, value in main.items():
        other = extra[key]
        if isinstance(value, torch.Tensor):
            merged[key] = torch.cat([value, other], dim=0)
        else:
            merged[key] = list(value) + list(other)
    return merged


class ContrastBatchSource:
    """Deterministic supply of complete contrast pairs, cycling each epoch."""

    def __init__(self, dataset, pairs_per_batch: int, seed: int):
        if pairs_per_batch <= 0:
            raise ValueError("pairs_per_batch must be positive")
        self.dataset = dataset
