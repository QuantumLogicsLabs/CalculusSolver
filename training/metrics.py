"""Exact-match evaluation by operation, via batched greedy decoding.

Token accuracy is a weak signal here: one wrong VAR/COEF/EXP token makes the
whole derivative wrong while barely moving per-token accuracy. These helpers
score whole sequences.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def _strip(tokens: Sequence[int], eos_id: int, pad_id: int) -> List[int]:
    """Cut at the first EOS (exclusive); drop trailing padding."""
    out: List[int] = []
    for t in tokens:
        if t == eos_id:
            break
        out.append(int(t))
    while out and out[-1] == pad_id:
        out.pop()
    return out


@torch.no_grad()
def greedy_decode_batch(model, src_seq, bos_id: int, eos_id: int, pad_id: int, max_len: int):
    """Greedy-decode a batch; returns ``(B, <=max_len)`` generated ids (no BOS)."""
    batch = src_seq.size(0)
    generated = torch.full((batch, 1), bos_id, dtype=torch.long, device=src_seq.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=src_seq.device)
    for _ in range(max_len):
        logits, _, _ = model(src_seq, generated, true_rule_ids=None)
        nxt = logits[:, -1, :].argmax(dim=-1)
        nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
        generated = torch.cat([generated, nxt.unsqueeze(1)], dim=1)
        finished = finished | (nxt == eos_id)
        if bool(finished.all()):
            break
    return generated[:, 1:]


def select_indices(dataset_ops: Sequence[str], max_examples: Optional[int]) -> List[int]:
    """Deterministic, operation-stratified subset (all rows when max is None)."""
    n = len(dataset_ops)
    if max_examples is None or max_examples >= n:
        return list(range(n))
    by_op: Dict[str, List[int]] = {}
    for i, op in enumerate(dataset_ops):
        by_op.setdefault(op, []).append(i)
    per_op = max(1, max_examples // max(len(by_op), 1))
    chosen: List[int] = []
    for op in sorted(by_op):
        rows = by_op[op]
        if len(rows) <= per_op:
            chosen.extend(rows)
        else:
            stride = len(rows) / per_op
            chosen.extend(rows[int(k * stride)] for k in range(per_op))
    return sorted(chosen)


