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


