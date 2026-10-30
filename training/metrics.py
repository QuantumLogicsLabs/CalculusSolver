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


def evaluate_exact_match(
    model,
    dataset,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    device="cpu",
    max_gen_len: int = 80,
    batch_size: int = 128,
    max_examples: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> Dict:
    """Greedy exact-match accuracy overall and per operation.

    ``dataset[i]`` must return a dict with ``src_seq``, ``tgt_out_seq`` and
    ``op``. The target is ``tgt_out_seq[1:]`` up to the first EOS, matching
    train.py's teacher-forcing alignment.
    """
    was_training = model.training
    model.eval()

    if indices is None:
        indices = select_indices(dataset.ops, max_examples)
    indices = list(indices)

    correct_flags: Dict[int, bool] = {}
    by_op_correct: Dict[str, int] = {}
    by_op_total: Dict[str, int] = {}
    tok_correct = 0
    tok_total = 0
    gen_len_total = 0

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        items = [dataset[i] for i in chunk]
        src = torch.stack([it["src_seq"] for it in items]).to(device)
        generated = greedy_decode_batch(model, src, bos_id, eos_id, pad_id, max_gen_len).cpu().tolist()

        for idx, item, gen in zip(chunk, items, generated):
            target = _strip(item["tgt_out_seq"].tolist()[1:], eos_id, pad_id)
            pred = _strip(gen, eos_id, pad_id)
            ok = pred == target
            correct_flags[idx] = ok
            op = item["op"]
            by_op_total[op] = by_op_total.get(op, 0) + 1
            by_op_correct[op] = by_op_correct.get(op, 0) + int(ok)
            n = min(len(pred), len(target))
            tok_correct += sum(1 for a, b in zip(pred[:n], target[:n]) if a == b)
            tok_total += n
            gen_len_total += len(pred)

    if was_training:
        model.train()

    total = len(indices)
    correct = sum(correct_flags.values())
    return {
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        },
        "by_op": {
            op: {
                "correct": by_op_correct.get(op, 0),
                "total": by_op_total[op],
                "accuracy": by_op_correct.get(op, 0) / by_op_total[op],
            }
            for op in sorted(by_op_total)
        },
        "token_accuracy": tok_correct / max(tok_total, 1),
        "avg_gen_len": gen_len_total / max(total, 1),
        "correct_flags": correct_flags,
    }

