"""Warmup + cosine LR schedule: one implementation, validated against config.

train.py builds its LambdaLR from ``build_lr_lambda`` below, and
``verify_schedule`` replays the very same function on a throwaway optimizer to
prove the configured values (warmup steps, cosine decay, min_lr_ratio) are
what the scheduler actually does -- before any real training step is taken.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float) -> Callable[[int], float]:
    """Linear warmup to 1.0, then cosine decay down to ``min_lr_ratio``."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return lr_lambda


def resolve_schedule(cfg: Dict, steps_per_epoch: int, epochs: int) -> Dict:
    """Read scheduler settings from config and validate them."""
    lr_decay = cfg.get("lr_decay", {}) or {}
    decay_type = lr_decay.get("type", "cosine")
    if decay_type != "cosine":
        raise ValueError(
            f"lr_decay.type={decay_type!r} is not implemented; the scheduler only "
            "supports 'cosine'. Refusing to train with a mislabeled schedule."
        )
    warmup_steps = int(cfg.get("warmup_steps", 1000))
    min_lr_ratio = float(lr_decay.get("min_lr_ratio", 0.1))
