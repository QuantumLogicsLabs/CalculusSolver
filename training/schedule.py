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
    base_lr = float(cfg["learning_rate"])
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"lr_decay.min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    if base_lr <= 0:
        raise ValueError(f"learning_rate must be > 0, got {base_lr}")

    total_steps = int(epochs) * int(steps_per_epoch)
    resolved = {
        "type": decay_type,
        "base_lr": base_lr,
        "warmup_steps": warmup_steps,
        "min_lr_ratio": min_lr_ratio,
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": total_steps,
        "warmup_fraction": warmup_steps / total_steps if total_steps else None,
        "warnings": [],
    }
    if total_steps <= warmup_steps:
        resolved["warnings"].append(
            f"total_steps ({total_steps}) <= warmup_steps ({warmup_steps}): "
            "the LR never leaves warmup and the cosine decay is never reached"
        )
    return resolved


def lr_at(resolved: Dict, step: int) -> float:
    fn = build_lr_lambda(resolved["warmup_steps"], resolved["total_steps"], resolved["min_lr_ratio"])
    return resolved["base_lr"] * fn(step)


def verify_schedule(resolved: Dict, tol: float = 1e-9) -> Dict:
    """Replay the scheduler on a dummy optimizer and assert the intended shape.

