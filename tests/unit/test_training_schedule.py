"""Training configuration and LR scheduler: intended values and real behaviour."""

import json
import math
from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR

from training.schedule import (
    build_lr_lambda,
    lr_at,
    resolve_schedule,
    scheduler_matches_expected,
    verify_schedule,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def test_config_holds_the_intended_schedule(cfg):
    assert cfg["epochs"] == 15
    assert cfg["warmup_steps"] == 1500
    assert cfg["lr_decay"]["type"] == "cosine"
    assert cfg["lr_decay"]["min_lr_ratio"] == 0.05


def test_config_declares_seed_and_loss_sections(cfg):
    assert isinstance(cfg["seed"], int)
    assert cfg["token_weighted_loss"]["weights"] == {
        "structural": 1.0, "numerical": 3.5, "variable": 4.0, "other": 1.0
    }
    assert cfg["contrast"]["pairs_per_batch"] >= 1


def test_scheduler_implements_the_configured_values(cfg):
    resolved = resolve_schedule(cfg, steps_per_epoch=2110, epochs=cfg["epochs"])
    assert resolved["warmup_steps"] == 1500
    assert resolved["min_lr_ratio"] == 0.05
    assert resolved["total_steps"] == 15 * 2110

    report = verify_schedule(resolved)
    assert report["verified"]
    assert report["lr_step_0"] == 0.0
    assert report["lr_end_of_warmup"] == pytest.approx(cfg["learning_rate"])
    assert report["lr_final"] == pytest.approx(cfg["learning_rate"] * 0.05)


def test_cosine_midpoint_is_halfway_between_peak_and_floor():
