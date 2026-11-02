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
    resolved = {"base_lr": 1.0, "warmup_steps": 10, "min_lr_ratio": 0.1, "total_steps": 110}
    mid = lr_at(resolved, 60)
    assert mid == pytest.approx(0.1 + 0.9 * 0.5)


def test_warmup_is_linear():
    fn = build_lr_lambda(100, 1000, 0.05)
    assert fn(0) == 0.0
    assert fn(50) == pytest.approx(0.5)
    assert fn(100) == pytest.approx(1.0)


def test_lr_never_rises_after_warmup_and_stays_at_floor_past_the_end():
    fn = build_lr_lambda(20, 200, 0.05)
    values = [fn(s) for s in range(20, 400)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
    assert fn(10_000) == pytest.approx(0.05)


def test_non_cosine_decay_is_refused(cfg):
    bad = {**cfg, "lr_decay": {"type": "linear", "min_lr_ratio": 0.05}}
    with pytest.raises(ValueError):
        resolve_schedule(bad, 100, 3)


def test_invalid_values_are_refused(cfg):
    with pytest.raises(ValueError):
        resolve_schedule({**cfg, "lr_decay": {"type": "cosine", "min_lr_ratio": 1.5}}, 100, 3)
    with pytest.raises(ValueError):
        resolve_schedule({**cfg, "warmup_steps": -1}, 100, 3)


def test_short_run_warns_that_warmup_never_finishes(cfg):
    resolved = resolve_schedule(cfg, steps_per_epoch=3, epochs=1)
    assert any("never leaves warmup" in w for w in resolved["warnings"])


def test_live_scheduler_matches_formula_after_stepping(cfg):
    resolved = resolve_schedule(cfg, steps_per_epoch=40, epochs=2)
    resolved = {**resolved, "warmup_steps": 10}
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=resolved["base_lr"])
    sched = LambdaLR(
        opt, build_lr_lambda(resolved["warmup_steps"], resolved["total_steps"], resolved["min_lr_ratio"])
    )
    for _ in range(33):
        opt.step()
        sched.step()
    assert scheduler_matches_expected(sched, resolved, 33)
    assert scheduler_matches_expected(sched, resolved, 0) is None
    assert not scheduler_matches_expected(sched, resolved, 10)
