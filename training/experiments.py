"""Collect, compare, and rank controlled-experiment runs.

An experiment arm is a directory produced by ``train.py --output-dir`` holding
best.pt, metrics.json, config_used.json, run_manifest.json, and (after
benchmarking) eval_pure_neural.json. Everything here reads those files; it
never trains or decodes anything itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from training.reproducibility import sha256_file

TARGET_ACCURACY = 0.85
BENCHMARK_JSON = "eval_pure_neural.json"
BENCHMARK_MD = "EVAL_pure_neural.md"

ARM_FACTORS: Dict[str, Dict[str, bool]] = {
    "baseline": {"token_weighted_loss": False, "contrast": False},
    "weighted_loss": {"token_weighted_loss": True, "contrast": False},
    "contrast": {"token_weighted_loss": False, "contrast": True},
    "combined": {"token_weighted_loss": True, "contrast": True},
}


def _load(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_arm_result(run_dir) -> Optional[Dict]:
    """Summarise one arm directory, or None if it has not finished training."""
    run_dir = Path(run_dir)
    metrics = _load(run_dir / "metrics.json")
    if metrics is None:
        return None
    config_used = _load(run_dir / "config_used.json") or {}
    manifest = _load(run_dir / "run_manifest.json") or {}
    bench = _load(run_dir / BENCHMARK_JSON)

    final = metrics.get("final_eval")
    last_epoch = (metrics.get("per_epoch") or [{}])[-1]
    if final:
        val = {
            "loss": final["val_loss"],
            "exact_match": final["exact_match"]["overall"]["accuracy"],
            "rows": final["exact_match"]["overall"]["total"],
            "exact_match_by_op": {
                op: row["accuracy"] for op, row in final["exact_match"]["by_op"].items()
            },
            "loss_by_op": final.get("val_loss_by_op", {}),
            "source": "final evaluation of saved checkpoint",
        }
        contrast = (final.get("contrast") or {}).get("pair_accuracy")
    else:
        val = {
            "loss": metrics.get("best_val_loss"),
            "exact_match": last_epoch.get("val_exact_match"),
            "rows": last_epoch.get("val_exact_match_rows"),
            "exact_match_by_op": last_epoch.get("val_exact_match_by_op", {}),
            "loss_by_op": last_epoch.get("val_loss_by_op", {}),
            "source": "last-epoch subset (final evaluation disabled)",
        }
        contrast = last_epoch.get("contrast_pair_acc")

    benchmark = None
    if bench:
        benchmark = {
            "mode": bench.get("evaluation_mode"),
            "overall": bench["overall"]["accuracy"],
            "exact_match": bench["overall"]["exact_match"],
            "total": bench["overall"]["total"],
            "by_op": {op: row["accuracy"] for op, row in bench["summary"].items()},
            "checkpoint_sha256": bench.get("inputs", {}).get("checkpoint_sha256"),
        }

    return {
        "arm": run_dir.name,
        "run_dir": run_dir.as_posix(),
        "factors": {
            "token_weighted_loss": bool((config_used.get("token_weighted_loss") or {}).get("enabled")),
            "contrast": bool((config_used.get("contrast") or {}).get("enabled")),
        },
        "seed": metrics.get("seed"),
        "epochs_configured": metrics.get("epochs_configured"),
        "epochs_run": metrics.get("epochs_run"),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_loss": metrics.get("best_val_loss"),
        "nonfinite_loss_steps": metrics.get("nonfinite_loss_steps"),
        "val": val,
        "contrast_pair_accuracy": contrast,
        "benchmark": benchmark,
        "checkpoint_sha256": (manifest.get("checkpoint") or {}).get("sha256"),
        "checkpoint_exists": (run_dir / "best.pt").is_file(),
    }


def selection_score(row: Dict) -> Optional[float]:
    """Mean of validation exact match and pure-neural benchmark accuracy.

    Both must exist: validation alone can be gamed by the training mix, the
    benchmark alone is a small, fixed set. Averaging keeps either from
    deciding the choice by itself.
    """
    if not row.get("benchmark") or row["val"].get("exact_match") is None:
        return None
    return 0.5 * row["val"]["exact_match"] + 0.5 * row["benchmark"]["overall"]


def rank_arms(rows: List[Dict]) -> List[Dict]:
    """Eligible arms, best first (ties broken by lower validation loss)."""
    eligible = [r for r in rows if selection_score(r) is not None and r["checkpoint_exists"]]
    return sorted(
        eligible,
        key=lambda r: (-selection_score(r), r["val"]["loss"] if r["val"]["loss"] is not None else 1e9),
    )
