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


def verify_checkpoint_binding(row: Dict) -> List[str]:
    """Problems that would make the benchmark numbers not belong to best.pt."""
    problems = []
    ckpt = Path(row["run_dir"]) / "best.pt"
    actual = sha256_file(ckpt)
    if actual is None:
        return [f"{ckpt.as_posix()} is missing"]
    if row.get("checkpoint_sha256") and row["checkpoint_sha256"] != actual:
        problems.append("run_manifest.json checkpoint hash does not match best.pt")
    bench_hash = (row.get("benchmark") or {}).get("checkpoint_sha256")
    if bench_hash and bench_hash != actual:
        problems.append("benchmark was measured on a different checkpoint than best.pt")
    return problems


def _pct(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.1%}"


def _delta(x: Optional[float], base: Optional[float]) -> str:
    if x is None or base is None:
        return "N/A"
    return f"{(x - base) * 100:+.1f} pp"


def render_comparison_markdown(rows: List[Dict], title: str = "Controlled Experiment Comparison") -> str:
    order = list(ARM_FACTORS)
    rows = sorted(rows, key=lambda r: order.index(r["arm"]) if r["arm"] in order else len(order))
    base = next((r for r in rows if r["arm"] == "baseline"), None)

    val_ops = sorted({op for r in rows for op in r["val"]["exact_match_by_op"]})
    bench_ops = sorted({op for r in rows if r["benchmark"] for op in r["benchmark"]["by_op"]})

    lines = [
        f"# {title}",
        "",
        "Every arm shares the seed, dataset, tokenizer, model, optimizer and LR schedule; only the listed "
        "factors differ from the baseline. Validation loss is the *unweighted* token cross-entropy so it is "
        "comparable across arms. Exact match is whole-sequence greedy decoding. The benchmark column is the "
        "**pure-neural, no-fallback** measurement (`eval/run_eval.py --no-fallback`).",
        "",
        "## Overview",
        "",
        "| Arm | Weighted loss | Contrast batches | Epochs run | Best epoch | Val loss | Val exact match | "
        "Contrast pair acc | Benchmark (no fallback) |",
        "|---|:-:|:-:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {'yes' if r['factors']['token_weighted_loss'] else 'no'} | "
            f"{'yes' if r['factors']['contrast'] else 'no'} | {r['epochs_run']} | {r['best_epoch']} | "
            f"{r['val']['loss']:.4f} | {_pct(r['val']['exact_match'])} | "
            f"{_pct(r['contrast_pair_accuracy'])} | "
            f"{_pct(r['benchmark']['overall']) if r['benchmark'] else 'not run'} |"
        )

    lines += [
        "",
        "## Validation exact match by operation",
        "",
        "| Arm | Overall | " + " | ".join(val_ops) + " |",
        "|---|--:|" + "|".join(["--:"] * len(val_ops)) + "|",
    ]
    for r in rows:
        cells = " | ".join(_pct(r["val"]["exact_match_by_op"].get(op)) for op in val_ops)
        lines.append(f"| {r['arm']} | {_pct(r['val']['exact_match'])} | {cells} |")

    lines += [
        "",
        "## Validation loss by operation (token-mean, unweighted)",
        "",
        "| Arm | Overall | " + " | ".join(val_ops) + " |",
        "|---|--:|" + "|".join(["--:"] * len(val_ops)) + "|",
    ]
    for r in rows:
        cells = " | ".join(
            f"{r['val']['loss_by_op'][op]:.4f}" if op in r["val"]["loss_by_op"] else "N/A" for op in val_ops
        )
        lines.append(f"| {r['arm']} | {r['val']['loss']:.4f} | {cells} |")

    if bench_ops:
        lines += [
            "",
            "## Benchmark accuracy by operation (pure neural, no fallback)",
            "",
            "| Arm | Overall | " + " | ".join(bench_ops) + " |",
            "|---|--:|" + "|".join(["--:"] * len(bench_ops)) + "|",
        ]
        for r in rows:
            b = r["benchmark"]
            if not b:
                lines.append(f"| {r['arm']} | not run |" + " |" * len(bench_ops))
                continue
            cells = " | ".join(_pct(b["by_op"].get(op)) for op in bench_ops)
            lines.append(f"| {r['arm']} | {_pct(b['overall'])} ({b['exact_match']}/{b['total']}) | {cells} |")

    if base:
        lines += [
            "",
            "## Effect of each change versus the baseline",
            "",
            "| Arm | Δ val exact match | Δ contrast pair acc | Δ benchmark | Δ val loss |",
            "|---|--:|--:|--:|--:|",
        ]
        for r in rows:
            if r is base:
                continue
            d_loss = (
                f"{r['val']['loss'] - base['val']['loss']:+.4f}"
                if r["val"]["loss"] is not None and base["val"]["loss"] is not None
                else "N/A"
            )
            lines.append(
                f"| {r['arm']} | {_delta(r['val']['exact_match'], base['val']['exact_match'])} | "
                f"{_delta(r['contrast_pair_accuracy'], base['contrast_pair_accuracy'])} | "
                f"{_delta(r['benchmark']['overall'] if r['benchmark'] else None, base['benchmark']['overall'] if base['benchmark'] else None)} | "
                f"{d_loss} |"
            )

    lines += ["", "## Provenance", "", "| Arm | Checkpoint SHA-256 | Seed | Non-finite loss steps |", "|---|---|--:|--:|"]
    for r in rows:
        lines.append(
            f"| {r['arm']} | `{r['checkpoint_sha256'] or 'N/A'}` | {r['seed']} | {r['nonfinite_loss_steps']} |"
        )
    lines.append("")
    return "\n".join(lines)
