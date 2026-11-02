"""Controlled experiments: baseline vs weighted loss vs contrast batches vs both.

Each arm trains in its own subprocess from the same seed, data, tokenizer,
model and LR schedule; only the two factors under test change, so a difference
between arms is attributable to a factor rather than to noise in the setup.
Arms train into ``experiments/<arm>/`` and never touch checkpoints/final/.

    python scripts/run_experiments.py                      # all four arms, 15 epochs
    python scripts/run_experiments.py --arms baseline combined
    python scripts/run_experiments.py --epochs 1 --max-steps 20 --set exact_match_eval.per_epoch_examples=32   # smoke test
    python scripts/run_experiments.py --skip-train         # re-benchmark / re-compare existing arms

Benchmarking uses ``eval/run_eval.py --no-fallback`` (pure neural), the
measurement the >85% target must be claimed against.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.experiments import (  # noqa: E402
    ARM_FACTORS,
    BENCHMARK_JSON,
    BENCHMARK_MD,
    collect_arm_result,
    render_comparison_markdown,
)
from training.reproducibility import sha256_file  # noqa: E402


def benchmark_is_current(run_dir: Path) -> bool:
    """True when the stored benchmark was measured on this arm's current best.pt."""
    path = run_dir / BENCHMARK_JSON
    if not path.is_file():
        return False
    recorded = json.loads(path.read_text(encoding="utf-8")).get("inputs", {}).get("checkpoint_sha256")
    return recorded is not None and recorded == sha256_file(run_dir / "best.pt")


def run(cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def train_arm(arm, run_dir, args):
    factors = ARM_FACTORS[arm]
    cmd = [
        sys.executable, "train.py",
        "--from-scratch",
        "--run-name", arm,
        "--output-dir", run_dir,
        "--seed", args.seed,
        "--set", f"token_weighted_loss.enabled={json.dumps(factors['token_weighted_loss'])}",
        "--set", f"contrast.enabled={json.dumps(factors['contrast'])}",
    ]
    if args.epochs is not None:
        cmd += ["--epochs", args.epochs]
    if args.max_steps is not None:
        cmd += ["--max-steps", args.max_steps]
    for item in args.set:
        cmd += ["--set", item]
    run(cmd)


def benchmark_arm(run_dir, args):
    run([
        sys.executable, "eval/run_eval.py",
        "--checkpoint", run_dir / "best.pt",
        "--no-fallback",
        "--beam-size", args.beam_size,
        "--max-len", args.max_len,
        "--output", run_dir / BENCHMARK_MD,
        "--json-output", run_dir / BENCHMARK_JSON,
    ] + (["--sample", args.benchmark_sample] if args.benchmark_sample else []))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=list(ARM_FACTORS), choices=list(ARM_FACTORS))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (default: config.json = 15)")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap steps per epoch (smoke tests)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra config override applied identically to every arm")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--benchmark-sample", type=int, default=None, help="Benchmark only first N per category")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retrain arms that already have metrics.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "EXPERIMENT_COMPARISON.md")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        run_dir = out_dir / arm
        done = (run_dir / "metrics.json").is_file()
        if not args.skip_train:
            if done and not args.force:
                print(f"[{arm}] metrics.json exists - skipping training (use --force to retrain)")
            else:
                train_arm(arm, run_dir, args)
        if not args.skip_benchmark and (run_dir / "best.pt").is_file():
            if benchmark_is_current(run_dir) and not args.force:
                print(f"[{arm}] benchmark already matches best.pt - skipping (use --force to redo)")
            else:
                benchmark_arm(run_dir, args)

    rows = [r for r in (collect_arm_result(out_dir / arm) for arm in ARM_FACTORS) if r]
    if not rows:
        print("No finished arms found; nothing to compare.")
        return 1

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    report = args.report if args.report.is_absolute() else ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_comparison_markdown(rows), encoding="utf-8")
    print(f"\nWrote {report.relative_to(ROOT).as_posix() if report.is_relative_to(ROOT) else report}")
    print(f"Wrote {(out_dir / 'summary.json').relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else out_dir / 'summary.json'}")
    for r in rows:
        bench = f"{r['benchmark']['overall']:.1%}" if r["benchmark"] else "not run"
        print(f"  {r['arm']:14s} val EM {r['val']['exact_match']:.1%} | benchmark(no fallback) {bench}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
