"""Run the canonical benchmark evaluation and record reproducibility metadata.

The checked-in baseline uses the same hybrid inference path as the application:
beam search followed by verifier-gated fallback. Use ``--no-fallback`` only for
a diagnostic measurement of the neural checkpoint in isolation.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "final" / "best.pt"
DEFAULT_BENCHMARK_DIR = ROOT / "eval" / "benchmarks"
DEFAULT_REPORT = ROOT / "docs" / "EVAL_RESULTS.md"
DEFAULT_JSON = ROOT / "docs" / "eval_results.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.eval_harness import is_equivalent
from inference.solve import CalculusSolverInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint and generate reproducible Markdown and JSON reports."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path (default: checkpoints/final/best.pt)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory containing benchmark_*.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help="Markdown report path (default: docs/EVAL_RESULTS.md)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON,
        help="Machine-readable summary path (default: docs/eval_results.json)",
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Evaluate only the first N records per category"
    )
    parser.add_argument(
        "--beam-size", type=int, default=3, help="Beam size for decoding (canonical baseline: 3)"
    )
    parser.add_argument(
        "--max-len", type=int, default=64, help="Maximum decoded token count (canonical baseline: 64)"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable verifier fallback for a pure-neural diagnostic run",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0; inference is deterministic)"
    )
    args = parser.parse_args()

    if args.sample is not None and args.sample <= 0:
        parser.error("--sample must be a positive integer")
    if args.beam_size <= 0:
        parser.error("--beam-size must be a positive integer")
    if args.max_len <= 0:
        parser.error("--max-len must be a positive integer")

    is_canonical_output = args.output.resolve() == DEFAULT_REPORT.resolve()
    is_canonical_json = args.json_output.resolve() == DEFAULT_JSON.resolve()
    if (args.sample is not None or args.no_fallback) and (is_canonical_output or is_canonical_json):
        parser.error(
            "Sampled or pure-neural diagnostic runs must use both --output and --json-output "
            "so they cannot overwrite the canonical hybrid baseline."
        )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        dirty = None
    return {"commit": commit, "dirty": dirty}


def load_benchmarks(
    benchmark_dir: Path, sample: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark_files = [
        Path(path) for path in sorted(glob.glob(str(benchmark_dir / "benchmark_*.json")))
    ]
    if not benchmark_files:
        raise FileNotFoundError(f"No benchmark_*.json files found in {benchmark_dir}")

    loaded: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for path in benchmark_files:
        with path.open("r", encoding="utf-8") as stream:
            all_problems = json.load(stream)
        if not isinstance(all_problems, list):
            raise ValueError(f"Benchmark must contain a JSON list: {path}")

        selected = all_problems[:sample] if sample is not None else all_problems
        operation = path.stem.removeprefix("benchmark_")
        loaded.append({"operation": operation, "path": path, "problems": selected})
        manifest.append(
            {
                "file": repo_relative(path),
                "operation": operation,
                "records_in_file": len(all_problems),
                "records_evaluated": len(selected),
                "sha256": sha256_file(path),
            }
        )
    return loaded, manifest


def evaluate(
    solver: CalculusSolverInference, benchmarks: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], int]:
    summary: dict[str, dict[str, Any]] = {}
    exception_count = 0

    for benchmark in benchmarks:
        operation = benchmark["operation"]
        problems = benchmark["problems"]
        exact_match_count = 0
        verified_count = 0
        print(f"Evaluating {operation} ({len(problems)} problems)...", flush=True)

        for index, problem in enumerate(problems):
            try:
                result = solver.solve(problem["expr"])
                prediction = result.get("output") or result.get("expr") or {}
                exact_match_count += int(is_equivalent(prediction, problem["target"]))
                verified_count += int(bool(result.get("verified", False)))
            except Exception as exc:  # Keep the full-suite report usable while surfacing failures.
                exception_count += 1
                print(f"  ERROR {operation}[{index}]: {exc}", file=sys.stderr, flush=True)

        total = len(problems)
        summary[operation] = {
            "total": total,
            "exact_match": exact_match_count,
            "verified": verified_count,
            "accuracy": exact_match_count / total if total else 0.0,
            "verification_rate": verified_count / total if total else 0.0,
        }
        print(
            f"  {operation}: exact={exact_match_count}/{total}, "
            f"verified={verified_count}/{total}",
            flush=True,
        )
    return summary, exception_count


def build_record(
    args: argparse.Namespace,
    manifest: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
    exception_count: int,
) -> dict[str, Any]:
    total = sum(row["total"] for row in summary.values())
    exact = sum(row["exact_match"] for row in summary.values())
    verified = sum(row["verified"] for row in summary.values())
    checkpoint = args.checkpoint.resolve()

    repository = git_state()
    return {
        "schema_version": 1,
        "evaluation_mode": "pure_neural" if args.no_fallback else "hybrid_verifier_fallback",
        "settings": {
            "beam_size": args.beam_size,
            "max_len": args.max_len,
            "seed": args.seed,
            "sample_per_category": args.sample,
            "fallback_enabled": not args.no_fallback,
        },
        "inputs": {
            "checkpoint": repo_relative(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "evaluator": repo_relative(Path(__file__)),
            "evaluator_sha256": sha256_file(Path(__file__)),
            "benchmarks": manifest,
        },
        "environment": {
            "git_commit": repository["commit"],
            "git_dirty": repository["dirty"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "summary": summary,
        "overall": {
            "total": total,
            "exact_match": exact,
            "verified": verified,
            "accuracy": exact / total if total else 0.0,
            "verification_rate": verified / total if total else 0.0,
            "exceptions": exception_count,
        },
        "comparison": {
            "stale_document": {
                "benchmark_total": 300,
                "partial": {"exact_match": 31, "total": 60, "accuracy": 31 / 60},
                "gradient": {"exact_match": 20, "total": 50, "accuracy": 20 / 50},
                "overall": {"exact_match": 219, "total": 300, "accuracy": 219 / 300},
            },
            "resolution": (
                "Commit e9b7923 reduced benchmark_gradient.json from 50 records to 18. "
                "Regenerating from the five current files changes the suite total from 300 to 268."
            ),
        },
    }


def fraction(count: int, total: int) -> str:
    rate = count / total if total else 0.0
    return f"{count}/{total} ({rate:.1%})"


def render_markdown(record: dict[str, Any]) -> str:
    settings = record["settings"]
    inputs = record["inputs"]
    environment = record["environment"]
    summary = record["summary"]
    overall = record["overall"]
    mode = record["evaluation_mode"]
    mode_label = (
        "hybrid inference (neural beam search + verifier-gated fallback)"
        if mode == "hybrid_verifier_fallback"
        else "pure neural inference (fallback disabled)"
    )

    lines = [
        "# Evaluation Results",
        "",
        "## Current reproducible baseline",
        "",
        f"- **Checkpoint:** `{inputs['checkpoint']}`",
        f"- **Checkpoint SHA-256:** `{inputs['checkpoint_sha256']}`",
        f"- **Evaluation mode:** {mode_label}",
        f"- **Decoder:** beam size {settings['beam_size']}, max length {settings['max_len']}",
        f"- **Seed:** {settings['seed']}",
        "",
        "| Operation | Problems | Exact match | Verification rate |",
        "|---|---:|---:|---:|",
    ]
    for operation, row in summary.items():
        lines.append(
            f"| {operation} | {row['total']} | "
            f"{fraction(row['exact_match'], row['total'])} | "
            f"{fraction(row['verified'], row['total'])} |"
        )
    lines.append(
        f"| **Overall** | **{overall['total']}** | "
        f"**{fraction(overall['exact_match'], overall['total'])}** | "
        f"**{fraction(overall['verified'], overall['total'])}** |"
    )
    lines.extend(
        [
            "",
            f"> Partial is not an active accuracy shortfall in this baseline: it is "
            f"{fraction(summary['partial']['exact_match'], summary['partial']['total'])}. "
            f"Gradient is {fraction(summary['gradient']['exact_match'], summary['gradient']['total'])}, "
            f"and the gradient benchmark contains {summary['gradient']['total']} records.",
            "",
            "> Exact match and verification rate are distinct measurements. In particular, the current "
            "gradient verifier accepts predictions with an extra component, so verification can exceed exact match. "
            "This report records both without treating verifier acceptance as exact correctness.",
            "",
            "## Correction of the stale report",
            "",
            "The previous table was generated for an older 300-record suite and older checkpoint behavior. "
            "Commit `e9b7923` replaced the 50-record gradient benchmark with the current 18-record file, "
            "but the table was not regenerated. The 32-record reduction explains the full total mismatch: "
            "300 - (50 - 18) = 268. The resolution is to report the benchmark files that actually exist, "
            "not to synthesize 32 missing cases.",
            "",
            "| Measurement | Before (stale document) | After (current inputs) |",
            "|---|---:|---:|",
            f"| Benchmark total | 300 | {overall['total']} |",
            f"| Partial exact match | 31/60 (51.7%) | {fraction(summary['partial']['exact_match'], summary['partial']['total'])} |",
            f"| Gradient exact match | 20/50 (40.0%) | {fraction(summary['gradient']['exact_match'], summary['gradient']['total'])} |",
            f"| Overall exact match | 219/300 (73.0%) | {fraction(overall['exact_match'], overall['total'])} |",
            "",
            "## Benchmark manifest",
            "",
            "The hashes below define the exact benchmark inputs. Record counts are read from the JSON arrays, "
            "not copied from documentation.",
            "",
            "| File | Records | SHA-256 |",
            "|---|---:|---|",
        ]
    )
    for item in inputs["benchmarks"]:
        lines.append(f"| `{item['file']}` | {item['records_in_file']} | `{item['sha256']}` |")
    lines.extend(
        [
            f"| **Total** | **{sum(item['records_in_file'] for item in inputs['benchmarks'])}** | |",
            "",
            "## Reproduce this evaluation",
            "",
            "Run from the repository root with Python 3.12:",
            "",
            "```powershell",
            "python -m venv .venv",
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-neural.txt",
            ".\\.venv\\Scripts\\python.exe eval/run_eval.py --checkpoint checkpoints/final/best.pt --beam-size 3 --max-len 64",
            "```",
            "",
            "The command regenerates this file and `docs/eval_results.json`. A matching run must use the "
            "checkpoint and benchmark hashes above and should finish with zero exceptions. For a checkpoint-only "
            "diagnostic, pass `--no-fallback` together with non-canonical `--output` and `--json-output` paths; "
            "do not compare that diagnostic directly with this hybrid baseline.",
            "",
            "## Recorded environment",
            "",
            f"- **Git commit:** `{environment['git_commit'] or 'unavailable'}`",
            f"- **Git working tree dirty:** `{environment['git_dirty']}`",
            f"- **Evaluator SHA-256:** `{inputs['evaluator_sha256']}`",
            f"- **Python:** `{environment['python']}`",
            f"- **PyTorch:** `{environment['torch']}`",
            f"- **NumPy:** `{environment['numpy']}`",
            f"- **Platform:** `{environment['platform']}`",
            f"- **Evaluation exceptions:** `{overall['exceptions']}`",
            "",
            "The machine-readable before/after evidence for future comparisons is committed in "
            "`docs/eval_results.json`; the console output reports the same per-category numerators and denominators.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.checkpoint = (
        (ROOT / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint.resolve()
    )
    args.benchmark_dir = (
        (ROOT / args.benchmark_dir).resolve()
        if not args.benchmark_dir.is_absolute()
        else args.benchmark_dir.resolve()
    )
    args.output = (
        (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    )
    args.json_output = (
        (ROOT / args.json_output).resolve()
        if not args.json_output.is_absolute()
        else args.json_output.resolve()
    )

    if not args.checkpoint.is_file():
        print(f"Error: checkpoint does not exist: {args.checkpoint}", file=sys.stderr)
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)

    try:
        benchmarks, manifest = load_benchmarks(args.benchmark_dir, args.sample)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    mode = "pure neural" if args.no_fallback else "hybrid verifier fallback"
    print(
        f"Loading {mode}: checkpoint={repo_relative(args.checkpoint)}, "
        f"beam_size={args.beam_size}, max_len={args.max_len}",
        flush=True,
    )
    solver = CalculusSolverInference(
        model_path=str(args.checkpoint),
        beam_size=args.beam_size,
        max_len=args.max_len,
        enable_fallback=not args.no_fallback,
    )
    try:
        summary, exception_count = evaluate(solver, benchmarks)
    finally:
        solver.close()

    record = build_record(args, manifest, summary, exception_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(record), encoding="utf-8")
    args.json_output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"Saved Markdown report to {repo_relative(args.output)}", flush=True)
    print(f"Saved JSON summary to {repo_relative(args.json_output)}", flush=True)
    print(
        f"Overall: exact={record['overall']['exact_match']}/{record['overall']['total']} "
        f"({record['overall']['accuracy']:.1%}), "
        f"verified={record['overall']['verified']}/{record['overall']['total']} "
        f"({record['overall']['verification_rate']:.1%}), "
        f"exceptions={record['overall']['exceptions']}",
        flush=True,
    )
    return 1 if exception_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
