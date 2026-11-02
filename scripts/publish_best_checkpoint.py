"""Pick the best experiment arm and publish it as the project checkpoint.

Selection uses validation AND benchmark performance: an arm is eligible only
when it has both a validation exact-match score and a pure-neural (no-fallback)
benchmark run, and arms are ranked by the mean of the two (ties: lower
validation loss). The published bundle contains the model, the exact config it
was trained with, the run manifest, the training report, the pure-neural
evaluation report, and a selection report that states -- without rounding up --
whether the >85% target was actually met on the no-fallback benchmark.

    python scripts/publish_best_checkpoint.py --dry-run     # show the choice and report, change nothing
    python scripts/publish_best_checkpoint.py --yes         # copy into checkpoints/final/ and docs/
    python scripts/publish_best_checkpoint.py --yes --refresh-canonical-eval

Publishing replaces checkpoints/final/best.pt, so the committed hybrid report
docs/EVAL_RESULTS.md / docs/eval_results.json no longer describes it (the
freshness unit test will fail until it is regenerated). Pass
--refresh-canonical-eval to regenerate it right away.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.experiments import (  # noqa: E402
    ARM_FACTORS,
    BENCHMARK_JSON,
    BENCHMARK_MD,
    TARGET_ACCURACY,
    collect_arm_result,
    rank_arms,
    render_comparison_markdown,
    selection_score,
    verify_checkpoint_binding,
)
from training.reproducibility import sha256_file  # noqa: E402


def pct(x):
    return "N/A" if x is None else f"{x:.1%}"


def build_selection_report(rows, ranked, chosen, published_at):
    bench = chosen["benchmark"]
    pure = bench is not None and "pure_neural" in str(bench.get("mode"))
    met = pure and bench["overall"] > TARGET_ACCURACY
    claim = (
        f"**MET** - the pure-neural benchmark accuracy is {pct(bench['overall'])} "
        f"({bench['exact_match']}/{bench['total']}), above {TARGET_ACCURACY:.0%}."
        if met
        else (
            f"**NOT MET** - the pure-neural benchmark accuracy is {pct(bench['overall']) if bench else 'unmeasured'}"
            + (f" ({bench['exact_match']}/{bench['total']})" if bench else "")
            + f", which does not exceed {TARGET_ACCURACY:.0%}. Do not quote a >85% figure for this checkpoint."
        )
    )

    lines = [
        "# Checkpoint Selection",
        "",
        f"Published: {published_at}",
        "",
        "## Decision",
        "",
        f"- **Selected arm:** `{chosen['arm']}` "
        f"(token-weighted loss: {'on' if chosen['factors']['token_weighted_loss'] else 'off'}, "
        f"contrast minibatches: {'on' if chosen['factors']['contrast'] else 'off'})",
        f"- **Checkpoint SHA-256:** `{sha256_file(Path(chosen['run_dir']) / 'best.pt')}`",
        f"- **Best epoch / epochs run:** {chosen['best_epoch']} / {chosen['epochs_run']}",
        f"- **Validation exact match:** {pct(chosen['val']['exact_match'])} "
        f"({chosen['val']['rows']} rows; {chosen['val']['source']})",
        f"- **Validation loss (unweighted):** {chosen['val']['loss']:.4f}",
        f"- **Contrast pair accuracy:** {pct(chosen['contrast_pair_accuracy'])}",
        f"- **Pure-neural benchmark:** {pct(bench['overall']) if bench else 'N/A'}",
        "",
        "## >85% target",
        "",
        "The target is claimed only against the standalone / no-fallback benchmark "
        "(`eval/run_eval.py --no-fallback`); hybrid results that lean on the verifier fallback do not count.",
        "",
        claim,
        "",
        "## Selection rule",
        "",
        "Eligible arms have both a validation exact-match score and a pure-neural benchmark run. "
        "Score = 0.5 x validation exact match + 0.5 x pure-neural benchmark accuracy; ties go to the lower "
        "validation loss. The benchmark is a small fixed set and validation is drawn from the training "
        "distribution, so neither alone decides.",
        "",
        "| Rank | Arm | Score | Val exact match | Benchmark (no fallback) | Val loss |",
        "|--:|---|--:|--:|--:|--:|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['arm']}{' (selected)' if r is chosen else ''} | {selection_score(r):.4f} | "
            f"{pct(r['val']['exact_match'])} | {pct(r['benchmark']['overall'])} | {r['val']['loss']:.4f} |"
        )
    skipped = [r["arm"] for r in rows if r not in ranked]
    if skipped:
        lines += ["", f"Ineligible (missing validation score, benchmark, or checkpoint): {', '.join(skipped)}"]

    lines += [
        "",
        "## Published files",
        "",
        "| File | Source |",
        "|---|---|",
        "| `checkpoints/final/best.pt` | selected arm `best.pt` |",
        "| `checkpoints/final/config_used.json` | exact config the arm trained with |",
        "| `checkpoints/final/run_manifest.json` | seeds, dataset/tokenizer hashes, optimizer/scheduler, environment |",
        "| `checkpoints/final/metrics.json` | per-epoch metrics and final evaluation |",
        "| `docs/TRAINING_RESULTS.md` | training report of the selected arm |",
        "| `docs/EVAL_PURE_NEURAL.md`, `docs/eval_pure_neural.json` | no-fallback benchmark of the selected checkpoint |",
        "| `docs/EXPERIMENT_COMPARISON.md` | all arms side by side |",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python problem_generator.py --contrast-only   # only if data/splits/contrast_*.jsonl are missing",
        "python scripts/run_experiments.py              # trains + benchmarks every arm into experiments/",
        "python scripts/publish_best_checkpoint.py --dry-run",
        "```",
        "",
        "---",
        "",
    ]
    comparison = render_comparison_markdown(rows, title="Full comparison of all arms").splitlines()
    lines += ["#" + l if l.startswith("#") else l for l in comparison]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--dest", type=Path, default=ROOT / "checkpoints" / "final")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    parser.add_argument("--arm", choices=list(ARM_FACTORS), help="Publish this arm instead of the top-ranked one")
    parser.add_argument("--dry-run", action="store_true", help="Show the decision and write nothing")
    parser.add_argument("--yes", action="store_true", help="Confirm replacing the existing published checkpoint")
    parser.add_argument("--refresh-canonical-eval", action="store_true",
                        help="Re-run the hybrid eval to regenerate docs/EVAL_RESULTS.md and eval_results.json")
    args = parser.parse_args()

    rows = [r for r in (collect_arm_result(args.experiments_dir / arm) for arm in ARM_FACTORS) if r]
    if not rows:
        print(f"No finished arms under {args.experiments_dir}. Run scripts/run_experiments.py first.", file=sys.stderr)
        return 2
    ranked = rank_arms(rows)
    if not ranked:
        print("No arm has both validation metrics and a pure-neural benchmark "
              "(run scripts/run_experiments.py without --skip-benchmark).", file=sys.stderr)
        return 2

    chosen = next((r for r in ranked if r["arm"] == args.arm), None) if args.arm else ranked[0]
    if chosen is None:
        print(f"Arm '{args.arm}' is not eligible for publishing.", file=sys.stderr)
        return 2

    problems = verify_checkpoint_binding(chosen)
    if problems:
        print("Refusing to publish:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 2

    published_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = build_selection_report(rows, ranked, chosen, published_at)
    bench = chosen["benchmark"]
    print(f"Selected arm: {chosen['arm']}  (score {selection_score(chosen):.4f}; "
          f"val EM {pct(chosen['val']['exact_match'])}; benchmark {pct(bench['overall'])})")

    if args.dry_run:
        print("\n--dry-run: nothing written. Selection report would read:\n")
        print(report)
        return 0

    dest = args.dest
    if (dest / "best.pt").exists() and not args.yes:
        print(f"{dest / 'best.pt'} already exists; re-run with --yes to replace it.", file=sys.stderr)
        return 2

    src = Path(chosen["run_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src / "best.pt", dest / "best.pt")
    if sha256_file(dest / "best.pt") != sha256_file(src / "best.pt"):
        print("Copied checkpoint hash mismatch; aborting.", file=sys.stderr)
        return 1
    for name in ("config_used.json", "metrics.json"):
        shutil.copy2(src / name, dest / name)

    manifest = json.loads((src / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["published"] = {
        "from_arm": chosen["arm"],
        "from_run_dir": chosen["run_dir"],
        "published_utc": published_at,
        "selection_score": selection_score(chosen),
        "published_path": (dest / "best.pt").relative_to(ROOT).as_posix()
        if (dest / "best.pt").is_relative_to(ROOT) else str(dest / "best.pt"),
    }
    (dest / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    shutil.copy2(src / "TRAINING_RESULTS.md", args.docs_dir / "TRAINING_RESULTS.md")
    shutil.copy2(src / BENCHMARK_MD, args.docs_dir / "EVAL_PURE_NEURAL.md")
    shutil.copy2(src / BENCHMARK_JSON, args.docs_dir / "eval_pure_neural.json")
    (args.docs_dir / "EXPERIMENT_COMPARISON.md").write_text(
        render_comparison_markdown(rows), encoding="utf-8"
    )
    (args.docs_dir / "CHECKPOINT_SELECTION.md").write_text(report, encoding="utf-8")
    print(f"Published '{chosen['arm']}' to {dest} and reports to {args.docs_dir}")

    if args.refresh_canonical_eval:
        print("Regenerating the canonical hybrid evaluation for the new checkpoint...", flush=True)
        subprocess.run([sys.executable, "eval/run_eval.py"], cwd=ROOT, check=True)
    else:
        print("\nNOTE: docs/EVAL_RESULTS.md and docs/eval_results.json still describe the previous "
              "checkpoint. Run `python eval/run_eval.py` (or re-run with --refresh-canonical-eval) "
              "before committing, or tests/unit/test_eval_report_freshness.py will fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
