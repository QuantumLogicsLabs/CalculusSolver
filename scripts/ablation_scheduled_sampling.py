import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.json"
CHECKPOINT = ROOT / "checkpoints" / "final" / "best.pt"
TRAINING_RESULTS = ROOT / "docs" / "TRAINING_RESULTS.md"
RUNS = [("A_ss_off", False), ("B_ss_on", True)]


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_config(base, ss_enabled, epochs, ramp):
    cfg = json.loads(json.dumps(base))
    cfg["epochs"] = epochs
    cfg["scheduled_sampling"]["enabled"] = ss_enabled
    cfg["scheduled_sampling"]["ramp_epochs"] = ramp
    cfg["curriculum"]["gradient_3var_ramp_epochs"] = ramp
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def train_run(name, ss_enabled, base, epochs, ramp):
    out_dir = ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(base, ss_enabled, epochs, ramp)
    run([sys.executable, "train.py", "--from-scratch"])
    shutil.copy2(CHECKPOINT, out_dir / "best.pt")
    shutil.copy2(TRAINING_RESULTS, out_dir / "TRAINING_RESULTS.md")


def evaluate_run(name, beam_size, max_len):
    run([
        sys.executable, "eval/run_eval.py",
        "--checkpoint", f"runs/{name}/best.pt",
        "--no-fallback",
        "--beam-size", str(beam_size),
        "--max-len", str(max_len),
        "--output", f"docs/EVAL_{name}.md",
        "--json-output", f"docs/eval_{name}.json",
    ])


def compare():
    for name, _ in RUNS:
        record = json.loads((ROOT / "docs" / f"eval_{name}.json").read_text(encoding="utf-8"))
        print(f"\n{name}")
        for op, v in record["summary"].items():
            print(f"  {op:14s} {v['exact_match']}/{v['total']}  ({v['accuracy']:.1%})")
        o = record["overall"]
        print(f"  {'OVERALL':14s} {o['exact_match']}/{o['total']}  ({o['accuracy']:.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--ramp", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    original = CONFIG.read_text(encoding="utf-8")
    base = json.loads(original)
    try:
        if not args.skip_train:
            for name, ss_enabled in RUNS:
                train_run(name, ss_enabled, base, args.epochs, args.ramp)
        for name, _ in RUNS:
            evaluate_run(name, args.beam_size, args.max_len)
    finally:
        CONFIG.write_text(original, encoding="utf-8")
    compare()


if __name__ == "__main__":
    main()
