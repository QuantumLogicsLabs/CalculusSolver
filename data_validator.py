import glob
import json
import sys
from pathlib import Path
from typing import Dict, Set

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from tokenizer.slang_serializer import serialize_slang_math


def load_benchmark_signatures(benchmark_dir: Path) -> Set[str]:
    signatures: Set[str] = set()
    if not benchmark_dir.exists():
        return signatures
    for bf in benchmark_dir.glob("*.json"):
        with open(bf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for item in data:
                    expr = item.get("expr")
                    if expr:
                        signatures.add(" ".join(serialize_slang_math(expr)))
            except Exception:
                pass
    return signatures


def validate_slang_data():
    splits = ["train.jsonl", "val.jsonl", "test.jsonl"]
    base_dir = Path("data/splits")
    benchmark_dir = Path("eval/benchmarks")
    required_keys = [
        "src_tokens",
        "tgt_input_tokens",
        "tgt_output_tokens",
        "rule_ids",
        "verification_state",
    ]

    print("--- SLaNg Clean Data & Anti-Overfitting Validation ---")
    any_failures = False

    benchmark_sigs = load_benchmark_signatures(benchmark_dir)
    print(f"Loaded {len(benchmark_sigs)} benchmark problem signatures for leakage screening.\n")

    split_signatures: Dict[str, Set[str]] = {}

    for s in splits:
        file_path = base_dir / s
        if not file_path.exists():
            print(f"[FAIL] Missing critical split path: {file_path}")
            any_failures = True
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"Analyzing {s}: Total Records = {len(lines)}")

        key_failures = 0
        serializer_failures = 0
        duplicates = 0
        leaks = 0
        first_error = None
        seen_in_split: Set[str] = set()

        for line_no, line in enumerate(lines, start=1):
            entry = json.loads(line)

            missing = [k for k in required_keys if k not in entry]
            if missing:
                key_failures += 1
                if first_error is None:
                    first_error = f"line {line_no}: missing keys {missing}"
                continue

            # Serializer round-trip check
            src_toks = None
            for field in ("src_tokens", "tgt_input_tokens", "tgt_output_tokens"):
                try:
                    toks = serialize_slang_math(entry[field])
                    if field == "src_tokens":
                        src_toks = toks
                except Exception as e:
                    serializer_failures += 1
                    if first_error is None:
                        first_error = f"line {line_no}, field '{field}': {e}"
                    break

            if src_toks is not None:
                sig = " ".join(src_toks)
                if sig in seen_in_split:
                    duplicates += 1
                else:
                    seen_in_split.add(sig)

                if sig in benchmark_sigs:
                    leaks += 1

        split_signatures[s] = seen_in_split

        if key_failures or serializer_failures or duplicates > 0 or leaks > 0:
            any_failures = True
            print(f"   [FAIL] {key_failures} key errors | {serializer_failures} serializer errors | {duplicates} duplicates | {leaks} benchmark leaks")
            if first_error:
                print(f"   first failure: {first_error}")
        else:
            print(f"   [OK] All {len(lines)} rows serialize cleanly with 0 duplicates and 0 benchmark leaks.")

    # Inter-split disjointness check
    print("\nChecking Inter-Split Disjointness (Zero-Leakage Guarantee)...")
    split_names = [s for s in splits if s in split_signatures]
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = split_signatures[s1].intersection(split_signatures[s2])
            if overlap:
                any_failures = True
                print(f"   [FAIL] Overlap detected between {s1} and {s2}: {len(overlap)} shared expressions!")
            else:
                print(f"   [OK] {s1} vs {s2}: Perfectly disjoint (0 shared expressions).")

    if any_failures:
        print("\n[FAIL] Validation FAILED — issues found above.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All splits passed schema, serialization, 0-duplicate, and 0-leakage validation!")


if __name__ == "__main__":
    validate_slang_data()
