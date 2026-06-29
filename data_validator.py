import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from tokenizer.slang_serializer import serialize_slang_math


def validate_slang_data():
    splits = ["train.jsonl", "val.jsonl", "test.jsonl"]
    base_dir = Path("data/splits")
    required_keys = ["src_tokens", "tgt_input_tokens", "tgt_output_tokens", "rule_ids", "verification_state"]

    print("--- 🩺 SLaNg Data Validation Reports ---")
    any_failures = False

    for s in splits:
        file_path = base_dir / s
        if not file_path.exists():
            print(f"❌ Missing critical split path: {file_path}")
            any_failures = True
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"📊 Analyzing {s}: Total Row Records = {len(lines)}")

        key_failures = 0
        serializer_failures = 0
        first_error = None

        for line_no, line in enumerate(lines, start=1):
            entry = json.loads(line)

            missing = [k for k in required_keys if k not in entry]
            if missing:
                key_failures += 1
                if first_error is None:
                    first_error = f"line {line_no}: missing keys {missing}"
                continue

            # The real check: does this row's envelopes actually serialize
            # against the SLaNg serializer the model is trained against?
            for field in ("src_tokens", "tgt_input_tokens", "tgt_output_tokens"):
                try:
                    serialize_slang_math(entry[field])
                except Exception as e:
                    serializer_failures += 1
                    if first_error is None:
                        first_error = f"line {line_no}, field '{field}': {e}"
                    break

        if key_failures or serializer_failures:
            any_failures = True
            print(f"   ❌ {key_failures} rows failed key checks, {serializer_failures} rows failed serializer round-trip")
            print(f"   ↳ first failure: {first_error}")
        else:
            print(f"   ✅ All {len(lines)} rows have required keys and serialize cleanly.")

    if any_failures:
        print("\n❌ Validation FAILED — see failures above.")
        sys.exit(1)
    else:
        print("\n✅ All splits passed schema and serializer validation.")


if __name__ == "__main__":
    validate_slang_data()