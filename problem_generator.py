import json
import random
from pathlib import Path

def generate_slang_dataset():
    print("⏳ [Dataset Engine] Programmatically synthesizing 100k-row authentic SLaNg dataset...")
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    dataset = []

    for i in range(100000):
        rule = random.randint(0, 3)
        var_name = "x"
        coeff = random.randint(1, 15)
        power = random.randint(2, 5)

        # Real SLaNg term shape, matching tokenizer/slang_serializer.py:
        #   - "coeff" is a number
        #   - "var" is a dict of {variable_name: exponent}
        # No "type"/"terms" wrapper — that shape doesn't exist in the real serializer.
        src_expr = {"coeff": coeff, "var": {var_name: power}}
        ans_expr = {"coeff": coeff * power, "var": {var_name: power - 1}}

        src_op_node = {
            "op": "diff",
            "var": var_name,
            "expr": src_expr
        }

        dataset.append({
            "src_tokens": src_op_node,
            "tgt_input_tokens": ans_expr,
            "tgt_output_tokens": ans_expr,
            "rule_ids": rule,
            "verification_state": 1
        })

    random.shuffle(dataset)

    with open("data/slang_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    for name, split_data in [("train", dataset[:90000]), ("val", dataset[90000:95000]), ("test", dataset[95000:])]:
        with open(splits_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")

    print(f"✅ [Dataset Engine] 100,000 authentic lines generated successfully.")

if __name__ == "__main__":
    generate_slang_dataset()
