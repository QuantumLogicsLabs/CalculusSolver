import json
import random
from pathlib import Path

def generate_slang_data():
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # 🎯 FIX 1 & 2: Generate scalable dataset using structural FRAC envelope layouts matching regression fixtures
    def create_frac_node(coeff, power):
        return {
            "numi": {
                "terms": [
                    {
                        "coeff": coeff,
                        "var": {"name": "x", "pow": power}
                    }
                ]
            },
            "deno": 1
        }

    dataset = []
    # Generation loop to build a diverse mathematical structure dataset instead of hardcoded rows
    for i in range(2500):
        # Sample 1: Power Rule Patterns
        c1 = random.randint(1, 10)
        p1 = random.randint(2, 6)
        dataset.append({
            "src_tokens": {"op": "diff", "var": "x", "expr": create_frac_node(c1, p1)},
            "tgt_input_tokens": {"op": "ans", "expr": create_frac_node(c1 * p1, p1 - 1)},
            "tgt_output_tokens": {"op": "ans", "expr": create_frac_node(c1 * p1, p1 - 1)},
            "rule_ids": 0,
            "verification_state": 1
        })
        
        # Sample 2: Deliberately incorrect patterns for sequence training head masking check
        dataset.append({
            "src_tokens": {"op": "diff", "var": "x", "expr": create_frac_node(c1, p1)},
            "tgt_input_tokens": {"op": "ans", "expr": create_frac_node(c1, p1)}, # Wrong output tracking
            "tgt_output_tokens": {"op": "ans", "expr": create_frac_node(c1, p1)},
            "rule_ids": 0,
            "verification_state": 0
        })

    # Distribute the generated samples cleanly into splits
    splits = {
        "train": dataset[:3000],
        "val": dataset[3000:4000],
        "test": dataset[4000:]
    }
    
    for split_name, split_data in splits.items():
        with open(splits_dir / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
                
    print(f"🎯 [Dataset Engine] Generated {len(dataset)} structural FRAC validation row items.")

if __name__ == "__main__":
    generate_slang_data()