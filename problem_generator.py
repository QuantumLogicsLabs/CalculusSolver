import json
import random
from pathlib import Path

def generate_slang_dataset():
    print("⏳ [Dataset Engine] Programmatically synthesizing 100k-row strict rule-bound SLaNg dataset...")
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    
    def make_frac(terms):
        return {
            "numi": {"terms": terms},
            "deno": 1
        }
        
    for i in range(100000):
        # Scope restricted purely to authentic polynomial derivative layers to prevent fake schema tokens
        rule = random.randint(0, 3)
        var_name = "x"
        
        if rule == 0: # Standard Base Monomial Form
            coeff = random.randint(1, 25)
            power = random.randint(2, 9)
            src_expr = make_frac([{"coeff": coeff, "var": {var_name: power}}])
            ans_expr = make_frac([{"coeff": coeff * power, "var": {var_name: power - 1}}])
            
        elif rule == 1: # Higher Order Quadratic Polynomial Expressions
            c1, c2 = random.randint(1, 10), random.randint(1, 15)
            src_expr = make_frac([
                {"coeff": c1, "var": {var_name: 3}},
                {"coeff": c2, "var": {var_name: 2}}
            ])
            ans_expr = make_frac([
                {"coeff": c1 * 3, "var": {var_name: 2}},
                {"coeff": c2 * 2, "var": {var_name: 1}}
            ])
            
        elif rule == 2: # Linear Term with Constant Shifts
            c1 = random.randint(2, 20)
            c2 = random.randint(1, 50)
            src_expr = make_frac([
                {"coeff": c1, "var": {var_name: 1}},
                {"coeff": c2} # Constant term has omit mapping design pattern
            ])
            ans_expr = make_frac([
                {"coeff": c1}
            ])
            
        else: # Multi-order Multi-term Algebraic Expansions
            c1, c2, c3 = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
            src_expr = make_frac([
                {"coeff": c1, "var": {var_name: 4}},
                {"coeff": -c2, "var": {var_name: 3}},
                {"coeff": c3, "var": {var_name: 2}}
            ])
            ans_expr = make_frac([
                {"coeff": c1 * 4, "var": {var_name: 3}},
                {"coeff": -c2 * 3, "var": {var_name: 2}},
                {"coeff": c3 * 2, "var": {var_name: 1}}
            ])

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
                
    print(f"✅ [Dataset Engine] 100,000 structural canonical polynomial rows successfully generated.")

if __name__ == "__main__":
    generate_slang_dataset()