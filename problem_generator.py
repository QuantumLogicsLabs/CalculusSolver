import json
import random
from pathlib import Path

def generate_slang_dataset():
    print("⏳ [Dataset Engine] Programmatically synthesizing 100k-row canonical SLaNg dataset...")
    
    # Base configuration directories matching schema tracking specifications
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Rule labels mapped according to classification head specifications
    # 0: power_rule, 1: trig, 2: exponential, 3: logarithmic, 4: sum_difference
    
    dataset = []
    
    # 1. Helper function to generate clean canonical FRAC nodes
    def make_frac(terms):
        return {
            "numi": {"terms": terms},
            "deno": 1
        }
        
    # Generate 100,000 highly diverse samples
    for i in range(100000):
        # Pick a rule class randomly to maintain perfect target feature distribution
        rule = random.randint(0, 4)
        var_name = "x"
        
        if rule == 0: # Power Rule
            coeff = random.randint(1, 20)
            power = random.randint(2, 8)
            
            src_expr = make_frac([{"coeff": coeff, "var": {var_name: power}}])
            ans_expr = make_frac([{"coeff": coeff * power, "var": {var_name: power - 1}}])
            rule_id = 0
            
        elif rule == 1: # Trig Derivatives (sin/cos representation within strict keys)
            # Modeling sin(x) -> cos(x) inside a standardized term wrapper
            src_expr = make_frac([{"coeff": 1, "var": {"sin_x": 1}}])
            ans_expr = make_frac([{"coeff": 1, "var": {"cos_x": 1}}])
            rule_id = 1
            
        elif rule == 2: # Exponential Rule (e^x derivatives representation)
            coeff = random.randint(1, 5)
            src_expr = make_frac([{"coeff": coeff, "var": {"e_x": 1}}])
            ans_expr = make_frac([{"coeff": coeff, "var": {"e_x": 1}}])
            rule_id = 2
            
        elif rule == 3: # Logarithmic Rule (ln(x) mapped tracking)
            src_expr = make_frac([{"coeff": 1, "var": {"ln_x": 1}}])
            ans_expr = make_frac([{"coeff": 1, "var": {"x": -1}}]) # 1/x representation
            rule_id = 3
            
        else: # Sum/Difference of Terms
            c1, c2 = random.randint(1, 15), random.randint(1, 15)
            p1, p2 = random.randint(2, 5), random.randint(2, 5)
            
            src_expr = make_frac([
                {"coeff": c1, "var": {var_name: p1}},
                {"coeff": -c2, "var": {var_name: p2}}
            ])
            ans_expr = make_frac([
                {"coeff": c1 * p1, "var": {var_name: p1 - 1}},
                {"coeff": -c2 * p2, "var": {var_name: p2 - 1}}
            ])
            rule_id = 4

        # Wrap problem into a fully qualified valid OP node envelope
        src_op_node = {
            "op": "diff",
            "var": var_name,
            "expr": src_expr
        }
        
        # 🎯 FIX: Set verification_state = 1 explicitly for true data patterns ONLY. 
        # No intentionally corrupted targets injected into teacher forcing token matrices.
        dataset.append({
            "src_tokens": src_op_node,
            "tgt_input_tokens": ans_expr,
            "tgt_output_tokens": ans_expr,
            "rule_ids": rule_id,
            "verification_state": 1
        })

    # Shuffle to eliminate sequence bias before splitting
    random.shuffle(dataset)
    
    # Split distributions: 90% Train, 5% Val, 5% Test
    train_split = dataset[:90000]
    val_split = dataset[90000:95000]
    test_split = dataset[95000:]
    
    # Save combined master baseline log
    with open("data/slang_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    # Save partitioned splits tracking blocks
    for name, split_data in [("train", train_split), ("val", val_split), ("test", test_split)]:
        with open(splits_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
                
    print(f"✨ [Dataset Engine] Successfully completed! 100,000 structural canonical rows generated.")

if __name__ == "__main__":
    generate_slang_data()