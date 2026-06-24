import json
import random
from pathlib import Path

def build_slang_generator(total_samples: int = 100000):
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    rules_map = {
        "power rule": 0,
        "trig derivative": 1,
        "exponential rule": 2,
        "logarithmic rule": 3
    }
    
    templates = [
        {"rule": "power rule", "input": "d/dx[x^{p}]", "correct": "{p}x^{p_minus}", "wrong": "{p}x^{p}"},
        {"rule": "power rule", "input": "d/dx[{c}x^{p}]", "correct": "{cp}x^{p_minus}", "wrong": "{c}x^{p_minus}"},
        {"rule": "trig derivative", "input": "d/dx[sin({c}x)]", "correct": "{c}cos({c}x)", "wrong": "cos({c}x)"},
        {"rule": "trig derivative", "input": "d/dx[cos({c}x)]", "correct": "-{c}sin({c}x)", "wrong": "{c}sin({c}x)"},
        {"rule": "exponential rule", "input": "d/dx[e^{{{c}x}}]", "correct": "{c}e^{{{c}x}}", "wrong": "e^{{{c}x}}"},
        {"rule": "logarithmic rule", "input": "d/dx[ln({c}x)]", "correct": "1/x", "wrong": "{c}/x"}
    ]
    
    dataset = []
    
    for i in range(total_samples):
        t = random.choice(templates)
        p = random.randint(2, 9)
        c = random.randint(2, 6)
        
        # Decide balance flag for verifier state head (50% true steps, 50% false steps)
        is_correct = random.choice([True, False])
        
        inp_str = t["input"].format(p=p, c=c)
        out_str = t["correct"].format(p=p, p_minus=p-1, c=c, cp=c*p) if is_correct else t["wrong"].format(p=p, p_minus=p-1, c=c)
        v_state = 1 if is_correct else 0
        v_tag = "verified" if is_correct else "corrupted"
        
        text_line = f"{inp_str} → {out_str}, {t['rule']}, {v_tag}."
        
        # SLaNg character level structural simulation lists
        src_tokens = list(inp_str)
        src_positions = list(range(len(src_tokens)))
        tgt_in = ["<s>"] + list(out_str)
        tgt_out = list(out_str) + ["</s>"]
        
        dataset.append({
            "src_tokens": src_tokens,
            "src_positions": src_positions,
            "tgt_input_tokens": tgt_in,
            "tgt_output_tokens": tgt_out,
            "rule_ids": rules_map[t["rule"]],
            "verification_state": v_state,
            "text": text_line
        })
        
    # Split distributions shuffle
    random.shuffle(dataset)
    train_idx = int(0.90 * total_samples)
    val_idx = int(0.95 * total_samples)
    
    train_split = dataset[:train_idx]
    val_split = dataset[train_idx:val_idx]
    test_split = dataset[val_idx:]
    
    def save_jsonl(path, data_list):
        with open(path, "w", encoding="utf-8") as f:
            for d in data_list:
                f.write(json.dumps(d) + "\n")
                
    save_jsonl("data/slang_dataset.jsonl", dataset)
    save_jsonl(splits_dir / "train.jsonl", train_split)
    save_jsonl(splits_dir / "val.jsonl", val_split)
    save_jsonl(splits_dir / "test.jsonl", test_split)
    print(f"✅ Generated {total_samples} samples across train/val/test splits.")

if __name__ == "__main__":
    build_slang_generator()
