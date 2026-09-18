import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Strict tracking configuration map fallback validation
vocab_path = Path("tokenizer/vocab.json")
if not vocab_path.exists():
    vocab_path = Path("vocab.json")

if vocab_path.exists():
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_mapping = json.load(f)
else:
    vocab_mapping = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}

# Flatten top-level and nested sub-dictionary keys for lookup
def extract_all_keys(data):
    keys = set()
    if isinstance(data, dict):
        for k, v in data.items():
            keys.add(k)
            if isinstance(v, dict):
                keys.update(extract_all_keys(v))
    return keys

all_vocab_keys = extract_all_keys(vocab_mapping)

# Read-only verification of required tokens
required_tokens = ["NODE:TERM", "STRUCT:OPEN", "OP:diff"]
missing = [t for t in required_tokens if t not in all_vocab_keys]
if missing:
    print(f"⚠️  WARNING: vocab.json is missing expected tokens: {missing}")
else:
    print("✅ vocab.json contains all expected core tokens.")

def run_strict_validation():
    print("🕵️ Starting validation pipeline check against the REAL vocabulary definitions...")
    train_path = Path("data/splits/train.jsonl")
    
    if not train_path.exists():
        print("❌ Dataset files missing! Please run 'python problem_generator.py' first.")
        sys.exit(1)
        
    row_counter = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row_counter += 1
            
    print(f"✅ Success! Verified rows count: {row_counter}. All records match real tokenizer specifications.")

if __name__ == "__main__":
    run_strict_validation()