import sys
import os
import json
from pathlib import Path

# Explicit location bindings
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 🎯 FIX: Direct strict path binding to tokenizer directory configuration map
vocab_path = Path("tokenizer/vocab.json")
if not vocab_path.exists():
    vocab_path = Path("vocab.json") # Workspace fallback resolution

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_mapping = json.load(f)

from tokenizer.slang_serializer import serialize_slang_math

def run_strict_validation():
    print("🕵️ Starting exhaustive structural check and vocabulary matching pipeline verification...")
    train_path = Path("data/splits/train.jsonl")
    
    if not train_path.exists():
        print("❌ Dataset files missing! Please run 'python problem_generator.py' first.")
        sys.exit(1)
        
    row_counter = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row_counter += 1
            
            # Extract structures via real schema
            src_tokens_str = serialize_slang_math(row["src_tokens"])
            tgt_tokens_str = serialize_slang_math(row["tgt_input_tokens"])
            
            src_list = src_tokens_str.split() if isinstance(src_tokens_str, str) else list(src_tokens_str)
            tgt_list = tgt_tokens_str.split() if isinstance(tgt_tokens_str, str) else list(tgt_tokens_str)
            
            # 🎯 FIX: Explicit loud crash fallback across data layers if unexpected tokens leak
            for token in (src_list + tgt_list):
                if token not in vocab_mapping:
                    print(f"❌ CRITICAL EXCEPTION: Token '{token}' at row {row_counter} is completely absent from vocab.json!")
                    print("Aborting pipeline to avoid silent tracking corruption.")
                    sys.exit(1)
                    
    print(f"✅ Success! Verified rows count: {row_counter}. All nodes are authentic structural entities!")

if __name__ == "__main__":
    run_strict_validation()