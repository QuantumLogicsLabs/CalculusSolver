import json
from pathlib import Path
from tokenizer.slang_serializer import serialize_slang_math

def run_strict_validation():
    print("🕵️ Starting execution sanity verification check against dataset paths...")
    train_path = Path("data/splits/train.jsonl")
    
    if not train_path.exists():
        print("❌ Dataset files missing! Please run 'python problem_generator.py' first.")
        return
        
    row_counter = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row_counter += 1
            try:
                # Validate source execution schema
                serialize_slang_math(row["src_tokens"])
                # Validate response/answer target schema shapes 
                serialize_slang_math(row["tgt_input_tokens"])
            except ValueError as e:
                print(f"❌ Exception raised at sample row line index {row_counter}: {e}")
                return
                
    print(f"✅ Success! Verified rows count: {row_counter}. Zero structural exceptions encountered!")

if __name__ == "__main__":
    run_strict_validation()