import json
from pathlib import Path

def validate_slang_data():
    splits = ["train.jsonl", "val.jsonl", "test.jsonl"]
    base_dir = Path("data/splits")
    
    print("--- 🩺 SLaNg Data Validation Reports ---")
    for s in splits:
        file_path = base_dir / s
        if not file_path.exists():
            print(f"❌ Missing critical split path: {file_path}")
            return
            
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        print(f"📊 Analyzing {s}: Total Row Records = {len(lines)}")
        
        # Verify first entry structural schema integrity
        first_entry = json.loads(lines[0])
        required_keys = ["src_tokens", "src_positions", "tgt_input_tokens", "tgt_output_tokens", "rule_ids", "verification_state", "text"]
        
        for k in required_keys:
            if k not in first_entry:
                print(f"   ❌ Schema validation failed on key: {k}")
                return
        print(f"   ✅ Schema signatures map perfectly.")
        
    print("\n✨ Phase 1 Documented Status: Success. Ready for training handoff pipeline.")

if __name__ == "__main__":
    validate_slang_data()