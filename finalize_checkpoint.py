"""
Copy the trained checkpoint (checkpoints/final/best.pt) into model/model.pkl,
which is the file inference/solve.py loads by default in production.

Run this AFTER training completes:
    python finalize_checkpoint.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "checkpoints" / "final" / "best.pt"
DST = ROOT / "model" / "model.pkl"

if not SRC.exists():
    raise SystemExit(f"No trained checkpoint found at {SRC} — run train.py first.")

DST.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(SRC, DST)
print(f"Copied {SRC} -> {DST}")
print("model/model.pkl now reflects the latest training run.")
print("(Loading no longer depends on the .pkl extension — the checkpoint")
print(" itself now states its architecture, so this file loads correctly")
print(" via inference/solve.py's default model_path.)")