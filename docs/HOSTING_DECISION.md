# Hosting Decision -- Neural Model vs Vercel Serverless

## Constraint
Vercel Python serverless functions cap deployment size at 250MB unzipped.
torch + numpy + model.pkl exceed this by 2-3x. Confirmed not feasible to
run the neural path (`inference/solve.py`) inside a Vercel function.

## Decision: Option (B) -- Fallback + Groq Only in Production

**Chosen approach**: Production (Vercel) runs `FallbackSolver` and optionally
`GroqSolver` only. The PyTorch neural model is scoped as a **local
development and research** tool.

### What runs in production
- **FallbackSolver** (deterministic, pure-Python): Handles `diff`, `partial`,
  `integrate`, `gradient`, `tangent_line` for polynomial expressions.
- **GroqSolver** (optional, LLM-based): Activated when `GROQ_API_KEY` env
  var is set. Uses Groq API for broader expression support.
- No torch/numpy in `requirements.txt`.

### What is local/research only
- **Neural mode** (`model/`, `train.py`, `checkpoints/`): Gated behind
  `requirements-neural.txt`. Used for training experiments and accuracy
  research.
- `checkpoints/final/best.pt`: Produced by `train.py`, used for local
  evaluation via `run_pipeline.py`. Not deployed to Vercel.

## Why this doesn't limit functionality today
`FallbackSolver` covers the full scope described in README.md's
"Key Features" for polynomial expressions. Neural mode is an accuracy
upgrade for non-polynomial rules (trig, exp, log) -- not a hard
requirement for the current launch.

## When to revisit (criteria for Option A)
Consider self-hosting the neural model (Render, Fly.io, HF Spaces) if:
1. Non-polynomial rule support (trig, exp, log, chain rule) becomes a
   hard product requirement.
2. The model demonstrates >90% accuracy on non-polynomial benchmarks.
3. Latency from an external model service is acceptable (<2s p95).

At that point, deploy the PyTorch model as a separate microservice with
`MODEL_PATH` pointing to the hosted checkpoint, and route neural requests
from the Vercel API via HTTP.
