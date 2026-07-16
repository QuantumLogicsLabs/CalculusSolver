# CalculusSolver — Vercel-Ready API

A production-ready Python API designed to solve calculus expressions represented in SLaNg AST format. Refactored for seamless deployment on Vercel Serverless Functions.

## Key Features

1. **Dual Execution Modes**:
   - **Production Mode (Vercel)**: Uses `FallbackSolver` (deterministic, pure-Python polynomial solver supporting `diff`, `partial`, `integrate`, `gradient`, and `tangent_line`). Optionally uses `GroqSolver` if `GROQ_API_KEY` is set. Requires zero heavy dependencies, making it extremely fast and lightweight.
   - **Neural Mode (Local/Research Only)**: Available for local development and training research. Automatically activated if a PyTorch model checkpoint is detected. Not deployed to Vercel due to the 250MB serverless size cap (see [HOSTING_DECISION.md](docs/HOSTING_DECISION.md)).
2. **Pure Python Architecture**:
   - Re-implemented SLaNg AST validation, algebraic verification, and LaTeX conversion in 100% pure Python.
   - Zero Node.js subprocesses. No dependency on local Node installations.
3. **Vercel Serverless Optimization**:
   - Organized as standalone serverless handlers under `api/` using Vercel's native `BaseHTTPRequestHandler` python runtime.
   - Supercharged build speeds with minimized dependencies (~50MB package limitation compliant).

---

## Directory Structure

```
CalculusSolver/
├── api/
│   ├── _shared.py       # Shared solver loading, LaTeX formatting & CORS helpers
│   ├── index.py         # GET /api (Health Check & Active Mode info)
│   ├── solve.py         # POST /api/solve (Calculus solving handler)
│   └── validate.py      # POST /api/validate (SLaNg AST structure validator)
│   └── app.py           # Starlette server (for local dev)
│
├── inference/
│   ├── beam_search.py   # Pure-Python search logic
│   ├── fallback_solver.py # Deterministic polynomial solver
│   ├── solve.py         # Neural model loader & solver pipeline
│   └── verifier.py      # Pure-Python math verifier & test-point evaluator
│
├── model/
│   ├── architecture.py  # PyTorch model definitions
│   └── ...              # Tree encoder, tree decoder, rule heads
│
├── tokenizer/
│   ├── slang_serializer.py # Pure-Python SLaNg AST serializer/deserializer
│   └── vocab.json       # SLaNg vocabulary mappings
│
├── requirements.txt      # Production-only/local dev dependencies (Starlette, Uvicorn)
├── requirements-neural.txt # Optional dependencies for neural weights (PyTorch, Numpy)
├── vercel.json           # Vercel configuration for Python serverless builds
└── README.md
```

---

## Installation & Local Development

### 1. Minimal Installation (Fallback Mode)
For development, testing, and standard Vercel deployments, install the lightweight dependencies. Local dev and Vercel use `requirements.txt` only. Neural/local training uses `requirements-neural.txt` (see below) — do not install it before deploying to Vercel.
```bash
pip install -r requirements.txt
```

Run the local development server:
```bash
uvicorn api.app:app --reload --port 8000
```

### 2. Full Neural Installation (Optional)
If you want to run neural inference locally with model weights, install the neural requirements (which includes PyTorch and NumPy):
```bash
pip install -r requirements-neural.txt
```

Place your trained model weight checkpoint (`best.pt`) in one of the priority locations:
1. `checkpoints/final/best.pt`
2. `checkpoints/sft/best.pt`
3. `checkpoints/pretrain/best.pt`
4. Or set the `MODEL_PATH` environment variable pointing to your `.pt` file.

---

## API Endpoints

### 1. GET `/api` (or `/health` on local dev)
Returns active solver configuration and health status.
**Response**:
```json
{
  "status": "ok",
  "solver_mode": "fallback",
  "solver_loaded": true,
  "checkpoint_error": "No checkpoint found..."
}
```

### 2. POST `/api/solve` (or `/solve` on local dev)
Solves a calculus operation on the given SLaNg AST expression.
**Request Body**:
```json
{
  "input": {
    "op": "diff",
    "var": "x",
    "expr": {
      "numi": {
        "terms": [{"coeff": 3, "var": {"x": 2}}]
      },
      "deno": 1
    }
  }
}
```
**Response**:
```json
{
  "status": "solved",
  "expr": {
    "numi": {
      "terms": [{"coeff": 6, "var": {"x": 1}}]
    },
    "deno": 1
  },
  "steps": [
    {
      "rule": "power_rule",
      "description": "Differentiated with respect to x using the power rule.",
      "before": "3x^{2}",
      "after": "6x"
    }
  ],
  "latex": "6x",
  "confidence": 1.0,
  "verified": true,
  "warning": "Fallback mode — no neural checkpoint loaded.",
  "rule": "power_rule",
  "mode": "fallback"
}
```

### 3. POST `/api/validate` (or `/validate` on local dev)
Validates that the provided SLaNg AST node or structure can be serialized.
**Request Body**:
```json
{
  "expression": {
    "numi": {
      "terms": [{"coeff": 3, "var": {"x": 2}}]
    },
    "deno": 1
  }
}
```
**Response**:
```json
{
  "valid": true
}
```

---

## Vercel Deployment

Deploying the CalculusSolver API to Vercel is simple. You can use the Vercel CLI:
```bash
vercel
```
Or connect your GitHub repository containing this codebase directly to your Vercel dashboard. Vercel will automatically read `vercel.json`, build the serverless functions using `@vercel/python`, and expose the endpoints.

### Hosting & Deployment Decision

> **Production scope**: The Vercel deployment runs `FallbackSolver` (deterministic polynomial solver) and optionally `GroqSolver` (LLM-based). The PyTorch neural model is excluded from production due to the 250MB serverless size cap. Neural mode is available for local training, research, and evaluation only. See [docs/HOSTING_DECISION.md](docs/HOSTING_DECISION.md) for the full rationale.
