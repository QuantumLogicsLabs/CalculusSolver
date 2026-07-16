"""
CalculusSolver Neural Inference Service.
Runs as a standalone microservice to host the PyTorch neural model.
Exposes POST /solve.
"""

import os
import sys
from pathlib import Path
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.solve import CalculusSolverInference

# Initialize the neural solver
checkpoint_path = os.environ.get("MODEL_PATH", str(ROOT / "checkpoints" / "final" / "best.pt"))
print(f"[Neural Service] Loading neural checkpoint from: {checkpoint_path}")
solver = CalculusSolverInference(model_path=checkpoint_path)


async def solve_handler(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)

    input_env = body.get("input", body)
    if not isinstance(input_env, dict):
        return JSONResponse(
            {"detail": "'input' must be a JSON object (SLaNg envelope)."},
            status_code=422,
        )

    try:
        result = solver.solve(input_env)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"detail": f"Neural Solver error: {exc}"}, status_code=500)


async def health_handler(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "mode": "neural",
        "checkpoint": checkpoint_path,
        "exists": os.path.exists(checkpoint_path)
    })


app = Starlette(
    routes=[
        Route("/solve", solve_handler, methods=["POST"]),
        Route("/health", health_handler, methods=["GET"]),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)
