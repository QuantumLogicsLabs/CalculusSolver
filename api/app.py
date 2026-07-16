"""
CalculusSolver API — Starlette only, no FastAPI, no pydantic.
Local dev entrypoint.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from api._shared import get_solver, get_solver_status, normalize_solver_result
from tokenizer.slang_serializer import serialize_slang_math

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app):
    # Eagerly initialize solver on startup
    get_solver()
    yield


# ── Route Handlers ────────────────────────────────────────────────────────────

async def root_handler(request: Request) -> JSONResponse:
    return JSONResponse({"message": "CalculusSolver API is running. Use POST /api/solve to solve equations."})

async def health_handler(request: Request) -> JSONResponse:
    return JSONResponse(get_solver_status())


async def solve_handler(request: Request) -> JSONResponse:
    solver, mode = get_solver()
    if solver is None:
        return JSONResponse({"detail": "Solver not initialised."}, status_code=503)

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
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"detail": f"Solver error: {exc}"}, status_code=500)

    normalized = normalize_solver_result(result, mode)
    return JSONResponse(normalized)


async def validate_handler(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)

    expression = body.get("expression", body)

    try:
        serialize_slang_math(expression)
        return JSONResponse({"valid": True})
    except Exception as error:
        return JSONResponse({"valid": False, "reason": str(error)})


# ── Application setup ─────────────────────────────────────────────────────────

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://127.0.0.1:8002",
).split(",")

app = Starlette(
    debug=False,
    routes=[
        Route("/", root_handler),
        Route("/api/health", health_handler),
        Route("/api/solve", solve_handler, methods=["POST"]),
        Route("/api/validate", validate_handler, methods=["POST"]),
        # Also support without /api for local fallback if needed
        Route("/health", health_handler),
        Route("/solve", solve_handler, methods=["POST"]),
        Route("/validate", validate_handler, methods=["POST"]),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
    lifespan=lifespan,
)
