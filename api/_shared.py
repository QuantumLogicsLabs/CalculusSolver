"""
Shared module for Vercel serverless functions.
Provides solver singleton, CORS headers, and JSON response helpers.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── CORS ──────────────────────────────────────────────────────────────────────

_DEFAULT_ORIGINS = (
    "*,"
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://127.0.0.1:8002"
)

ALLOWED_ORIGINS = set(
    os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
)


def cors_headers(origin: Optional[str] = None) -> Dict[str, str]:
    """Return CORS headers for the given origin (or wildcard if allowed)."""
    headers = {
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "86400",
    }
    if origin and (origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS):
        headers["Access-Control-Allow-Origin"] = origin
    elif "*" in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = "*"
    else:
        # Default to first allowed origin for non-browser requests
        headers["Access-Control-Allow-Origin"] = next(iter(ALLOWED_ORIGINS), "*")
    return headers


# ── JSON helpers ──────────────────────────────────────────────────────────────


def json_body(body: bytes) -> Dict[str, Any]:
    """Parse a JSON request body, raising ValueError on failure."""
    try:
        return json.loads(body)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc


def json_response(
    handler,
    data: Any,
    status: int = 200,
    origin: Optional[str] = None,
) -> None:
    """Write a JSON response with CORS headers to an HTTP handler."""
    payload = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    for key, value in cors_headers(origin).items():
        handler.send_header(key, value)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def handle_options(handler, origin: Optional[str] = None) -> None:
    """Handle CORS preflight OPTIONS request."""
    handler.send_response(204)
    for key, value in cors_headers(origin).items():
        handler.send_header(key, value)
    handler.end_headers()


# ── Solver singleton ──────────────────────────────────────────────────────────

_solver = None
_solver_mode = "unloaded"
_solver_error = None

CHECKPOINT_PRIORITY = [
    ("final", ROOT / "checkpoints" / "final" / "best.pt"),
    ("sft", ROOT / "checkpoints" / "sft" / "best.pt"),
    ("pretrain", ROOT / "checkpoints" / "pretrain" / "best.pt"),
]


def _resolve_model_path():
    """Find the best available checkpoint."""
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        p = p if p.is_absolute() else ROOT / p
        if p.exists():
            return str(p), "env"
    for stage, path in CHECKPOINT_PRIORITY:
        if path.exists():
            return str(path), stage
    return None, None


# NOTE: keep `groq` and `torch` imports lazy (inside the branches below).
# The Vercel build only installs requirements.txt (no torch/groq-heavy deps).
# A top-level import here will break every serverless function at import time.
class HttpNeuralSolverProxy:
    """Proxy that forwards solve requests to a remote neural inference service."""
    def __init__(self, url: str):
        self.url = url

    def solve(self, payload: dict) -> dict:
        import urllib.request
        import json
        
        # Ensure we send the correct format
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req) as res:
                return json.loads(res.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"[Neural Proxy] HTTP solve request failed: {exc}")


# NOTE: keep `groq` and `torch` imports lazy (inside the branches below).
# The Vercel build only installs requirements.txt (no torch/groq-heavy deps).
# A top-level import here will break every serverless function at import time.
def get_solver():
    """
    Lazy-load and return the solver singleton.
    Returns (solver, mode, stage, error).
    """
    global _solver, _solver_mode, _solver_error

    if _solver is not None:
        return _solver, _solver_mode

    # 1. First priority: Remote neural inference url
    neural_url = os.environ.get("NEURAL_INFERENCE_URL")
    if neural_url:
        try:
            _solver = HttpNeuralSolverProxy(neural_url)
            _solver_mode = "neural"
            _solver_error = None
            print(
                f"[CalculusSolver] Proxying neural mode to remote service: {neural_url}",
                flush=True,
            )
            return _solver, _solver_mode
        except Exception as exc:
            _solver_error = str(exc)
            print(f"[CalculusSolver] Neural proxy init failed: {exc}", flush=True)

    # 2. Second priority: Local neural solver (resolve checkpoint path first)
    model_path, stage = _resolve_model_path()
    if model_path is not None:
        try:
            from inference.solve import CalculusSolverInference
            _solver = CalculusSolverInference(model_path=model_path)
            _solver_mode = "neural"
            _solver_error = None
            print(
                f"[CalculusSolver] Neural model loaded from stage='{stage}' path='{model_path}'",
                flush=True,
            )
            return _solver, _solver_mode
        except Exception as exc:
            _solver_error = str(exc)
            print(
                f"[CalculusSolver] Neural load failed (stage='{stage}', path='{model_path}'): {exc}",
                flush=True,
            )

    # 3. Third priority: Try to load GroqSolver (Fallback intelligent model)
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            from inference.groq_solver import GroqSolver

            _solver = GroqSolver()
            _solver_mode = "groq"
            _solver_error = None
            print(
                f"[CalculusSolver] Groq model loaded — using {os.environ.get('GROQ_MODEL', 'llama3-70b-8192')}",
                flush=True,
            )
            return _solver, _solver_mode
        except Exception as exc:
            _solver_error = str(exc)
            print(f"[CalculusSolver] Groq load failed: {exc}", flush=True)

    # 4. Final priority: Fallback
    from inference.fallback_solver import FallbackSolver
    _solver = FallbackSolver()
    _solver_mode = "fallback"
    if not _solver_error:
        _solver_error = (
            "No NEURAL_INFERENCE_URL, neural checkpoint, or GROQ_API_KEY provided. "
            "Falling back to deterministic solver."
        )
    print(
        "[CalculusSolver] Running in FALLBACK mode — "
        "supports diff, partial, integrate, gradient, tangent_line.",
        flush=True,
    )
    return _solver, _solver_mode


def get_solver_status() -> Dict[str, Any]:
    """Return solver health status dict."""
    solver, mode = get_solver()
    return {
        "status": "ok",
        "solver_mode": mode,
        "solver_loaded": solver is not None,
        "checkpoint_error": _solver_error if mode != "groq" else None,
    }


def term_to_latex(term: dict) -> str:
    """Format a SLaNg term object as LaTeX."""
    coeff = term.get("coeff", 0)
    variables = term.get("var", {})
    if isinstance(coeff, float) and coeff.is_integer():
        coeff = int(coeff)
    if not variables:
        return str(coeff)
    parts = []
    if coeff == -1:
        parts.append("-")
    elif coeff != 1:
        parts.append(str(coeff))
    for name, power in variables.items():
        if isinstance(power, float) and power.is_integer():
            power = int(power)
        parts.append(name if power == 1 else f"{name}^{{{power}}}")
    return "".join(parts)


def fraction_to_latex(expr: dict) -> str:
    """Format a SLaNg fraction object (or gradient dict) as LaTeX."""
    if not isinstance(expr, dict):
        return str(expr)
    if "gradient" in expr:
        grad_dict = expr["gradient"]
        vars_sorted = sorted(grad_dict.keys())
        parts = [fraction_to_latex(grad_dict[v]) for v in vars_sorted]
        return "\\nabla f = (" + ", ".join(parts) + ")"
        
    terms = expr.get("numi", {}).get("terms", []) if isinstance(expr.get("numi"), dict) else expr.get("numi", [])
    if not isinstance(terms, list):
        terms = [terms] if terms else []
    if not terms:
        return "0"
    parts = []
    for t in terms:
        if isinstance(t, dict):
            s = term_to_latex(t)
        else:
            s = str(t)
        if parts and not s.startswith("-"):
            parts.append("+")
        parts.append(s)
    numerator = " ".join(parts)
    
    deno = expr.get("deno", 1)
    if isinstance(deno, dict):
        deno_latex = fraction_to_latex(deno)
    elif isinstance(deno, list):
        deno_latex = " ".join(term_to_latex(t) if isinstance(t, dict) else str(t) for t in deno)
    else:
        deno_latex = str(deno)
    return numerator if deno_latex == "1" else f"\\frac{{{numerator}}}{{{deno_latex}}}"


def normalize_solver_result(result: dict, mode: str) -> dict:
    """Normalize/unwrap solver output into the standard API response format."""
    if mode in ("neural", "inference"):
        output = result.get("output") or {}
        if isinstance(output, dict) and "expr" in output:
            expr = output["expr"]
            steps = output.get("steps", [])
        else:
            expr = output
            steps = []
        latex = fraction_to_latex(expr)
        return {
            "status": result.get("status", "unverified"),
            "expr": expr,
            "steps": steps,
            "latex": latex,
            "confidence": float(result.get("confidence", 0.0)),
            "verified": bool(result.get("verified")),
            "warning": result.get("warning"),
            "rule": result.get("rule"),
            "mode": mode,
        }
    else:
        # Fallback and Groq solver results already have the correct structure
        return {**result, "mode": mode}

