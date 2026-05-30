import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class SolveRequest(BaseModel):
    input: Dict[str, Any]


class SolveResponse(BaseModel):
    status: str
    expr: Any
    steps: List[Any]
    latex: Optional[Any]
    confidence: float
    verified: Optional[bool] = None
    warning: Optional[str] = None
    rule: Optional[str] = None


def _format_latex(expression: Any) -> Optional[Any]:
    script_path = Path(__file__).resolve().parents[1] / "format_slang_expression.js"
    proc = subprocess.run(
        ["node", "--input-type=module", str(script_path)],
        input=json.dumps({"expression": expression}),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
        return payload.get("latex")
    except json.JSONDecodeError:
        return None


def _unwrap_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict) and "expr" in output:
        return {
            "expr": output["expr"],
            "steps": output.get("steps", []),
        }
    return {"expr": output, "steps": []}


@router.post("/solve", response_model=SolveResponse)
def solve(request: Request, body: SolveRequest) -> Dict[str, Any]:
    solver = getattr(request.app.state, "solver", None)
    if solver is None:
        detail = getattr(request.app.state, "solver_error", None)
        if detail:
            raise HTTPException(
                status_code=503, detail=f"Solver is not available: {detail}"
            )
        raise HTTPException(status_code=503, detail="Solver is not available.")

    result = solver.solve(body.input)
    unpacked = _unwrap_output(result.get("output"))
    latex = _format_latex(unpacked["expr"])

    return {
        "status": result.get("status", "unverified"),
        "expr": unpacked["expr"],
        "steps": unpacked["steps"],
        "latex": latex,
        "confidence": float(result.get("confidence", 0.0)),
        "verified": result.get("verified"),
        "warning": result.get("warning"),
        "rule": result.get("root_rule_label"),
    }
