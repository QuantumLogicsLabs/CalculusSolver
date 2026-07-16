"""
Vercel serverless function: POST /api/solve → run the calculus solver.

Accepts a SLaNg input envelope and returns the solution with steps,
LaTeX output, confidence, and verification status.

Request body:
    {"input": {"op": "diff", "var": "x", "expr": {...}}}

Response:
    {"status": "solved", "expr": {...}, "steps": [...], "latex": "...", ...}
"""

from http.server import BaseHTTPRequestHandler

from api._shared import (
    get_solver,
    handle_options,
    json_body,
    json_response,
    normalize_solver_result,
)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        origin = self.headers.get("Origin")

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        # Parse JSON
        try:
            body = json_body(raw_body)
        except ValueError:
            json_response(
                self, {"detail": "Invalid JSON body."}, status=400, origin=origin
            )
            return

        # Extract input envelope — accept both {"input": {...}} and direct envelope
        input_env = body.get("input", body)
        if not isinstance(input_env, dict):
            json_response(
                self,
                {"detail": "'input' must be a JSON object (SLaNg envelope)."},
                status=422,
                origin=origin,
            )
            return

        # Solve
        solver, mode = get_solver()
        if solver is None:
            json_response(
                self,
                {"detail": "Solver not initialised."},
                status=503,
                origin=origin,
            )
            return

        try:
            result = solver.solve(input_env)
        except ValueError as exc:
            json_response(
                self, {"detail": str(exc)}, status=422, origin=origin
            )
            return
        except Exception as exc:
            json_response(
                self, {"detail": f"Solver error: {exc}"}, status=500, origin=origin
            )
            return

        # Normalize and tag result
        normalized = normalize_solver_result(result, mode)
        json_response(self, normalized, origin=origin)

    def do_OPTIONS(self):
        handle_options(self, origin=self.headers.get("Origin"))

    def log_message(self, format, *args):
        pass
