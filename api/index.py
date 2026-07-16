"""
Vercel serverless function: GET /api → health check.
Returns solver status, mode, and readiness info.
"""

from http.server import BaseHTTPRequestHandler

from api._shared import get_solver_status, handle_options, json_response


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        origin = self.headers.get("Origin")
        json_response(self, get_solver_status(), origin=origin)

    def do_OPTIONS(self):
        handle_options(self, origin=self.headers.get("Origin"))

    def log_message(self, format, *args):
        # Suppress default stderr logging in serverless
        pass
