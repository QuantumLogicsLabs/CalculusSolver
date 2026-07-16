"""
Vercel serverless function: POST /api/validate → validate SLaNg expression.
Accepts: {"expression": {...}} or direct expression.
Returns: {"valid": true} or {"valid": false, "reason": "..."}
"""

from http.server import BaseHTTPRequestHandler

from api._shared import handle_options, json_body, json_response
from tokenizer.slang_serializer import serialize_slang_math


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

        expression = body.get("expression", body)

        try:
            serialize_slang_math(expression)
            json_response(self, {"valid": True}, origin=origin)
        except Exception as error:
            # SLaNg serialization error means invalid format/tokens
            json_response(
                self,
                {"valid": False, "reason": str(error)},
                origin=origin,
            )

    def do_OPTIONS(self):
        handle_options(self, origin=self.headers.get("Origin"))

    def log_message(self, format, *args):
        pass
