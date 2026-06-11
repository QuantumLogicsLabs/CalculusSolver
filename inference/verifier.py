import json
import subprocess
from pathlib import Path

# Points to inference/verifier.js
VERIFY_JS = Path(__file__).resolve().parent / "verifier.js"


def verify_answer(input_envelope: dict, output_tokens: list, timeout: float = 5.0) -> dict:
    """
    Calls inference/verifier.js via stdin/stdout.

    Sends:   { "input": {...SLaNg envelope...}, "output_tokens": [...] }
    Returns: { "status": "solved"/"unverified", "verified": true/false, "confidence": float, "output": {...SLaNg expr...} }
    """
    if not VERIFY_JS.exists():
        return {
            "verified":   False,
            "confidence": 0.0,
            "status":     "unverified",
            "error":      f"verifier.js not found at {VERIFY_JS}",
        }

    payload = json.dumps({
        "input":        input_envelope,
        "output_tokens": output_tokens,
    })

    try:
        proc = subprocess.run(
            ["node", "--input-type=module", str(VERIFY_JS)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.stdout.strip():
            return json.loads(proc.stdout.strip())
        return {
            "verified":   False,
            "confidence": 0.0,
            "status":     "unverified",
            "error":      proc.stderr.strip() or "No output from verifier",
        }
    except subprocess.TimeoutExpired:
        return {"verified": False, "confidence": 0.0,
                "status": "unverified",
                "error": f"Verifier timed out after {timeout}s"}
    except Exception as e:
        return {"verified": False, "confidence": 0.0,
                "status": "unverified", "error": str(e)}


def verify_answer_safe(input_envelope: dict, output_tokens: list) -> dict:
    """Never raises — safe to call inside the API."""
    try:
        return verify_answer(input_envelope, output_tokens)
    except Exception as e:
        return {"verified": False, "confidence": 0.0,
                "status": "unverified", "error": str(e)}