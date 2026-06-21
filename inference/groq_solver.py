"""
GroqSolver — Intelligent calculus solver using Groq API.

Replaces the local PyTorch model to save space for Vercel deployment.
"""

import json
import os
from typing import Any, Dict

from groq import Groq


class GroqSolver:
    """
    Intelligent model solver using the Groq API.
    Implements the same .solve(payload) interface as FallbackSolver.
    """

    mode = "groq"
    stage = "production"

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
        
        self.client = Groq(api_key=self.api_key)
        self.model_name = os.environ.get("GROQ_MODEL", "llama3-70b-8192")

    def solve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send the SLaNg payload to Groq and expect a JSON response back.
        """
        system_prompt = (
            "You are an advanced calculus solver API. "
            "You will be given a JSON object representing a calculus problem (SLaNg envelope). "
            "Your task is to solve it step-by-step and return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            "  \"status\": \"solved\",\n"
            "  \"expr\": { \"numi\": { \"terms\": [...] }, \"deno\": ... },\n"
            "  \"steps\": [\n"
            "    { \"rule\": \"...\", \"description\": \"...\", \"before\": \"...\", \"after\": \"...\" }\n"
            "  ],\n"
            "  \"latex\": \"...\",\n"
            "  \"confidence\": 1.0,\n"
            "  \"verified\": true,\n"
            "  \"rule\": \"...\"\n"
            "}\n"
            "No markdown blocks, no extra text, just raw JSON."
        )

        user_prompt = f"Solve this:\n{json.dumps(payload, indent=2)}"

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model_name,
                temperature=0.0, # Deterministic answers
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content
            if not raw_content:
                raise ValueError("Empty response from Groq API")

            parsed = json.loads(raw_content)
            
            # Ensure required fields
            parsed["mode"] = self.mode
            if "status" not in parsed:
                parsed["status"] = "solved"
            if "confidence" not in parsed:
                parsed["confidence"] = 1.0
            if "verified" not in parsed:
                parsed["verified"] = True

            return parsed
        except Exception as e:
            # Re-raise to be handled by the endpoint
            raise ValueError(f"Groq API error or invalid JSON: {str(e)}")
