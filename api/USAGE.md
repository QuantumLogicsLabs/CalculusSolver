# CalculusSolver API — Usage

## Start
```
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## GET /health
Returns which mode the server is in.
```json
{
  "status": "ok",
  "solver_mode": "fallback",
  "solver_stage": null,
  "solver_loaded": true,
  "checkpoint_error": "No checkpoint found..."
}
```
solver_mode values: "neural" | "fallback"

---

## POST /solve

⚠ Important: wrap the SLaNg envelope inside an "input" key.

Request:
```json
{
  "input": {
    "op": "diff",
    "var": "x",
    "expr": {
      "numi": { "terms": [{ "coeff": 3, "var": { "x": 2 } }] },
      "deno": { "terms": [{ "coeff": 1 }] }
    }
  }
}
```

Response:
```json
{
  "status":     "solved",
  "rule":       "power_rule",
  "confidence": 0.97,
  "verified":   true,
  "expr":       { ... },
  "latex":      "6x",
  "steps":      [{ "rule": "power_rule", "description": "d/dx[x^n] = n*x^(n-1)" }],
  "mode":       "fallback"
}
```

status values: "solved" | "unverified" | "error"
mode values:   "neural" | "fallback"

---

## POST /validate
```json
{ "expression": { "numi": {...}, "deno": {...} } }
```
Returns: `{ "valid": true }` or `{ "valid": false, "reason": "..." }`

---

## JavaScript fetch for Member 4
```javascript
// calculusSolverClient.js — update your solve() function to this:

export async function solve(text, opType, variable) {
  const res = await fetch(`${API_BASE}/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: {              // ← must be nested inside "input"
        op:         opType,
        var:        variable,
        text_input: text.trim(),
      }
    }),
  });
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return res.json();
}
```

## HTTP Status Codes
| Code | Meaning |
|------|---------|
| 200  | Success — check status field |
| 400  | Invalid JSON body |
| 422  | Missing "input" field or unsupported op |
| 503  | Solver not initialised |
| 500  | Solver internal error |
