# Known Issues and Tracked Gaps

This file records bugs and discrepancies that were discovered, their root cause,
and their resolution. Each entry includes the fix reference so the full history
is preserved even after the issue is closed.

---

## [RESOLVED] STRUCT:OPEN missing from tokenizer/vocab.json

**Discovered:** During unit test implementation (Task 1, PR adding test suite)  
**Fixed:** Fix 3 (vocab.json v1.1)  
**Severity:** High — silent data corruption in neural training and inference  
**Affected path:** Neural solver only (FallbackSolver and GroqSolver unaffected)

### What was wrong

`tokenizer/slang_serializer.py` emits `"STRUCT:OPEN"` as the opening bracket
token for every fraction node and op-node it serializes. This token is defined
as a module-level constant:

```python
OPEN = "STRUCT:OPEN"
```

`tokenizer/vocab.json` defined `STRUCT:CLOSE` (ID 7) but had no entry for
`STRUCT:OPEN`. The `structure_tokens` section contained six tokens (IDs 4–9)
with no gap available for insertion without renumbering.

### Why it mattered

In `inference/solve.py`, `CalculusSolverInference._serialize_input()` converts
token strings to integer IDs using:

```python
self.vocab_map["token_to_id"].get(token, self.pad_id)
```

Because `"STRUCT:OPEN"` was absent from the vocab map, every occurrence of it
silently mapped to `self.pad_id` (ID 0, `[PAD]`). This corrupted the entire
input token sequence before it reached the transformer encoder — every
structural opening bracket was encoded as padding.

### Why it was not caught earlier

The FallbackSolver and GroqSolver do not use the vocab at all — they operate
on raw SLaNg dicts. The discrepancy only affects the neural path
(`CalculusSolverInference`), which requires a trained checkpoint to exercise.
In CI and local development without a checkpoint, the neural path is never
reached, so the corruption was invisible.

The unit test for `test_fraction_contains_struct_open` correctly asserted that
`"STRUCT:OPEN"` appears in the serialized token list, but the test had no
assertion that the token also exists in the vocab — it only tested the
serializer output, not the vocab lookup.

### The fix

`STRUCT:OPEN` was assigned ID **23** in `vocab.json` v1.1. ID 23 was chosen
because:

- IDs 4–9 (structure tokens) were fully occupied; inserting there would
  require renumbering existing tokens and invalidating all trained model weights
- ID 23 is the first unused ID after operation tokens end at 22
- Assigning it here shifts nothing and breaks no existing weights

### Files changed

| File | Change |
|---|---|
| `tokenizer/vocab.json` | Added `"STRUCT:OPEN": 23` to `structure_tokens`; version bumped to `1.1` |
| `tests/unit/test_slang_serializer.py` | Updated two comments that described this as a known discrepancy |
| `docs/KNOWN_ISSUES.md` | This file created |

### Verification

After the fix, the following confirms `STRUCT:OPEN` is correctly round-trippable
through the vocab:

```python
import json

with open("tokenizer/vocab.json") as f:
    vocab = json.load(f)

token_to_id = {}
for category in vocab.values():
    if isinstance(category, dict):
        token_to_id.update(category)

assert "STRUCT:OPEN" in token_to_id, "STRUCT:OPEN must be in vocab"
assert token_to_id["STRUCT:OPEN"] == 23, "STRUCT:OPEN must have ID 23"

id_to_token = {v: k for k, v in token_to_id.items()}
assert id_to_token[23] == "STRUCT:OPEN", "ID 23 must map back to STRUCT:OPEN"

print("STRUCT:OPEN correctly registered at ID 23")
```

---

## [RESOLVED] Vercel build fails due to empty website/ directory

**Discovered:** During deployment packaging audit (Task 4)  
**Fixed:** Removed website build steps from vercel.json  
**Severity:** High — completely blocks Vercel deployment  
**Affected path:** Deployment / API hosting

### What was wrong

`vercel.json` contained a `buildCommand` (`cd website && npm install && npm run build`) and an `outputDirectory` (`website/dist`). However, the `website/` directory in this repository is an uninitialized or empty submodule with no `package.json`.

### Why it mattered

Vercel's build process executes the `buildCommand` before attempting to deploy any serverless functions. Because `npm install` fails in an empty directory without a package definition, the entire build process would crash. This prevented the Python API functions under `api/` from ever being built or deployed, resulting in a completely broken deployment pipeline.

### The fix

Since the focus is currently on an API-only deployment (and the frontend is either non-existent or managed elsewhere), the `buildCommand` and `outputDirectory` keys were entirely removed from `vercel.json`. Vercel now correctly defaults to only building the Python functions defined in the `builds` array.

### Files changed

| File | Change |
|---|---|
| `vercel.json` | Removed `buildCommand` and `outputDirectory` keys |
| `docs/KNOWN_ISSUES.md` | Added this entry |

---

## [RESOLVED] tokenizer/vocab.json missing function tokens for trig/exp/log (Phase 2 vocab expansion)

**Discovered:** Flagged as deferred scope in `NEURAL_DEPLOYMENT.md` ("Phase 2 — Trig/exp/log vocabulary expansion") and in `DATASET_REPORT.md`'s coverage gap section  
**Fixed:** vocab.json v1.2  
**Severity:** Medium — blocks dataset/model coverage of non-polynomial expressions, not a data-corruption risk like the STRUCT:OPEN issue  
**Affected path:** Neural solver dataset generation and training only (FallbackSolver and GroqSolver unaffected — they operate on raw SLaNg dicts, not the vocab)

### What was wrong

The training dataset only covered polynomial expressions (power rule, sum rule, constant terms,
partial derivatives). `tokenizer/vocab.json` had no tokens representing the mathematical functions
`sin`, `cos`, `tan`, `exp`, or `ln`, so the dataset generator and tokenizer had no way to represent
trigonometric, exponential, or logarithmic expressions even if problem templates were written for them.

### Why it mattered

Without these tokens, the model could never learn to solve anything beyond polynomials, regardless
of training quality — the vocabulary itself set a hard ceiling on what expressions could be
represented, tokenized, and fed to the transformer encoder.

### The fix

Added a new `function_tokens` category to `vocab.json`, assigning IDs **100–104**:

| Token | ID |
|---|---|
| `FUNC:sin` | 100 |
| `FUNC:cos` | 101 |
| `FUNC:tan` | 102 |
| `FUNC:exp` | 103 |
| `FUNC:ln` | 104 |

Following the same precedent as the `STRUCT:OPEN` fix above:

- IDs were appended strictly after the current highest existing ID (99, `RULE:integration_by_parts`)
- No existing token was renumbered or reused, so no previously trained model weights are invalidated
- A new top-level category (`function_tokens`) was used rather than inserting into `operation_tokens`,
  since `sin`/`cos`/`tan`/`exp`/`ln` are mathematical functions, not operations like `diff`/`integrate`,
  keeping the same semantic separation already used between `OP:`, `RULE:`, and `VAR:` namespaces

### Files changed

| File | Change |
|---|---|
| `tokenizer/vocab.json` | Added `function_tokens` block (`FUNC:sin`, `FUNC:cos`, `FUNC:tan`, `FUNC:exp`, `FUNC:ln`, IDs 100–104); version bumped to `1.2` |
| `problem_generator.py` | New trig/exp/log problem templates added alongside existing polynomial templates (pending — see PR) |
| `DATASET_REPORT.md` | Coverage section updated to reflect new trig/exp/log support (pending — see PR) |
| `docs/KNOWN_ISSUES.md` | This entry added |

### Verification

```python
import json

with open("tokenizer/vocab.json") as f:
    vocab = json.load(f)

token_to_id = {}
for category in vocab.values():
    if isinstance(category, dict):
        token_to_id.update(category)

expected = {
    "FUNC:sin": 100,
    "FUNC:cos": 101,
    "FUNC:tan": 102,
    "FUNC:exp": 103,
    "FUNC:ln": 104,
}
for tok, expected_id in expected.items():
    assert token_to_id[tok] == expected_id, f"{tok} should be {expected_id}"

id_to_token = {v: k for k, v in token_to_id.items()}
for expected_id, tok in {v: k for k, v in expected.items()}.items():
    assert id_to_token[expected_id] == tok, f"ID {expected_id} should map to {tok}"

print("All 5 function tokens correctly registered at IDs 100-104")
```

---

## Filing new issues

To add a new entry, copy the template below and fill it in:

```markdown
## [STATUS] Short description

**Discovered:** When/how found  
**Fixed:** Fix reference or "Pending"  
**Severity:** Low / Medium / High  
**Affected path:** Which solver modes / components are affected

### What was wrong
...

### Why it mattered
...

### The fix
...

### Files changed
...
```