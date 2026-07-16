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

## [RESOLVED] model/model.pkl checkpoint architecture mismatch

**Discovered:** Member A task brief — inference raised a state_dict key mismatch on load
**Fixed:** Checkpoint retired; retrained from scratch on current architecture
**Severity:** High — blocks the neural inference path entirely
**Affected path:** Neural solver only (`inference/solve.py` non-`.pt` branch)

### What was wrong

`model/model.pkl` was inspected directly (raw pickle key extraction, without
needing to construct the model) and its `model_state_dict` contains keys like:

```
transformer.encoder.layers.0.self_attn.in_proj_weight
transformer.decoder.layers.0.multihead_attn.in_proj_weight
```

This is the parameter naming produced by plain `torch.nn.Transformer` /
`nn.TransformerEncoderLayer` (a single combined `in_proj_weight` per attention
layer). The current architecture (`model/tree_encoder.py`,
`model/tree_decoder.py`) is a custom implementation with separate
`q_proj` / `k_proj` / `v_proj` / `out_proj` weights and an additional
`parent_child_bias` parameter that has no equivalent in the checkpoint at all.

### Why it mattered

This is not a renaming issue — the two models are structurally different
(different attention implementation, different parameter count and shapes).
No key-mapping dict can correctly translate one into the other; attempting to
force-load would silently place unrelated weights into the wrong layers.

### The fix

Per task brief instructions, the checkpoint was treated as stale and retired.
`train.py` retrains `model/transformer.py`'s `CalculusSolverModel` (which
wraps the current `TreeEncoder` / `TreeDecoder`) from scratch and saves to
`checkpoints/final/best.pt`. Use `inference/solve.py`'s `.pt` loading path
(`CalculusSolverInference(model_path="checkpoints/final/best.pt")`) rather
than the old `model/model.pkl`.

### Files changed

| File | Change |
|---|---|
| `model/model.pkl` | Retired (superseded by `checkpoints/final/best.pt`) |
| `docs/KNOWN_ISSUES.md` | This entry added |

---

## [RESOLVED] Decoder trained without target shift (teacher forcing bug)

**Discovered:** `docs/TRAINING_RESULTS.md` showed val loss dropping to
near-zero (0.0070 → 0.0011 over 5 epochs) while `docs/EVAL_RESULTS.md` showed
0.0% exact match on all 300 benchmark problems — a mismatch between the
training signal and real performance
**Fixed:** `train.py`, `SlangDatasetLoader.__getitem__`
**Severity:** High — model learned nothing transferable to real inference
**Affected path:** Neural solver training only (`train.py`)

### What was wrong

In `data/slang_dataset.jsonl`, every one of the 125,000 rows has
`tgt_input_tokens` and `tgt_output_tokens` set to the *same* SLaNg tree.
`train.py` tokenized each field independently, producing two identical token
sequences, and fed one to the decoder as input while using the other as the
loss target. Because `TreeDecoder`'s causal mask still lets position `i`
attend to its own position-`i` input (only future positions are blocked), the
decoder could satisfy the loss by copying its current input straight through
instead of learning to predict the next token from the tokens before it.

### Why it mattered

Training and validation loss looked good (and `val_seq` — actually a loss
value, not an accuracy, despite the column name — kept decreasing) purely
because the copy shortcut is trivial to learn. But `eval/run_eval.py`
generates autoregressively: at inference time the ground-truth answer is
never available as decoder input, so a model that only learned to copy its
input produces garbage, hence 0.0% exact match on every category.

### The fix

`SlangDatasetLoader.__getitem__` now tokenizes the target once (with
`[BOS]`/`[EOS]` boundaries) and shifts it by one position instead of
tokenizing `tgt_input_tokens` and `tgt_output_tokens` separately:

- decoder input:  `[BOS, t1, ..., t_{n-1}]`
- decoder target: `[t1, ..., t_n, EOS]`

This forces the decoder to predict the token that comes *after* what it has
seen so far, matching how generation actually works at inference time.

### Files changed

| File | Change |
|---|---|
| `train.py` | `SlangDatasetLoader._tokenize` split into `_tokenize_ids` + `_pad`; `__getitem__` now tokenizes the target once and shifts it instead of tokenizing `tgt_input_tokens`/`tgt_output_tokens` separately |
| `config.json` | `hidden_dim` 128→256, `max_steps` →1000, `epochs` →15, early-stopping `patience` 5→6 (Step 3 tuning) |
| `docs/KNOWN_ISSUES.md` | This entry added |

### Verification

After retraining, confirm with:

```bash
python train.py
python eval/run_eval.py
```

`docs/TRAINING_RESULTS.md`'s `Val Seq` column should now reflect real
learning progress, and `docs/EVAL_RESULTS.md` should show non-zero exact
match on every operation category (diff, integrate, gradient, partial,
tangent_line) — not just overall.

---

## [RESOLVED] Sequence loss counted PAD positions

**Discovered:** After fixing the teacher-forcing target shift, a retrain still
showed `Val Seq` collapsing to near-zero within a few epochs (0.1161 → 0.0003)
while `eval/run_eval.py` still scored 0.0% exact match on all 300 problems
**Fixed:** `train.py` — both the training loop and `evaluate_validation`
**Severity:** High — a second, independent bug masking real learning signal
**Affected path:** Neural solver training only (`train.py`)

### What was wrong

`criterion_sequence = nn.CrossEntropyLoss(reduction='none')` had no
`ignore_index` set, so cross-entropy was computed over **every** position in
the padded `max_len=32` decoder output — including `[PAD]` positions. Real
SLaNg token sequences are much shorter than 32 tokens, so most positions in
every training example are `[PAD]`. The model could cheaply minimize the
averaged loss by learning to predict `[PAD]` almost everywhere, without ever
learning to predict the real content tokens.

### Why it mattered

This produced the same symptom as the target-shift bug — loss dropping fast
while `eval/run_eval.py` stayed at 0% — for a different reason. Fixing the
target shift alone was not enough; the loss function itself needed to ignore
padding, otherwise "predict PAD everywhere" remained the easiest way to
minimize loss.

### The fix

- `nn.CrossEntropyLoss(reduction='none', ignore_index=pad_idx)` so PAD
  positions contribute zero loss.
- Per-sequence loss is now averaged over the **real (non-pad) token count**
  only (`raw_loss_seq.sum(dim=-1) / non_pad.sum(dim=-1)`), instead of a plain
  mean over the full fixed-length sequence — otherwise a mean over mostly-zero
  positions would still under-report the true per-token loss.

### Files changed

| File | Change |
|---|---|
| `train.py` | `ignore_index=pad_idx` added to `criterion_sequence`; both the training loop and `evaluate_validation` now normalize sequence loss by real (non-pad) token count |
| `docs/KNOWN_ISSUES.md` | This entry added |

---

## [RESOLVED] Beam search never accepts a single token (BOS/EOS not in grammar)

**Discovered:** After fixing the teacher-forcing target-shift bug, `train.py`
showed healthy-looking `Val Seq` loss (down to 0.0003) but
`eval/run_eval.py` still reported 0.0% exact match on all 300 problems —
a second, independent bug on the generation side
**Fixed:** `inference/beam_search.py`
**Severity:** High — makes generation always collapse to just `[BOS]`
regardless of model quality
**Affected path:** Neural solver inference/eval (`inference/beam_search.py`)

### What was wrong

`beam_search()` grammar-checks every candidate next token with
`is_valid_prefix(token_strings + [candidate])`, where `token_strings`
includes the leading `[BOS]` control token. The grammar parser
(`is_valid_prefix` / `parse_node`) only understands AST tokens
(`NODE:TERM`, `NODE:FRAC`, `OP:...`, `STRUCT:...`) — it has no case for
`[BOS]`, so `parse_node` immediately returns `"invalid"` for any sequence
starting with `[BOS]`. That marks **every** candidate token invalid at the
very first decoding step, `safe_logits` becomes all `-inf`, the beam is
dropped, `candidates` ends up empty, and the loop breaks immediately —
generation always collapses to just `[BOS]` no matter how well the model
was trained. The same grammar has no token for `[EOS]` either, so appending
`[EOS]` to a genuinely complete tree also always failed the check.

### Why it mattered

This made eval accuracy structurally 0% independent of model quality —
even a perfectly trained model could never generate anything past `[BOS]`,
which is why the fix for the teacher-forcing bug alone did not move
`docs/EVAL_RESULTS.md` off 0.0%.

### The fix

In the beam search loop, `[BOS]` is stripped from `token_strings` before
grammar-checking (the grammar only describes the AST, not the control
tokens). `[EOS]` is handled separately via a new `is_complete_tree()`
helper, which is only true once the tokens generated so far form a fully
closed AST — `[EOS]` is allowed exactly then, instead of being checked
through `is_valid_prefix`, which the grammar can never satisfy for it.

### Files changed

| File | Change |
|---|---|
| `inference/beam_search.py` | Added `is_complete_tree()`; generation loop now grammar-checks `[BOS]`-stripped tokens and allows `[EOS]` via `is_complete_tree()` instead of `is_valid_prefix(tokens + ["[EOS]"])` |
| `docs/KNOWN_ISSUES.md` | This entry added |

### Verification

```bash
python eval/run_eval.py
```

`docs/EVAL_RESULTS.md` should now show non-zero exact-match on categories
the model has actually learned, instead of a structural 0.0% everywhere.

---

## [RESOLVED] PR review round: model.pkl stale, unstable training, fragile model-class detection

**Discovered:** PR #24 review
**Fixed:** `train.py`, `inference/solve.py`, new `finalize_checkpoint.py`
**Severity:** Medium — did not block training/eval, but made the committed
`model/model.pkl` misleading and made checkpoint loading fragile

### Issue A — model/model.pkl was never updated after training

`train.py` saves to `checkpoints/final/best.pt`, but `inference/solve.py`'s
default `model_path` is `model/model.pkl` — the two were never the same
file, so the committed `model.pkl` kept reflecting an old, unrelated
checkpoint no matter how many times `train.py` was re-run.

**Fix:** New `finalize_checkpoint.py` copies the trained checkpoint into
`model/model.pkl` after training. Run it once training is done:
```bash
python train.py
python finalize_checkpoint.py
```

### Issue B — training loss spiked/dropped instead of improving steadily

No gradient clipping was applied before `optimizer.step()`, so a handful of
batches with unusually large gradients could produce a destructive update
(seen as e.g. train loss jumping from 0.51 to 1.21 at epoch 9 in one run).

**Fix:** Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
in `train.py`'s training loop, right after `backward()` and before
`optimizer.step()`.

### Issue C — model class was guessed from the file extension

`inference/solve.py` decided whether to load `CalculusModel`
(`model/architecture.py`) or `CalculusSolverModel` (`model/transformer.py`)
based on whether the checkpoint's filename ended in `.pt`/`.pth`. A file
extension is not a reliable guarantee of which model produced the weights —
this is part of how the original checkpoint/architecture mismatch went
unnoticed.

**Fix:** `train.py`'s new `save_checkpoint()` helper now stores
`"architecture"` (e.g. `"CalculusSolverModel"`) and `"config"` (hidden_dim,
vocab_size, num_rules) directly in the checkpoint dict.
`CalculusSolverInference` now reads that field to decide which model class
to instantiate, regardless of filename. Legacy checkpoints saved before this
fix (no `"architecture"` field) still load via the old extension-based
guess, with a visible warning telling you to re-save them.

### Files changed

| File | Change |
|---|---|
| `train.py` | Added `save_checkpoint()` helper (embeds architecture/config metadata); added gradient clipping before `optimizer.step()` |
| `inference/solve.py` | Loads model class from checkpoint's `"architecture"` field instead of file extension; `_load_checkpoint` tries `torch.load` before falling back to `joblib` |
| `finalize_checkpoint.py` | New — copies `checkpoints/final/best.pt` into `model/model.pkl` after training |
| `docs/KNOWN_ISSUES.md` | This entry added |

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