# ARCHITECTURE.md — CalculusSolver

> Complete structural reference for the CalculusSolver ML system. Covers every layer, every file, every data flow, and the design decisions behind each choice.

---

## Table of contents

1. [System philosophy](#1-system-philosophy)
2. [High-level system map](#2-high-level-system-map)
3. [Repository structure](#3-repository-structure)
4. [Layer 0 — SLaNg (external dependency)](#4-layer-0--slang-external-dependency)
5. [Layer 1 — Tokenizer](#5-layer-1--tokenizer)
6. [Layer 2 — Model](#6-layer-2--model)
   - 6.1 Tree Encoder
   - 6.2 Rule Head
   - 6.3 Tree Decoder
   - 6.4 Step Tracer
   - 6.5 Full forward pass
7. [Layer 3 — Data pipeline](#7-layer-3--data-pipeline)
8. [Layer 4 — Training](#8-layer-4--training)
9. [Layer 5 — Inference](#9-layer-5--inference)
10. [Layer 6 — API](#10-layer-6--api)
11. [Layer 7 — Evaluation](#11-layer-7--evaluation)
12. [Cross-cutting concerns](#12-cross-cutting-concerns)
13. [Data flow: end to end](#13-data-flow-end-to-end)
14. [Dependency graph](#14-dependency-graph)
15. [Design decisions and trade-offs](#15-design-decisions-and-trade-offs)

---

## 1. System philosophy

Three principles govern every architectural decision in CalculusSolver.

**SLaNg is the only I/O format.** The model never sees LaTeX strings, plain-text equations, or any intermediate representation at training or inference time. Every input is a `slangmath` expression object. Every output is a `slangmath` expression object. This eliminates an entire class of parsing errors and means the model's output can always be plugged directly back into `slangmath` without a conversion step.

**The model predicts what to do; slangmath verifies whether it is correct.** CalculusSolver does not implement any calculus rules. It learns to predict which `slangmath` rule applies at each node in the input tree, and in what order to apply them. After generating an answer, it hands the answer back to `slangmath` for numerical verification. The model is the strategist; `slangmath` is the ground truth.

**Interpretability is structural, not post-hoc.** The Rule Head produces explicit rule predictions (`quotient_rule`, `chain_rule`, etc.) that map one-to-one with `slangmath`'s internal function names. The step trace in the output envelope is generated from these predictions, not from a separate language model. Any human reading `result.steps` is reading the model's actual decision process, not a summary of it.

---

## 2. High-level system map

```
┌────────────────────────────────────────────────────────────────────────┐
│  Caller                                                                │
│  cs.solve({ op: "diff", var: "x", expr: createFraction(...) })        │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ SLaNg expression object
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Tokenizer                                                   │
│  slang_serializer.js  →  vocab.json  →  positional_encoding.py        │
│  SLaNg tree  ──DFS──►  token sequence  +  tree position embeddings    │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ (tokens, positions)
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Model  (model/)                                             │
│                                                                        │
│   tree_encoder.py          ─────────────────────────────────────────  │
│   8-layer Transformer                                                  │
│   + parent-child attention bias                                        │
│           │                                                            │
│           ├──────────────────────┐                                     │
│           ▼                      ▼                                     │
│   rule_head.py           tree_decoder.py                               │
│   Classifier             Autoregressive Transformer                    │
│   → rule labels          + cross-attention to encoder                 │
│           │              + SLaNg validity mask                         │
│           │                      │                                     │
│           └──────────┬───────────┘                                     │
│                      ▼                                                 │
│              step_tracer.py                                            │
│              Auxiliary head → step descriptions                        │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ output token sequence
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 5 — Inference                                                   │
│  beam_search.py  →  verifier.js                                        │
│  Deserialize tokens back to SLaNg tree                                 │
│  Run slangmath to numerically verify the answer                        │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ output envelope
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 6 — API                                                         │
│  FastAPI  /solve  /validate                                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository structure

```
CalculusSolver/
│
├── model/                          # Neural network — Python
│   ├── architecture.py             # Top-level model class, ties all heads together
│   ├── tree_encoder.py             # 8-layer Transformer encoder + attention bias
│   ├── tree_decoder.py             # Autoregressive decoder + validity mask
│   ├── rule_head.py                # Per-node rule classifier (classifier head)
│   └── step_tracer.py              # Step description generator (auxiliary head)
│
├── tokenizer/                      # Serialization layer — JS + Python
│   ├── slang_serializer.js         # SLaNg tree ↔ DFS token sequence
│   ├── vocab.json                  # All token types and their integer IDs
│   └── positional_encoding.py      # (depth, sibling_idx, path_hash) → vector
│
├── data/                           # Datasets — generated, not committed to git
│   ├── raw/                        # Scraped LaTeX problem sets
│   ├── slang_pairs/                # Verified (input SLaNg, output SLaNg) pairs
│   ├── synthetic/                  # Self-play generated pairs
│   └── splits/                     # train.jsonl / val.jsonl / test.jsonl
│
├── data_pipeline/                  # Data generation — JS
│   ├── generate_synthetic.js       # Random SLaNg trees → slangmath solves them
│   ├── latex_to_slang.js           # LaTeX problems → SLaNg pairs via latexToSlang()
│   └── verify_with_slang.js        # Numerical equivalence check on any pair
│
├── training/                       # Training loops — Python
│   ├── pretrain.py                 # Stage 1: masked SLaNg tree reconstruction
│   ├── finetune.py                 # Stage 2: supervised fine-tuning on pairs
│   ├── verifier_loop.py            # Stage 3: slangmath-in-the-loop hard example mining
│   └── config/
│       ├── pretrain.yaml           # Hyperparameters for Stage 1
│       └── finetune.yaml           # Hyperparameters for Stage 2
│
├── inference/                      # Inference runtime — JS + Python
│   ├── CalculusSolver.js           # Public JS class (browser-ready entry point)
│   ├── solve.py                    # Python inference wrapper, loads checkpoint
│   ├── beam_search.py              # Beam search with SLaNg validity mask
│   └── verifier.js                 # Post-hoc numerical check via slangmath
│
├── api/                            # HTTP server — Python
│   ├── app.py                      # FastAPI application, lifespan, middleware
│   └── routes/
│       ├── solve.py                # POST /solve — main inference endpoint
│       └── validate.py             # POST /validate — verify any SLaNg expression
│
├── eval/                           # Evaluation scripts — JS
│   ├── slang_equivalence.js        # Numerical equivalence on model vs ground truth
│   ├── step_accuracy.js            # Per-rule accuracy across the benchmark
│   └── benchmarks/
│       ├── ap_calculus.json        # AP Calc AB + BC problems as SLaNg trees
│       ├── mit_ocw.json            # MIT 18.01 + 18.02 problems
│       └── multivariable.json      # Gradient, hessian, Lagrange problems
│
├── experiments/                    # One-off experiment scripts — JS
│   ├── test_diff.js
│   ├── test_integration.js
│   ├── test_optimization.js
│   └── test_multivariable.js
│
├── checkpoints/                    # Model weights — not committed to git
│   ├── pretrain/                   # Stage 1 checkpoints
│   ├── sft/                        # Stage 2 checkpoints
│   └── final/                      # Stage 3 checkpoints (production)
│
├── package.json                    # Node dependencies (slangmath, etc.)
├── requirements.txt                # Python dependencies
├── ARCHITECTURE.md                 # This file
└── README.md
```

---

## 4. Layer 0 — SLaNg (external dependency)

SLaNg (`npm i slangmath`) is not part of the CalculusSolver codebase. It is an external dependency that plays three distinct roles in the system.

**Role 1: The only data format.** Every expression that enters or exits CalculusSolver is a `slangmath` object. The input envelope wraps a `slangmath` expression; the output envelope contains one. No conversion to or from any other format happens anywhere in the hot path.

**Role 2: The ground-truth oracle during data generation.** The data pipeline calls `slangmath` functions (`differentiateFraction`, `gradient`, `lagrangeMultipliers`, etc.) to produce the correct output for every training pair. The model never sees a training label that was not verified by `slangmath`.

**Role 3: The post-hoc verifier during inference.** After the model generates an answer, `verifier.js` calls the relevant `slangmath` function on the original input and compares the result numerically using `evaluateFraction`. This happens on every single inference call before the result is returned to the caller.

The functions CalculusSolver uses from `slangmath`:

| Function                                     | Used for                                           |
| -------------------------------------------- | -------------------------------------------------- |
| `createTerm(coef, vars)`                     | Building input expressions                         |
| `createFraction(num, den)`                   | Building fractional expressions                    |
| `createFunction(name, args)`                 | Building function nodes (sin, cos, etc.)           |
| `differentiateFraction(expr, var)`           | Verifying `diff` and `partial` outputs             |
| `gradient(expr, vars)`                       | Verifying `gradient` outputs                       |
| `hessian(expr, vars)`                        | Verifying `hessian` outputs                        |
| `lagrangeMultipliers(f, g, vars)`            | Verifying `lagrange` outputs                       |
| `tangentPlane(expr, vars, at)`               | Verifying `tangent_plane` outputs                  |
| `directionalDerivative(expr, vars, pt, dir)` | Verifying `dir_deriv` outputs                      |
| `findCriticalPoints(expr, vars)`             | Verifying `optimize` outputs                       |
| `evaluateFraction(expr, point)`              | Numerical equivalence checks                       |
| `slangToLatex(expr)`                         | Producing the `latex` field in the output envelope |
| `latexToSlang(latex)`                        | LaTeX bootstrap in the data pipeline               |

---

## 5. Layer 1 — Tokenizer

**Files:** `tokenizer/slang_serializer.js`, `tokenizer/vocab.json`, `tokenizer/positional_encoding.py`

The tokenizer converts a `slangmath` expression tree into a flat integer sequence suitable for a Transformer, and back again. It has two parts: the serializer (JS, because it works with `slangmath` objects directly) and the positional encoder (Python, because it runs inside the training loop).

### 5.1 `slang_serializer.js` — SLaNg tree ↔ token sequence

The serializer performs a depth-first traversal of the `slangmath` expression tree and emits one token per node. The traversal is deterministic: left children before right children, depth before breadth. This means the same expression always produces the same token sequence, and the token sequence encodes enough structure to reconstruct the original tree exactly.

Structural tokens:

| Token   | Meaning                                                                   |
| ------- | ------------------------------------------------------------------------- |
| `OPEN`  | Begin a child list                                                        |
| `CLOSE` | End a child list                                                          |
| `SEP`   | Separator between siblings (e.g. between numerator and denominator terms) |

Example — `2x / (x² + 1)` with `op: diff, var: x`:

```
[OP:diff] [VAR:x] [OPEN]
  [FRAC] [OPEN]
    [TERM] [COEF:2] [VAR:x] [EXP:1]
  [SEP]
    [TERM] [COEF:1] [VAR:x] [EXP:2]
    [SEP]
    [TERM] [COEF:1]
  [CLOSE]
[CLOSE]
```

The serializer also runs in reverse: given the model's output token sequence, it reconstructs a `slangmath` object by consuming `OPEN`/`CLOSE`/`SEP` tokens as tree structure and numeric tokens as node attributes. If the sequence is structurally invalid (mismatched brackets, impossible node type in a given position), reconstruction raises an error that is caught by `beam_search.py` and treated as a failed beam.

### 5.2 `vocab.json` — token vocabulary

`vocab.json` maps every possible token to an integer ID and back. Token families:

- `OP:*` — operation types from the input envelope (`OP:diff`, `OP:gradient`, `OP:lagrange`, etc.)
- `VAR:*` — variable names (`VAR:x`, `VAR:y`, `VAR:z`, and so on)
- `COEF:*` — rational coefficients, stored as reduced fractions (`COEF:2`, `COEF:1/3`, `COEF:-7/2`)
- `EXP:*` — integer exponents (`EXP:1`, `EXP:2`, `EXP:-1`, etc.)
- `FN:*` — function names (`FN:sin`, `FN:cos`, `FN:exp`, `FN:ln`)
- `NODE:*` — node type tokens (`NODE:TERM`, `NODE:FRAC`, `NODE:FUNCTION`)
- `STRUCT:*` — structural tokens (`STRUCT:OPEN`, `STRUCT:CLOSE`, `STRUCT:SEP`)
- `RULE:*` — rule labels used by the Rule Head (`RULE:quotient_rule`, `RULE:chain_rule`, etc.)
- Special tokens: `[PAD]`, `[BOS]`, `[EOS]`, `[MASK]`

To extend the vocabulary for a new operation, add the relevant `OP:*` and `RULE:*` entries to `vocab.json` and re-run `python tokenizer/build_vocab.py` to rebuild the embedding matrix.

### 5.3 `positional_encoding.py` — tree-aware position embeddings

Standard sinusoidal encodings encode _sequence position_ (token 0, token 1, token 2...). For a DFS-serialized tree, sequence position does not capture tree structure — two tokens at the same depth but in different subtrees will have very different sequence positions but identical structural roles. The tree positional encoding replaces sequence position with three structural signals:

**Depth** — how many edges from the root to this node. Projected to `hidden_dim/3` dimensions via a learned linear layer.

**Sibling index** — the zero-based position of this node among its siblings. A node that is the second term in a polynomial has `sibling_idx=1`. Projected to `hidden_dim/3` dimensions.

**Path hash** — a deterministic hash of the sequence of left/right branch decisions from the root to this node. This gives each structural position a unique identity even when depth and sibling index alone would be ambiguous. Projected to the remaining `hidden_dim/3` dimensions.

The three projected vectors are concatenated and added to the token embedding before the first encoder layer. This gives the encoder a rich representation of where each token sits in the tree, not just where it sits in the flat sequence.

---

## 6. Layer 2 — Model

**Files:** `model/architecture.py`, `model/tree_encoder.py`, `model/tree_decoder.py`, `model/rule_head.py`, `model/step_tracer.py`

### 6.1 `tree_encoder.py` — Tree Encoder

The Tree Encoder is an 8-layer Transformer encoder. Its input is the token sequence produced by the serializer, with tree positional embeddings added. Its output is a sequence of contextual embedding vectors — one per input token — that the Decoder and Rule Head read via cross-attention.

The single architectural modification relative to a standard Transformer encoder is the **parent-child attention bias**. Before softmax is applied in each attention head, a learned scalar bias is added to the attention logit between any two tokens where one is the direct parent of the other in the original SLaNg tree. This bias is per-head (8 scalars total, one per head) and is learned during training. It does not affect tokens that are not in a parent-child relationship.

The effect is that structurally adjacent tokens attend more strongly to each other regardless of how far apart they are in the flat sequence. Without this bias, a deeply nested node in a large expression would be 30+ positions away from its parent in the sequence, and the encoder would need many layers to propagate information between them. With the bias, this happens in the first layer.

```python
# Simplified illustration from tree_encoder.py
def forward(self, tokens, positions, parent_child_pairs):
    x = self.embed(tokens) + self.pos_enc(positions)
    for layer in self.layers:
        # Standard multi-head self-attention, modified:
        attn_bias = self.make_parent_child_bias(parent_child_pairs)
        # attn_bias: (batch, heads, seq, seq) — zero everywhere except parent-child pairs
        x = layer(x, attn_bias=attn_bias)
    return x  # (batch, seq, hidden_dim)
```

Hyperparameters:

| Parameter        | Value |
| ---------------- | ----- |
| Layers           | 8     |
| Hidden dimension | 512   |
| Attention heads  | 8     |
| Head dimension   | 64    |
| FFN dimension    | 2048  |
| Activation       | GeLU  |
| Dropout          | 0.1   |

### 6.2 `rule_head.py` — Rule Head

The Rule Head is a linear classifier that runs on the encoder's output. For each token in the input sequence that corresponds to an operator node (i.e. tokens of type `OP:*` or `NODE:FRAC` or `NODE:FUNCTION`), the Rule Head predicts the most likely calculus rule to apply at that node.

The Rule Head does not run on every token — only on operator-type tokens, identified by their `vocab.json` token family. This is enforced by a mask that sets the logits for non-operator tokens to zero before the classifier head.

```python
# rule_head.py
class RuleHead(nn.Module):
    def forward(self, encoder_out, operator_mask):
        # operator_mask: (batch, seq) — True at operator token positions
        operator_hidden = encoder_out[operator_mask]   # (n_operators, hidden_dim)
        logits = self.classifier(operator_hidden)       # (n_operators, n_rules)
        return logits
```

The rule vocabulary is the set of all `RULE:*` tokens in `vocab.json`. Rules map to `slangmath` functions:

| Rule label           | slangmath function      | Notes                                    |
| -------------------- | ----------------------- | ---------------------------------------- |
| `quotient_rule`      | `differentiateFraction` | Applied when expr is a fraction          |
| `product_rule`       | `slang-advanced.js`     | Applied when expr is a product           |
| `chain_rule`         | `slang-extended.js`     | Applied when expr is a composed function |
| `power_rule`         | `differentiateFraction` | Applied when expr is a power             |
| `sum_rule`           | `differentiateFraction` | Applied term-by-term                     |
| `partial_x`          | `gradient`              | First partial, multivariable             |
| `partial_y`          | `gradient`              | Second partial, multivariable            |
| `form_lagrangian`    | `lagrangeMultipliers`   | First step of Lagrange problems          |
| `solve_system`       | `lagrangeMultipliers`   | Solving the KKT system                   |
| `evaluate_objective` | `lagrangeMultipliers`   | Substituting critical points             |
| `simplify`           | internal                | Algebraic simplification                 |
| `undefined`          | —                       | For limits that do not exist, etc.       |

During inference, the Rule Head's predictions are passed to the Decoder as a conditioning signal (see 6.3) and are also used directly to populate the `rule` field in each step of `result.steps`.

### 6.3 `tree_decoder.py` — Tree Decoder

The Tree Decoder is an 8-layer autoregressive Transformer decoder. It generates the output SLaNg token sequence one token at a time. At each step it attends to:

- Its own previously generated tokens (causal self-attention with a triangular mask)
- The full encoder output (cross-attention)
- The Rule Head's predictions for the current subtree (injected as a learned prefix embedding at the start of each subtree boundary)

**SLaNg validity mask.** Before sampling at each decoding step, the validity mask sets to `-inf` the logits for any token that would produce a structurally invalid SLaNg expression in the current decoding state. The mask is computed by a stateful automaton that tracks the current parse state — what node type is open, how many children have been emitted, whether a `SEP` is expected — and uses this state to whitelist only the tokens that `slangmath` would accept at this position.

Because the validity mask is applied _before_ sampling, the decoder cannot produce an invalid expression even with greedy decoding. Invalid expressions are impossible at the token level, not just filtered out at the end. This is the key correctness guarantee of the architecture.

```python
# tree_decoder.py (simplified)
def decode_step(self, prev_tokens, encoder_out, rule_embeddings, parse_state):
    hidden = self.self_attn(prev_tokens)
    hidden = self.cross_attn(hidden, encoder_out)
    hidden = hidden + rule_embeddings[current_subtree_id]
    logits = self.lm_head(hidden[:, -1])

    # Apply validity mask — computed from parse_state
    validity_mask = compute_slang_validity_mask(parse_state)
    logits[~validity_mask] = float('-inf')

    return logits, next_parse_state(parse_state, sampled_token)
```

The validity mask is implemented in `inference/beam_search.py` and communicates with `slangmath` through a lightweight JS bridge. The bridge exposes a synchronous `isValidNextToken(state, token)` function that the Python side calls via a subprocess pool.

Hyperparameters: same as the encoder (8 layers, 512 hidden, 8 heads), plus a causal attention mask and cross-attention to the encoder.

### 6.4 `step_tracer.py` — Step Tracer

The Step Tracer is a small auxiliary head that generates the natural-language `description` field for each step in `result.steps`. It is not a separate language model. It takes two inputs — the Rule Head's prediction for a given operator node and the Decoder's hidden state at the corresponding point in the output sequence — and produces a description string by looking up a learned template for each rule and filling in the relevant sub-expressions from the Decoder's hidden state.

Templates are stored in `tokenizer/vocab.json` under the `RULE:*` entries and contain placeholders (`{u}`, `{v}`, `{var}`, etc.) that are filled by projecting the Decoder's hidden state into expression-slot embeddings and nearest-neighbor matching to vocabulary tokens.

The Step Tracer is trained with teacher-forcing on the step descriptions in the training data, with a weight of 0.5 relative to the main decoder loss (see training config).

### 6.5 Full forward pass

During training, the full model runs as follows:

```
input tokens  →  Tree Encoder  →  encoder_out            (batch, seq_in, hidden)
encoder_out   →  Rule Head     →  rule_logits             (n_operators, n_rules)
rule_logits   →  argmax        →  rule_ids                (n_operators,)
rule_ids      →  rule_embed    →  rule_embeddings         (n_operators, hidden)
encoder_out
+ rule_embeddings
+ target tokens  →  Tree Decoder  →  decoder_logits      (batch, seq_out, vocab)
decoder_logits
+ rule_ids       →  Step Tracer   →  step_descriptions   (n_steps, max_desc_len)

Loss = decoder_ce(decoder_logits, target_tokens)     × 1.0
     + rule_head_ce(rule_logits, target_rules)        × 1.0
     + step_tracer_ce(step_descriptions, target_desc) × 0.5
```

All four components share the encoder's parameters. The Rule Head and Step Tracer are small enough (one linear layer each) that their gradient contribution is negligible relative to the encoder and decoder.

---

## 7. Layer 3 — Data pipeline

**Files:** `data_pipeline/generate_synthetic.js`, `data_pipeline/latex_to_slang.js`, `data_pipeline/verify_with_slang.js`

All training data flows through one of two paths, both of which end with `slangmath` as the final arbiter of correctness.

### 7.1 Synthetic self-play (`generate_synthetic.js`)

A `SlangTreeGenerator` samples random `slangmath` expression trees with controllable depth and variable set. For each sampled tree, it constructs an input envelope with a random `op` type and calls the corresponding `slangmath` function to compute the output. If `slangmath` returns a valid result (no exceptions, no `NaN` terms), the (input, output) pair is written to `data/synthetic/`.

This is the primary data source: 5 million pairs, covering all supported operations. The generator is biased toward the operation types that appear least in the LaTeX bootstrap data (currently `lagrange` and `series`) to keep the training distribution balanced.

### 7.2 LaTeX bootstrap (`latex_to_slang.js`)

Existing problem sets (AP Calculus AB/BC, MIT OCW 18.01/18.02) are distributed as LaTeX strings. `latexToSlang()` from `slangmath` converts each problem and each answer into a `slangmath` object. The resulting input-output pair is then verified: `slangmath` is run on the input and its output is compared numerically to the parsed answer using `evaluateFraction`. Only pairs where `slangmath` agrees with the provided answer are kept.

Rejection rates by source:

| Source            | Attempted | Kept | Rejection rate |
| ----------------- | --------- | ---- | -------------- |
| AP Calculus AB/BC | ~65K      | 40K  | ~38%           |
| MIT 18.01         | ~180K     | 120K | ~33%           |
| MIT 18.02         | ~280K     | 200K | ~29%           |
| Taylor series     | ~110K     | 80K  | ~27%           |

Rejections come from: LaTeX that `latexToSlang()` cannot parse, answers that are algebraically equivalent but structurally different enough that `evaluateFraction` does not agree (resolved by using more test points), and problems outside the current op vocabulary.

### 7.3 Verification (`verify_with_slang.js`)

`verify_with_slang.js` is a standalone utility that takes any (input SLaNg, output SLaNg) pair and checks numerical equivalence at `N` random test points (default 50). It is used by both pipeline scripts and can also be run independently on any pair:

```bash
node data_pipeline/verify_with_slang.js \
  --input '{"op":"diff","var":"x","expr":{...}}' \
  --output '{"expr":{...}}' \
  --points 100
```

---

## 8. Layer 4 — Training

**Files:** `training/pretrain.py`, `training/finetune.py`, `training/verifier_loop.py`, `training/config/pretrain.yaml`, `training/config/finetune.yaml`

Training runs in three sequential stages. Each stage produces a checkpoint that is the starting point for the next.

### 8.1 Stage 1 — Masked SLaNg tree pretraining (`pretrain.py`)

**What it does:** Randomly masks 20% of operator nodes in SLaNg trees (replacing them with the `[MASK]` token) and trains the encoder-decoder to reconstruct the original token. No calculus is learned here — the model is only learning the structural grammar of valid SLaNg expressions.

**Why it exists:** Without pretraining, the decoder would need to simultaneously learn SLaNg syntax and calculus reasoning from scratch. The masked pretraining stage front-loads the syntax learning so that Stage 2 can focus entirely on calculus.

**What is trained:** Encoder + Decoder only. The Rule Head and Step Tracer are not part of the Stage 1 loss.

**Config (`pretrain.yaml`):**

```yaml
model:
  encoder_layers: 8
  decoder_layers: 8
  hidden_dim: 512
  heads: 8

training:
  batch_size: 128
  lr: 2e-4
  warmup_steps: 5000
  max_steps: 300000
  mask_ratio: 0.20
  fp16: true
```

**Output:** `checkpoints/pretrain/best.pt`

### 8.2 Stage 2 — Supervised fine-tuning (`finetune.py`)

**What it does:** Trains the full model (Encoder + Rule Head + Decoder + Step Tracer) on complete (input SLaNg → output SLaNg + steps) pairs using teacher-forcing. The combined loss is a weighted sum of the four component losses.

**Starting point:** `checkpoints/pretrain/best.pt`

**What is trained:** All parameters. The Rule Head and Step Tracer are initialized fresh and trained from scratch at this stage.

**Config (`finetune.yaml`):**

```yaml
training:
  batch_size: 64
  lr: 5e-5
  warmup_steps: 1000
  max_steps: 150000
  fp16: true

loss:
  decoder_weight: 1.0
  rule_head_weight: 1.0
  step_tracer_weight: 0.5
```

**Output:** `checkpoints/sft/best.pt`

### 8.3 Stage 3 — SLaNg-in-the-loop hard example training (`verifier_loop.py`)

**What it does:** Runs the Stage 2 model on the training data in inference mode, collects all problems where the model's answer fails `slangmath` verification, adds these to a hard example pool, and retrains with 40% of each batch drawn from the hard pool.

**Why it exists:** The model that Stage 2 produces gets easy problems right nearly all the time and hard problems wrong at a consistent rate. Stage 3 over-samples the hard problems so the model is forced to learn them. The hard pool is recomputed every 5,000 training steps.

**Starting point:** `checkpoints/sft/best.pt`

**Key flag:** `--hard-example-ratio 0.4` — the fraction of each batch that comes from the hard pool.

**Output:** `checkpoints/final/best.pt` — this is the production checkpoint.

### 8.4 Monitoring

All stages log to Weights & Biases. The project name is `calculussolver`. Key metrics:

| Metric                 | Logged by  | What to watch                                                           |
| ---------------------- | ---------- | ----------------------------------------------------------------------- |
| `val/numerical_equiv`  | All stages | Primary correctness signal; should rise steeply in Stage 2              |
| `val/rule_accuracy`    | Stage 2, 3 | Rule Head correctness; plateau here means the Rule Head needs more data |
| `val/step_accuracy`    | Stage 2, 3 | Step Tracer quality                                                     |
| `train/hard_pool_size` | Stage 3    | Should stabilize; unbounded growth = Stage 2 did not converge           |
| `train/loss`           | All stages | Should decrease monotonically; spikes indicate data issues              |

---

## 9. Layer 5 — Inference

**Files:** `inference/CalculusSolver.js`, `inference/solve.py`, `inference/beam_search.py`, `inference/verifier.js`

### 9.1 `CalculusSolver.js` — public entry point

The JS class that callers import. It is browser-ready (no Node-only APIs). It wraps the Python inference server at a configurable endpoint, or falls back to a bundled WASM model for offline use (not yet implemented; see roadmap).

```javascript
import { CalculusSolver } from "calculussolver";
import { createFraction, createTerm } from "slangmath";

const cs = new CalculusSolver({ endpoint: "http://localhost:8000" });
const result = await cs.solve({
  op: "diff",
  var: "x",
  expr: createFraction(
    [createTerm(2, { x: 1 })],
    [createTerm(1, { x: 2 }), createTerm(1)],
  ),
});
```

`cs.solve()` serializes the input envelope to JSON, posts it to `/solve`, deserializes the response back to a `slangmath` object, runs the post-hoc verifier, and returns the output envelope. The verifier runs client-side in JS, not on the server, so it works even when the server is remote.

### 9.2 `solve.py` — Python inference wrapper

Loads the checkpoint at startup (once, not per request), tokenizes the input, runs beam search, deserializes the output tokens back to a `slangmath`-compatible JSON object, and returns it to the FastAPI route handler. Checkpoint loading uses `torch.load` with `map_location` defaulting to CUDA if available, CPU otherwise.

### 9.3 `beam_search.py` — beam search with validity mask

Runs beam search with `beam_size=5`. At each decoding step:

1. Expand all live beams by one token.
2. Apply the SLaNg validity mask — invalid tokens are set to `-inf` before scoring.
3. Score all candidates with the decoder's log-probability.
4. Keep the top `beam_size` candidates.
5. Terminate beams that emit `[EOS]`.

Return the highest-scoring completed beam. If no beam completes within `max_len=512` tokens, return the highest-scoring incomplete beam and set `result.status = "partial"`.

The validity mask communicates with `slangmath` through a subprocess pool of Node processes. Each process runs a tiny JS script that exposes `isValidNextToken(parseState, tokenId)` as a synchronous function. The subprocess pool is initialized once at server startup and kept warm.

### 9.4 `verifier.js` — post-hoc numerical verifier

Runs after every `cs.solve()` call. Maps the `op` field of the result to the corresponding `slangmath` function, calls it on the original input expression, and compares the result to the model's output using `evaluateFraction` at 50 random test points.

```javascript
// verifier.js (simplified)
const VERIFIER_MAP = {
  diff: (input) => differentiateFraction(input.expr, input.var),
  partial: (input) => differentiateFraction(input.expr, input.var),
  gradient: (input) => gradient(input.expr, input.vars),
  hessian: (input) => hessian(input.expr, input.vars),
  lagrange: (input) =>
    lagrangeMultipliers(input.objective, input.constraints, input.vars),
  tangent_plane: (input) => tangentPlane(input.expr, input.vars, input.at),
  dir_deriv: (input) =>
    directionalDerivative(input.expr, input.vars, input.point, input.direction),
  optimize: (input) => findCriticalPoints(input.expr, input.vars),
};

export function verify(input, modelOutput) {
  const oracle = VERIFIER_MAP[input.op](input);
  const testPoints = sampleTestPoints(input.vars, 50);
  const agrees = testPoints.every(
    (pt) =>
      Math.abs(
        evaluateFraction(oracle, pt) - evaluateFraction(modelOutput.expr, pt),
      ) < 1e-9,
  );
  return agrees ? "verified" : "unverified";
}
```

If the verifier returns `"unverified"`, `result.status` is changed from `"solved"` to `"unverified"` and the disagreement details are appended to `result.warnings`. The model's answer is still returned to the caller — it is not discarded.

---

## 10. Layer 6 — API

**Files:** `api/app.py`, `api/routes/solve.py`, `api/routes/validate.py`

### 10.1 `app.py`

FastAPI application. On startup, loads the checkpoint and initializes the `solve.py` inference wrapper and the Node subprocess pool for the validity mask. On shutdown, terminates the subprocess pool cleanly.

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Each worker loads its own copy of the model into GPU/CPU memory. If running on a single GPU, use `--workers 1` to avoid VRAM contention.

### 10.2 `POST /solve`

**Request body:** A JSON-serialized input envelope (see README for full schema).

**Response body:** A JSON-serialized output envelope.

**Error responses:**

| Status                          | Meaning                                          |
| ------------------------------- | ------------------------------------------------ |
| 400                             | Input envelope failed schema validation          |
| 422                             | Input SLaNg expression is structurally invalid   |
| 500                             | Model inference failed unexpectedly              |
| 200 with `status: "unverified"` | Model answered but slangmath disagrees           |
| 200 with `status: "partial"`    | Beam search hit `max_len` before completing      |
| 200 with `status: "unsolvable"` | Rule Head predicted `undefined` at the root node |

### 10.3 `POST /validate`

Accepts any SLaNg expression JSON and runs it through `slangmath`'s structural validator. Returns `{ valid: true }` or `{ valid: false, reason: "..." }`. Used by the browser playground and by external callers who want to validate an expression before submitting it to `/solve`.

---

## 11. Layer 7 — Evaluation

**Files:** `eval/slang_equivalence.js`, `eval/step_accuracy.js`, `eval/benchmarks/`

### 11.1 `slang_equivalence.js`

Loads a benchmark file (newline-delimited JSON, same format as training data), runs `cs.solve()` on each problem, and checks each answer with `evaluateFraction` at `--points N` random test points. Reports per-operation and overall numerical equivalence rates.

```bash
node eval/slang_equivalence.js \
  --checkpoint checkpoints/final/best.pt \
  --benchmark eval/benchmarks/ap_calculus.json \
  --points 50
```

### 11.2 `step_accuracy.js`

Runs the same inference loop but evaluates the step trace rather than the final answer. For each step in the model's output, checks whether the `rule` field matches the ground truth. Reports per-rule accuracy, which is more informative than overall accuracy for identifying which operations need more training data.

### 11.3 Benchmark files

All benchmarks are stored as SLaNg expression objects, not LaTeX. Each entry has the same format as a training pair, allowing the same evaluation code to run on both the dev set and the held-out benchmarks.

| File                 | Source            | Problems    | Operations covered                                    |
| -------------------- | ----------------- | ----------- | ----------------------------------------------------- |
| `ap_calculus.json`   | AP Calc AB + BC   | 40K         | diff, integrate, limit, series                        |
| `mit_ocw.json`       | MIT 18.01 + 18.02 | 120K + 200K | diff, partial, gradient, hessian                      |
| `multivariable.json` | Hand-curated      | 5K          | gradient, hessian, lagrange, tangent_plane, dir_deriv |

---

## 12. Cross-cutting concerns

### Language split

CalculusSolver uses two languages for deliberate reasons.

**JavaScript** is used for everything that directly touches `slangmath` objects: the serializer, the verifier, the data pipeline scripts, the eval scripts, the public `CalculusSolver.js` class, and all experiment scripts. JS is used here because `slangmath` is a JS library — running the verification and serialization in the same runtime as `slangmath` eliminates a serialization boundary and ensures that the objects the model's output is checked against are exactly the same type as the objects it was trained on.

**Python** is used for the model, training, and the FastAPI server. PyTorch has no JS equivalent for production training. The boundary between the two languages is the tokenizer: the JS serializer converts `slangmath` objects to integer sequences, and the Python side sees only integer sequences from that point onward.

### The JS-Python bridge

The only runtime communication between JS and Python is in `beam_search.py`, where the validity mask queries the `slangmath` structural validator at each decoding step. This is implemented as a pool of Node subprocesses that receive token state as JSON on stdin and return a boolean validity decision on stdout. The pool is sized to `num_beams * num_decoder_workers` to ensure no beam ever waits for a free subprocess.

### Checkpoints

Checkpoints are PyTorch `.pt` files containing the full model state dict, the optimizer state, the current step, and the vocabulary size (to detect vocab mismatches on load). The JS components (serializer, verifier) are stateless and version-independent.

### No LaTeX in the hot path

`latexToSlang()` and `slangToLatex()` from `slangmath` are used only at the edges: in the data pipeline (converting raw problem sets) and in the output envelope (computing the display-only `latex` field from `result.expr`). They are never called during model inference. The model itself has no awareness of LaTeX.

---

## 13. Data flow: end to end

### Training data flow

```
raw LaTeX problem sets
        │
        ▼
data_pipeline/latex_to_slang.js
  latexToSlang() → SLaNg expression
  slangmath() → correct answer
  verify_with_slang.js → numerical check
        │
        ├── pass → data/slang_pairs/
        └── fail → rejected (logged)

data_pipeline/generate_synthetic.js
  SlangTreeGenerator → random SLaNg tree
  slangmath() → correct answer
  verify_with_slang.js → numerical check
        │
        ├── pass → data/synthetic/
        └── fail → discarded

data/slang_pairs/ + data/synthetic/
        │
        ▼
data_pipeline/split.py
        │
        ├── data/splits/train.jsonl
        ├── data/splits/val.jsonl
        └── data/splits/test.jsonl

data/splits/train.jsonl
        │
        ▼
tokenizer/slang_serializer.js → integer token sequences
tokenizer/positional_encoding.py → position embeddings
        │
        ▼
training/pretrain.py  →  training/finetune.py  →  training/verifier_loop.py
        │                                                      │
checkpoints/pretrain/              checkpoints/sft/    checkpoints/final/
```

### Inference data flow

```
caller: cs.solve({ op, var, expr })
        │
        ▼
inference/CalculusSolver.js
  JSON serialize input envelope
  POST /solve
        │
        ▼
api/routes/solve.py
  schema validation (Pydantic)
        │
        ▼
inference/solve.py
  tokenizer/slang_serializer.js → token sequence
  tokenizer/positional_encoding.py → position embeddings
        │
        ▼
model/tree_encoder.py → encoder_out
        │
        ├── model/rule_head.py → rule predictions
        └── model/tree_decoder.py
              + SLaNg validity mask (beam_search.py ↔ Node subprocess pool)
              + rule embeddings from rule_head
              → output token sequence
              │
              ▼
        model/step_tracer.py → step descriptions
              │
              ▼
        tokenizer/slang_serializer.js (reverse) → SLaNg expression object
              │
              ▼
inference/verifier.js
  slangmath(input) → oracle answer
  evaluateFraction(model_output, test_points) vs evaluateFraction(oracle, test_points)
        │
        ├── agree → status: "solved"
        └── disagree → status: "unverified", warnings: [...]
        │
        ▼
output envelope returned to caller
```

---

## 14. Dependency graph

```
CalculusSolver.js
    ├── slangmath  (npm)
    └── api/routes/solve.py (HTTP)
            └── inference/solve.py
                    ├── model/architecture.py
                    │       ├── model/tree_encoder.py
                    │       ├── model/rule_head.py
                    │       ├── model/tree_decoder.py
                    │       └── model/step_tracer.py
                    ├── inference/beam_search.py
                    │       └── [Node subprocess] slangmath (validity mask)
                    └── tokenizer/positional_encoding.py

inference/verifier.js
    └── slangmath

data_pipeline/generate_synthetic.js
    └── slangmath

data_pipeline/latex_to_slang.js
    └── slangmath

data_pipeline/verify_with_slang.js
    └── slangmath

training/pretrain.py
training/finetune.py
training/verifier_loop.py
    ├── model/architecture.py
    ├── tokenizer/positional_encoding.py
    └── [reads] data/splits/

eval/slang_equivalence.js
eval/step_accuracy.js
    ├── slangmath
    └── inference/CalculusSolver.js
```

`slangmath` is a leaf in this graph — it has no CalculusSolver dependencies. Every other component depends on it, directly or transitively.

---

## 15. Design decisions and trade-offs

### SLaNg-only I/O vs. accepting LaTeX

**Decision:** The model only accepts and produces `slangmath` objects, never LaTeX strings.

**Trade-off:** Callers must convert their problems to `slangmath` objects before calling `cs.solve()`. `slangmath` provides `latexToSlang()` for this, but it is a one-way cost the caller must pay.

**Reason:** If the model accepted LaTeX, every training pair would need LaTeX as the intermediate format, introducing LaTeX parsing errors into the ground truth. By using `slangmath` objects throughout, the training labels are always verified by `slangmath` itself. The caller's conversion cost is a one-time cost per problem; the training data quality benefit is permanent.

### Rule Head before Decoder vs. end-to-end sequence generation

**Decision:** The Rule Head predicts calculus rules as a separate classifier head before the Decoder generates the output token sequence.

**Trade-off:** The Rule Head adds parameters and a separate training loss. It requires step-level rule annotations in the training data.

**Reason:** Without the Rule Head, the Decoder would need to implicitly learn which rule to apply and then generate the correct output entirely from the cross-attention signal. The Rule Head makes this explicit: the Decoder receives the predicted rule as a conditioning signal, which gives it a strong prior for what kind of expression it should generate next. This also makes the model's decisions interpretable — the `result.steps` array reflects what the Rule Head actually predicted, not a post-hoc summary.

### Validity mask at decoding time vs. post-hoc filtering

**Decision:** Invalid tokens are masked to `-inf` at every decoding step so that structurally invalid SLaNg expressions cannot be generated.

**Trade-off:** The validity mask requires a JS-Python bridge (subprocess pool) that adds latency to each decoding step. The mask automaton must be kept in sync with `slangmath`'s structural rules.

**Reason:** Post-hoc filtering would mean some beams complete invalid expressions, consuming compute. Worse, if all beams produce invalid expressions, the model returns nothing. The validity mask guarantees that every beam is a valid SLaNg expression at every step, so the model always returns something the caller can use.

### Three-stage training vs. end-to-end training

**Decision:** Training is split into masked pretraining (Stage 1), supervised fine-tuning (Stage 2), and hard example mining (Stage 3).

**Trade-off:** Three stages means three separate training runs, three checkpoints to manage, and more total compute than a single end-to-end run.

**Reason:** Stage 1 teaches SLaNg syntax without the distraction of calculus. Without it, Stage 2 would need to learn both simultaneously from scratch, which in practice causes the decoder to produce valid-looking but mathematically wrong expressions (it learns syntax faster than semantics). Stage 3 addresses the long tail of hard problems that Stage 2 gets right at low frequency — without it, the hard problem accuracy stays flat after Stage 2.

### JS subprocess pool for validity mask vs. rewriting the mask in Python

**Decision:** The SLaNg validity mask is implemented in JS (because it uses `slangmath`'s structural API) and called from Python via a subprocess pool.

**Trade-off:** Subprocess communication adds ~1–2ms per decoding step. Keeping a pool of Node processes alive adds memory overhead.

**Reason:** Rewriting `slangmath`'s structural rules in Python would create a duplicate implementation that would diverge from `slangmath` every time `slangmath` is updated. The subprocess approach means the validity mask is always exactly as strict as `slangmath` itself — no more, no less.


---

## 4. Systems Architecture Enhancements (Task B5 Addendum)

### 4.1 Four-Component Pipeline Flow
The execution framework cascades expressions deterministically across a tightly-integrated stack:
$$\text{Abstract Syntax Tree Data Input} \longrightarrow \text{Tree Encoder (Structural Relational Bias)}$$
$$\text{Tree Encoder Output Maps} \longrightarrow \text{Rule Head Dense Classifier Logits}$$
$$\text{Conditioned Rule Embeddings + Memory} \longrightarrow \text{Tree Decoder Autoregressive Beam Search Loop}$$
$$\text{Decoder States Space} \longrightarrow \text{Step Tracer Template Vector String Decoders}$$

### 4.2 Symbolic Verification Loop Subprocess
Every multi-candidate path extracted out of the target token spaces evaluates dynamically against isolated validation micro-interpreters. The operational cycle instantiates structured JSON expressions inside separate OS subprocess instances running `python -m slang.verifier` parameters, protecting interface systems from syntactic structural hallucinations.

### 4.3 Standard 3-Stage Structural Training Mechanics
Optimizations evolve linearly using segmented validation layers:
1. **Stage 1 (Sequence Pre-training)**: Baseline mapping optimization maximizing exact-token retrieval probability weights.
2. **Stage 2 (Teacher Forcing Adjustment)**: Conditioning hidden decoders directly over golden rule labels to safely evaluate gradients during architectural adjustments.
3. **Stage 3 (Constraint-Reinforced Verification)**: Tuning generation scores using verifier feedback passes.

### 4.4 Cross-Repository Architecture Layout
The engine bridges foundational syntax configurations from external structural token modules down to core model parameters. The model matches incoming tokens directly with dynamic reverse mappings ensuring complete system integration during downstream data validation cycles.