# SLaNg Dataset Schema Definitions

This document outlines the structured JSONL data format utilized by the CalculusSolver tokenization, pretraining, and fine-tuning pipelines.

## 1. JSONL Data Entry Schema

Each entry in the `.jsonl` files represents a complete calculus derivation step or problem mapping instance containing the following structured keys:

| Field Key | Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `src_tokens` | `List[str]` | The character or sub-word tokenized array representing the source math input string. | `["d", "/", "d", "x", "[", "x", "^", "2", "]"]` |
| `src_positions`| `List[int]` | Positional structural indexing for custom transformer/tree tracking models. | `[0, 1, 2, 3, 4, 5, 6, 7, 8]` |
| `tgt_input_tokens`| `List[str]` | The shifted target input tokens utilized for auto-regressive teacher-forcing decoding loops. | `["<s>", "2", "x"]` |
| `tgt_output_tokens`| `List[str]` | The target labels for structural output generation sequences. | `["2", "x", "</s>"]` |
| `rule_ids` | `int` | The multi-head classification target maps matching unique integer labels to rules. | `0` |
| `verification_state`| `int` | Binary classification mapping flag: `1` for mathematically true derivations, `0` for false steps. | `1` |
| `text` | `str` | Raw string representation containing input, output, and ground truth tags. | `"d/dx[x^2] → 2x, power rule, verified."` |

## 2. Rule Enumeration Multi-Head Map
* `0`: power rule
* `1`: trig derivative
* `2`: exponential rule
* `3`: logarithmic rule