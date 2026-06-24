# SLaNg Dataset Schema Definitions

Each entry in the `.jsonl` files contains the following keys:
- `src_tokens`: Character level source math input.
- `src_positions`: Positional tracking list.
- `tgt_input_tokens`: Shifted target input tokens for decoder forcing.
- `tgt_output_tokens`: Target labels for structural output generation.
- `rule_ids`: Multi-head classification integer targeting math rules.
- `verification_state`: Binary flag (1 for true derivations, 0 for false steps).
- `text`: String log containing complete equation block text.