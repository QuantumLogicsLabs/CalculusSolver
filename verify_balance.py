import json
from collections import Counter

gradient_pairs = Counter()
vanish_positions = Counter()

with open("data/splits/train.jsonl") as f:
    for line in f:
        row = json.loads(line)
        src = row.get("src_tokens", {})
        op = src.get("op")

        if op == "gradient":
            grad_vars = tuple(sorted(row["tgt_input_tokens"]["gradient"].keys()))
            if len(grad_vars) == 2:
                gradient_pairs[grad_vars] += 1

        if op == "partial":
            bound_var = src.get("var")
            terms = src.get("expr", {}).get("numi", {}).get("terms", [])
            for idx, term in enumerate(terms):
                term_vars = term.get("var", {})
                if bound_var not in term_vars:
                    vanish_positions[idx] += 1

print("Gradient 2-var pair counts:", gradient_pairs)
print("Vanish-term position counts:", vanish_positions)