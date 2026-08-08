"""
SLaNg serializer and deserializer in pure Python.
Matches the behavior of the original JS slang_serializer.
"""

from typing import Any, List, Dict, Tuple, Union

OPEN = "STRUCT:OPEN"
CLOSE = "STRUCT:CLOSE"
SEP = "STRUCT:SEP"
NUMI = "STRUCT:NUMI"
DENO = "STRUCT:DENO"
FRAC = "NODE:FRAC"
TERM = "NODE:TERM"
# Gradient node type. Represents a dict keyed by variable name, each value
# itself a SLaNg expression (e.g. {"x": <d/dx expr>, "y": <d/dy expr>}).
# Added to support inference/verifier.py's gradient_oracle(), which returns
# exactly this shape. See docs/KNOWN_ISSUES.md.
GRADIENT = "NODE:GRADIENT"


OP_PREFIX = "OP:"
OPVAR_PREFIX = "OPVAR:"
VAR_PREFIX = "VAR:"
COEF_PREFIX = "COEF:"
EXP_PREFIX = "EXP:"
# NEW: point decorator prefix for tangent_line's x0 value. Kept as a
# distinct namespace from COEF:/EXP: (rather than reusing either) to avoid
# any parsing ambiguity in parse_op_node's fixed decorator-check order.
POINT_PREFIX = "POINT:"


def serialize_slang_math(node: Any) -> List[str]:
    """Serialize a SLaNg AST node (dict or list) to a list of token strings."""
    tokens: List[str] = []

    def serialize(n: Any) -> None:
        if n is None:
            raise ValueError("Cannot serialize null or undefined slang node.")
        if isinstance(n, list):
            serialize_term_list(n)
            return
        if isinstance(n, dict):
            if isinstance(n.get("op"), str):
                serialize_op_node(n)
                return
            if "numi" in n and "deno" in n:
                serialize_fraction(n)
                return
            if "coeff" in n:
                serialize_term(n)
                return
            # NEW: unwrap the {"gradient": {...}} envelope used by
            # eval/benchmarks' target format -- serialize the inner
            # {var: expr} dict directly, not the wrapper key itself.
            # Checked BEFORE the general gradient-dict branch below, since
            # {"gradient": {...}} would otherwise itself match that
            # branch's shape check (a dict whose sole value is a dict).
            if set(n.keys()) == {"gradient"} and isinstance(n["gradient"], dict):
                serialize_gradient_node(n["gradient"])
                return
            # NEW: gradient node -- a non-empty dict whose values are all
            # themselves SLaNg-serializable expressions, keyed by variable
            # name. Checked LAST among dict branches since it is the most
            # permissive shape-match (any dict lacking op/numi+deno/coeff).
            if n and all(isinstance(v, (dict, list, int, float)) for v in n.values()):
                serialize_gradient_node(n)
                return
        raise ValueError(
            f"Unsupported slang node type during serialization: {n}"
        )

    def serialize_op_node(n: Dict[str, Any]) -> None:
        tokens.append(f"{OP_PREFIX}{n['op']}")

        # NEW (backward-compatible, opt-in): optional scale/sign decorator on
        # an op-node, e.g. {"coeff": -1, "op": "sin", ...} for -sin(x). Reuses
        # the existing COEF: token range -- no new vocab IDs needed. Existing
        # op-nodes (diff, integrate, gradient, product_rule, ...) never set
        # 'coeff', so this branch is never triggered for them and their token
        # output is completely unchanged.
        if "coeff" in n and n["coeff"] != 1:
            coeff_val = float(n["coeff"])
            if coeff_val.is_integer():
                coeff_val = int(coeff_val)
            tokens.append(f"{COEF_PREFIX}{coeff_val}")

        # NEW (backward-compatible, opt-in): optional power decorator on an
        # op-node, e.g. {"op": "sec", "power": 2, ...} for sec^2(x). Reuses
        # the existing EXP: token range -- no new vocab IDs needed. Existing
        # op-nodes never set 'power', so this is unaffected for them too.
        if "power" in n and n["power"] != 1:
            power_val = float(n["power"])
            if power_val.is_integer():
                power_val = int(power_val)
            tokens.append(f"{EXP_PREFIX}{power_val}")

        # NEW (backward-compatible, opt-in): optional point decorator on an
        # op-node, e.g. {"op": "tangent_line", "point": 3, ...} for the x0
        # value at which the tangent line is evaluated. Uses a distinct
        # POINT: token range (vocab.json v1.7) rather than reusing COEF:,
        # to keep parse_op_node's fixed decorator order unambiguous. Only
        # tangent_line sets this field; all other op-nodes are unaffected.
        if "point" in n:
            point_val = float(n["point"])
            if point_val.is_integer():
                point_val = int(point_val)
            tokens.append(f"{POINT_PREFIX}{point_val}")

        if n.get("var") is not None:
            tokens.append(f"{OPVAR_PREFIX}{n['var']}")
        if isinstance(n.get("vars"), list):
            for variable in n["vars"]:
                tokens.append(f"{OPVAR_PREFIX}{variable}")
        tokens.append(OPEN)

        children = []
        if n.get("expr") is not None:
            children.append(n["expr"])
        if n.get("u") is not None:
            children.append(n["u"])
        if n.get("v") is not None:
            children.append(n["v"])
        if n.get("left") is not None:
            children.append(n["left"])
        if n.get("right") is not None:
            children.append(n["right"])
        if isinstance(n.get("args"), list):
            children.extend(n["args"])

        for i, child in enumerate(children):
            if i > 0:
                tokens.append(SEP)
            serialize(child)
        tokens.append(CLOSE)

    def serialize_fraction(n: Dict[str, Any]) -> None:
        tokens.append(FRAC)
        tokens.append(OPEN)
        tokens.append(NUMI)
        tokens.append(OPEN)
        serialize_term_list(extract_terms(n["numi"]))
        tokens.append(CLOSE)
        tokens.append(SEP)
        tokens.append(DENO)
        tokens.append(OPEN)
        serialize_term_list(extract_terms(n["deno"]))
        tokens.append(CLOSE)
        tokens.append(CLOSE)

    def serialize_gradient_node(n: Dict[str, Any]) -> None:
        """Serialize a {var: expr, ...} dict as NODE:GRADIENT OPEN
        OPVAR:<name> <expr> [SEP OPVAR:<name> <expr> ...] CLOSE. Variables
        are sorted alphabetically for deterministic, reproducible token
        output (matching the existing var-sort convention in serialize_term)."""
        tokens.append(GRADIENT)
        tokens.append(OPEN)
        sorted_items = sorted(n.items(), key=lambda kv: kv[0])
        for i, (var_name, expr) in enumerate(sorted_items):
            if i > 0:
                tokens.append(SEP)
            tokens.append(f"{OPVAR_PREFIX}{var_name}")
            serialize(expr)
        tokens.append(CLOSE)

    def serialize_term_list(terms: List[Any]) -> None:
        if not isinstance(terms, list):
            raise ValueError(f"Expected an array of terms, got {terms}")
        if len(terms) == 0:
            tokens.append(TERM)
            tokens.append(f"{COEF_PREFIX}0")
            return
        for i, term in enumerate(terms):
            if i > 0:
                tokens.append(SEP)
            serialize(term)

    def serialize_term(n: Dict[str, Any]) -> None:
        tokens.append(TERM)
        coeff = n.get("coeff")
        if coeff is None:
            raise ValueError(f"TERM node missing coeff: {n}")
        try:
            coeff_val = float(coeff)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"TERM node missing numeric coeff: {n}") from exc

        # Convert float to int if it's a whole number
        if coeff_val.is_integer():
            coeff_val = int(coeff_val)
        tokens.append(f"{COEF_PREFIX}{coeff_val}")

        var_dict = n.get("var")
        if var_dict and isinstance(var_dict, dict):
            # Sort variables alphabetically to match JS sort
            sorted_vars = sorted(var_dict.items(), key=lambda x: x[0])
            for name, exp in sorted_vars:
                tokens.append(f"{VAR_PREFIX}{name}")
                try:
                    exp_val = float(exp)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid exponent in term: {n}") from exc
                if exp_val.is_integer():
                    exp_val = int(exp_val)
                tokens.append(f"{EXP_PREFIX}{exp_val}")

    def extract_terms(container: Any) -> List[Any]:
        if container is None:
            return []
        if isinstance(container, list):
            return container
        if isinstance(container, dict) and isinstance(container.get("terms"), list):
            return container["terms"]
        if isinstance(container, (int, float)):
            return [{"coeff": container}]
        raise ValueError(f"Unsupported fraction term container: {container}")

    serialize(node)
    return tokens


def deserialize_slang_math(tokens: List[str]) -> Any:
    """Deserialize a list of SLaNg token strings back to a SLaNg AST node."""
    if not isinstance(tokens, list):
        raise ValueError("deserialize_slang_math expects an array of tokens.")

    def parse_node(index: int) -> Tuple[Any, int]:
        if index >= len(tokens):
            raise ValueError(f"Unexpected end of tokens at index {index}")
        token = tokens[index]
        if token == TERM:
            return parse_term(index)
        if token == FRAC:
            return parse_fraction(index)
        if token == GRADIENT:
            return parse_gradient_node(index)
        if isinstance(token, str) and token.startswith(OP_PREFIX):
            return parse_op_node(index)
        raise ValueError(
            f"Unexpected token while parsing node at index {index}: {token}"
        )

    def parse_op_node(index: int) -> Tuple[Dict[str, Any], int]:
        op_token = tokens[index]
        node: Dict[str, Any] = {"op": op_token[len(OP_PREFIX):]}
        index += 1

        # NEW (backward-compatible): optional COEF: decorator immediately
        # after the OP: token, mirroring serialize_op_node's emission order.
        # Existing token streams (diff, integrate, ...) never have a COEF:
        # token in this position, so this only ever triggers for the new
        # sin/cos/tan/exp/ln/sec nodes that actually emit it.
        if (
            index < len(tokens)
            and isinstance(tokens[index], str)
            and tokens[index].startswith(COEF_PREFIX)
        ):
            coef_str = tokens[index][len(COEF_PREFIX):]
            try:
                node["coeff"] = int(coef_str)
            except ValueError:
                node["coeff"] = float(coef_str)
            index += 1

        # NEW (backward-compatible): optional EXP: decorator immediately
        # after the optional COEF:, mirroring serialize_op_node's order.
        if (
            index < len(tokens)
            and isinstance(tokens[index], str)
            and tokens[index].startswith(EXP_PREFIX)
        ):
            exp_str = tokens[index][len(EXP_PREFIX):]
            try:
                node["power"] = int(exp_str)
            except ValueError:
                node["power"] = float(exp_str)
            index += 1

        # NEW (backward-compatible): optional POINT: decorator immediately
        # after the optional COEF:/EXP:, mirroring serialize_op_node's
        # emission order. Existing op-nodes never have a POINT: token in
        # this position, so this only ever triggers for tangent_line nodes.
        if (
            index < len(tokens)
            and isinstance(tokens[index], str)
            and tokens[index].startswith(POINT_PREFIX)
        ):
            point_str = tokens[index][len(POINT_PREFIX):]
            try:
                node["point"] = int(point_str)
            except ValueError:
                node["point"] = float(point_str)
            index += 1

        while (
            index < len(tokens)
            and isinstance(tokens[index], str)
            and tokens[index].startswith(OPVAR_PREFIX)
        ):
            var_name = tokens[index][len(OPVAR_PREFIX):]
            if "var" not in node:
                node["var"] = var_name
            elif "vars" not in node:
                node["vars"] = [node["var"], var_name]
            else:
                node["vars"].append(var_name)
            index += 1

        index = expect_token(index, OPEN)
        children = []
        while index < len(tokens) and tokens[index] != CLOSE:
            child_node, next_idx = parse_node(index)
            children.append(child_node)
            index = next_idx
            if index < len(tokens) and tokens[index] == SEP:
                index += 1
        index = expect_token(index, CLOSE)

        if len(children) == 1:
            node["expr"] = children[0]
        elif len(children) == 2 and node["op"] in ("product_rule", "quotient_rule"):
            node["u"] = children[0]
            node["v"] = children[1]
        elif len(children) > 0:
            node["children"] = children
        return node, index

    def parse_fraction(index: int) -> Tuple[Dict[str, Any], int]:
        index = expect_token(index, FRAC)
        index = expect_token(index, OPEN)
        index = expect_token(index, NUMI)
        numerator_terms, index = parse_wrapped_term_list(index)
        index = expect_token(index, SEP)
        index = expect_token(index, DENO)
        denominator_terms, index = parse_wrapped_term_list(index)
        index = expect_token(index, CLOSE)

        return {
            "numi": {"terms": numerator_terms},
            "deno": {"terms": denominator_terms},
        }, index

    def parse_gradient_node(index: int) -> Tuple[Dict[str, Any], int]:
        """Parse NODE:GRADIENT OPEN OPVAR:<name> <expr> [SEP ...] CLOSE
        back into a {var: expr, ...} dict, mirroring serialize_gradient_node.
        Note: this returns the BARE {var: expr} dict, not wrapped in
        {"gradient": {...}} -- callers comparing against
        eval/benchmarks' target format must wrap it themselves, since
        inference/verifier.py's own gradient_oracle() also produces the
        bare shape internally."""
        index = expect_token(index, GRADIENT)
        index = expect_token(index, OPEN)
        result: Dict[str, Any] = {}
        while index < len(tokens) and tokens[index] != CLOSE:
            var_token = tokens[index]
            if not isinstance(var_token, str) or not var_token.startswith(OPVAR_PREFIX):
                raise ValueError(
                    f"Expected OPVAR token in gradient node at index {index}, got {var_token}"
                )
            var_name = var_token[len(OPVAR_PREFIX):]
            index += 1
            child_node, index = parse_node(index)
            result[var_name] = child_node
            if index < len(tokens) and tokens[index] == SEP:
                index += 1
        index = expect_token(index, CLOSE)
        return result, index

    def parse_wrapped_term_list(index: int) -> Tuple[List[Any], int]:
        index = expect_token(index, OPEN)
        terms = []
        while index < len(tokens) and tokens[index] != CLOSE:
            term_node, next_idx = parse_node(index)
            terms.append(term_node)
            index = next_idx
            if index < len(tokens) and tokens[index] == SEP:
                index += 1
        index = expect_token(index, CLOSE)
        return terms, index

    def parse_term(index: int) -> Tuple[Dict[str, Any], int]:
        index = expect_token(index, TERM)
        coef_token = tokens[index]
        if not isinstance(coef_token, str) or not coef_token.startswith(COEF_PREFIX):
            raise ValueError(
                f"Expected COEF after TERM at index {index}, got {coef_token}"
            )
        coef_str = coef_token[len(COEF_PREFIX):]
        try:
            coeff: Union[int, float] = int(coef_str)
        except ValueError:
            coeff = float(coef_str)
        node: Dict[str, Any] = {"coeff": coeff}
        index += 1

        while (
            index < len(tokens)
            and isinstance(tokens[index], str)
            and tokens[index].startswith(VAR_PREFIX)
        ):
            var_name = tokens[index][len(VAR_PREFIX):]
            index += 1
            exp_token = tokens[index]
            if not isinstance(exp_token, str) or not exp_token.startswith(EXP_PREFIX):
                raise ValueError(
                    f"Expected EXP token after VAR at index {index}, got {exp_token}"
                )
            exp_str = exp_token[len(EXP_PREFIX):]
            try:
                exp: Union[int, float] = int(exp_str)
            except ValueError:
                exp = float(exp_str)
            if "var" not in node:
                node["var"] = {}
            node["var"][var_name] = exp
            index += 1

        return node, index

    def expect_token(index: int, expected: str) -> int:
        if index >= len(tokens):
            raise ValueError(f"Expected token {expected} but reached end of tokens")
        if tokens[index] != expected:
            raise ValueError(
                f"Expected token {expected} at index {index}, got {tokens[index]}"
            )
        return index + 1

    node, next_idx = parse_node(0)
    if next_idx != len(tokens):
        raise ValueError(
            f"Extra tokens found after deserialization at position {next_idx}."
        )
    return node