def serialize_slang_math(node):
    """
    Faithful Python port of tokenizer/slang_serializer.js.
    Returns (tokens, parent_ids, child_ids):
      - tokens: list[str] of the real SLaNg token grammar
      - parent_ids[i]: id of the node that token i's enclosing node belongs to
      - child_ids[i]: id of the node that token i itself belongs to
    parent_ids/child_ids let TreeEncoder apply real structural attention bias
    instead of a placeholder zero tensor.
    """
    tokens, parents, children = [], [], []
    _serialize(node, tokens, parents, children, parent_id=-1, node_counter=[0])
    return tokens, parents, children


def _new_id(counter):
    nid = counter[0]
    counter[0] += 1
    return nid


def _serialize(node, tokens, parents, children, parent_id, node_counter):
    if node is None:
        raise ValueError("Cannot serialize null or undefined slang node.")

    if isinstance(node, list):
        _serialize_term_list(node, tokens, parents, children, parent_id, node_counter)
        return

    if isinstance(node, dict):
        if isinstance(node.get("op"), str):
            _serialize_op_node(node, tokens, parents, children, parent_id, node_counter)
            return
        if node.get("numi") is not None and node.get("deno") is not None:
            _serialize_fraction(node, tokens, parents, children, parent_id, node_counter)
            return
        if isinstance(node.get("coeff"), (int, float)):
            _serialize_term(node, tokens, parents, children, parent_id, node_counter)
            return

    raise ValueError(f"Unsupported slang node type during serialization: {node!r}")


def _emit(tok, tokens, parents, children, parent_id, own_id):
    tokens.append(tok)
    parents.append(parent_id)
    children.append(own_id)


def _serialize_op_node(node, tokens, parents, children, parent_id, node_counter):
    own_id = _new_id(node_counter)
    _emit(f"OP:{node['op']}", tokens, parents, children, parent_id, own_id)

    if node.get("var") is not None:
        _emit(f"OPVAR:{node['var']}", tokens, parents, children, parent_id, own_id)
    if isinstance(node.get("vars"), list):
        for v in node["vars"]:
            _emit(f"OPVAR:{v}", tokens, parents, children, parent_id, own_id)

    _emit("STRUCT:OPEN", tokens, parents, children, parent_id, own_id)

    kids = []
    if node.get("expr") is not None:
        kids.append(node["expr"])
    if node.get("u") is not None:
        kids.append(node["u"])
    if node.get("v") is not None:
        kids.append(node["v"])
    if node.get("left") is not None:
        kids.append(node["left"])
    if node.get("right") is not None:
        kids.append(node["right"])
    if isinstance(node.get("args"), list):
        kids.extend(node["args"])

    for i, kid in enumerate(kids):
        if i > 0:
            _emit("STRUCT:SEP", tokens, parents, children, parent_id, own_id)
        _serialize(kid, tokens, parents, children, own_id, node_counter)

    _emit("STRUCT:CLOSE", tokens, parents, children, parent_id, own_id)


def _serialize_fraction(node, tokens, parents, children, parent_id, node_counter):
    own_id = _new_id(node_counter)
    _emit("NODE:FRAC", tokens, parents, children, parent_id, own_id)
    _emit("STRUCT:OPEN", tokens, parents, children, parent_id, own_id)
    _emit("STRUCT:NUMI", tokens, parents, children, parent_id, own_id)
    _emit("STRUCT:OPEN", tokens, parents, children, parent_id, own_id)
    _serialize_term_list(_extract_terms(node["numi"]), tokens, parents, children, own_id, node_counter)
    _emit("STRUCT:CLOSE", tokens, parents, children, parent_id, own_id)
    _emit("STRUCT:SEP", tokens, parents, children, parent_id, own_id)
    _emit("STRUCT:DENO", tokens, parents, children, parent_id, own_id)
    _emit("STRUCT:OPEN", tokens, parents, children, parent_id, own_id)
    _serialize_term_list(_extract_terms(node["deno"]), tokens, parents, children, own_id, node_counter)
    _emit("STRUCT:CLOSE", tokens, parents, children, parent_id, own_id)
    _emit("STRUCT:CLOSE", tokens, parents, children, parent_id, own_id)


def _serialize_term_list(terms, tokens, parents, children, parent_id, node_counter):
    if not isinstance(terms, list):
        raise ValueError(f"Expected an array of terms, got {terms!r}")
    if len(terms) == 0:
        own_id = _new_id(node_counter)
        _emit("NODE:TERM", tokens, parents, children, parent_id, own_id)
        _emit("COEF:0", tokens, parents, children, parent_id, own_id)
        return
    for i, t in enumerate(terms):
        if i > 0:
            _emit("STRUCT:SEP", tokens, parents, children, parent_id, parent_id)
        _serialize(t, tokens, parents, children, parent_id, node_counter)


def _serialize_term(node, tokens, parents, children, parent_id, node_counter):
    own_id = _new_id(node_counter)
    _emit("NODE:TERM", tokens, parents, children, parent_id, own_id)
    if not isinstance(node.get("coeff"), (int, float)):
        raise ValueError(f"TERM node missing numeric coeff: {node!r}")
    _emit(f"COEF:{node['coeff']}", tokens, parents, children, parent_id, own_id)

    var = node.get("var")
    if isinstance(var, dict):
        for name, exp in sorted(var.items()):
            _emit(f"VAR:{name}", tokens, parents, children, parent_id, own_id)
            _emit(f"EXP:{exp}", tokens, parents, children, parent_id, own_id)


def _extract_terms(container):
    if container is None:
        return []
    if isinstance(container, list):
        return container
    if isinstance(container, dict) and isinstance(container.get("terms"), list):
        return container["terms"]
    if isinstance(container, (int, float)):
        return [{"coeff": container}]
    raise ValueError(f"Unsupported fraction term container: {container!r}")
