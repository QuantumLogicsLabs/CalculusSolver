def serialize_slang_math(envelope):
    """
    Authentic standard parser for mathematical polynomial expressions, tree structures, 
    and sum rules adhering strictly to v1.1 format constraints.
    """
    if not isinstance(envelope, dict):
        return ""

    tokens = []

    # 1. Handle Operators / Core Nodes (Recursive Layout)
    if "op" in envelope:
        op_val = str(envelope["op"]).upper()
        tokens.append(f"OP:{op_val}")
        
        # Open structural tracking layer context
        tokens.append("STRUCT:OPEN")
        
        # Recursively parse child branches or arguments if they exist
        if "args" in envelope and isinstance(envelope["args"], list):
            for arg in envelope["args"]:
                sub_res = serialize_slang_math(arg)
                if sub_res:
                    tokens.append(sub_res)
                    
        elif "children" in envelope and isinstance(envelope["children"], list):
            for child in envelope["children"]:
                sub_res = serialize_slang_math(child)
                if sub_res:
                    tokens.append(sub_res)
                    
        tokens.append("STRUCT:CLOSE")

    # 2. Handle Base Leaf Components (Terms, Fractions, Variables, Coefficients, Exponents)
    elif "type" in envelope:
        type_val = str(envelope["type"]).upper()
        tokens.append(f"MODE:{type_val}")
        
        if "coef" in envelope and envelope["coef"] is not None:
            tokens.append(f"COEF:{envelope['coef']}")
            
        if "var" in envelope and envelope["var"] is not None:
            tokens.append(f"VAR:{envelope['var']}")
            
        if "exp" in envelope and envelope["exp"] is not None:
            tokens.append(f"EXP:{envelope['exp']}")
            
        if "num" in envelope and envelope["num"] is not None:
            tokens.append(f"NUM:{envelope['num']}")
            
        if "deno" in envelope and envelope["deno"] is not None:
            tokens.append(f"DENO:{envelope['deno']}")

    # Fallback to handle generic sub-object key iterations safely if nested differently
    else:
        for k, v in envelope.items():
            if isinstance(v, dict):
                sub_res = serialize_slang_math(v)
                if sub_res:
                    tokens.append(sub_res)

    return " ".join(tokens)
