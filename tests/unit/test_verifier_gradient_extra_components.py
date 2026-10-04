import sys
import os

# Ensure root repository directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from inference.verifier import verify
from tokenizer.slang_serializer import serialize_slang_math


def test_rejects_invented_z_component():
    """An answer adding a non-zero z component when the input has no z
    must fail verification, even if the x and y components are correct."""
    input_env = {
        "op": "gradient",
        "var": "x",
        "expr": {
            "numi": {
                "terms": [
                    {"coeff": -1, "var": {"x": 1}},
                    {"coeff": 1, "var": {"y": 1}},
                ]
            },
            "deno": 1,
        },
    }

    spurious_output = {
        "x": {"numi": {"terms": [{"coeff": -7}]}, "deno": 1},
        "y": {"numi": {"terms": [{"coeff": 12}]}, "deno": 1},
        "z": {"numi": {"terms": [{"coeff": 12}]}, "deno": 1},
    }

    tokens = serialize_slang_math(spurious_output)
    result = verify(input_env, tokens)
    assert result["verified"] is False, "Invented z component must be rejected"


def test_accepts_exact_variable_match():
    """A correct answer with exactly the right variables must still pass."""
    input_env = {
        "op": "gradient",
        "var": "x",
        "expr": {
            "numi": {
                "terms": [
                    {"coeff": -1, "var": {"x": 1}},
                    {"coeff": 1, "var": {"y": 1}},
                ]
            },
            "deno": 1,
        },
    }

    correct_output = {
        "x": {"numi": {"terms": [{"coeff": -1}]}, "deno": 1},
        "y": {"numi": {"terms": [{"coeff": 1}]}, "deno": 1},
    }

    tokens = serialize_slang_math(correct_output)
    result = verify(input_env, tokens)
    assert result["verified"] is True, "Exact-match gradient should still verify"


if __name__ == "__main__":
    test_rejects_invented_z_component()
    test_accepts_exact_variable_match()
    print("All tests passed!")