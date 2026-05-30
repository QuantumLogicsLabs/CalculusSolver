import json
from pathlib import Path

import streamlit as st

from inference.solve import CalculusSolverInference


ROOT = Path(__file__).resolve().parent


def resolve_model_path() -> Path:
    candidates = [
        ROOT / "checkpoints" / "final" / "best.pt",
        ROOT / "checkpoints" / "sft" / "best.pt",
        ROOT / "checkpoints" / "pretrain" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No model checkpoint found. Tried: {tried}")


@st.cache_resource
def load_solver():
    try:
        return (
            CalculusSolverInference(
                model_path=str(resolve_model_path()),
                vocab_path=str(ROOT / "tokenizer" / "vocab.json"),
                beam_size=5,
                max_len=256,
            ),
            None,
        )
    except Exception as exc:
        return None, str(exc)


st.set_page_config(page_title="CalculusSolver")
st.title("CalculusSolver")

solver, solver_error = load_solver()
if solver_error:
    st.warning(f"Solver is not available: {solver_error}")

raw_input = st.text_area(
    "Input envelope",
    value=json.dumps(
        {
            "op": "diff",
            "var": "x",
            "expr": {
                "numi": {"terms": [{"coeff": 1, "var": {"x": 2}}]},
                "deno": 1,
            },
        },
        indent=2,
    ),
    height=220,
)

if st.button("Solve", type="primary"):
    if solver is None:
        st.error("Add a compatible checkpoint before running /solve.")
    else:
        try:
            payload = json.loads(raw_input)
            result = solver.solve(payload)
            st.json(result)
        except Exception as exc:
            st.error(str(exc))
