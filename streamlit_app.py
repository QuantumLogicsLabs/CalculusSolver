"""
CalculusSolver — Streamlit UI
Loads the best available checkpoint and exposes:
  • Interactive solver with example selector
  • Checkpoint status + training metrics panel
  • Inline accuracy test runner (calls eval/evaluate_model.py)
  • Fallback deterministic solver when no checkpoint exists
"""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PRIORITY = [
    ("final", ROOT / "checkpoints" / "final" / "best.pt"),
    ("sft", ROOT / "checkpoints" / "sft" / "best.pt"),
    ("pretrain", ROOT / "checkpoints" / "pretrain" / "best.pt"),
]


def resolve_model_path():
    import os

    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        p = p if p.is_absolute() else ROOT / p
        if p.exists():
            return p, "env"
    for stage, path in CHECKPOINT_PRIORITY:
        if path.exists():
            return path, stage
    tried = ", ".join(str(p) for _, p in CHECKPOINT_PRIORITY)
    raise FileNotFoundError(f"No checkpoint found. Tried: {tried}")


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK SOLVER  (pure Python polynomial math — no neural model needed)
# ─────────────────────────────────────────────────────────────────────────────


def _copy(expr):
    return json.loads(json.dumps(expr))


def _norm_term(term):
    clean = {"coeff": term.get("coeff", 0)}
    variables = {k: v for k, v in term.get("var", {}).items() if v != 0}
    if variables:
        clean["var"] = variables
    return clean


def _diff_fraction(expr, variable):
    if expr.get("deno", 1) != 1:
        raise ValueError("Fallback solver only supports denominator 1.")
    terms = []
    for term in expr.get("numi", {}).get("terms", []):
        power = term.get("var", {}).get(variable, 0)
        if power == 0:
            continue
        t = _copy(term)
        t["coeff"] = t.get("coeff", 0) * power
        t.setdefault("var", {})[variable] = power - 1
        terms.append(_norm_term(t))
    return {"numi": {"terms": terms or [{"coeff": 0}]}, "deno": 1}


def _integrate_fraction(expr, variable):
    if expr.get("deno", 1) != 1:
        raise ValueError("Fallback solver only supports denominator 1.")
    terms = []
    for term in expr.get("numi", {}).get("terms", []):
        power = term.get("var", {}).get(variable, 0)
        if power == -1:
            raise ValueError("Fallback solver does not support logarithmic integrals.")
        t = _copy(term)
        np1 = power + 1
        t["coeff"] = t.get("coeff", 0) / np1
        t.setdefault("var", {})[variable] = np1
        terms.append(_norm_term(t))
    return {"numi": {"terms": terms or [{"coeff": 0}]}, "deno": 1}


def _term_to_text(term):
    coeff = term.get("coeff", 0)
    variables = term.get("var", {})
    if not variables:
        return str(coeff)
    pieces = []
    if coeff == -1:
        pieces.append("-")
    elif coeff != 1:
        pieces.append(str(coeff))
    for name, power in variables.items():
        pieces.append(name if power == 1 else f"{name}^{power}")
    return "".join(pieces)


def _fraction_to_text(expr):
    terms = expr.get("numi", {}).get("terms", [])
    numerator = " + ".join(_term_to_text(t) for t in terms).replace("+ -", "- ")
    denominator = expr.get("deno", 1)
    return numerator if denominator == 1 else f"({numerator}) / ({denominator})"


class FallbackSolver:
    mode = "fallback"
    stage = "none"

    def solve(self, payload):
        operation = payload.get("op")
        variable = payload.get("var", "x")
        expr = payload.get("expr")
        if not isinstance(expr, dict):
            raise ValueError("Input must include expr as a SLaNg fraction object.")

        if operation in ("diff", "partial"):
            output = _diff_fraction(expr, variable)
            rule = "power_rule"
        elif operation == "integrate":
            output = _integrate_fraction(expr, variable)
            rule = "power_rule_integral"
        else:
            raise ValueError(
                "Fallback solver supports diff, partial, and integrate. "
                "Train a checkpoint for full model inference."
            )

        return {
            "status": "solved",
            "expr": output,
            "steps": [
                {
                    "rule": rule,
                    "description": "Solved with deterministic fallback calculus.",
                }
            ],
            "latex": _fraction_to_text(output),
            "confidence": 1.0,
            "verified": True,
            "warning": "Fallback mode: no neural checkpoint is loaded.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT METADATA  (reads training metrics stored alongside best.pt)
# ─────────────────────────────────────────────────────────────────────────────


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    """
    Reads <checkpoint_dir>/metrics.json if it exists.
    training/pretrain.py, finetune.py, and verifier_loop.py should write
    this file at the end of each run with the final val metrics.
    """
    meta_path = checkpoint_path.parent / "metrics.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER LOADER
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_solver():
    try:
        model_path, stage = resolve_model_path()
        from inference.solve import CalculusSolverInference

        solver = CalculusSolverInference(
            model_path=str(model_path),
            vocab_path=str(ROOT / "tokenizer" / "vocab.json"),
            beam_size=5,
            max_len=256,
        )
        solver.mode = "neural"
        solver.stage = stage
        solver.checkpoint_path = model_path
        metadata = load_checkpoint_metadata(model_path)
        return solver, None, metadata
    except Exception as exc:
        fb = FallbackSolver()
        return fb, str(exc), {}


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = {
    "d/dx x²": {
        "op": "diff",
        "var": "x",
        "expr": {"numi": {"terms": [{"coeff": 1, "var": {"x": 2}}]}, "deno": 1},
    },
    "d/dx 3x³ + 2x": {
        "op": "diff",
        "var": "x",
        "expr": {
            "numi": {
                "terms": [
                    {"coeff": 3, "var": {"x": 3}},
                    {"coeff": 2, "var": {"x": 1}},
                ]
            },
            "deno": 1,
        },
    },
    "∫ 6x dx": {
        "op": "integrate",
        "var": "x",
        "expr": {"numi": {"terms": [{"coeff": 6, "var": {"x": 1}}]}, "deno": 1},
    },
    "d/dx 5x⁴ − 3x² + 7": {
        "op": "diff",
        "var": "x",
        "expr": {
            "numi": {
                "terms": [
                    {"coeff": 5, "var": {"x": 4}},
                    {"coeff": -3, "var": {"x": 2}},
                    {"coeff": 7},
                ]
            },
            "deno": 1,
        },
    },
    "∫ (x³ + 2x) dx": {
        "op": "integrate",
        "var": "x",
        "expr": {
            "numi": {
                "terms": [
                    {"coeff": 1, "var": {"x": 3}},
                    {"coeff": 2, "var": {"x": 1}},
                ]
            },
            "deno": 1,
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY TEST RUNNER  (runs eval/evaluate_model.py as a subprocess)
# ─────────────────────────────────────────────────────────────────────────────


def run_accuracy_test(checkpoint_path: Path, num_samples: int, per_rule: bool) -> str:
    """
    Calls eval/evaluate_model.py and streams back its stdout.
    Returns the combined output as a string.
    """
    cmd = [
        sys.executable,
        str(ROOT / "eval" / "evaluate_model.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--num_samples",
        str(num_samples),
    ]
    if per_rule:
        cmd.append("--per_rule")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[stderr]\n{result.stderr}"
        return output
    except Exception as exc:
        return f"Error running evaluation: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="CalculusSolver", layout="wide", page_icon="∫")

solver, solver_error, metadata = load_solver()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="margin-bottom:4px;font-size:2rem;">
        CalculusSolver <span style="font-size:1rem;color:#888;">neural calculus engine</span>
    </h1>
    """,
    unsafe_allow_html=True,
)

# ── Model Status Banner ───────────────────────────────────────────────────────
if solver_error:
    st.info(
        "⚠️ **Fallback mode** — No trained checkpoint found. "
        "Only basic polynomial differentiation and integration are available. "
        "Run the three-stage GPU training pipeline to enable full neural inference.",
        icon=None,
    )
    with st.expander("🔍 Checkpoint search details"):
        st.code(solver_error)
else:
    stage_label = getattr(solver, "stage", "unknown")
    checkpoint_path = getattr(solver, "checkpoint_path", None)
    st.success(
        f"✅ **Neural checkpoint loaded** — stage: `{stage_label}` · "
        f"path: `{checkpoint_path}`"
    )

st.divider()

# ── Training Metrics Panel ────────────────────────────────────────────────────
if metadata:
    st.subheader("📈 Training Metrics")
    st.caption("Metrics from the last training run stored alongside the checkpoint.")

    metric_cols = st.columns(4)
    metric_map = [
        ("val/numerical_equiv", "Numerical Equiv", "%.3f"),
        ("val/rule_accuracy", "Rule Accuracy", "%.3f"),
        ("val/step_accuracy", "Step Accuracy", "%.3f"),
        ("train/hard_pool_size", "Hard Pool Size", "%d"),
    ]
    for col, (key, label, fmt) in zip(metric_cols, metric_map):
        value = metadata.get(key)
        if value is not None:
            try:
                col.metric(label, fmt % float(value))
            except (TypeError, ValueError):
                col.metric(label, str(value))
        else:
            col.metric(label, "—")

    if "training_steps" in metadata:
        st.caption(f"Checkpoint trained for **{metadata['training_steps']:,}** steps.")

    st.divider()
elif not solver_error:
    st.info(
        "💡 No `metrics.json` found next to the checkpoint. "
        "Your training scripts can write one with keys `val/numerical_equiv`, "
        "`val/rule_accuracy`, `val/step_accuracy`, and `train/hard_pool_size` "
        "to display live metrics here.",
        icon=None,
    )
    st.divider()

# ── Interactive Solver ────────────────────────────────────────────────────────
st.subheader("🧮 Interactive Solver")

left, right = st.columns([0.45, 0.55])

with left:
    selected = st.selectbox("Load example", list(EXAMPLES.keys()))
    default_payload = json.dumps(EXAMPLES[selected], indent=2)
    raw_input = st.text_area(
        "SLaNg input envelope (JSON)",
        value=default_payload,
        height=320,
        help="Edit any field or paste a custom SLaNg op envelope.",
    )
    run = st.button("▶  Solve", type="primary", use_container_width=True)

with right:
    st.subheader("Result")
    if run:
        try:
            payload = json.loads(raw_input)
            result = solver.solve(payload)
            st.json(result)

            # Surface warning prominently
            if result.get("warning"):
                st.warning(result["warning"])

            # Show confidence and verified badges
            conf = result.get("confidence", 0.0)
            verified = result.get("verified")
            badge_cols = st.columns(3)
            badge_cols[0].metric("Confidence", f"{conf:.2%}")
            badge_cols[1].metric("Verified", "✓ Yes" if verified else "✗ No")
            badge_cols[2].metric("Mode", getattr(solver, "mode", "—"))

        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        except Exception as exc:
            st.error(str(exc))
    else:
        st.caption("Choose an example or edit the JSON, then press **Solve**.")

st.divider()

# ── Accuracy Test Runner ──────────────────────────────────────────────────────
st.subheader("🔬 Accuracy Test Runner")
st.caption(
    "Runs `eval/evaluate_model.py` against the loaded checkpoint. "
    "Results stream into the panel below."
)

acc_col1, acc_col2, acc_col3 = st.columns([0.3, 0.3, 0.4])

with acc_col1:
    num_samples = st.number_input(
        "Samples to evaluate",
        min_value=10,
        max_value=10000,
        value=100,
        step=10,
        help="More samples = more accurate results but slower. 100 is a good quick check.",
    )

with acc_col2:
    per_rule = st.checkbox(
        "Per-rule breakdown",
        value=False,
        help="Show accuracy separately for each calculus rule.",
    )

with acc_col3:
    run_eval = st.button(
        "▶  Run accuracy test", use_container_width=True, disabled=bool(solver_error)
    )
    if solver_error:
        st.caption("⚠️ Accuracy test requires a trained checkpoint.")

if run_eval:
    checkpoint_path = getattr(solver, "checkpoint_path", None)
    if checkpoint_path is None:
        st.error("No checkpoint path available. Train a model first.")
    else:
        with st.spinner(f"Evaluating {num_samples} samples …"):
            output = run_accuracy_test(
                checkpoint_path=checkpoint_path,
                num_samples=num_samples,
                per_rule=per_rule,
            )
        st.subheader("Evaluation Output")
        st.code(output, language="text")

st.divider()

# ── GPU Training Reference ────────────────────────────────────────────────────
with st.expander("⚙️ GPU Training Quick Reference"):
    st.markdown("### Three-Stage Training Pipeline")
    st.markdown(
        "Run these commands in order. Each stage reads from the previous checkpoint."
    )

    st.markdown("**Stage 1 — Grammar Pretraining** (~18 h on A100)")
    st.code(
        "python training/pretrain.py \\\n"
        "  --config training/config/pretrain.yaml \\\n"
        "  --data   data/splits/train.jsonl \\\n"
        "  --output checkpoints/pretrain/",
        language="bash",
    )

    st.markdown("**Stage 2 — Supervised Fine-Tuning** (~10 h on A100)")
    st.code(
        "python training/finetune.py \\\n"
        "  --checkpoint checkpoints/pretrain/best.pt \\\n"
        "  --config     training/config/finetune.yaml \\\n"
        "  --data       data/splits/train.jsonl \\\n"
        "  --val        data/splits/val.jsonl \\\n"
        "  --output     checkpoints/sft/",
        language="bash",
    )

    st.markdown("**Stage 3 — Verifier-Loop Hard Examples** (~8 h on A100)")
    st.code(
        "python training/verifier_loop.py \\\n"
        "  --checkpoint        checkpoints/sft/best.pt \\\n"
        "  --data              data/splits/train.jsonl \\\n"
        "  --hard-example-ratio 0.4 \\\n"
        "  --output            checkpoints/final/",
        language="bash",
    )

    st.markdown("**Multi-GPU (Accelerate)**")
    st.code(
        "accelerate config    # run once\n"
        "accelerate launch training/pretrain.py   [same flags as above]\n"
        "accelerate launch training/finetune.py   [same flags]\n"
        "accelerate launch training/verifier_loop.py [same flags]",
        language="bash",
    )

    st.markdown("**Data Generation**")
    st.code(
        "node data_pipeline/generate_slang_data.js --count 100000 --out data/slang_dataset.jsonl\n"
        "node data_pipeline/split_data.js",
        language="bash",
    )

    st.markdown(
        "**Save metrics.json** (add to your training scripts to enable the metrics panel above)"
    )
    st.code(
        "import json\n"
        "metrics = {\n"
        '    "val/numerical_equiv": float(best_num_equiv),\n'
        '    "val/rule_accuracy":   float(best_rule_acc),\n'
        '    "val/step_accuracy":   float(best_step_acc),\n'
        '    "train/hard_pool_size": int(hard_pool_size),   # Stage 3 only\n'
        '    "training_steps": int(total_steps),\n'
        "}\n"
        'with open(output_dir / "metrics.json", "w") as f:\n'
        "    json.dump(metrics, f, indent=2)",
        language="python",
    )
