import streamlit as st
import json
import os
import sys
from pathlib import Path
import torch

# Ensure current directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from inference.solve import CalculusSolverInference
from api._shared import fraction_to_latex, normalize_solver_result

# ── Page Configuration ──
st.set_page_config(
    page_title="SLaNg Calculus Solver",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Styling (Premium Aesthetics) ──
st.markdown("""
<style>
    /* Premium Fonts and Backgrounds */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@300;400&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Sleek gradient top border */
    .stApp {
        border-top: 8px solid;
        border-image: linear-gradient(135deg, #FF3366, #FF9933, #33CCFF, #3366FF) 1;
    }
    
    /* Glow effect for cards */
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    /* Vibrant highlight text */
    .highlight-text {
        background: linear-gradient(135deg, #33CCFF, #3366FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── Preset Selection Problems ──
PRESETS = {
    "Differentiate 3x^2": {
        "op": "diff",
        "var": "x",
        "expr": {
            "numi": {"terms": [{"coeff": 3, "var": {"x": 2}}]},
            "deno": 1
        }
    },
    "Integrate 6x": {
        "op": "integrate",
        "var": "x",
        "expr": {
            "numi": {"terms": [{"coeff": 6, "var": {"x": 1}}]},
            "deno": 1
        }
    },
    "Differentiate 5x^3 (Partial w.r.t x)": {
        "op": "partial",
        "var": "x",
        "expr": {
            "numi": {"terms": [{"coeff": 5, "var": {"x": 3}}]},
            "deno": 1
        }
    },
    "Gradient of x^2 + y^2": {
        "op": "gradient",
        "vars": ["x", "y"],
        "expr": {
            "numi": {"terms": [
                {"coeff": 1, "var": {"x": 2}},
                {"coeff": 1, "var": {"y": 2}}
            ]},
            "deno": 1
        }
    }
}

# ── Sidebar Configuration ──
st.sidebar.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=70)
st.sidebar.markdown("<h2 class='highlight-text'>Calculus AI Settings</h2>", unsafe_allow_html=True)
st.sidebar.write("Choose the neural checkpoint configuration to load for solving.")

# Checkpoint paths
default_ckpt = "checkpoints/checkpoint_epoch_1.pt"
ckpt_input = st.sidebar.text_input("Model Checkpoint Path", default_ckpt)

if not os.path.exists(ckpt_input):
    st.sidebar.warning(f"⚠️ Checkpoint not found at {ckpt_input}. Solver will fall back to Fallback/Deterministic mode.")
else:
    st.sidebar.success("🎯 Neural checkpoint detected!")

# Vocab Path
vocab_path = st.sidebar.text_input("Vocabulary path", "tokenizer/vocab.json")

# ── Main Application Body ──
st.markdown("<h1 class='highlight-text' style='text-align: center;'>🎯 SLaNg Neural Calculus Solver</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #888;'>High-fidelity neural calculus engine running on Streamlit Cloud</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🎛️ Input Calculus Envelope")
    
    # Preset Selector
    preset_choice = st.selectbox("Load a Preset Problem", ["-- Custom --"] + list(PRESETS.keys()))
    
    # Raw JSON text input
    if preset_choice != "-- Custom--":
        initial_val = json.dumps(PRESETS.get(preset_choice, PRESETS["Differentiate 3x^2"]), indent=2)
    else:
        initial_val = json.dumps(PRESETS["Differentiate 3x^2"], indent=2)
        
    envelope_json = st.text_area("SLaNg JSON Envelope", value=initial_val, height=220)
    
    solve_btn = st.button("🚀 Run Neural Inference", use_container_width=True)

with col2:
    st.markdown("### 📝 Solver Output & Verification")
    
    if solve_btn:
        try:
            input_envelope = json.loads(envelope_json)
        except Exception as e:
            st.error(f"❌ Invalid JSON envelope format: {e}")
            input_envelope = None
            
        if input_envelope:
            with st.spinner("🧠 Running neural inference and verifier..."):
                try:
                    # Instantiate inference solver
                    # If checkpoint does not exist, it will fall back
                    if os.path.exists(ckpt_input):
                        solver = CalculusSolverInference(model_path=ckpt_input, vocab_path=vocab_path)
                        result = solver.solve(input_envelope)
                        mode = "neural"
                    else:
                        from inference.fallback_solver import FallbackSolver
                        solver = FallbackSolver()
                        result = solver.solve(input_envelope)
                        mode = "fallback"
                    
                    # Normalize response format
                    normalized = normalize_solver_result(result, mode)
                    
                    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                    
                    # Metrics columns
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Status", normalized["status"].upper())
                    with m2:
                        st.metric("Mode", normalized["mode"].upper())
                    with m3:
                        st.metric("Confidence", f"{normalized['confidence'] * 100:.1f}%")
                    
                    # LaTeX Output
                    st.markdown("#### 📐 Mathematically Equivalent Answer (LaTeX)")
                    st.latex(normalized["latex"])
                    
                    # Rule identified
                    st.markdown(f"**Identified Rule:** `{normalized.get('rule', 'unknown')}`")
                    
                    # Step-by-step trace
                    if normalized.get("steps"):
                        st.markdown("#### 🪜 Derivation Breakdown")
                        for idx, step in enumerate(normalized["steps"], 1):
                            st.write(f"**Step {idx}:** {step.get('description')}")
                            if "before" in step and "after" in step:
                                st.write(f"  * `{step['before']}` → `{step['after']}`")
                                
                    if normalized.get("warning"):
                        st.warning(f"⚠️ {normalized['warning']}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show Raw JSON
                    with st.expander("🔍 Show Raw Response JSON"):
                        st.json(normalized)
                        
                except Exception as exc:
                    st.exception(exc)
    else:
        st.info("👈 Enter/select a problem on the left and click **Run Neural Inference**.")
