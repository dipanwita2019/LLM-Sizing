import streamlit as st
import pandas as pd
import math
import hashlib

# ===== PASSWORD PROTECTION =====
def check_password():
    """Returns True if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Replace this hash with your own generated hash
        # Current hash is for password: "password"
        correct_password_hash = "643435139fa71bb855e0e4375d5e77268fcffaa97f118300d349e746414f93e2"

        entered_hash = hashlib.sha256(st.session_state["password"].encode()).hexdigest()

        if entered_hash == correct_password_hash:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Return True if password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show login screen
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🔒 LLM Sizer - Login")
    st.markdown("**Enter password to access the tool**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.text_input(
        "Password",
        type="password",
        on_change=password_entered,
        key="password",
        help="Contact admin if you don't have access"
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("❌ Incorrect password. Please try again.")

    return False


# Check password before showing app
if not check_password():
    st.stop()
# ===== END PASSWORD PROTECTION =====


# Professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }

    .block-container {
        background: white;
        border-radius: 24px;
        padding: 3rem 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        max-width: 800px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border-radius: 16px;
        padding: 1rem 2rem;
        width: 100%;
        border: none;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.6);
    }

    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    h2 {
        color: #667eea;
        font-size: 1.8rem;
        margin-top: 2rem;
        font-weight: 700;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }

    h3 {
        color: #764ba2;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }

    .stRadio > label, .stSelectbox > label {
        font-weight: 600;
        color: #2d3748;
        font-size: 1.1rem;
    }

    .stRadio > div {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }

    .stRadio > div:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }

    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    [data-testid="stMetricLabel"] {
        color: #4a5568;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    .stSelectbox > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }

    .stExpander {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        margin: 1rem 0;
    }

    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }

    .success-box {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }

    [data-testid="stExpander"] {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
    }

    .caption-text {
        color: #718096;
        font-size: 0.9rem;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("⚡ GPU Sizing Tool")
st.markdown(
    "<p style='text-align: center; color: #718096; font-size: 1.2rem; margin-top: -1rem;'><strong>Lightning-fast hardware recommendations for AI workloads</strong></p>",
    unsafe_allow_html=True)
st.markdown("---")

# Question 1: Workload Type
st.markdown("### 1. Workload Type")
workload = st.radio(
    "Select workload type",
    ["Inference", "Fine-Tuning", "Training"],
    horizontal=True,
    key="workload",
    label_visibility="collapsed"
)

st.markdown("")

# Question 2: Model Size
st.markdown("### 2. Model Size")
model_options = {
    "Qwen2.5-0.5B": 0.5,
    "Qwen2.5-1.5B": 1.5,
    "Qwen2.5-3B": 3.0,
    "Qwen2.5-7B": 7.0,
    "Qwen2.5-14B": 14.0,
    "Qwen2.5-32B": 32.0,
    "Qwen2.5-72B": 72.0,
    "Llama 3.1 8B": 8.0,
    "Llama 3.1 70B": 70.0,
    "Llama 3.1 405B": 405.0
}

model_name = st.selectbox("Select model", list(model_options.keys()), label_visibility="collapsed")
params = model_options[model_name]

st.markdown("")

# Question 3: Precision
st.markdown("### 3. Precision")
precision = st.radio(
    "Select precision",
    ["FP16", "INT8", "INT4"],
    horizontal=True,
    key="precision",
    label_visibility="collapsed"
)

st.markdown("")

# Advanced Settings (collapsible)
with st.expander("⚙️ Advanced Settings (Optional)"):
    st.markdown("**Batch Size**")
    st.caption("Number of requests processed simultaneously. Higher = more throughput but more VRAM.")
    batch_size = st.slider("Batch size", min_value=1, max_value=32, value=1, key="batch", label_visibility="collapsed")

    st.markdown("")
    st.markdown("**Sequence Length**")
    st.caption("Maximum length of text context (in tokens). Longer context = more VRAM. 1 token ≈ 0.75 words.")
    sequence_length = st.slider("Sequence length", min_value=512, max_value=8192, value=2048, step=512, key="sequence",
                                label_visibility="collapsed")

st.markdown("---")


# VRAM calculation based on ApXML formulas
def calculate_vram(params, precision, workload, batch_size=1, sequence_length=2048):
    """Calculate VRAM based on ApXML methodology"""

    # Bytes per parameter
    bytes_map = {"FP16": 2, "INT8": 1, "INT4": 0.5}
    bytes_per_param = bytes_map[precision]

    # Model weights in GB
    model_weights = (params * 1e9 * bytes_per_param) / (1024 ** 3)

    # KV Cache (scales with batch size and sequence length)
    # Base calculation: 12% for batch=1, seq=2048
    # Scale proportionally for different values
    base_kv = model_weights * 0.12
    kv_cache = base_kv * (batch_size * sequence_length) / (1 * 2048)

    # Activation memory varies by workload and batch size
    if workload == "Inference":
        activations = model_weights * 0.2 * batch_size
    elif workload == "Fine-Tuning":
        activations = model_weights * 1.5 * batch_size
    else:  # Training
        activations = model_weights * 3.5 * batch_size

    # Total with 20% overhead
    total = (model_weights + kv_cache + activations) * 1.2

    return {
        "model_weights": model_weights,
        "kv_cache": kv_cache,
        "activations": activations,
        "total": total
    }


# Calculate button
if st.button("Calculate Requirements"):
    vram = calculate_vram(params, precision, workload, batch_size, sequence_length)

    st.markdown("<div class='success-box'>✨ Calculation Complete!</div>", unsafe_allow_html=True)

    # Results section
    st.markdown("---")
    st.markdown("## 💾 VRAM Requirements")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total VRAM Needed", f"{vram['total']:.1f} GB")
    with col2:
        st.metric("Model Weights", f"{vram['model_weights']:.1f} GB")

    # Breakdown
    with st.expander("View Breakdown"):
        st.markdown(f"""
        **Memory Components:**
        - Model Weights: {vram['model_weights']:.1f} GB
        - KV Cache: {vram['kv_cache']:.1f} GB
        - Activations: {vram['activations']:.1f} GB
        - Overhead (20%): {(vram['total'] - (vram['model_weights'] + vram['kv_cache'] + vram['activations'])):.1f} GB

        **Total: {vram['total']:.1f} GB**
        """)

    st.markdown("---")

    # System Requirements
    st.markdown("## System Requirements")
    ram_needed = math.ceil(vram['total'] * 2)
    st.markdown(f"""
    **Recommended System RAM:** {ram_needed} GB minimum

    *General rule: 2× VRAM for optimal performance*
    """)

# Footer
st.markdown("---")
st.caption("Calculations based on ApXML VRAM methodology")
st.caption(
    f"Current settings: Batch size = {batch_size if 'batch_size' in locals() else 1}, Sequence length = {sequence_length if 'sequence_length' in locals() else 2048}")