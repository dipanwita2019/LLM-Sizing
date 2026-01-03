import streamlit as st
import pandas as pd
import math
import hashlib
from datetime import datetime


# ===== PASSWORD PROTECTION =====
def check_password():
    """Returns True if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        correct_password_hash = "643435139fa71bb855e0e4375d5e77268fcffaa97f118300d349e746414f93e2"

        entered_hash = hashlib.sha256(st.session_state["password"].encode()).hexdigest()

        if entered_hash == correct_password_hash:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

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


if not check_password():
    st.stop()

# Professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
        margin: 0;
    }

    .block-container {
        background: white;
        border-radius: 0;
        padding: 3rem 5%;
        box-shadow: none;
        max-width: 100%;
        margin: 0;
    }

    [data-testid="stAppViewContainer"] {
        padding: 0;
        margin: 0;
    }

    section[data-testid="stSidebar"] {
        display: none;
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

    .platform-info-box {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #667eea;
    }

    .future-proof-box {
        background: linear-gradient(135deg, #e6f7ff 0%, #d1f0ff 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #4ba3c7;
    }

    .multi-node-warning {
        background: linear-gradient(135deg, #f6ad55 0%, #ed8936 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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

# Question 2: Model Selection (Open Source Models - 2026 Updated)
st.markdown("### 2. Model Selection")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Model Family**")
    model_family = st.selectbox(
        "Select model family",
        [
            "Llama (Meta)",
            "DeepSeek",
            "Qwen (Alibaba)",
            "Mistral AI",
            "GPT-OSS (OpenAI)",
            "Phi (Microsoft)",
            "Gemma (Google)",
            "MiMo (Xiaomi)",
            "Granite (IBM)",
            "Falcon (TII)",
            "Custom"
        ],
        index=0,  # Llama is default
        key="model_family",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**Version**")
    if "Llama" in model_family:
        version_options = ["3.1", "4 Behemoth (Preview)", "4 Maverick", "4 Scout", "3.3", "3.0", "2"]
    elif "DeepSeek" in model_family:
        version_options = ["V3.2", "R1 (Reasoning)", "V3", "V2.5", "Coder-V2"]
    elif "Qwen" in model_family:
        version_options = ["3-Max", "2.5", "2.5-Coder", "QwQ (Reasoning)"]
    elif "Mistral" in model_family:
        version_options = ["Large 2", "Mixtral 8x22B", "Mixtral 8x7B", "Small 2409", "7B v0.3"]
    elif "GPT-OSS" in model_family:
        version_options = ["120B", "20B"]
    elif "Phi" in model_family:
        version_options = ["4", "4-mini-flash", "3.5", "3 Medium", "3 Mini"]
    elif "Gemma" in model_family:
        version_options = ["2 (27B)", "2 (9B)", "2 (2B)"]
    elif "MiMo" in model_family:
        version_options = ["V2-Flash"]
    elif "Granite" in model_family:
        version_options = ["4.0-H-Small", "4.0-H-Tiny", "4.0-H-Micro", "3.2", "3.1", "3.0", "8B", "2B"]
    elif "Falcon" in model_family:
        version_options = ["3 (10B)", "3 (7B)", "3 (3B)", "3 (1B)", "2 11B", "Mamba 7B", "180B", "40B"]
    else:
        version_options = ["Custom"]

    version = st.selectbox(
        "Select version",
        version_options,
        key="version",
        label_visibility="collapsed"
    )

with col3:
    st.markdown("**Parameters (B)**")

    if "Llama" in model_family:
        if "4 Behemoth" in version:
            param_options = ["~2000 [288 active]"]
        elif "4 Maverick" in version:
            param_options = ["400 [17 active]"]
        elif "4 Scout" in version:
            param_options = ["109 [17 active]"]
        elif version == "3.3":
            param_options = ["70"]
        elif version == "3.1":
            param_options = ["8", "70", "405"]
        elif version == "3.0":
            param_options = ["8", "70"]
        else:
            param_options = ["7", "13", "70"]
    elif "DeepSeek" in model_family:
        if "V3.2" in version:
            param_options = ["685 [37 active]"]
        elif "R1" in version:
            param_options = ["671 [37 active]"]
        elif "V3" in version:
            param_options = ["671"]
        elif "V2.5" in version:
            param_options = ["236"]
        elif "Coder" in version:
            param_options = ["16", "236"]
        else:
            param_options = ["671"]
    elif "Qwen" in model_family:
        if "3-Max" in version:
            param_options = ["235 [22 active]"]
        elif "Coder" in version:
            param_options = ["32"]
        elif "QwQ" in version:
            param_options = ["32"]
        else:
            param_options = ["0.5", "1.5", "3", "7", "14", "32", "72"]
    elif "Mistral" in model_family:
        if "Large" in version:
            param_options = ["123"]
        elif "8x22B" in version:
            param_options = ["141 [39 active]"]
        elif "8x7B" in version:
            param_options = ["47 [13 active]"]
        elif "Small" in version:
            param_options = ["22"]
        else:
            param_options = ["7"]
    elif "GPT-OSS" in model_family:
        if "120B" in version:
            param_options = ["117 [5.1 active]"]
        else:
            param_options = ["20"]
    elif "Phi" in model_family:
        if version == "4":
            param_options = ["14.7"]
        elif "mini-flash" in version:
            param_options = ["3.8"]
        elif version == "3.5":
            param_options = ["3.8"]
        elif "Medium" in version:
            param_options = ["14"]
        else:
            param_options = ["3.8"]
    elif "Gemma" in model_family:
        if "27B" in version:
            param_options = ["27"]
        elif "9B" in version:
            param_options = ["9"]
        else:
            param_options = ["2"]
    elif "MiMo" in model_family:
        param_options = ["309 [15 active]"]
    elif "Granite" in model_family:
        if "4.0-H-Small" in version:
            param_options = ["32 [9 active]"]
        elif "4.0-H-Tiny" in version:
            param_options = ["7 [1 active]"]
        elif "4.0-H-Micro" in version:
            param_options = ["3"]
        elif version == "8B":
            param_options = ["8"]
        elif version == "2B":
            param_options = ["2"]
        else:
            param_options = ["8"]
    elif "Falcon" in model_family:
        if "180B" in version:
            param_options = ["180"]
        elif "40B" in version:
            param_options = ["40"]
        elif "2 11B" in version:
            param_options = ["11"]
        elif "Mamba" in version:
            param_options = ["7"]
        elif "(10B)" in version:
            param_options = ["10"]
        elif "(7B)" in version:
            param_options = ["7"]
        elif "(3B)" in version:
            param_options = ["3"]
        elif "(1B)" in version:
            param_options = ["1"]
        else:
            param_options = ["40"]
    else:
        params = st.number_input(
            "Enter parameters (B)",
            min_value=0.1,
            max_value=2000.0,
            value=7.0,
            step=0.5,
            key="params_custom",
            label_visibility="collapsed"
        )
        param_options = None

    if param_options:
        params_str = st.selectbox(
            "Select parameters",
            param_options,
            key="params",
            label_visibility="collapsed"
        )
        params_clean = params_str.replace("~", "").split(" ")[0]
        params = float(params_clean)

if model_family != "Custom":
    family_clean = model_family.split(" (")[0]
    if "[" in params_str and "active" in params_str:
        model_name = f"{family_clean} {version} {params_str}"
    else:
        model_name = f"{family_clean} {version} {params}B"
else:
    model_name = f"Custom {version} {params}B"

st.caption(f"✅ Selected: **{model_name}**")
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

# Advanced Settings
with st.expander("⚙️ Advanced Settings (Optional)"):
    st.markdown("**Batch Size**")
    st.caption("Number of requests processed simultaneously.")
    batch_size = st.slider("Batch size", min_value=1, max_value=32, value=1, key="batch", label_visibility="collapsed")

    st.markdown("")
    st.markdown("**Sequence Length**")
    st.caption("Maximum context length (tokens). 1 token ≈ 0.75 words.")
    sequence_length = st.slider("Sequence length", min_value=512, max_value=8192, value=2048, step=512, key="sequence",
                                label_visibility="collapsed")

st.markdown("---")


# VRAM calculation - ORIGINAL ApXML METHOD
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


# Workstation data
workstations = [
    {"name": "Z2 Mini G1i", "max_vram": 20, "config": "1x RTX 4000 SFF Ada (20GB)", "gpus": 1,
     "supported": ["1x A400 4GB", "1x A1000 8GB", "1x RTX 2000 Ada 16GB", "1x RTX 4000 SFF Ada 20GB"]},
    {"name": "Z2 Tower G1i", "max_vram": 96, "config": "1x RTX PRO 6000 Blackwell Max-Q (96GB)", "gpus": 1,
     "supported": ["2x A400 4GB", "2x A1000 8GB", "2x RTX 2000 Ada 16GB", "2x RTX 4000 Ada 20GB",
                   "1x RTX 4500 Ada 24GB", "1x RTX 5000 Ada 32GB", "1x RTX 5880 Ada 48GB", "1x RTX 6000 Ada 48GB",
                   "1x RTX PRO 4000 Blackwell 24GB", "1x RTX PRO 4500 Blackwell 32GB", "1x RTX PRO 5000 Blackwell 48GB",
                   "1x RTX PRO 6000 Blackwell Max-Q 96GB"]},
    {"name": "Z4 G5", "max_vram": 192, "config": "2x RTX PRO 6000 Blackwell Max-Q (192GB total)", "gpus": 2,
     "supported": ["2x RTX A400 4GB", "2x RTX A1000 8GB", "2x RTX A4000 16GB", "2x RTX A4500 20GB",
                   "2x RTX 2000 Ada 16GB", "2x RTX 4000 Ada 20GB", "2x RTX 4500 Ada 24GB", "2x RTX 5000 Ada 32GB",
                   "2x RTX 5880 Ada 48GB", "2x RTX 6000 Ada 48GB", "2x RTX PRO 4000 Blackwell 24GB",
                   "2x RTX PRO 4500 Blackwell 32GB", "2x RTX PRO 5000 Blackwell 48GB",
                   "2x RTX PRO 6000 Blackwell Max-Q 96GB"]},
    {"name": "Z6 G5", "max_vram": 288, "config": "3x RTX PRO 6000 Blackwell Max-Q (288GB total)", "gpus": 3,
     "supported": ["3x RTX A400 4GB", "2x NVIDIA A800 40GB", "3x RTX A1000 8GB", "3x RTX A4000 16GB",
                   "3x RTX A4500 20GB", "3x RTX 2000 Ada 16GB", "3x RTX 4000 Ada 20GB", "3x RTX 4500 Ada 24GB",
                   "3x RTX 5000 Ada 32GB", "3x RTX 5880 Ada 48GB", "3x RTX 6000 Ada 48GB",
                   "3x RTX PRO 4000 Blackwell 24GB", "3x RTX PRO 4500 Blackwell 32GB", "3x RTX PRO 5000 Blackwell 48GB",
                   "3x RTX PRO 6000 Blackwell Max-Q 96GB"]},
    {"name": "Z8 G5", "max_vram": 192, "config": "2x RTX PRO 6000 Blackwell Max-Q (192GB total)", "gpus": 2,
     "supported": ["2x RTX A400 4GB", "2x RTX A1000 8GB", "2x RTX A4000 16GB", "2x RTX A4500 20GB",
                   "2x RTX 2000 Ada 16GB", "2x RTX 4000 Ada 20GB", "2x RTX 4500 Ada 24GB", "2x RTX 5000 Ada 32GB",
                   "2x RTX 5880 Ada 48GB", "2x RTX 6000 Ada 48GB", "2x RTX PRO 4000 Blackwell 24GB",
                   "2x RTX PRO 4500 Blackwell 32GB", "2x RTX PRO 5000 Blackwell 48GB",
                   "2x RTX PRO 6000 Blackwell Max-Q 96GB"]},
    {"name": "Z8 Fury G5", "max_vram": 384, "config": "4x RTX PRO 6000 Blackwell Max-Q (384GB total)", "gpus": 4,
     "supported": ["3x NVIDIA A800 40GB", "4x RTX A400 4GB", "4x RTX A1000 8GB", "4x RTX A4000 16GB",
                   "4x RTX A4500 20GB", "4x RTX 2000 Ada 16GB", "4x RTX 4000 Ada 20GB", "4x RTX 4500 Ada 24GB",
                   "4x RTX 5000 Ada 32GB", "4x RTX 5880 Ada 48GB", "4x RTX 6000 Ada 48GB",
                   "4x RTX PRO 4000 Blackwell 24GB", "4x RTX PRO 4500 Blackwell 32GB", "4x RTX PRO 5000 Blackwell 48GB",
                   "4x RTX PRO 6000 Blackwell Max-Q 96GB"]},
]


def find_platform(vram_needed):
    """Find the minimum platform that fits the VRAM requirement"""
    for ws in workstations:
        if ws["max_vram"] >= vram_needed:
            return ws
    # Multi-node if exceeds single workstation
    num_nodes = math.ceil(vram_needed / 384)
    return {
        "name": f"{num_nodes}x Z8 Fury G5",
        "max_vram": num_nodes * 384,
        "config": f"{num_nodes} nodes × 4 GPUs × 96GB = {num_nodes * 384}GB total",
        "gpus": num_nodes * 4,
        "multi_node": True
    }


def recommend_with_headroom(vram_required, workload):
    """Two-tier recommendation: Minimum + Future-Proof with workload-specific headroom"""

    # Headroom percentages by workload
    headroom_map = {
        "Inference": 0.30,
        "Fine-Tuning": 0.30,
        "Training": 0.50
    }

    headroom_pct = headroom_map.get(workload, 0.30)
    future_proof_vram = vram_required * (1 + headroom_pct)

    # Find platforms
    minimum = find_platform(vram_required)
    future_proof = find_platform(future_proof_vram)

    # If same platform, bump future-proof to next tier
    if minimum["name"] == future_proof["name"]:
        # Find next tier
        idx = next((i for i, ws in enumerate(workstations) if ws["name"] == minimum["name"]), -1)
        if idx < len(workstations) - 1:
            future_proof = workstations[idx + 1]
        else:
            # Already at max, suggest multi-node
            future_proof = {
                "name": "2x Z8 Fury G5",
                "max_vram": 768,
                "config": "2 nodes × 4 GPUs × 96GB = 768GB total",
                "gpus": 8,
                "multi_node": True
            }

    # Calculate actual headroom
    min_headroom_gb = minimum["max_vram"] - vram_required
    min_headroom_pct = (min_headroom_gb / vram_required) * 100

    fp_headroom_gb = future_proof["max_vram"] - vram_required
    fp_headroom_pct = (fp_headroom_gb / vram_required) * 100

    return {
        "minimum": {
            **minimum,
            "headroom_gb": round(min_headroom_gb, 1),
            "headroom_pct": int(min_headroom_pct)
        },
        "future_proof": {
            **future_proof,
            "headroom_gb": round(fp_headroom_gb, 1),
            "headroom_pct": int(fp_headroom_pct)
        },
        "target_headroom_pct": int(headroom_pct * 100)
    }


# Calculate button
if st.button("Calculate Requirements"):
    vram = calculate_vram(params, precision, workload, batch_size, sequence_length)

    st.markdown("<div class='success-box'>✨ Calculation Complete!</div>", unsafe_allow_html=True)

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

    # Recommendations
    recommendations = recommend_with_headroom(vram['total'], workload)

    minimum = recommendations['minimum']
    future_proof = recommendations['future_proof']
    target_headroom = recommendations['target_headroom_pct']

    st.markdown("## 🖥️ Platform Recommendations")

    # Multi-node warning
    if minimum.get("multi_node"):
        st.markdown(f"""
        <div class='multi-node-warning'>
            ⚠️ <strong>Multi-Node Setup Required</strong><br>
            Your workload requires a cluster configuration.<br>
            <em>Consultation recommended for optimal setup.</em>
        </div>
        """, unsafe_allow_html=True)

    # MINIMUM CONFIGURATION
    st.markdown("### ✅ Minimum Configuration")
    st.markdown(f"**{minimum['name']}**")

    st.markdown(f"""
    <div class='platform-info-box'>
        <p><strong>Configuration:</strong> {minimum['config']}</p>
        <p><strong>Total VRAM:</strong> {minimum['max_vram']} GB</p>
        <p><strong>Headroom:</strong> +{minimum['headroom_gb']} GB ({minimum['headroom_pct']}%)</p>
        <p><strong>Timeline:</strong> Fits current workload</p>
    </div>
    """, unsafe_allow_html=True)

    if not minimum.get("multi_node") and "supported" in minimum:
        with st.expander("🎮 View Supported GPUs"):
            for gpu in minimum['supported']:
                st.markdown(f"• {gpu}")

    st.markdown("")

    # FUTURE-PROOF CONFIGURATION
    st.markdown(f"### 🚀 Future-Proof Configuration (+{target_headroom}% Headroom)")
    st.markdown(f"{future_proof['name']} - Recommended for {workload}")

    st.markdown(f"""
    <div class='future-proof-box'>
        <p><strong>Configuration:</strong> {future_proof['config']}</p>
        <p><strong>Total VRAM:</strong> {future_proof['max_vram']} GB</p>
        <p><strong>Headroom:</strong> +{future_proof['headroom_gb']} GB ({future_proof['headroom_pct']}%)</p>
        <p><strong>Timeline:</strong> Future-proof for 18-30 months</p>
    </div>
    """, unsafe_allow_html=True)

    # Workload-specific benefits
    with st.expander(f"💡 Why {target_headroom}% Headroom for {workload}?"):
        if workload == "Inference":
            st.markdown("""
            **For Inference workloads, 30% headroom provides:**
            • Handle 2-4x batch size increase for growing traffic
            • Accommodate context window expansion (2K → 8K+ tokens)
            • Support next-generation model upgrades
            """)
        elif workload == "Fine-Tuning":
            st.markdown("""
            **For Fine-Tuning workloads, 30% headroom provides:**
            • Upgrade from LoRA to full fine-tuning without hardware change
            • Experiment with larger base models as they release
            • Run multiple fine-tuning experiments simultaneously
            """)
        else:  # Training
            st.markdown("""
            **For Training workloads, 50% headroom provides:**
            • Train models 2-3x larger without infrastructure upgrade
            • Experiment with distributed training techniques
            • Build multi-stage training pipelines (pre-training + fine-tuning)
            """)

        st.caption("* Based on industry best practices and AI model growth trends (2018-2026)")

    if not future_proof.get("multi_node") and "supported" in future_proof:
        with st.expander("🎮 View Supported GPUs"):
            for gpu in future_proof['supported']:
                st.markdown(f"• {gpu}")

    st.markdown("---")

    # System RAM
    st.markdown("## 💻 System RAM Requirements")
    ram_multipliers = {
        "Inference": 1.5,
        "Fine-Tuning": 2.0,
        "Training": 2.5
    }
    ram_multiplier = ram_multipliers.get(workload, 2.0)
    ram_needed = max(16, math.ceil(vram['total'] * ram_multiplier))

    st.markdown(f"""
    **Recommended System RAM:** {ram_needed} GB minimum

    *Rule for {workload}: {ram_multiplier}× VRAM*
    """)

    st.caption("System RAM (motherboard memory) is separate from VRAM (GPU memory)")

    # Footer
    st.markdown("---")
    st.caption("VRAM calculations based on ApXML methodology | Platform recommendations for HP Z Workstations")
    st.caption(f"Batch size: {batch_size} | Sequence length: {sequence_length} tokens")