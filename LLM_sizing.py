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
        background: white !important;
        border-radius: 24px;
        padding: 3rem 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        max-width: 800px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
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
        color: #4c51bf !important;
        font-size: 1.8rem;
        margin-top: 2rem;
        font-weight: 700;
        border-bottom: 3px solid #4c51bf;
        padding-bottom: 0.5rem;
    }

    h3 {
        color: #000000 !important;
        font-size: 1.3rem;
        font-weight: 700 !important;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
    }

    .stRadio > label, .stSelectbox > label, .stSlider > label, .stNumberInput > label, label {
        font-weight: 600 !important;
        color: #000000 !important;
        font-size: 1.1rem !important;
    }

    .stMarkdown p strong, .stMarkdown strong {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    .stRadio div[role="radiogroup"] label {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    .stRadio > div {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #a0aec0;
        transition: all 0.3s ease;
    }

    .stRadio > div:hover {
        border-color: #4c51bf;
        box-shadow: 0 4px 12px rgba(76, 81, 191, 0.3);
    }

    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    [data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stSelectbox > div > div {
        background: white;
        border: 2px solid #a0aec0;
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    .stSelectbox > div > div:hover {
        border-color: #4c51bf;
        box-shadow: 0 4px 12px rgba(76, 81, 191, 0.3);
    }

    .stExpander {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%);
        border-radius: 16px;
        border: 2px solid #a0aec0;
        margin: 1rem 0;
    }

    .streamlit-expanderHeader {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4c51bf, transparent);
    }

    .success-box {
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }

    [data-testid="stExpander"] {
        background: white;
        border: 2px solid #a0aec0;
        border-radius: 16px;
    }

    .caption-text, [data-testid="stCaptionContainer"], .stCaption {
        color: #2d3748 !important;
        font-size: 0.9rem;
        font-style: italic;
        font-weight: 500 !important;
    }

    .platform-info-box {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #4c51bf;
    }

    .platform-info-box p {
        color: #000000 !important;
        font-weight: 500;
    }

    .multi-node-warning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
    }

    /* Additional high contrast fixes */
    p, span, div {
        color: #1a202c !important;
    }

    /* Subtitle text */
    .main p {
        color: #2d3748 !important;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Hide Streamlit menu and footer - CLOUD DEPLOYMENT VERSION
hide_streamlit_style = """
    <style>
    /* Hide all Streamlit branding */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}

    /* Hide toolbar and manage button - Cloud specific */
    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }

    .stActionButton {
        display: none !important;
        visibility: hidden !important;
    }

    [data-testid="manage-app-button"] {
        display: none !important;
    }

    [data-testid="collapsedControl"] {
        display: none !important;
    }

    .stDeployButton {
        display: none !important;
    }

    button[kind="header"] {
        display: none !important;
    }

    /* Cloud-specific floating action button */
    .stApp > header {
        display: none !important;
    }

    .stApp [data-testid="stHeader"] {
        display: none !important;
    }

    /* Hide the three-dot menu and settings */
    button[data-testid="baseButton-header"] {
        display: none !important;
    }

    /* Nuclear option - hide everything in top right */
    .css-1dp5vir, .eyeqlp51, .css-vurnku, .st-emotion-cache-1wrcr25 {
        display: none !important;
    }

    /* Hide floating action buttons (Manage app, etc) */
    div[class*="floating"] {
        display: none !important;
    }

    /* Alternative class names Streamlit Cloud uses */
    .styles_viewerBadge__1yB5_, 
    .viewerBadge_container__1QSob,
    .styles_viewerBadge__CvC9N,
    button[title="View app menu"] {
        display: none !important;
    }
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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


# ===== WORKSTATION DATA =====
workstations = [
    {
        "name": "Z2 Mini G1i",
        "max_gpus": 1,
        "max_vram_per_gpu": 20,
        "max_vram_total": 20,
        "max_vram_config": "1x RTX 4000 SFF Ada 20GB",
        "supported_gpus": [
            "1x A400 4GB",
            "1x A1000 8GB",
            "1x RTX 2000 Ada 16GB",
            "1x RTX 4000 SFF Ada 20GB"
        ]
    },
    {
        "name": "Z2 Tower G1i",
        "max_gpus": 2,
        "max_vram_per_gpu": 96,
        "max_vram_total": 96,
        "max_vram_config": "1x NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB",
        "supported_gpus": [
            "2x A400 4GB",
            "2x A1000 8GB",
            "2x RTX 2000 Ada 16GB",
            "2x RTX 4000 Ada 20GB",
            "1x RTX 4500 Ada 24GB",
            "1x RTX 5000 Ada 32GB",
            "1x RTX 5880 Ada 48GB",
            "1X NVIDIA RTX™ 6000 Ada 48GB",
            "1X NVIDIA RTX PRO 4000 Blackwell 24GB",
            "1X NVIDIA RTX PRO 4500 Blackwell 32GB",
            "1x RTX PRO 5000 Blackwell 48GB",
            "1x NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB"
        ]
    },
    {
        "name": "Z4 G5",
        "max_gpus": 2,
        "max_vram_per_gpu": 96,
        "max_vram_total": 192,
        "max_vram_config": "2x NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB",
        "supported_gpus": [
            "2x RTX A400 4GB",
            "2x RTX A1000 8GB",
            "2x RTX A4000 16GB",
            "2x RTX A4500 20GB",
            "2x RTX 2000 Ada 16GB",
            "2x RTX 4000 Ada 20GB",
            "2x RTX 4500 Ada 24GB",
            "2x RTX 5000 Ada 32GB",
            "2x RTX 5880 Ada 48GB",
            "2x RTX 6000 Ada 48GB",
            "2x RTX PRO 4000 Blackwell 24GB",
            "2x RTX PRO 4500 Blackwell 32GB",
            "2x RTX PRO 5000 Blackwell 48GB",
            "2x NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB"
        ]
    },
    {
        "name": "Z6 G5",
        "max_gpus": 3,
        "max_vram_per_gpu": 96,
        "max_vram_total": 288,
        "max_vram_config": "3x NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB",
        "supported_gpus": [
            "3x RTX A400 4GB",
            "2X NVIDIA® A800 40GB",
            "3x RTX A1000 8GB",
            "3x RTX A4000 16GB",
            "3x RTX A4500 20GB",
            "3x RTX 2000 Ada 16GB",
            "3x RTX 4000 Ada 20GB",
            "3x RTX 4500 Ada 24GB",
            "3x RTX 5000 Ada 32GB",
            "3x RTX 5880 Ada 48GB",
            "3x RTX 6000 Ada 48GB",
            "3x RTX PRO 4000 Blackwell 24GB",
            "3x RTX PRO 4500 Blackwell 32GB",
            "3x RTX PRO 5000 Blackwell 48GB",
            "3x NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB"
        ]
    },
    {
        "name": "Z8 G5",
        "max_gpus": 2,
        "max_vram_per_gpu": 96,
        "max_vram_total": 192,
        "max_vram_config": "2x NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB",
        "supported_gpus": [
            "2x RTX A400 4GB",
            "2x RTX A1000 8GB",
            "2x RTX A4000 16GB",
            "2x RTX A4500 20GB",
            "2x RTX 2000 Ada 16GB",
            "2x RTX 4000 Ada 20GB",
            "2x RTX 4500 Ada 24GB",
            "2x RTX 5000 Ada 32GB",
            "2x RTX 5880 Ada 48GB",
            "2x RTX 6000 Ada 48GB",
            "2x RTX PRO 4000 Blackwell 24GB",
            "2x RTX PRO 4500 Blackwell 32GB",
            "2x RTX PRO 5000 Blackwell 48GB",
            "2x NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB"
        ]
    },
    {
        "name": "Z8 Fury G5",
        "max_gpus": 4,
        "max_vram_per_gpu": 96,
        "max_vram_total": 384,
        "max_vram_config": "4x NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB",
        "supported_gpus": [
            "3X NVIDIA® A800 40GB",
            "4x RTX A400 4GB",
            "4x RTX A1000 8GB",
            "4x RTX A4000 16GB",
            "4x RTX A4500 20GB",
            "4x RTX 2000 Ada 16GB",
            "4x RTX 4000 Ada 20GB",
            "4x RTX 4500 Ada 24GB",
            "4x RTX 5000 Ada 32GB",
            "4x RTX 5880 Ada 48GB",
            "4x RTX 6000 Ada 48GB",
            "4x RTX PRO 4000 Blackwell 24GB",
            "4x RTX PRO 4500 Blackwell 32GB",
            "4x RTX PRO 5000 Blackwell 48GB",
            "4x NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB"
        ]
    }
]


def recommend_workstation_smart(vram_required):
    """
    Smart workstation recommendation with TP=2/4 preference
    Avoids TP=3, suggests multi-node for very large models
    """

    # Single GPU workloads (TP=1)
    if vram_required <= 20:
        return {
            "platform": "Z2 Mini G1i",
            "config": "1x GPU (up to 20GB)",
            "tp": 1,
            "nodes": 1,
            "total_gpus": 1,
            "reasoning": "Fits in single small GPU",
            "workstation": workstations[0]
        }

    if vram_required <= 96:
        return {
            "platform": "Z2 Tower G1i",
            "config": "1x RTX PRO 6000 Blackwell Max-Q (96GB)",
            "tp": 1,
            "nodes": 1,
            "total_gpus": 1,
            "reasoning": "Fits in single high-end GPU",
            "workstation": workstations[1]
        }

    # TP=2 workloads (2 GPUs)
    if vram_required <= 192:
        return {
            "platform": "Z4 G5 or Z8 G5",
            "config": "2x RTX PRO 6000 Blackwell Max-Q (96GB each) = 192GB total",
            "tp": 2,
            "nodes": 1,
            "total_gpus": 2,
            "reasoning": "Model distributed across 2 GPUs using Tensor Parallelism (TP=2)",
            "workstation": workstations[2]  # Z4 G5
        }

    # TP=4 workloads (4 GPUs) - SKIP Z6 G5 (TP=3)
    if vram_required <= 384:
        return {
            "platform": "Z8 Fury G5",
            "config": "4x RTX PRO 6000 Blackwell Max-Q (96GB each) = 384GB total",
            "tp": 4,
            "nodes": 1,
            "total_gpus": 4,
            "reasoning": "Model distributed across 4 GPUs using Tensor Parallelism (TP=4). Z6 G5 skipped (TP=3 is inefficient)",
            "workstation": workstations[5]  # Z8 Fury G5
        }

    # Multi-node: TP=8 (2 workstations × 4 GPUs)
    if vram_required <= 768:
        return {
            "platform": "2x Z8 Fury G5",
            "config": "2 platforms × 4 GPUs × 96GB = 768GB total",
            "tp": 8,
            "nodes": 2,
            "total_gpus": 8,
            "reasoning": "Multi-platform setup with model distributed across 8 GPUs (TP=8)",
            "workstation": workstations[5],  # Z8 Fury G5
            "multi_node": True
        }

    # Multi-node: TP=16 (4 workstations × 4 GPUs)
    if vram_required <= 1536:
        return {
            "platform": "4x Z8 Fury G5",
            "config": "4 platforms × 4 GPUs × 96GB = 1,536GB total",
            "tp": 16,
            "nodes": 4,
            "total_gpus": 16,
            "reasoning": "Large multi-platform cluster with model distributed across 16 GPUs (TP=16)",
            "workstation": workstations[5],  # Z8 Fury G5
            "multi_node": True
        }

    # Beyond workstation scale
    return {
        "platform": "GPU Cluster / Cloud Infrastructure",
        "config": f"Requires {math.ceil(vram_required / 96)} GPUs minimum (96GB each)",
        "tp": "Custom",
        "nodes": "> 4",
        "total_gpus": math.ceil(vram_required / 96),
        "reasoning": "Model exceeds workstation capacity, requires GPU cluster",
        "workstation": None,
        "multi_node": True
    }


def gpu_count_options(ws):
    """Return the specific GPU count options for each platform"""
    name = ws["name"]
    if name == "Z2 Mini G1i":
        return "1 GPU"
    if name == "Z2 Tower G1i":
        return "1, 2 GPUs"
    if name == "Z4 G5":
        return "2 GPUs"
    if name == "Z6 G5":
        return "2, 3 GPUs"
    if name == "Z8 G5":
        return "1, 2 GPUs"
    if name == "Z8 Fury G5":
        return "3, 4 GPUs"
    return f"{ws['max_gpus']} GPUs"


def generate_report(model_name, params, precision, workload, batch_size, sequence_length, vram,
                   min_rec, min_vram, min_headroom_gb, min_headroom_pct,
                   fp_rec=None, fp_vram=None, fp_headroom_gb=None, fp_headroom_pct=None,
                   target_headroom=30):
    """Generate a sales-friendly downloadable text report"""

    report = f"""
╔═══════════════════════════════════════════════════════════════════╗
║          HP Z WORKSTATION GPU SIZING RECOMMENDATION               ║
╚═══════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

═══════════════════════════════════════════════════════════════════
CUSTOMER REQUIREMENTS
═══════════════════════════════════════════════════════════════════

AI Model:           {model_name}
Model Size:         {params} Billion Parameters
Precision:          {precision}
Use Case:           {workload}
Batch Size:         {batch_size}
Context Length:     {sequence_length} tokens

═══════════════════════════════════════════════════════════════════
VRAM CALCULATION (ApXML Methodology)
═══════════════════════════════════════════════════════════════════

How we calculated VRAM requirements:

1. Model Weights:       {vram['model_weights']:.1f} GB
   └─ Storage for model parameters at {precision} precision

2. KV Cache:            {vram['kv_cache']:.1f} GB
   └─ Memory for attention mechanism (scales with context length)

3. Activation Memory:   {vram['activations']:.1f} GB
   └─ Temporary computation memory (varies by workload type)

4. System Overhead:     {(vram['total'] - (vram['model_weights'] + vram['kv_cache'] + vram['activations'])):.1f} GB
   └─ 20% buffer for framework and system operations

───────────────────────────────────────────────────────
TOTAL VRAM REQUIRED:    {vram['total']:.1f} GB
───────────────────────────────────────────────────────

═══════════════════════════════════════════════════════════════════
PLATFORM RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════

"""

    # Good Fit (Minimum)
    report += f"""
┌─────────────────────────────────────────────────────────────────┐
│ ✅ GOOD FIT - {min_rec['platform']}
└─────────────────────────────────────────────────────────────────┘

Configuration:      {min_rec['config']}
Total VRAM:         {min_vram:.0f} GB
Available Headroom: +{min_headroom_gb:.1f} GB ({min_headroom_pct:.0f}%)

"""

    if min_rec.get('workstation'):
        ws = min_rec['workstation']
        report += f"""GPU Configuration Options:
"""
        for i, gpu in enumerate(ws['supported_gpus'][:6], 1):  # Show first 6 options
            report += f"  {i}. {gpu}\n"
        if len(ws['supported_gpus']) > 6:
            report += f"  ... and {len(ws['supported_gpus']) - 6} more options\n"

    if min_headroom_pct < target_headroom:
        report += f"""
⚠️  NOTE: This configuration provides {min_headroom_pct:.0f}% headroom.
    {workload} workloads typically need {target_headroom}% for optimal performance.
    Consider the "Better Choice" option below for future growth.
"""

    # Better Choice (Future-proof) - if provided
    if fp_rec and fp_vram:
        report += f"""

┌─────────────────────────────────────────────────────────────────┐
│ 🚀 BETTER CHOICE - {fp_rec['platform']}
└─────────────────────────────────────────────────────────────────┘

Configuration:      {fp_rec['config']}
Total VRAM:         {fp_vram:.0f} GB
Available Headroom: +{fp_headroom_gb:.1f} GB ({fp_headroom_pct:.0f}%)

✓ Meets {target_headroom}% headroom target for {workload} workloads
✓ Future-proof for 18-30 months
✓ Handles growing workloads and model upgrades

"""
        if fp_rec.get('workstation'):
            fp_ws = fp_rec['workstation']
            report += f"""GPU Configuration Options:
"""
            for i, gpu in enumerate(fp_ws['supported_gpus'][:6], 1):
                report += f"  {i}. {gpu}\n"
            if len(fp_ws['supported_gpus']) > 6:
                report += f"  ... and {len(fp_ws['supported_gpus']) - 6} more options\n"

    report += f"""
═══════════════════════════════════════════════════════════════════
WHY HEADROOM MATTERS FOR {workload.upper()}
═══════════════════════════════════════════════════════════════════
"""

    if workload == "Inference":
        report += """
Recommended: 30% headroom provides:
  • Handle 2-4× batch size increases as usage grows
  • Accommodate longer context windows (2K → 8K+ tokens)
  • Upgrade to next-generation models without hardware changes
  • Serve multiple models simultaneously
"""
    elif workload == "Fine-Tuning":
        report += """
Recommended: 30% headroom provides:
  • Upgrade from LoRA to full fine-tuning methods
  • Experiment with larger base models as they're released
  • Run multiple fine-tuning experiments in parallel
  • Handle larger training datasets
"""
    else:  # Training
        report += """
Recommended: 50% headroom provides:
  • Train models 2-3× larger without infrastructure upgrade
  • Experiment with distributed training techniques (TP, PP, DP)
  • Build multi-stage pipelines (pre-training + fine-tuning)
  • Accommodate gradient accumulation and larger batch sizes
"""

    report += f"""
═══════════════════════════════════════════════════════════════════
SYSTEM REQUIREMENTS
═══════════════════════════════════════════════════════════════════

System RAM:         {max(16, math.ceil(vram['total'] * 2))} GB minimum
                    (Recommended: 2× VRAM for optimal performance)

Storage:            High-speed NVMe SSD recommended
                    - Model storage: ~{vram['model_weights'] * 2:.0f} GB
                    - Dataset storage: Project-dependent

Network:            10GbE or faster for multi-node setups
                    1GbE sufficient for single workstation

═══════════════════════════════════════════════════════════════════
TECHNICAL NOTES
═══════════════════════════════════════════════════════════════════

Calculation Method:
  • Based on ApXML VRAM estimation methodology
  • Accounts for model weights, KV cache, activations, and overhead
  • Validated against real-world LLM deployments

Tensor Parallelism (TP):
  • TP=1: Single GPU (≤96GB models)
  • TP=2: 2 GPUs working together (96-192GB models)
  • TP=4: 4 GPUs working together (192-384GB models)
  • TP=8+: Multi-node clusters for larger models

Important Considerations:
  • Actual requirements may vary by framework (PyTorch, TensorFlow, etc.)
  • Quantization (INT8, INT4) can significantly reduce VRAM needs
  • Flash Attention and other optimizations may lower requirements
  • Multi-node setups require high-speed interconnect (NVLink, InfiniBand)

═══════════════════════════════════════════════════════════════════
NEXT STEPS
═══════════════════════════════════════════════════════════════════

1. Review the recommended configurations above
2. Consider your growth plans over the next 18-30 months
3. Contact your HP representative to:
   - Discuss specific GPU configurations
   - Review pricing and availability
   - Schedule a technical consultation if needed
   - Explore financing options

For Questions or Orders:
  Contact your HP Sales Representative
  or visit: www.hp.com/workstations

═══════════════════════════════════════════════════════════════════
Report Settings: Batch={batch_size}, Context={sequence_length} tokens
═══════════════════════════════════════════════════════════════════
"""
    return report


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

    # ===== WORKSTATION RECOMMENDATION WITH HEADROOM LOGIC =====

    # Get minimum platform
    minimum_rec = recommend_workstation_smart(vram['total'])

    # Define headroom targets
    headroom_targets = {
        "Inference": 30,
        "Fine-Tuning": 30,
        "Training": 50
    }
    target_headroom_pct = headroom_targets.get(workload, 30)

    # Calculate actual headroom for minimum platform
    if minimum_rec.get("workstation"):
        min_platform_vram = minimum_rec["workstation"]["max_vram_total"]
    else:
        # For multi-node setups, extract total VRAM
        min_platform_vram = minimum_rec.get("max_vram_total", vram['total'])

    min_headroom_gb = min_platform_vram - vram['total']
    min_headroom_pct = (min_headroom_gb / vram['total']) * 100 if vram['total'] > 0 else 0

    # Determine if we need to show future-proof option
    meets_target = min_headroom_pct >= target_headroom_pct

    # Find next tier up for future-proof (if needed)
    if not meets_target:
        # Calculate target VRAM with headroom
        future_proof_vram_target = vram['total'] * (1 + target_headroom_pct / 100)

        # Get recommendation for future-proof target - this returns the right platform automatically
        future_proof_rec = recommend_workstation_smart(future_proof_vram_target)

        # If same platform as minimum, force bump to next tier by adding 1 GB to push it over
        if future_proof_rec["platform"] == minimum_rec["platform"]:
            # Bump VRAM requirement slightly to force next tier
            if min_platform_vram <= 20:
                future_proof_rec = recommend_workstation_smart(21)
            elif min_platform_vram <= 96:
                future_proof_rec = recommend_workstation_smart(97)
            elif min_platform_vram <= 192:
                future_proof_rec = recommend_workstation_smart(193)
            elif min_platform_vram <= 384:
                future_proof_rec = recommend_workstation_smart(385)
            elif min_platform_vram <= 768:
                future_proof_rec = recommend_workstation_smart(769)
            else:
                future_proof_rec = recommend_workstation_smart(1537)

        # Calculate future-proof platform VRAM
        # Check multi-node FIRST before checking workstation
        if future_proof_rec.get("multi_node") and future_proof_rec.get("total_gpus"):
            # Multi-node setup - calculate from total GPUs
            fp_platform_vram = future_proof_rec["total_gpus"] * 96
        elif future_proof_rec.get("workstation"):
            # Single workstation
            fp_platform_vram = future_proof_rec["workstation"]["max_vram_total"]
        elif future_proof_rec.get("total_gpus"):
            # Cluster without workstation - calculate from total GPUs
            fp_platform_vram = future_proof_rec["total_gpus"] * 96
        else:
            # Fallback
            fp_platform_vram = min_platform_vram * 2

        fp_headroom_gb = fp_platform_vram - vram['total']
        fp_headroom_pct = (fp_headroom_gb / vram['total']) * 100 if vram['total'] > 0 else 0

    # ===== DISPLAY LOGIC =====
    st.markdown("## 🖥️ Platform Recommendations")

    # Check if this is beyond single workstation (>384 GB)
    if vram['total'] > 384:
        st.markdown(f"""
        <div class='multi-node-warning'>
            ⚠️ <strong>Enterprise GPU Cluster Required</strong><br>
            Your model requires approximately <strong>{math.ceil(vram['total'] / 96)} GPUs</strong> (96GB each).<br>
            Estimated Total VRAM: <strong>{vram['total']:.1f} GB</strong><br>
        </div>
        """, unsafe_allow_html=True)

        st.info(
            "**For models of this size:** We recommend consulting with HP's AI infrastructure team to design an optimal multi-node setup tailored to your specific requirements.")

        # Show example configurations
        with st.expander("💡 Example Multi-Node Configurations"):
            num_gpus = math.ceil(vram['total'] / 96)

            if num_gpus <= 8:
                st.markdown("""
                **Possible Setup: 2x Z8 Fury G5**
                - 2 nodes × 4 GPUs = 8 GPUs total
                - Total VRAM: 768 GB
                - Tensor Parallelism: TP=8
                """)
            elif num_gpus <= 16:
                st.markdown("""
                **Possible Setup: 4x Z8 Fury G5**
                - 4 nodes × 4 GPUs = 16 GPUs total
                - Total VRAM: 1,536 GB
                - Tensor Parallelism: TP=16
                """)
            else:
                st.markdown(f"""
                **Custom GPU Cluster**
                - Estimated GPUs needed: {num_gpus}+
                - Recommended: Cloud or on-premises cluster
                - Contact HP for enterprise AI infrastructure solutions
                """)

    else:
        # ===== NORMAL WORKSTATION LOGIC (≤384 GB) =====

        # Multi-node warning for 192-384 GB range
        if minimum_rec.get("multi_node", False):
            st.markdown(f"""
            <div class='multi-node-warning'>
                ⚠️ <strong>Multi-Node Setup May Be Required</strong><br>
                Estimated: {minimum_rec.get('nodes', 'Multiple')} nodes with {minimum_rec.get('total_gpus', 'multiple')} GPUs total.<br>
                <em>Consultation recommended for optimal setup.</em>
            </div>
            """, unsafe_allow_html=True)

        # ===== SCENARIO 1 & 2: MEETS TARGET (Show only minimum) =====
        if meets_target:
            st.markdown("### ✅ Good Fit")
            st.success(f"**{minimum_rec['platform']}**")

            if minimum_rec.get("workstation"):
                ws = minimum_rec["workstation"]

                # Extract config without total
                config_clean = minimum_rec['config'].replace(f" = {min_platform_vram}GB total", "").replace(
                    "(up to 20GB)", "(20GB)")

                st.markdown(f"""
                <div class='platform-info-box'>
                    <p><strong>Configuration:</strong> {config_clean}</p>
                    <p><strong>Total VRAM:</strong> {min_platform_vram} GB</p>
                    <p><strong>Headroom:</strong> +{min_headroom_gb:.1f} GB ({min_headroom_pct:.0f}%)</p>
                </div>
                """, unsafe_allow_html=True)

                # Simple status message
                if min_headroom_pct >= target_headroom_pct * 2:
                    st.info(
                        f"💡 This platform provides well above the {target_headroom_pct}% target for {workload} workloads.")
                else:
                    st.info(
                        f"💡 This platform meets the {target_headroom_pct}% headroom target for {workload} workloads.")

                # Supported GPUs
                with st.expander("🎮 View Supported GPUs"):
                    for gpu in ws['supported_gpus']:
                        st.markdown(f"• {gpu}")

                # Workload-specific benefits
                with st.expander(f"💡 Why {target_headroom_pct}% Headroom for {workload}?"):
                    if workload == "Inference":
                        st.markdown("""
                        **Headroom benefits:**
                        - Handle 2-4× batch size increase for growing traffic
                        - Accommodate context window expansion (2K → 8K+ tokens)
                        - Support next-generation model upgrades
                        """)
                    elif workload == "Fine-Tuning":
                        st.markdown("""
                        **Headroom benefits:**
                        - Upgrade from LoRA to full fine-tuning
                        - Experiment with larger base models
                        - Run multiple experiments simultaneously
                        """)
                    else:  # Training
                        st.markdown("""
                        **Headroom benefits:**
                        - Train models 2-3× larger without upgrade
                        - Experiment with distributed training
                        - Build multi-stage training pipelines
                        """)

                # Optional upgrade
                with st.expander("💡 Want Even More Headroom?"):
                    current_idx = next((i for i, w in enumerate(workstations) if w["name"] == ws["name"]), -1)
                    if current_idx < len(workstations) - 1:
                        next_ws = workstations[current_idx + 1]
                        next_headroom_gb = next_ws["max_vram_total"] - vram['total']
                        next_headroom_pct = (next_headroom_gb / vram['total']) * 100

                        st.markdown(f"""
                        **{next_ws['name']}**
                        - Configuration: {next_ws['max_vram_config']}
                        - Total VRAM: {next_ws['max_vram_total']} GB
                        - Headroom: +{next_headroom_gb:.1f} GB ({next_headroom_pct:.0f}%)
                        """)

                        if workload == "Inference":
                            st.caption("Use case: Multi-model serving, high-concurrency deployments")
                        elif workload == "Fine-Tuning":
                            st.caption("Use case: Parallel experiments, larger datasets")
                        else:
                            st.caption("Use case: Distributed training, multi-stage pipelines")
                    else:
                        st.markdown("**Contact HP for multi-node configurations**")

        # ===== SCENARIO 3: DOES NOT MEET TARGET =====
        else:
            # MINIMUM CONFIGURATION
            st.markdown("### ✅ Good Fit")
            st.warning(f"**{minimum_rec['platform']}**")

            if minimum_rec.get("workstation"):
                ws = minimum_rec["workstation"]

                # Extract config without total
                config_clean = minimum_rec['config'].replace(f" = {min_platform_vram}GB total", "").replace(
                    "(up to 20GB)", "(20GB)")

                st.markdown(f"""
                <div class='platform-info-box'>
                    <p><strong>Configuration:</strong> {config_clean}</p>
                    <p><strong>Total VRAM:</strong> {min_platform_vram} GB</p>
                    <p><strong>Headroom:</strong> +{min_headroom_gb:.1f} GB ({min_headroom_pct:.0f}%)</p>
                </div>
                """, unsafe_allow_html=True)

                st.warning(
                    f"⚠️ This platform provides only {min_headroom_pct:.0f}% headroom. {workload} needs {target_headroom_pct}% for optimal performance.")

                with st.expander("🎮 View Supported GPUs"):
                    for gpu in ws['supported_gpus']:
                        st.markdown(f"• {gpu}")

            st.markdown("")

            # FUTURE-PROOF CONFIGURATION
            st.markdown(f"### 🚀 Better Choice")
            st.success(f"**{future_proof_rec['platform']}**")

            # Check multi-node FIRST before checking workstation
            if future_proof_rec.get("multi_node"):
                # Multi-node setup - use the explicit max_vram_total
                fp_config_clean = future_proof_rec['config'].replace(f" = {fp_platform_vram}GB total", "")

                st.markdown(f"""
                <div class='platform-info-box' style='border-color: #4ba3c7; background: linear-gradient(135deg, #e6f7ff 0%, #d1f0ff 100%);'>
                    <p><strong>Configuration:</strong> {fp_config_clean}</p>
                    <p><strong>Total VRAM:</strong> {fp_platform_vram} GB</p>
                    <p><strong>Headroom:</strong> +{fp_headroom_gb:.1f} GB ({fp_headroom_pct:.0f}%)</p>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"💡 This multi-node platform meets the {target_headroom_pct}% headroom target for {workload} workloads.")

                # Workload-specific benefits
                with st.expander(f"💡 Why {target_headroom_pct}% Headroom for {workload}?"):
                    if workload == "Inference":
                        st.markdown("""
                        **Headroom benefits:**
                        - Handle 2-4× batch size increase for growing traffic
                        - Accommodate context window expansion (2K → 8K+ tokens)
                        - Support next-generation model upgrades
                        """)
                    elif workload == "Fine-Tuning":
                        st.markdown("""
                        **Headroom benefits:**
                        - Upgrade from LoRA to full fine-tuning
                        - Experiment with larger base models
                        - Run multiple experiments simultaneously
                        """)
                    else:  # Training
                        st.markdown("""
                        **Headroom benefits:**
                        - Train models 2-3× larger without upgrade
                        - Experiment with distributed training
                        - Build multi-stage training pipelines
                        """)

                # Show base workstation GPUs if available
                if future_proof_rec.get("workstation"):
                    fp_ws = future_proof_rec["workstation"]
                    with st.expander("🎮 View Supported GPUs (per node)"):
                        for gpu in fp_ws['supported_gpus']:
                            st.markdown(f"• {gpu}")

            elif future_proof_rec.get("workstation"):
                # Single workstation with workstation data
                fp_ws = future_proof_rec["workstation"]

                # Extract config without total
                fp_config_clean = future_proof_rec['config'].replace(f" = {fp_platform_vram}GB total", "").replace(
                    "(up to 20GB)", "(20GB)")

                st.markdown(f"""
                <div class='platform-info-box' style='border-color: #4ba3c7; background: linear-gradient(135deg, #e6f7ff 0%, #d1f0ff 100%);'>
                    <p><strong>Configuration:</strong> {fp_config_clean}</p>
                    <p><strong>Total VRAM:</strong> {fp_platform_vram} GB</p>
                    <p><strong>Headroom:</strong> +{fp_headroom_gb:.1f} GB ({fp_headroom_pct:.0f}%)</p>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"💡 This platform meets the {target_headroom_pct}% headroom target for {workload} workloads.")

                # Workload-specific benefits
                with st.expander(f"💡 Why {target_headroom_pct}% Headroom for {workload}?"):
                    if workload == "Inference":
                        st.markdown("""
                        **Headroom benefits:**
                        - Handle 2-4× batch size increase for growing traffic
                        - Accommodate context window expansion (2K → 8K+ tokens)
                        - Support next-generation model upgrades
                        """)
                    elif workload == "Fine-Tuning":
                        st.markdown("""
                        **Headroom benefits:**
                        - Upgrade from LoRA to full fine-tuning
                        - Experiment with larger base models
                        - Run multiple experiments simultaneously
                        """)
                    else:  # Training
                        st.markdown("""
                        **Headroom benefits:**
                        - Train models 2-3× larger without upgrade
                        - Experiment with distributed training
                        - Build multi-stage training pipelines
                        """)

                with st.expander("🎮 View Supported GPUs"):
                    for gpu in fp_ws['supported_gpus']:
                        st.markdown(f"• {gpu}")

            else:
                # Multi-node setup - use calculated fp_platform_vram
                # Extract config without total for cleaner display
                fp_config_clean = future_proof_rec['config'].replace(" = 768GB total", "").replace(" = 1536GB total",
                                                                                                   "").replace(
                    " = 1,536GB total", "")

                st.markdown(f"""
                <div class='platform-info-box' style='border-color: #4ba3c7; background: linear-gradient(135deg, #e6f7ff 0%, #d1f0ff 100%);'>
                    <p><strong>Configuration:</strong> {fp_config_clean}</p>
                    <p><strong>Total VRAM:</strong> {fp_platform_vram} GB</p>
                    <p><strong>Headroom:</strong> +{fp_headroom_gb:.1f} GB ({fp_headroom_pct:.0f}%)</p>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"💡 This platform meets the {target_headroom_pct}% headroom target for {workload} workloads.")

    # System Requirements
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

    # Download Report
    st.markdown("## 📥 Export Report")

    # Prepare report data - include future-proof if doesn't meet target
    if meets_target:
        # Only minimum recommendation
        report_content = generate_report(
            model_name, params, precision, workload, batch_size, sequence_length, vram,
            minimum_rec, min_platform_vram, min_headroom_gb, min_headroom_pct,
            target_headroom=target_headroom_pct
        )
    else:
        # Both minimum and future-proof recommendations
        report_content = generate_report(
            model_name, params, precision, workload, batch_size, sequence_length, vram,
            minimum_rec, min_platform_vram, min_headroom_gb, min_headroom_pct,
            future_proof_rec, fp_platform_vram, fp_headroom_gb, fp_headroom_pct,
            target_headroom=target_headroom_pct
        )

    st.download_button(
        label="📄 Download Sizing Report (TXT)",
        data=report_content,
        file_name=f"gpu_sizing_report_{model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        help="Download a detailed text report of this GPU sizing recommendation"
    )

    # Footer
    st.markdown("---")
    st.caption("Calculations based on ApXML VRAM methodology | Platform recommendations for HP Z Workstations")
    st.caption(f"Current settings: Batch size = {batch_size}, Sequence length = {sequence_length} tokens")