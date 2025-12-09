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

    .platform-info-box {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #667eea;
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
    "Llama 3.1 405B": 405.0,
    "Custom": None  # Placeholder for custom input
}

model_name = st.selectbox("Select model", list(model_options.keys()), label_visibility="collapsed")

# If Custom is selected, show input field
if model_name == "Custom":
    st.markdown("**Enter Custom Model Parameters (in billions)**")
    params = st.number_input(
        "Model size in billions of parameters (e.g., 13 for 13B model)",
        min_value=0.1,
        max_value=1000.0,
        value=7.0,
        step=0.5,
        label_visibility="collapsed",
        help="Enter the number of parameters in billions (B). For example: 7 for 7B, 13 for 13B, 70 for 70B"
    )
else:
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


def generate_report(model_name, params, precision, workload, batch_size, sequence_length, vram, recommendation):
    """Generate a downloadable text report"""
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║            GPU SIZING RECOMMENDATION REPORT                   ║
╚══════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

─────────────────────────────────────────────────────────────
MODEL CONFIGURATION
─────────────────────────────────────────────────────────────
Model:              {model_name}
Parameters:         {params}B
Precision:          {precision}
Workload Type:      {workload}
Batch Size:         {batch_size}
Sequence Length:    {sequence_length} tokens

─────────────────────────────────────────────────────────────
VRAM REQUIREMENTS
─────────────────────────────────────────────────────────────
Model Weights:      {vram['model_weights']:.2f} GB
KV Cache:           {vram['kv_cache']:.2f} GB
Activations:        {vram['activations']:.2f} GB
Overhead (20%):     {(vram['total'] - (vram['model_weights'] + vram['kv_cache'] + vram['activations'])):.2f} GB
───────────────────────────────────
TOTAL VRAM:         {vram['total']:.2f} GB

─────────────────────────────────────────────────────────────
RECOMMENDED WORKSTATION
─────────────────────────────────────────────────────────────
Platform:           {recommendation['platform']}
Configuration:      {recommendation['config']}
"""

    if recommendation.get('workstation'):
        ws = recommendation['workstation']
        report += f"""
Platform Specs:
  • GPUs Supported:  {gpu_count_options(ws)}
  • Max VRAM:        {ws['max_vram_total']} GB ({ws['max_vram_config']})

Supported GPU Options:
"""
        for gpu in ws['supported_gpus']:
            report += f"  • {gpu}\n"

    report += f"""
─────────────────────────────────────────────────────────────
RECOMMENDATION REASONING
─────────────────────────────────────────────────────────────
{recommendation['reasoning']}

─────────────────────────────────────────────────────────────
SYSTEM REQUIREMENTS
─────────────────────────────────────────────────────────────
Recommended RAM:    {max(8, math.ceil(vram['total'] * 2))} GB minimum
                    (2× VRAM for optimal performance)

─────────────────────────────────────────────────────────────
NOTES
─────────────────────────────────────────────────────────────
- This sizing is based on ApXML VRAM calculation methodology
- Actual requirements may vary based on framework and optimization
- For multi-platform setups, consultation is recommended
- Contact your HP representative for detailed configuration

╔══════════════════════════════════════════════════════════════╗
║  HP Z Workstations - GPU Sizing Tool                         ║
╚══════════════════════════════════════════════════════════════╝
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

    # ===== WORKSTATION RECOMMENDATION =====
    recommendation = recommend_workstation_smart(vram['total'])

    st.markdown("## 🖥️ Recommended Workstation")

    # Multi-node warning if applicable
    if recommendation.get("multi_node", False):
        # Get platform name from recommendation
        platform_name = recommendation['platform'].split('x')[1].strip() if 'x' in recommendation['platform'] else \
        recommendation['platform']

        st.markdown(f"""
        <div class='multi-node-warning'>
            ⚠️ <strong>Multi-Platform Setup Likely Required</strong><br>
            Likely need {platform_name} cluster.<br>
            Estimated configuration: {recommendation['nodes']} {platform_name} with approximately {recommendation['total_gpus']} NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB GPUs.<br>
            <em>Further consultation recommended to optimize configuration.</em>
        </div>
        """, unsafe_allow_html=True)

    st.success(f"**{recommendation['platform']}**")

    # Show workstation specs if available
    if recommendation.get("workstation"):
        ws = recommendation["workstation"]

        # Determine platform name for display
        if recommendation.get("multi_node", False):
            platform_display = f"{ws['name']} "
        else:
            platform_display = ws['name']


        st.markdown(f"""
        <div class='platform-info-box'>
            <p><strong>Platform Specifications:</strong></p>
            <p>• <strong>No. of GPUs Supported:</strong> {gpu_count_options(ws)}</p>
            <p>• <strong>Max Total VRAM per {platform_display}:</strong> {ws['max_vram_total']} GB ({ws['max_vram_config']})</p>
        </div>
        """, unsafe_allow_html=True)

    # Supported GPUs
    with st.expander("🎮 View Supported GPUs"):
        st.markdown("**Compatible GPU Options:**")
        for gpu in ws['supported_gpus']:
            st.markdown(f"• {gpu}")

    st.markdown("---")

    # System Requirements
    st.markdown("## 💻 System RAM Requirements")
    ram_needed = max(8, math.ceil(vram['total'] * 2))  # Minimum 8 GB
    st.markdown(f"""
    **Recommended System RAM:** {ram_needed} GB minimum

    *General rule: 2× VRAM for optimal performance*
    """)


    # Download Report
    st.markdown("## 📥 Export Report")
    report_content = generate_report(model_name, params, precision, workload, batch_size, sequence_length, vram,
                                     recommendation)

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
    st.caption(
        f"Current settings: Batch size = {batch_size if 'batch_size' in locals() else 1}, Sequence length = {sequence_length if 'sequence_length' in locals() else 2048}")