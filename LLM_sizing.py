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

# Initialize session state
if 'calculated' not in st.session_state:
    st.session_state.calculated = False

# Page configuration
st.set_page_config(
    page_title="LLM Sizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #0066cc;
    }
    .formula-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9em;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("LLM Sizer")
st.markdown("**Simple GPU sizing for LLM deployment**")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("Configuration")

# Model Selection
st.sidebar.subheader("1. Model Selection")
model_presets = {
    "Llama-3-8B": {"params": 8, "layers": 32, "hidden": 4096, "max_context": 8192},
    "Llama-3-70B": {"params": 70, "layers": 80, "hidden": 8192, "max_context": 8192},
    "Llama-3.1-8B": {"params": 8, "layers": 32, "hidden": 4096, "max_context": 131072},
    "Llama-3.1-70B": {"params": 70, "layers": 80, "hidden": 8192, "max_context": 131072},
    "Mistral-7B": {"params": 7, "layers": 32, "hidden": 4096, "max_context": 32768},
    "Mixtral-8x7B": {"params": 13, "layers": 32, "hidden": 4096, "max_context": 32768},  # active params
    "Custom": {"params": 8, "layers": 32, "hidden": 4096, "max_context": 8192}
}

selected_model = st.sidebar.selectbox("Select Model", list(model_presets.keys()))
model = model_presets[selected_model].copy()

if selected_model == "Custom":
    model["params"] = st.sidebar.number_input("Parameters (Billions)", 1, 1000, 8)
    model["layers"] = st.sidebar.number_input("Number of Layers", 1, 200, 32)
    model["hidden"] = st.sidebar.number_input("Hidden Dimension", 128, 16384, 4096)
    model["max_context"] = st.sidebar.number_input("Max Context Length", 512, 200000, 8192)
else:
    st.sidebar.markdown(f"""
    **Model Specs:**
    - Parameters: {model['params']}B
    - Layers: {model['layers']}
    - Hidden Dim: {model['hidden']}
    - Max Context: {model['max_context']:,}
    """)

st.sidebar.markdown("---")

# Workload Configuration
st.sidebar.subheader("2. Workload")
avg_context_tokens = st.sidebar.number_input(
    "Average Context Length (tokens)",
    128,
    model["max_context"],
    min(4096, model["max_context"]),
    help="Average number of tokens per request (prompt + response)"
)
n_concurrent = st.sidebar.number_input(
    "Concurrent Requests",
    1,
    1000,
    10,
    help="Number of simultaneous requests you want to handle"
)

st.sidebar.markdown("---")

# GPU Selection
st.sidebar.subheader("3. GPU Configuration")
gpu_options = {
    "RTX Ada 6000 (48GB)": {"memory": 48, "bandwidth": 0.96, "compute": 91},
    "Blackwell 5000 PCIe16 (96GB)": {"memory": 96, "bandwidth": 1.5, "compute": 250},
    "Blackwell 6000 PCIe16 (96GB)": {"memory": 96, "bandwidth": 1.8, "compute": 380},
}

selected_gpu = st.sidebar.selectbox("GPU Type", list(gpu_options.keys()))
gpu = gpu_options[selected_gpu]

st.sidebar.markdown(f"""
**GPU Specs:**
- Memory: {gpu['memory']} GB
- Bandwidth: {gpu['bandwidth']} TB/s
- Compute: {gpu['compute']} TFLOPS (FP16)
""")

st.sidebar.markdown("---")

# Precision Selection
st.sidebar.subheader("4. Precision")
precision_options = {
    "FP16": 2,
    "FP8": 1,
    "INT8": 1,
    "INT4": 0.5
}
precision = st.sidebar.selectbox(
    "Model Precision",
    list(precision_options.keys()),
    help="FP16 is standard, FP8/INT8 for quantized models"
)
precision_bytes = precision_options[precision]

st.sidebar.markdown("---")

# Calculate Button
st.sidebar.markdown("---")
if st.sidebar.button("Calculate Requirements", type="primary", use_container_width=True):
    st.session_state.calculated = True

# Add reset button when calculated
if st.session_state.calculated:
    if st.sidebar.button("🔄 Reset", use_container_width=True):
        st.session_state.calculated = False
        st.rerun()

# Main Calculations
if st.session_state.calculated:

    st.markdown("---")

    # ========================================
    # PERFORM ALL CALCULATIONS FIRST (Silent - no display)
    # ========================================

    # STEP 1: KV Cache Size per Token
    kv_cache_per_token_bytes = 2 * precision_bytes * model["layers"] * model["hidden"]
    kv_cache_per_token_gb = kv_cache_per_token_bytes / 1e9

    # STEP 2: GPU Memory Footprint
    model_weights_gb = model["params"] * precision_bytes
    kv_cache_total_gb = kv_cache_per_token_gb * avg_context_tokens * n_concurrent
    total_memory_gb = model_weights_gb + kv_cache_total_gb

    # STEP 3: Maximum Capacity
    available_memory = gpu["memory"] - model_weights_gb
    max_kv_tokens = int(available_memory / kv_cache_per_token_gb) if kv_cache_per_token_gb > 0 else 0

    # STEP 4: Concurrent Requests
    max_concurrent_worst = int(max_kv_tokens / model["max_context"]) if model["max_context"] > 0 and max_kv_tokens > 0 else 0
    max_concurrent_avg = int(max_kv_tokens / avg_context_tokens) if avg_context_tokens > 0 and max_kv_tokens > 0 else 0

    # STEP 5: Prefill Time per Token
    prefill_time_ms = (model["params"] * 2) / gpu["compute"]

    # STEP 6: Generation Time per Token (TPOT)
    generation_time_ms = (model["params"] * 2) / gpu["bandwidth"]
    tokens_per_second = 1000 / generation_time_ms

    # STEP 7: Total Latency (with defaults)
    default_prompt = int(avg_context_tokens * 0.75)
    prompt_tokens = min(default_prompt, model["max_context"])
    default_response = int(avg_context_tokens * 0.25)
    response_tokens = min(default_response, model["max_context"])
    ttft_target = 1.0
    latency_target = 5.0
    tps_target = 30
    prefill_time_total = (prompt_tokens * prefill_time_ms) / 1000
    generation_time_total = (response_tokens * generation_time_ms) / 1000
    total_latency = prefill_time_total + generation_time_total
    ttft = prefill_time_total

    # GPU Requirements
    gpus_needed = math.ceil(total_memory_gb / gpu["memory"])

    # ========================================
    # NOW DISPLAY RESULTS - RECOMMENDATIONS FIRST
    # ========================================

    st.header("📋 Recommendations & Summary")
    st.markdown("---")

    # ========================================
    # GPU Requirements Summary
    # ========================================
    st.subheader("GPU Requirements")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("No. of GPUs Required", gpus_needed, help="Based on memory requirements")
    with col2:
        st.metric("Memory per GPU", f"{total_memory_gb / gpus_needed:.1f} GB" if gpus_needed > 0 else "N/A")
    with col3:
        st.metric("Total Memory Needed", f"{total_memory_gb:.1f} GB")
    with col4:
        st.metric("GPU Type", selected_gpu.split()[0])

    st.markdown("---")

    # ========================================
    # Recommendations
    # ========================================
    st.subheader("💡 Deployment Recommendations")

    if gpus_needed == 1 and available_memory > 0:
        st.markdown(f"""
        <div class="success-box">
        <strong>✅ Single GPU Deployment</strong><br>
        Your workload fits on a single {selected_gpu}!<br>
        <br>
        <strong>Configuration:</strong><br>
        • Model: {selected_model}<br>
        • Concurrent Requests: {max_concurrent_avg} (avg case)<br>
        • Response Time: {total_latency:.2f}s<br>
        • Tokens/Second: {tokens_per_second:.1f}<br>
        </div>
        """, unsafe_allow_html=True)

    elif gpus_needed > 1:
        st.markdown(f"""
        <div class="warning-box">
        <strong> Multi-GPU Setup Required</strong><br>
        <br>
        <strong>Minimum Configuration:</strong><br>
        • GPUs Needed: <strong>{gpus_needed}x {selected_gpu}</strong><br>
        • Memory per GPU: {total_memory_gb / gpus_needed:.1f} GB<br>
        • Deployment Strategy: <strong>Tensor Parallelism (TP)</strong><br>
        <br>
        <strong>What is Tensor Parallelism?</strong><br>
        Model weights are split across GPUs. Each GPU holds a portion of the model,
        and they communicate during inference. Modern frameworks like vLLM and TensorRT-LLM
        handle this automatically.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="warning-box">
        <strong>❌ Configuration Issue</strong><br>
        Model weights ({model_weights_gb:.2f} GB) exceed single GPU capacity ({gpu['memory']} GB).<br>
        Consider: Larger GPU or model quantization (FP8/INT8).
        </div>
        """, unsafe_allow_html=True)

    # Capacity Warning
    if max_concurrent_avg < n_concurrent and max_concurrent_avg > 0:
        shortage = n_concurrent - max_concurrent_avg
        additional_gpus = math.ceil(shortage / max_concurrent_avg)

        st.markdown(f"""
        <div class="warning-box">
        <strong>⚠️ Throughput Warning</strong><br>
        <br>
        <strong>Your Target:</strong> {n_concurrent} concurrent requests<br>
        <strong>Current Capacity:</strong> {max_concurrent_avg} concurrent requests<br>
        <strong>Shortage:</strong> {shortage} requests<br>
        <br>
        <strong>Options:</strong><br>
        1. Add {additional_gpus} more GPU(s) for Data Parallelism (DP)<br>
        2. Reduce average context length to {int(avg_context_tokens * max_concurrent_avg / n_concurrent)} tokens<br>
        3. Accept lower concurrency
        </div>
        """, unsafe_allow_html=True)

    # ========================================
    # Performance Assessment
    # ========================================
    st.markdown("---")
    st.subheader("⚡ Performance Assessment")

    # Performance Metrics
    st.markdown("**Current Performance vs Targets:**")

    perf_summary = pd.DataFrame({
        "Metric": [
            "Time to First Token (TTFT)",
            "Total Response Time",
            "Tokens per Second",
            "Max Concurrent Requests"
        ],
        "Actual": [
            f"{ttft:.2f}s",
            f"{total_latency:.2f}s",
            f"{tokens_per_second:.1f}",
            f"{max_concurrent_avg}"
        ],
        "Target": [
            f"{ttft_target}s",
            f"{latency_target}s",
            f"{tps_target}",
            f"{n_concurrent}"
        ],
        "Status": [
            "✅" if ttft < ttft_target else "❌",
            "✅" if total_latency < latency_target else "❌",
            "✅" if tokens_per_second > tps_target else "❌",
            "✅" if max_concurrent_avg >= n_concurrent else "❌"
        ]
    })

    st.dataframe(perf_summary, hide_index=True, use_container_width=True)

    # Optimization Strategies
    st.markdown("---")
    st.markdown("**Optimization Strategies:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>Reduce Latency</strong><br><br>
        • Higher compute GPUs<br>
        • Enable FlashAttention<br>
        • Speculative decoding<br>
        • Reduce prompt length
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
        <strong>Increase Throughput</strong><br><br>
        • Continuous batching (vLLM)<br>
        • Add more GPUs (DP)<br>
        • PagedAttention<br>
        • Optimize batch sizes
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-box">
        <strong>Reduce Memory</strong><br><br>
        • FP8/INT8 quantization<br>
        • PagedAttention<br>
        • Reduce context length<br>
        • Lower batch size
        </div>
        """, unsafe_allow_html=True)

    # Multi-GPU Configuration Table (if needed)
    if gpus_needed > 1:
        st.markdown("---")
        st.subheader("🔧 Multi-GPU Configuration Options")

        gpu_configs = []
        for n_gpus in range(gpus_needed, min(gpus_needed + 5, 9)):
            mem_per_gpu = total_memory_gb / n_gpus
            fits = mem_per_gpu <= gpu["memory"]

            # Calculate capacity for this GPU configuration
            if fits:
                available_per_gpu = gpu["memory"] - mem_per_gpu
                total_available = available_per_gpu * n_gpus
                max_tokens_multi = int(total_available / kv_cache_per_token_gb) if kv_cache_per_token_gb > 0 else 0
                max_concurrent_multi = int(max_tokens_multi / avg_context_tokens) if avg_context_tokens > 0 else 0
            else:
                max_concurrent_multi = 0

            gpu_configs.append({
                "# GPUs": n_gpus,
                "Memory/GPU": f"{mem_per_gpu:.1f} GB",
                "GPU Memory": f"{gpu['memory']} GB",
                "Fits?": "✅" if fits else "❌",
                "Strategy": "Tensor Parallelism (TP)" if fits else "Requires TP",
                "Max Concurrent Requests": max_concurrent_multi
            })

        st.dataframe(pd.DataFrame(gpu_configs), hide_index=True, use_container_width=True)

    # ========================================
    # TECHNICAL DEEP DIVE SECTION
    # ========================================
    st.markdown("---")
    st.markdown("---")
    st.header("🔬 Technical Deep Dive")
    st.markdown("*Detailed calculations and formulas behind the recommendations*")
    st.markdown("---")

    # Add expandable section for technical details
    with st.expander("📊 **View Detailed Calculations**", expanded=False):

        # ========================================
        # STEP 1: KV Cache Size per Token
        # ========================================
        st.subheader("Step 1: KV Cache Size per Token")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="formula-box">
            <strong>Formula:</strong><br>
            KV Cache per token = 2 × precision_bytes × n_layers × d_model<br>
            = 2 × {precision_bytes} × {model['layers']} × {model['hidden']}<br>
            = {kv_cache_per_token_bytes:,} bytes/token<br>
            ≈ <strong>{kv_cache_per_token_gb:.6f} GB/token</strong>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="info-box">
            <strong>Why this formula?</strong><br>
            • First 2: Key + Value matrices<br>
            • precision_bytes: {precision} precision ({precision_bytes} bytes)<br>
            • n_layers: Each layer has its own cache<br>
            • d_model: Size of hidden representation
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ========================================
        # STEP 2: GPU Memory Footprint
        # ========================================
        st.subheader("Step 2: GPU Memory Footprint")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="formula-box">
            <strong>Formula:</strong><br>
            Total Memory = Model Weights + KV Cache<br>
            <br>
            Model Weights = {model['params']}B × {precision_bytes} bytes = <strong>{model_weights_gb:.2f} GB</strong><br>
            <br>
            KV Cache = {kv_cache_per_token_gb:.6f} × {avg_context_tokens} × {n_concurrent}<br>
            = <strong>{kv_cache_total_gb:.2f} GB</strong><br>
            <br>
            Total = {model_weights_gb:.2f} + {kv_cache_total_gb:.2f} = <strong>{total_memory_gb:.2f} GB</strong>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            memory_breakdown = pd.DataFrame({
                "Component": ["Model Weights", "KV Cache", "Total Required"],
                "Size (GB)": [
                    f"{model_weights_gb:.2f}",
                    f"{kv_cache_total_gb:.2f}",
                    f"{total_memory_gb:.2f}"
                ],
                "Percentage": [
                    f"{(model_weights_gb / total_memory_gb * 100):.1f}%",
                    f"{(kv_cache_total_gb / total_memory_gb * 100):.1f}%",
                    "100%"
                ]
            })
            st.dataframe(memory_breakdown, hide_index=True, use_container_width=True)

        st.markdown("---")

        # ========================================
        # STEP 3: Maximum Capacity
        # ========================================
        st.subheader("Step 3: Maximum Capacity")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="formula-box">
            <strong>Formula:</strong><br>
            Max KV Cache Tokens = (GPU Memory - Model Weights) ÷ KV per token<br>
            <br>
            = ({gpu['memory']} GB - {model_weights_gb:.2f} GB) ÷ {kv_cache_per_token_gb:.6f} GB/token<br>
            = {available_memory:.2f} GB ÷ {kv_cache_per_token_gb:.6f} GB/token<br>
            = <strong>{max_kv_tokens:,} tokens</strong>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if available_memory < 0:
                st.markdown(f"""
                <div class="warning-box">
                <strong>⚠️ Out of Memory!</strong><br>
                Model weights ({model_weights_gb:.2f} GB) exceed GPU memory ({gpu['memory']} GB).<br>
                You need multiple GPUs with Tensor Parallelism.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric("Available for KV Cache", f"{available_memory:.2f} GB")
                st.metric("Max Cacheable Tokens", f"{max_kv_tokens:,}")

        st.markdown("---")

        # ========================================
        # STEP 4: Concurrent Requests
        # ========================================
        st.subheader("Step 4: Concurrent Request Capacity")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="formula-box">
            <strong>Formula:</strong><br>
            Concurrent Requests = Max Tokens ÷ Context per Request<br>
            <br>
            <strong>Worst Case (max context {model['max_context']:,}):</strong><br>
            = {max_kv_tokens:,} ÷ {model['max_context']:,} = <strong>{max_concurrent_worst}</strong><br>
            <br>
            <strong>Average Case (avg context {avg_context_tokens:,}):</strong><br>
            = {max_kv_tokens:,} ÷ {avg_context_tokens:,} = <strong>{max_concurrent_avg}</strong>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            capacity_data = pd.DataFrame({
                "Scenario": ["Worst Case (max context)", "Average Case (your config)", "Your Target"],
                "Concurrent Requests": [
                    max_concurrent_worst,
                    max_concurrent_avg,
                    n_concurrent
                ]
            })
            st.dataframe(capacity_data, hide_index=True, use_container_width=True)

            if max_concurrent_avg < n_concurrent:
                st.markdown(f"""
                <div class="warning-box">
                <strong>⚠️ Insufficient Capacity</strong><br>
                Target: {n_concurrent} concurrent | Actual: {max_concurrent_avg} concurrent<br>
                Consider: Shorter contexts or more GPUs
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ========================================
        # STEP 5: Prefill Time per Token
        # ========================================
        st.subheader("Step 5: Prefill Time (Compute-Bound)")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="formula-box">
            <strong>Formula:</strong><br>
            Prefill Time per Token = (Params × 2) ÷ GPU Compute<br>
            <br>
            = ({model['params']}B params × 2) ÷ {gpu['compute']} TFLOPS<br>
            = {model['params'] * 2}B FLOPs ÷ {gpu['compute']} TFLOPS<br>
            = <strong>{prefill_time_ms:.3f} ms/token</strong>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
            <strong>Prefill Phase:</strong><br>
            • Processes input tokens in parallel<br>
            • Compute-bound (limited by TFLOPS)<br>
            • Happens once per request<br>
            • Populates KV cache
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ========================================
        # STEP 6: Generation Time per Token (TPOT)
        # ========================================
        st.subheader("Step 6: Generation Time per Token (Memory-Bound)")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="formula-box">
            <strong>Formula:</strong><br>
            Generation Time per Token = Bytes Moved ÷ Memory Bandwidth<br>
            <br>
            = ({model['params']}B params × 2 bytes) ÷ {gpu['bandwidth']} TB/s<br>
            = {model['params'] * 2} GB ÷ {gpu['bandwidth']} TB/s<br>
            = <strong>{generation_time_ms:.1f} ms/token</strong><br>
            <br>
            Tokens per Second = <strong>{tokens_per_second:.1f} tokens/s</strong>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="info-box">
            <strong>Generation Phase:</strong><br>
            • Generates tokens one at a time<br>
            • Memory-bound (limited by bandwidth)<br>
            • This is the bottleneck!<br>
            • Reuses KV cache from prefill
            </div>
            """, unsafe_allow_html=True)

            st.metric("Time Per Output Token (TPOT)", f"{generation_time_ms:.1f} ms")
            st.metric("Tokens per Second", f"{tokens_per_second:.1f}")

        st.markdown("---")

        # ========================================
        # STEP 7: Total Latency (with interactive inputs)
        # ========================================
        st.subheader("Step 7: Total Response Time")

        st.markdown("**Customize Your Scenario:**")
        st.info(f"""
        Your average context: **{avg_context_tokens:,} tokens total**
        We use 75% for prompt and 25% for response by default, but feel free to customize below.
        """)

        col1, col2 = st.columns(2)
        with col1:
            # Allow user to customize prompt tokens
            prompt_tokens_custom = st.number_input(
                "Prompt Size (tokens)",
                min_value=1,
                max_value=model["max_context"],
                value=prompt_tokens,
                step=100,
                help="Number of input tokens",
                key="prompt_tokens_deep_dive"
            )
        with col2:
            # Allow user to customize response tokens
            response_tokens_custom = st.number_input(
                "Response Size (tokens)",
                min_value=1,
                max_value=model["max_context"],
                value=response_tokens,
                step=10,
                help="Number of output tokens",
                key="response_tokens_deep_dive"
            )

        st.markdown("**Set Performance Targets:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            ttft_target_custom = st.number_input(
                "TTFT Target (seconds)",
                min_value=0.1,
                max_value=10.0,
                value=ttft_target,
                step=0.1,
                help="Time to First Token target",
                key="ttft_target_deep_dive"
            )
        with col2:
            latency_target_custom = st.number_input(
                "Total Latency Target (seconds)",
                min_value=0.1,
                max_value=30.0,
                value=latency_target,
                step=0.5,
                help="End-to-end response time target",
                key="latency_target_deep_dive"
            )
        with col3:
            tps_target_custom = st.number_input(
                "Tokens/Sec Target",
                min_value=1,
                max_value=200,
                value=tps_target,
                step=5,
                help="Minimum tokens per second",
                key="tps_target_deep_dive"
            )

        # Recalculate with custom inputs
        prefill_time_total_custom = (prompt_tokens_custom * prefill_time_ms) / 1000
        generation_time_total_custom = (response_tokens_custom * generation_time_ms) / 1000
        total_latency_custom = prefill_time_total_custom + generation_time_total_custom
        ttft_custom = prefill_time_total_custom

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="formula-box">
            <strong>Formula:</strong><br>
            Total Time = Prefill Time + Generation Time<br>
            <br>
            <strong>Your Scenario: {prompt_tokens_custom:,} token prompt + {response_tokens_custom} token response</strong><br>
            <br>
            Prefill = {prompt_tokens_custom:,} × {prefill_time_ms:.3f} ms = {prefill_time_total_custom:.2f} s<br>
            Generation = {response_tokens_custom} × {generation_time_ms:.1f} ms = {generation_time_total_custom:.2f} s<br>
            <br>
            Total = {prefill_time_total_custom:.2f} + {generation_time_total_custom:.2f} = <strong>{total_latency_custom:.2f} seconds</strong>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            perf_data_custom = pd.DataFrame({
                "Metric": [
                    "Time to First Token (TTFT)",
                    "Total Response Time",
                    "Tokens per Second"
                ],
                "Value": [
                    f"{ttft_custom:.2f} s",
                    f"{total_latency_custom:.2f} s",
                    f"{tokens_per_second:.1f}"
                ],
                "Target": [
                    f"< {ttft_target_custom}s",
                    f"< {latency_target_custom}s",
                    f"> {tps_target_custom}"
                ],
                "Status": [
                    "✅" if ttft_custom < ttft_target_custom else "⚠️",
                    "✅" if total_latency_custom < latency_target_custom else "⚠️",
                    "✅" if tokens_per_second > tps_target_custom else "⚠️"
                ]
            })
            st.dataframe(perf_data_custom, hide_index=True, use_container_width=True)

else:
    # Welcome Screen
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to LLM Sizer</h3>

    This tool calculates GPU requirements for LLM deployment </strong>.

    <h4>What You'll Get:</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Capacity Planning
        - **Memory Requirements** - Model weights + KV cache
        - **GPU Count** - How many GPUs you need
        - **Concurrent Requests** - Maximum capacity
        - **Multi-GPU Strategy** - TP/DP recommendations
        """)

        st.markdown("""
        ### Performance Estimates
        - **Time to First Token (TTFT)**
        - **Response Time** - End-to-end latency
        - **Throughput** - Tokens per second
        """)

    with col2:
        st.markdown("""
        ###     Getting Started

        1. **Select a Model** - Choose preset or custom
        2. **Define Workload** - Context length & concurrent requests
        3. **Pick GPU** - Select from available options
        4. **Calculate** - Get instant sizing results

        """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    st.title("LLM Sizer - Formula Reference")

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **1. KV Cache per Token**
        ```
        2 × precision_bytes × n_layers × d_model
        ```
        *Memory needed to cache each token*

        **2. Total Memory**
        ```
        Model Weights + KV Cache
        ```
        *Total GPU memory required*

        **3. Max Capacity**
        ```
        (GPU Memory - Weights) ÷ KV per token
        ```
        *Maximum tokens you can cache*

        **4. Concurrent Requests**
        ```
        Max Tokens ÷ Context per Request
        ```
        *Maximum simultaneous requests*
        """)

    with col2:
        st.markdown("""
        **5. Prefill Time (compute-bound)**
        ```
        (Params × 2) ÷ GPU Compute
        ```
        *Time to process input tokens*

        **6. Generation Time (memory-bound)**
        ```
        (Params × 2) ÷ GPU Bandwidth
        ```
        *Time per output token (TPOT)*

        **7. Total Latency**
        ```
        (Prompt × Prefill) + (Response × Generation)
        ```
        *End-to-end response time*
        """)

    # Divider
    st.markdown("---")

    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <strong>For planning and estimation purposes only. Real performance varies by implementation.<br>
    </div>
    """, unsafe_allow_html=True)

