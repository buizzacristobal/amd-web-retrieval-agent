"""
Streamlit Web Dashboard for AMD Web Retrieval Agent.
Features Live Multi-Hop Agent Trace, DOM Table Viewer, and AMD Telemetry HUD.
Designed for AMD AI League Match 4 evaluation and demo video recording.
"""

import os
import sys
import time

try:
    import streamlit as st
except ImportError:
    st = None
    if __name__ == "__main__":
        print("\n=======================================================")
        print("  AMD Web Retrieval Agent - Streamlit Application      ")
        print("=======================================================")
        print("Streamlit is required to launch the graphical dashboard.")
        print("Please install requirements and launch with:")
        print("    pip install -r requirements.txt")
        print("    streamlit run app.py")
        print("=======================================================\n")
        sys.exit(0)
    else:
        raise ImportError(
            "Streamlit is required to run the interactive dashboard in app.py. "
            "Please install streamlit via 'pip install streamlit' or 'pip install -r requirements.txt'."
        )

from config import Settings, get_hardware_target, get_vram_utilization_est, detect_rocm_runtime
from agent.orchestrator import RetrievalOrchestrator

# -----------------------------------------------------------------------------
# Page Configuration & AMD Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AMD Web Retrieval Agent | AMD AI League",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS for Video Recording
st.markdown("""
<style>
    /* AMD Crimson & Obsidian Dark Theme */
    :root {
        --amd-red: #ED1C24;
        --amd-dark: #0B0F19;
        --card-bg: #141A29;
        --border-color: #26334D;
        --text-primary: #F3F4F6;
        --text-muted: #9CA3AF;
    }
    .main-header {
        background: linear-gradient(90deg, #111827 0%, #1F2937 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border-left: 6px solid #ED1C24;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #9CA3AF;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #141A29;
        border: 1px solid #26334D;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        flex: 1;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ED1C24;
        margin-bottom: 0.2rem;
    }
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #9CA3AF;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .status-badge {
        display: inline-block;
        background-color: #222938;
        border: 1px solid #ED1C24;
        color: #F87171;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .abstain-box {
        background-color: #3b1b1b;
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #fee2e2;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: AMD Acceleration & Agent Control Center
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg", width=140)
    st.markdown("### ⚡ AMD Hardware Control Center")

    # Backend Selector
    backend_mode = st.selectbox(
        "Inference Backend Router",
        options=["amd_cloud_vllm", "local_rocm", "universal_fallback"],
        index=2,
        format_func=lambda x: {
            "amd_cloud_vllm": "AMD Developer Cloud (MI300X vLLM)",
            "local_rocm": "Local AMD ROCm (HIP/PyTorch)",
            "universal_fallback": "Universal Fallback / Extractive Engine"
        }[x]
    )

    # Dynamic inputs based on backend
    if backend_mode == "amd_cloud_vllm":
        api_url = st.text_input("AMD vLLM Endpoint URL", value=os.getenv("AMD_BASE_URL", "https://cloud.amd.com/v1"))
        api_key = st.text_input("AMD Cloud API Key", type="password", value=os.getenv("AMD_API_KEY", ""))
        model_name = st.selectbox("AMD Cloud Model", ["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"])
    elif backend_mode == "local_rocm":
        api_url = ""
        api_key = ""
        model_name = st.text_input("Local ROCm Model", value="Qwen/Qwen2.5-3B-Instruct")
    else:
        api_url = st.text_input("Local vLLM / Ollama URL (Optional)", value="http://localhost:8000/v1")
        api_key = st.text_input("API Key (Optional)", type="password", value="")
        model_name = st.text_input("Model ID", value="Qwen/Qwen2.5-7B-Instruct")

    st.markdown("---")
    st.markdown("### 🎛️ Agent Retrieval Tuning")
    max_search_results = st.slider("Max Search Results", min_value=2, max_value=8, value=5)
    max_hops = st.slider("Max Retrieval Hops", min_value=1, max_value=3, value=2)
    strictness_threshold = st.slider("Strictness Threshold", min_value=0.1, max_value=0.8, value=0.35, step=0.05)

    # Active Hardware Badge
    st.markdown("---")
    st.markdown("### 🖥️ Compute Telemetry")
    hw_target = get_hardware_target(backend_mode)
    vram_est = get_vram_utilization_est()
    st.markdown(f"<div class='status-badge'>TARGET: {hw_target}</div>", unsafe_allow_html=True)
    st.caption(f"**VRAM / RAM Status:** {vram_est}")

# Create runtime settings instance with UI overrides
app_settings = Settings(
    inference_backend=backend_mode,
    amd_base_url=api_url if backend_mode == "amd_cloud_vllm" else "https://cloud.amd.com/v1",
    amd_api_key=api_key if backend_mode == "amd_cloud_vllm" else "",
    amd_model=model_name if backend_mode == "amd_cloud_vllm" else "Qwen/Qwen2.5-72B-Instruct",
    openai_compatible_base_url=api_url,
    openai_compatible_api_key=api_key,
    openai_compatible_model=model_name,
    max_search_results=max_search_results,
    max_retrieval_hops=max_hops,
    grounding_strictness_threshold=strictness_threshold
)

# -----------------------------------------------------------------------------
# Main Interface Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>AMD Intelligent Web Retrieval Agent</h1>
    <p>Autonomous Multi-Hop Search, DOM Table Extractor & Anti-Hallucination Grounding Engine | AMD AI League Match 4</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Preset 1-Click Demo Buttons for 90-Second Video
# -----------------------------------------------------------------------------
st.markdown("##### 🚀 Quick Demo Presets (1-Click Evaluation):")
col1, col2, col3 = st.columns(3)

preset_query = None
with col1:
    if st.button("📊 Hardware Specs & Table Extraction", use_container_width=True):
        preset_query = "Compare AMD Instinct MI300X vs MI325X memory capacity, bandwidth, and FP8 performance specs."
with col2:
    if st.button("⚡ ROCm 6.2 Architecture Highlights", use_container_width=True):
        preset_query = "What are the latest key updates and architecture highlights in AMD ROCm 6.2+ for LLM inference?"
with col3:
    if st.button("🛑 Anti-Hallucination Abstention Test", use_container_width=True):
        preset_query = "What is the official release date and benchmark score of the fictional AMD Radeon RX 19990 XTX quantum GPU?"

# Query Input Field
user_query = st.text_input(
    "Enter a research query or select a preset above:",
    value=preset_query if preset_query else "Compare AMD Instinct MI300X vs MI325X memory capacity, bandwidth, and FP8 performance specs.",
    placeholder="e.g. Compare AMD Instinct MI300X vs MI325X memory and bandwidth..."
)

run_button = st.button("🚀 Execute Autonomous Retrieval Loop", type="primary")

# -----------------------------------------------------------------------------
# Execution & Results Panel
# -----------------------------------------------------------------------------
if run_button or preset_query:
    with st.spinner("🤖 Agent orchestrator executing: Planning -> Multi-Engine Search -> Deep DOM Scraping -> Hybrid BM25/Semantic Ranking..."):
        orchestrator = RetrievalOrchestrator(app_settings)
        response = orchestrator.run(user_query)

    # -------------------------------------------------------------------------
    # Top-Level KPI Dashboard
    # -------------------------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{response.grounding_score * 100:.0f}%</div>
            <div class="kpi-label">Grounding Score</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{len(response.citations)}</div>
            <div class="kpi-label">Verified Sources</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{response.hops_executed}</div>
            <div class="kpi-label">Retrieval Hops</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{response.telemetry.get('total_latency_ms', 0):.0f}ms</div>
            <div class="kpi-label">End-to-End Latency</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{response.telemetry.get('tables_extracted_total', 0)}</div>
            <div class="kpi-label">DOM Tables Extracted</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Grounded Answer Display
    # -------------------------------------------------------------------------
    if response.abstained:
        st.markdown(f"""
        <div class="abstain-box">
            <h4>🛑 Abstention Guardrail Triggered</h4>
            <p>The agent rejected speculation because verified web evidence does not support this entity or metric.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📝 Grounded Intelligence Synthesis")
    st.markdown(response.answer_markdown)

    # Verified Citations Bar
    if response.citations:
        st.markdown("##### 🔗 Verified Primary Sources & Citations:")
        c_cols = st.columns(min(len(response.citations), 3))
        for idx, cit in enumerate(response.citations):
            col = c_cols[idx % 3]
            with col:
                st.info(f"**[{cit.citation_id}] {cit.title}**\n\n[Open Verified Source]({cit.url})\n\n*{cit.supporting_quote}*")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Interactive Inspection Tabs (Trace, Tables, Telemetry)
    # -------------------------------------------------------------------------
    tab_trace, tab_chunks, tab_telemetry = st.tabs([
        "🧠 Agent Reasoning & Hop Trace",
        "📊 Extracted Chunks & DOM Tables",
        "⚡ AMD Inference & Pipeline Telemetry"
    ])

    with tab_trace:
        st.markdown("#### Autonomous Multi-Hop Trace & Decision Graph")
        for step in response.telemetry.get("hop_trace", []):
            with st.expander(f"Hop {step.get('hop', 1)}: {step.get('stage')}", expanded=True):
                st.write(step)

    with tab_chunks:
        st.markdown("#### Ranked Chunks & Extracted HTML Tables")
        st.caption("Extracted Markdown tables are treated as atomic units to preserve numerical spec comparisons intact.")
        for chunk in response.ranked_chunks:
            badge = "📊 Structured Table" if chunk.is_table else "📄 Text Passage"
            with st.expander(f"Chunk #{chunk.chunk_id} [{badge}] — Combined Score: {chunk.combined_score} (BM25: {chunk.bm25_score}, Semantic: {chunk.semantic_score})"):
                st.caption(f"Source: [{chunk.title}]({chunk.url})")
                if chunk.is_table:
                    st.markdown(chunk.text)
                else:
                    st.write(chunk.text)

    with tab_telemetry:
        st.markdown("#### AMD Acceleration & System Telemetry HUD")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.json({
                "hardware_target": response.telemetry.get("hardware_target"),
                "active_backend": response.telemetry.get("active_backend"),
                "vram_utilization_est": response.telemetry.get("vram_utilization_est"),
                "tokens_per_second": response.telemetry.get("tokens_per_second"),
                "total_latency_ms": response.telemetry.get("total_latency_ms"),
                "grounding_score_percent": response.telemetry.get("grounding_score_percent")
            })
        with col_t2:
            st.markdown("##### Phase Latencies (ms)")
            st.bar_chart(response.telemetry.get("phase_latencies", {}))

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("AMD AI League Match 4 — Intelligent Web Information Retrieval | Powered by AMD Instinct MI300X, ROCm 6.2 & vLLM")
