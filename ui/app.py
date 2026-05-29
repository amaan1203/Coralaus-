"""
coralaus — Streamlit UI
Interactive web interface for the paper analysis pipeline.
"""

import os
import sys
import json
import logging
import streamlit as st
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from agents.ingest import parse_paper
from agents.pwc_search import search_implementation
from agents.repo_health import check_repo_health
from agents.compat_check import check_compatibility
from agents.conflict_resolver import resolve_conflicts
from agents.no_impl_generator import generate_from_scratch
from agents.output_builder import build_output, save_output, format_summary
from agents.coral_utils import get_coral_client
from agents.pwc_mcp_client import get_s2_client

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(
    page_title="Coralaus — Research Paper Implementation Finder",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS (Premium Dark Theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');

    /* Global Typography & Colors */
    .stApp {
        font-family: 'Outfit', sans-serif;
        background-color: #0B0E14;
        color: #E2E8F0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0B0E14;
    }
    ::-webkit-scrollbar-thumb {
        background: #2D3748;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4A5568;
    }

    /* Animated Header */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header {
        background: linear-gradient(-45deg, #0f172a, #1e293b, #0c4a6e, #1e1b4b);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 10px 30px -10px rgba(14, 165, 233, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.15), transparent 70%);
        pointer-events: none;
    }

    .main-header h1 {
        color: #f8fafc;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-header h1 span {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1.25rem;
        margin-top: 0.75rem;
        font-weight: 300;
    }

    /* Sidebar Badges */
    .status-badge {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        box-shadow: 0 0 10px currentColor;
    }
    .dot.green { background-color: #10b981; color: #10b981; }
    .dot.red { background-color: #ef4444; color: #ef4444; }
    .dot.yellow { background-color: #f59e0b; color: #f59e0b; }

    /* Compact Metrics */
    .metric-tile {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-tile:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -8px rgba(56, 189, 248, 0.2);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #f8fafc; line-height: 1; margin-bottom: 4px; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }

    /* Pipeline Stepper */
    .stepper {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(15, 23, 42, 0.6);
        padding: 16px 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .step-num {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 1rem;
    }
    .step-active .step-num { background: #38bdf8; color: #0f172a; box-shadow: 0 0 15px rgba(56, 189, 248, 0.5); }
    .step-done .step-num { background: #10b981; color: #0f172a; }
    .step-error .step-num { background: #ef4444; color: white; }
    .step-pending .step-num { background: #334155; color: #94a3b8; }
    
    .step-text { font-weight: 500; font-size: 1.05rem; color: #f8fafc; }
    
    /* Code Blocks */
    .stCodeBlock {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .stCodeBlock code {
        font-family: 'Fira Code', monospace !important;
        font-size: 0.9rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 16px;
        padding-bottom: 16px;
        padding-left: 24px;
        padding-right: 24px;
        background: rgba(30, 41, 59, 0.3);
        border-radius: 12px 12px 0 0;
        border: 1px solid transparent;
        border-bottom: none;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #38bdf8 !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
        border-bottom-color: rgba(30, 41, 59, 0.8) !important;
    }

    /* Empty State Features */
    .feature-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        height: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    .feature-card:hover {
        transform: translateY(-5px);
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 20px 40px -15px rgba(56, 189, 248, 0.15);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Health Badge */
    .health-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 24px;
        font-weight: 600;
        font-size: 1rem;
    }
    .health-green { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .health-yellow { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .health-orange { background: rgba(249, 115, 22, 0.15); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.3); }
    .health-red { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Status Spinner */
    .stSpinner > div > div {
        border-color: #38bdf8 transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)


def render_stepper(step_num, title, state="active", details=""):
    """Render a styled pipeline step."""
    icon = {"active": "⚡", "done": "✓", "error": "✕", "pending": "⋯"}[state]
    st.markdown(f"""
    <div class="stepper step-{state}">
        <div class="step-num">{icon if state != 'active' else step_num}</div>
        <div style="display: flex; flex-direction: column;">
            <div class="step-text">{title}</div>
            {f'<div style="font-size: 0.85rem; color: #94a3b8; margin-top: 2px;">{details}</div>' if details else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_results(final_output, paper_json=None):
    if not paper_json:
        paper_json = {
            "title": final_output["paper"]["title"],
            "arxiv_id": final_output["paper"].get("arxiv_id"),
            "authors": [{"first": name.split()[0] if name.split() else "", "last": name.split()[-1] if len(name.split()) > 1 else ""} for name in final_output["paper"].get("authors", [])],
            "year": final_output["paper"].get("year"),
            "abstract": final_output["paper"].get("abstract", "")
        }

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("## ✨ Analysis Complete")
    
    # Render Tabs
    tab_summary, tab_docker, tab_json, tab_dl = st.tabs([
        "📋 Summary", "🐳 Dockerfile", "📊 JSON Data", "📥 Downloads"
    ])

    with tab_summary:
        st.markdown(f"### {paper_json.get('title', 'Unknown Paper')}")
        st.info(format_summary(final_output))
        
        # If we have repo health, show it nicely
        if final_output.get("health"):
            h = final_output["health"]
            score = h.get("health_score", -1)
            emoji = h.get("health_emoji", "⚪")
            label = h.get("health_label", "Unknown")
            health_class = "green" if score >= 80 else "yellow" if score >= 60 else "orange" if score >= 30 else "red"
            st.markdown(f'<div class="health-badge health-{health_class}">{emoji} Repo Health: {label} ({score}/100)</div>', unsafe_allow_html=True)

    with tab_docker:
        if final_output.get("dockerfile"):
            st.markdown(f"*Suggested Base Image: `{final_output.get('selected_base_image', 'N/A')}`*")
            if final_output.get("base_image_reason"):
                st.caption(f"Reasoning: {final_output['base_image_reason']}")
            st.code(final_output["dockerfile"], language="dockerfile")
        else:
            st.warning("No Dockerfile was generated.")
            
        if final_output.get("implementation_script"):
            st.markdown("### 🐍 Reference Implementation")
            st.code(final_output["implementation_script"], language="python")

    with tab_json:
        st.json(final_output)

    with tab_dl:
        st.markdown("### Export Artifacts")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.download_button("📄 Paper JSON", data=json.dumps(paper_json, indent=2), file_name="paper.json", mime="application/json", use_container_width=True)
        with col2:
            st.download_button("📊 Full Result JSON", data=json.dumps(final_output, indent=2), file_name="result.json", mime="application/json", use_container_width=True)
        with col3:
            if final_output.get("dockerfile"):
                st.download_button("🐳 Dockerfile", data=final_output["dockerfile"], file_name="Dockerfile", mime="text/plain", use_container_width=True, type="primary")
        with col4:
            if final_output.get("implementation_script"):
                st.download_button("🐍 Implementation", data=final_output["implementation_script"], file_name="implementation.py", mime="text/x-python", use_container_width=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ System Status")

    gemini_ok = bool(os.environ.get("GEMINI_API_KEY"))
    groq_ok = bool(os.environ.get("GROQ_API_KEY"))
    github_ok = bool(os.environ.get("GITHUB_TOKEN"))

    st.markdown(f"""
    <div class="status-badge"><div class="dot {'green' if gemini_ok else 'red'}"></div> Gemini API Key</div>
    <div class="status-badge"><div class="dot {'green' if groq_ok else 'red'}"></div> Groq API Key</div>
    <div class="status-badge"><div class="dot {'green' if github_ok else 'yellow'}"></div> GitHub Token</div>
    """, unsafe_allow_html=True)

    coral = get_coral_client()
    st.markdown(f"""
    <div class="status-badge"><div class="dot {'green' if coral.available else 'yellow'}"></div> Coral CLI {'(Ready)' if coral.available else '(Fallback)'}</div>
    """, unsafe_allow_html=True)

    pwc = get_s2_client()
    st.markdown(f"""
    <div class="status-badge"><div class="dot {'green' if pwc.available else 'red'}"></div> Semantic Scholar</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Activity")
    
    if "coral_queries" not in st.session_state:
        st.session_state.coral_queries = 0
    if "pwc_calls" not in st.session_state:
        st.session_state.pwc_calls = 0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-val">{st.session_state.coral_queries}</div>
            <div class="metric-label">Coral SQL</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-val">{st.session_state.pwc_calls}</div>
            <div class="metric-label">API Calls</div>
        </div>
        """, unsafe_allow_html=True)


# --- Header ---
st.markdown("""
<div class="main-header">
    <h1><span>🚢</span> Coralaus</h1>
    <p>Seamlessly transition from ML research paper to a fully reproducible Docker environment.</p>
</div>
""", unsafe_allow_html=True)


# --- Main Content ---
uploaded_file = st.file_uploader(
    "📄 Upload Research Paper (PDF)",
    type=["pdf"],
    help="Upload an ML/AI research paper PDF to begin analysis",
)

if uploaded_file:
    os.makedirs("output", exist_ok=True)
    pdf_path = os.path.join("output", uploaded_file.name)
    
    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if "final_output" in st.session_state:
            del st.session_state.final_output
        if "paper_json" in st.session_state:
            del st.session_state.paper_json
            
    analyze_clicked = st.button("🚀 Analyze & Generate Environment", type="primary", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if analyze_clicked:
        if "final_output" in st.session_state:
            del st.session_state.final_output
        if "paper_json" in st.session_state:
            del st.session_state.paper_json

        coral.reset_query_count()
        pwc.reset_call_count()
        total_llm_calls = {"gemini_flash": 0, "groq_llama": 0}

        # === Step 1: PDF Ingestion ===
        step1 = st.empty()
        with step1:
            render_stepper(1, "Parsing PDF Structure", "active", "Extracting text, sections, and metadata via GROBID")
        
        try:
            paper_json = parse_paper(pdf_path, os.path.join("output", "current_paper.json"))
            with step1:
                render_stepper(1, f"Parsed: {paper_json.get('title', 'Unknown Paper')}", "done", f"Found {len(paper_json.get('sections', []))} sections")
        except Exception as e:
            with step1:
                render_stepper(1, "PDF parsing failed", "error", str(e))
            st.stop()

        # === Step 2: Search ===
        step2 = st.empty()
        with step2:
            render_stepper(2, "Searching Implementations", "active", "Querying Semantic Scholar & PapersWithCode")
            
        search_result = search_implementation(paper_json)
        st.session_state.pwc_calls = pwc.get_call_count()
        
        if search_result["found"]:
            with step2:
                render_stepper(2, "Implementations Found", "done", f"Discovered {len(search_result.get('candidates', []))} potential repositories")
        else:
            with step2:
                render_stepper(2, "No Implementation Found", "done", "Will generate reference code from scratch")

        health_result = None
        compat_result = None
        resolver_result = None
        generator_result = None

        if search_result["found"] and search_result.get("candidates"):
            # === Step 2.5: Repo Validation ===
            step25 = st.empty()
            with step25:
                render_stepper(3, "Validating Repositories", "active", "Evaluating match confidence")
                
            from agents.repo_validator import validate_and_rank_candidates
            best_match, ranked_results = validate_and_rank_candidates(paper_json, search_result["candidates"])

            if best_match["confidence_score"] < 0.20:
                search_result["found"] = False
                search_result["repo_url"] = None
                search_result["validation"] = best_match
                with step25:
                    render_stepper(3, "Low Confidence Match", "done", f"Best score {best_match['confidence_score']:.1%} is too low. Treating as mismatch.")
            else:
                search_result["repo_url"] = best_match["repo_url"]
                search_result["stars"] = best_match.get("stars", 0)
                search_result["is_official"] = best_match.get("is_official", False)
                search_result["search_method"] = best_match.get("search_method")
                search_result["validation"] = best_match
                with step25:
                    render_stepper(3, f"Selected Repository: {best_match['repo_url'].split('/')[-1]}", "done", f"Confidence: {best_match['confidence_score']:.1%} ({best_match['classification']})")

        if search_result["found"]:
            # === Step 3: Health Check ===
            step3 = st.empty()
            with step3:
                render_stepper(4, "Checking Repo Health", "active", "Analyzing commit history and issues via Coral SQL")
                
            health_result = check_repo_health(search_result["repo_url"])
            st.session_state.coral_queries = coral.get_query_count()
            
            score = health_result.get("health_score", -1)
            emoji = health_result.get("health_emoji", "⚪")
            with step3:
                render_stepper(4, f"Health Score: {score}/100", "done", f"{emoji} {health_result.get('health_label', 'Unknown')}")

            # === Step 4: Dependencies ===
            step4 = st.empty()
            with step4:
                render_stepper(5, "Analyzing Dependencies", "active", "Scanning for conflicts and constraints")
                
            compat_result = check_compatibility(
                search_result["repo_url"],
                health_result.get("owner"),
                health_result.get("repo"),
            )
            st.session_state.coral_queries = coral.get_query_count()
            
            status_label = "Clean" if compat_result.get("clean") else f"{len(compat_result.get('conflicts', []))} conflicts detected"
            with step4:
                render_stepper(5, "Dependencies Analyzed", "done", status_label)

            # === Step 5: Conflict Resolution ===
            step5 = st.empty()
            with step5:
                render_stepper(6, "Generating Environment", "active", "Selecting base image and drafting Dockerfile (Groq Llama 3)")
                
            req_content = ""
            for name, content in compat_result.get("dep_files", {}).items():
                if "requirement" in name.lower() and name.endswith('.txt'):
                    req_content = content
                    break
                    
            resolver_result = resolve_conflicts(
                req_content,
                compat_result.get("conflicts", []),
                paper_json.get("title", ""),
                entrypoint=compat_result.get("entrypoint"),
                readme_content=compat_result.get("readme_content", ""),
                dep_files=compat_result.get("dep_files", {}),
                repo_year=health_result.get("signals", {}).get("year") if health_result else None,
            )
            total_llm_calls["groq_llama"] += 1
            st.session_state.coral_queries = coral.get_query_count()

            with step5:
                base_img = resolver_result.get('selected_base_image', 'Unknown')
                render_stepper(6, "Environment Generated", "done", f"Base: {base_img}")

        else:
            # === Step 6: Generate from Scratch ===
            step6 = st.empty()
            with step6:
                render_stepper(4, "Generating from Scratch", "active", "Cross-repo JOINs and Gemini 2.0 Flash")
                
            generator_result = generate_from_scratch(paper_json)
            st.session_state.coral_queries = coral.get_query_count()
            total_llm_calls.update(generator_result.get("llm_calls", {}))
            
            with step6:
                render_stepper(4, "Code & Dockerfile Generated", "done", f"Synthesized from {len(generator_result.get('related_repos', []))} related repos")

        # === Step 7: Final Output ===
        final_output = build_output(
            paper_json=paper_json,
            search_result=search_result,
            health_result=health_result,
            compat_result=compat_result,
            resolver_result=resolver_result,
            generator_result=generator_result,
            coral_queries_total=coral.get_query_count(),
            llm_calls=total_llm_calls,
        )
        save_output(final_output)

        st.session_state.coral_queries = coral.get_query_count()
        st.session_state.llm_calls = total_llm_calls
        st.session_state.final_output = final_output
        st.session_state.paper_json = paper_json

    if "final_output" in st.session_state:
        render_results(st.session_state.final_output, st.session_state.get("paper_json"))

else:
    if "final_output" in st.session_state:
        st.info(f"Showing results from last analyzed paper: **{st.session_state.final_output['paper']['title']}**")
        render_results(st.session_state.final_output, st.session_state.get("paper_json"))
    else:
        # Empty State
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(3)
        
        features = [
            ("🔍", "Intelligent Discovery", "Automatically cross-references PapersWithCode and GitHub to find the most official, highly-rated implementation."),
            ("🛡️", "Conflict Resolution", "Uses advanced RAG and LLM reasoning to detect and fix dependency hell before you ever type 'docker build'."),
            ("🐳", "Production Ready", "Generates optimized Dockerfiles with precise base images, proper caching layers, and minimal attack surfaces.")
        ]
        
        for col, (icon, title, desc) in zip(cols, features):
            with col:
                st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <h3 style="margin: 0 0 1rem 0; font-size: 1.25rem; font-weight: 600;">{title}</h3>
                    <p style="margin: 0; color: #94a3b8; line-height: 1.6;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)