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
from agents.pwc_mcp_client import get_pwc_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(
    page_title="Coralaus — Research Paper Implementation Finder",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }

    .main-header h1 {
        color: #d4af37;
        font-size: 2.5rem;
        margin: 0;
    }

    .main-header p {
        color: #93a1a1;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .health-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 24px;
        font-weight: 600;
        font-size: 1rem;
    }

    .health-green { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .health-yellow { background: rgba(234, 179, 8, 0.15); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.3); }
    .health-orange { background: rgba(249, 115, 22, 0.15); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.3); }
    .health-red { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

    .stat-card {
        background: rgba(15, 32, 39, 0.6);
        border: 1px solid rgba(212, 175, 55, 0.15);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #d4af37;
    }

    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #586e75;
        margin-top: 4px;
    }

    .step-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .step-active { background: rgba(42, 161, 152, 0.15); color: #2aa198; border: 1px solid rgba(42, 161, 152, 0.3); }
    .step-done { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .step-pending { background: rgba(88, 110, 117, 0.15); color: #586e75; border: 1px solid rgba(88, 110, 117, 0.3); }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🚢 Coralaus</h1>
    <p>Upload a research paper PDF → Find implementations → Check health → Resolve conflicts → Get a working Dockerfile</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Check environment
    gemini_ok = bool(os.environ.get("GEMINI_API_KEY"))
    groq_ok = bool(os.environ.get("GROQ_API_KEY"))
    github_ok = bool(os.environ.get("GITHUB_TOKEN"))

    st.markdown("**API Status:**")
    st.markdown(f"{'✅' if gemini_ok else '❌'} Gemini API Key")
    st.markdown(f"{'✅' if groq_ok else '❌'} Groq API Key")
    st.markdown(f"{'✅' if github_ok else '❌'} GitHub Token")

    coral = get_coral_client()
    st.markdown(f"{'✅' if coral.available else '⚠️'} Coral CLI {'(ready)' if coral.available else '(fallback mode)'}")

    pwc = get_pwc_client()
    st.markdown(f"{'✅' if pwc.available else '❌'} PapersWithCode API")

    st.divider()
    st.markdown("### 📊 Query Counter")
    if "coral_queries" not in st.session_state:
        st.session_state.coral_queries = 0
    if "pwc_calls" not in st.session_state:
        st.session_state.pwc_calls = 0
    if "llm_calls" not in st.session_state:
        st.session_state.llm_calls = {"gemini_flash": 0, "groq_llama": 0}

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Coral SQL", st.session_state.coral_queries)
    with col2:
        st.metric("PWC API", st.session_state.pwc_calls)

    st.divider()

# --- Main Content ---

# File upload
uploaded_file = st.file_uploader(
    "📄 Upload a Research Paper (PDF)",
    type=["pdf"],
    help="Upload an ML/AI research paper PDF to analyze",
)

if uploaded_file:
    # Save uploaded file
    os.makedirs("output", exist_ok=True)
    pdf_path = os.path.join("output", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.success(f"Uploaded: {uploaded_file.name} ({len(uploaded_file.getvalue()) / 1024:.1f} KB)")

    # Run pipeline button
    if st.button(" Analyze Paper", type="primary", use_container_width=True):
        # Initialize tracking
        coral.reset_query_count()
        pwc.reset_call_count()
        total_llm_calls = {"gemini_flash": 0, "groq_llama": 0}

        # === Component 1: PDF Ingestion ===
        with st.status("📄 Component 1: Parsing PDF with GROBID...", expanded=True) as status:
            try:
                paper_json = parse_paper(pdf_path, os.path.join("output", "current_paper.json"))
                st.write(f"**Title:** {paper_json.get('title', 'Unknown')}")
                st.write(f"**Authors:** {', '.join(a.get('last', '') for a in paper_json.get('authors', [])[:5])}")
                st.write(f"**Year:** {paper_json.get('year', 'N/A')}")
                st.write(f"**Sections:** {len(paper_json.get('sections', []))}")
                st.write(f"**References:** {len(paper_json.get('references', []))}")
                st.write(f"**Parsed by:** {paper_json.get('parsed_by', 'unknown')} (no LLM!)")
                status.update(label="✅ Component 1: PDF parsed successfully", state="complete")
            except Exception as e:
                st.error(f"PDF parsing failed: {e}")
                status.update(label="❌ Component 1: PDF parsing failed", state="error")
                st.stop()

        # === Component 2: PapersWithCode Search ===
        with st.status("🔍 Component 2: Searching PapersWithCode...", expanded=True) as status:
            search_result = search_implementation(paper_json)
            st.session_state.pwc_calls = pwc.get_call_count()

            if search_result["found"]:
                st.write(f"**Found:** {search_result['repo_url']}")
                st.write(f"**Stars:** ⭐ {search_result.get('stars', 'N/A')}")
                st.write(f"**Official:** {'Yes' if search_result.get('is_official') else 'No'}")
                st.write(f"**Method:** {search_result.get('search_method', 'N/A')}")
                status.update(label="✅ Component 2: Implementation found!", state="complete")
            else:
                st.write("No implementation found on PapersWithCode")
                status.update(label="⚠️ Component 2: No implementation found", state="complete")

        health_result = None
        compat_result = None
        resolver_result = None
        generator_result = None

        if search_result["found"]:
            # === Component 3: Repo Health Check ===
            with st.status("🏥 Component 3: Checking repo health via Coral GitHub...", expanded=True) as status:
                health_result = check_repo_health(search_result["repo_url"])
                st.session_state.coral_queries = coral.get_query_count()

                score = health_result.get("health_score", -1)
                emoji = health_result.get("health_emoji", "⚪")
                label = health_result.get("health_label", "Unknown")

                health_class = "green" if score >= 80 else "yellow" if score >= 60 else "orange" if score >= 30 else "red"
                st.markdown(f'<div class="health-badge health-{health_class}">{emoji} {label} — {score}/100</div>', unsafe_allow_html=True)

                signals = health_result.get("signals", {})
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Last Commit", f"{signals.get('last_commit_days_ago', '?')}d ago")
                with cols[1]:
                    st.metric("Open Issues", signals.get("open_issues", "?"))
                with cols[2]:
                    st.metric("Stars", signals.get("stars", "?"))
                with cols[3]:
                    st.metric("Archived", "Yes" if signals.get("archived") else "No")

                status.update(label=f"✅ Component 3: Health {emoji} {score}/100", state="complete")

            # === Component 4: Dependency Check ===
            with st.status("📦 Component 4: Checking dependencies...", expanded=True) as status:
                compat_result = check_compatibility(
                    search_result["repo_url"],
                    health_result.get("owner"),
                    health_result.get("repo"),
                )
                st.session_state.coral_queries = coral.get_query_count()

                if compat_result.get("clean"):
                    st.write("✅ No dependency conflicts detected")
                else:
                    st.write(f"⚠️ {len(compat_result.get('conflicts', []))} conflicts found:")
                    for c in compat_result.get("conflicts", [])[:5]:
                        st.write(f"  - {c}")

                status_label = "Clean" if compat_result.get("clean") else f"{len(compat_result.get('conflicts', []))} conflicts"
                status.update(
                    label=f"✅ Component 4: {status_label}",
                    state="complete"
                )

            # === Component 5: Conflict Resolution (if needed) ===
            if not compat_result.get("clean"):
                with st.status("🔧 Component 5: Resolving conflicts with Groq...", expanded=True) as status:
                    req_content = ""
                    for name, content in compat_result.get("dep_files", {}).items():
                        if "requirement" in name.lower():
                            req_content = content
                            break

                    resolver_result = resolve_conflicts(
                        req_content,
                        compat_result.get("conflicts", []),
                        paper_json.get("title", ""),
                        entrypoint=compat_result.get("entrypoint"),
                    )
                    total_llm_calls["groq_llama"] += 1

                    if resolver_result.get("dockerfile"):
                        st.write("✅ Dockerfile generated")
                        with st.expander("View Dockerfile"):
                            st.code(resolver_result["dockerfile"], language="dockerfile")

                    status.update(label="✅ Component 5: Conflicts resolved", state="complete")
            else:
                # Generate basic Dockerfile for clean repos too
                req_content = ""
                for name, content in compat_result.get("dep_files", {}).items():
                    if "requirement" in name.lower():
                        req_content = content
                        break
                if req_content:
                    resolver_result = resolve_conflicts(
                        req_content, [], paper_json.get("title", ""),
                        entrypoint=compat_result.get("entrypoint"),
                    )

        else:
            # === Component 6: Generate from Scratch ===
            with st.status("🔨 Component 6: Generating implementation from scratch...", expanded=True) as status:
                st.write("No implementation found — generating reference code using Gemini + Coral cross-repo JOINs")
                generator_result = generate_from_scratch(paper_json)
                st.session_state.coral_queries = coral.get_query_count()
                total_llm_calls.update(generator_result.get("llm_calls", {}))

                if generator_result.get("keyterms"):
                    st.write(f"**Keywords:** {', '.join(generator_result['keyterms'][:5])}")
                if generator_result.get("related_repos"):
                    st.write(f"**Related repos:** {', '.join(generator_result['related_repos'][:3])}")

                st.write(f"**Implementation:** {len(generator_result.get('implementation_py', ''))} chars generated")

                status.update(label="✅ Component 6: Implementation generated", state="complete")

        # === Component 7: Final Output ===
        with st.status("📋 Component 7: Assembling output...", expanded=True) as status:
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
            saved = save_output(final_output)

            st.session_state.coral_queries = coral.get_query_count()
            st.session_state.llm_calls = total_llm_calls

            status.update(label="✅ Component 7: Output assembled", state="complete")

        # === Results Display ===
        st.divider()
        st.markdown("## 📊 Results")

        # Summary
        st.text(format_summary(final_output))

        # Download buttons
        st.markdown("### 📥 Downloads")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.download_button(
                "📄 Paper JSON",
                data=json.dumps(paper_json, indent=2),
                file_name="current_paper.json",
                mime="application/json",
            )

        with col2:
            st.download_button(
                "📊 Full Result",
                data=json.dumps(final_output, indent=2),
                file_name="result.json",
                mime="application/json",
            )

        with col3:
            if final_output.get("dockerfile"):
                st.download_button(
                    "🐳 Dockerfile",
                    data=final_output["dockerfile"],
                    file_name="Dockerfile",
                    mime="text/plain",
                )

        with col4:
            if final_output.get("implementation_script"):
                st.download_button(
                    "🐍 Implementation",
                    data=final_output["implementation_script"],
                    file_name="implementation.py",
                    mime="text/x-python",
                )

        # Collapsible raw data
        with st.expander("🔍 Full Output JSON"):
            st.json(final_output)

else:
    # No file uploaded — show instructions
    st.markdown("### 🗺️ How It Works")

    cols = st.columns(4)
    steps = [
        ("🔍", "Find Code", "PapersWithCode MCP + Coral GitHub search"),
        ("🏥", "Health Check", "Coral SQL queries score the repo 0–100"),
        ("🐳", "Get Dockerfile", "Conflicts resolved, Dockerfile generated"),
    ]

    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 2rem">{icon}</div>
                <div class="stat-value" style="font-size: 1rem; margin-top: 8px">{title}</div>
                <div class="stat-label" style="font-size: 0.7rem; text-transform: none; margin-top: 4px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.info("👆 Upload a research paper PDF to get started!")
