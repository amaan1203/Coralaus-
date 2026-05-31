"""
coralaus — Streamlit UI  (Premium Redesign)
"""

import os
import sys
import json
import logging
import streamlit as st
from pathlib import Path

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
from agents.dockerfile_validator import validate_dockerfile, format_validation_summary

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Coralaus — ML Paper → Docker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL CSS + JS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ─── RESET & BASE ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body {
    margin: 0; padding: 0;
    background: #030712 !important;
    font-family: 'Space Grotesk', 'Outfit', sans-serif;
    color: #e2e8f0;
    overflow-x: hidden;
    font-size: 16px;
    letter-spacing: -0.01em;
}

/* Streamlit scaffolding overrides */
.stApp {
    background: transparent !important;
    font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
}
/* Scope font rules to page content only — never touch Streamlit's own chrome */
/* Exclude stIconMaterial spans: those use the Material Symbols ligature font;
   overriding font-family on them renders icon glyphs as raw text (e.g. "upload") */
.main p, .main div, .main li, .main a, .main label,
.block-container p, .block-container div,
.block-container li, .block-container a, .block-container label {
    font-family: 'Space Grotesk', 'Outfit', sans-serif;
}
.main span:not([data-testid="stIconMaterial"]),
.block-container span:not([data-testid="stIconMaterial"]) {
    font-family: 'Space Grotesk', 'Outfit', sans-serif;
}
h1, h2, h3, h4, h5, h6 { font-family: 'Outfit', sans-serif !important; letter-spacing: -0.03em; }

/* ─── RESTORE STREAMLIT TOOLBAR / HEADER ───────────────── */
[data-testid="stToolbar"],
[data-testid="stToolbar"] *,
header[data-testid="stHeader"],
header[data-testid="stHeader"] * {
    white-space: nowrap !important;
    word-break: normal !important;
    overflow-wrap: normal !important;
    font-family: inherit;
    display: revert;
    flex-wrap: nowrap;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
    position: relative;
    z-index: 1;
}
[data-testid="stSidebar"] { position: relative; z-index: 10; }
[data-testid="stSidebar"] > div:first-child {
    background: rgba(3, 7, 18, 0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(20px);
}

/* ─── CURSOR AURA (fixed, painted on body via JS) ──────────── */
/* We paint directly on body background via JS — no extra div needed */

/* ─── GRID BACKGROUND ──────────────────────────────────────── */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
        linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}

/* ─── SCROLLBAR ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 99px; }

/* ─── HERO / HEADER ─────────────────────────────────────────── */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-8px); }
}
@keyframes shimmer-text {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes badgePop {
    from { opacity: 0; transform: scale(0.8) translateY(10px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(99,102,241,0.25); box-shadow: 0 0 0 0 rgba(99,102,241,0.1); }
    50%       { border-color: rgba(99,102,241,0.5); box-shadow: 0 0 30px 4px rgba(99,102,241,0.12); }
}

.hero {
    position: relative;
    padding: 4rem 3rem 3.5rem;
    border-radius: 28px;
    margin-bottom: 2.5rem;
    background: rgba(9, 12, 25, 0.7);
    border: 1px solid rgba(99,102,241,0.2);
    backdrop-filter: blur(24px);
    overflow: hidden;
    animation: borderGlow 5s ease-in-out infinite;
}

/* Multi-colour blobs inside hero */
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(99,102,241,0.22) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    filter: blur(40px);
    animation: float 8s ease-in-out infinite;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; right: -40px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(14,165,233,0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    filter: blur(50px);
    animation: float 11s ease-in-out infinite reverse;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 18px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 700;
    color: #a5b4fc;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.6rem;
    font-family: 'Outfit', sans-serif !important;
    animation: badgePop 0.6s ease both;
}
.hero-badge .dot-pulse {
    width: 7px; height: 7px; border-radius: 50%;
    background: #6366f1;
    box-shadow: 0 0 8px #6366f1;
    animation: pulseDot 1.8s ease-in-out infinite;
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}

.hero-title {
    font-size: clamp(4rem, 8vw, 7.5rem);
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -0.04em;
    margin: 0 0 1.2rem 0;
    background: linear-gradient(135deg,
        #ffffff 0%,
        #c7d2fe 30%,
        #818cf8 55%,
        #38bdf8 80%,
        #ffffff 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer-text 5s linear infinite, fadeUp 0.7s ease both 0.1s;
}

.hero-sub {
    font-size: 1.25rem;
    color: #94a3b8;
    font-weight: 400;
    line-height: 1.75;
    max-width: 640px;
    margin: 0;
    font-family: 'Space Grotesk', sans-serif;
    animation: fadeUp 0.7s ease both 0.25s;
}

.hero-chips {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 2rem;
    animation: fadeUp 0.7s ease both 0.4s;
}
.chip {
    display: flex; align-items: center; gap: 6px;
    padding: 8px 18px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 99px;
    font-size: 0.88rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.01em;
    transition: all 0.25s;
}
.chip:hover {
    background: rgba(99,102,241,0.12);
    border-color: rgba(99,102,241,0.35);
    color: #c7d2fe;
    transform: translateY(-1px);
}

/* ─── GLASS CARD ───────────────────────────────────────────── */
.glass-card {
    background: rgba(9, 12, 25, 0.6);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.8rem;
    backdrop-filter: blur(16px);
    transition: border-color 0.3s, box-shadow 0.3s, transform 0.3s;
}
.glass-card:hover {
    border-color: rgba(99,102,241,0.3);
    box-shadow: 0 0 40px -10px rgba(99,102,241,0.15);
    transform: translateY(-3px);
}

/* ─── SIDEBAR ──────────────────────────────────────────────── */
.sys-label {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 14px;
    margin-top: 24px;
    font-family: 'Outfit', sans-serif;
}
.status-row {
    display: flex; align-items: center; gap: 10px;
    padding: 11px 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    margin-bottom: 7px;
    font-size: 0.95rem;
    font-weight: 500;
    color: #94a3b8;
    letter-spacing: -0.01em;
    transition: background 0.2s, border-color 0.2s;
}
.status-row:hover {
    background: rgba(99,102,241,0.07);
    border-color: rgba(99,102,241,0.18);
}
.led {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.led.g { background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,0.7); }
.led.r { background: #ef4444; box-shadow: 0 0 8px rgba(239,68,68,0.7); }
.led.y { background: #eab308; box-shadow: 0 0 8px rgba(234,179,8,0.7); }

.metric-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-box:hover {
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.25);
}
.metric-num   { font-size: 2.5rem; font-weight: 900; color: #f1f5f9; line-height: 1.1; }
.metric-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; margin-top: 8px; }

/* ─── PIPELINE STEPPER ─────────────────────────────────────── */
@keyframes pulseRing {
    0%   { box-shadow: 0 0 0 0 rgba(99,102,241,0.7); }
    70%  { box-shadow: 0 0 0 14px rgba(99,102,241,0); }
    100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
}
@keyframes stepSlide {
    from { opacity: 0; transform: translateX(-22px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes countFlash {
    0%   { color: #f1f5f9; transform: scale(1); }
    30%  { color: #6366f1; transform: scale(1.18); text-shadow: 0 0 20px rgba(99,102,241,0.8); }
    100% { color: #f1f5f9; transform: scale(1); }
}
@keyframes repoSlideIn {
    from { opacity: 0; transform: translateX(-30px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes ragPulse {
    0%, 100% { opacity: 0.4; transform: scaleX(0.95); }
    50%       { opacity: 1;   transform: scaleX(1); }
}

.step-card {
    display: flex; align-items: flex-start; gap: 18px;
    padding: 18px 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 10px;
    backdrop-filter: blur(14px);
    animation: stepSlide 0.45s cubic-bezier(0.34,1.2,0.64,1) both;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.step-card.active {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(14,165,233,0.04));
    border-color: rgba(99,102,241,0.3);
    box-shadow: 0 0 30px -8px rgba(99,102,241,0.2);
}
.step-card.done {
    background: linear-gradient(135deg, rgba(34,197,94,0.06), rgba(9,12,25,0.6));
    border-color: rgba(34,197,94,0.18);
}
.step-card.error {
    background: rgba(239,68,68,0.06);
    border-color: rgba(239,68,68,0.25);
}
.step-card.pending {
    background: rgba(255,255,255,0.02);
    border-color: rgba(255,255,255,0.04);
}

.step-icon {
    width: 38px; height: 38px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; font-weight: 800; flex-shrink: 0;
    margin-top: 2px;
    font-family: 'Outfit', sans-serif;
}
.step-card.active  .step-icon { background: linear-gradient(135deg,#6366f1,#4f46e5); color: #fff; animation: pulseRing 1.4s infinite; }
.step-card.done    .step-icon { background: linear-gradient(135deg,#22c55e,#16a34a); color: #fff; }
.step-card.error   .step-icon { background: linear-gradient(135deg,#ef4444,#dc2626); color: #fff; }
.step-card.pending .step-icon { background: rgba(255,255,255,0.06); color: #475569; }

.step-title  { font-size: 1.05rem; font-weight: 700; color: #f1f5f9; font-family: 'Outfit', sans-serif; letter-spacing: -0.02em; }
.step-detail { font-size: 0.88rem; color: #64748b; margin-top: 5px; line-height: 1.6; font-family: 'Space Grotesk', sans-serif; }

/* ─── REPO LIST ANIMATION ──────────────────────────────── */
.repo-list-wrap {
    margin-top: 12px;
    display: flex; flex-direction: column; gap: 6px;
    max-height: 260px; overflow: hidden;
}
.repo-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 14px;
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.14);
    border-radius: 10px;
    font-size: 0.88rem; font-family: 'JetBrains Mono', monospace;
    color: #a5b4fc;
    opacity: 0;
    animation: repoSlideIn 0.4s ease forwards;
}
.repo-row .repo-stars {
    margin-left: auto; font-size: 0.78rem; color: #64748b;
    font-family: 'Space Grotesk', sans-serif;
}

/* ─── RAG PIPELINE VISUALIZATION ──────────────────────── */
.rag-pipeline {
    display: flex; flex-direction: column; gap: 8px;
    margin-top: 12px;
    padding: 16px 18px;
    background: rgba(9,12,25,0.7);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px;
}
.rag-stage {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 0.9rem; font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    transition: all 0.4s;
}
.rag-stage.waiting  { background: rgba(255,255,255,0.02); color: #475569; border: 1px solid rgba(255,255,255,0.04); }
.rag-stage.active   { background: rgba(99,102,241,0.1);  color: #c7d2fe;  border: 1px solid rgba(99,102,241,0.25); }
.rag-stage.done     { background: rgba(34,197,94,0.07);  color: #86efac;  border: 1px solid rgba(34,197,94,0.18); }
.rag-bar {
    flex: 1; height: 3px; border-radius: 99px;
    background: rgba(255,255,255,0.06); overflow: hidden; position: relative;
}
.rag-bar-fill {
    position: absolute; top: 0; height: 100%; width: 55%;
    background: linear-gradient(90deg, transparent, #6366f1, #0ea5e9, transparent);
    animation: ragPulse 1.2s ease-in-out infinite, barFlow 1.4s linear infinite;
    border-radius: 99px;
}
.metric-num.flash { animation: countFlash 0.55s ease; }

/* ─── ORBITAL LOADER ───────────────────────────────────────── */
@keyframes spin1 { from { transform: rotate(0deg); }   to { transform: rotate(360deg); } }
@keyframes spin2 { from { transform: rotate(0deg); }   to { transform: rotate(-360deg); } }
@keyframes corePulse {
    0%, 100% { transform: translate(-50%,-50%) scale(1); opacity: 1; }
    50%       { transform: translate(-50%,-50%) scale(0.75); opacity: 0.6; }
}
@keyframes barFlow {
    0%   { left: -60%; }
    100% { left: 120%; }
}

.loader-shell {
    display: flex; flex-direction: column; align-items: center;
    padding: 2.4rem 2rem;
    background: rgba(9,12,25,0.85);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 22px;
    backdrop-filter: blur(20px);
    margin: 0.75rem 0;
}
.orbit-wrap {
    position: relative; width: 72px; height: 72px; margin-bottom: 1.4rem;
}
.orbit-ring {
    position: absolute; inset: 0;
    border-radius: 50%;
    border: 2px solid transparent;
}
.orbit-ring.r1 {
    border-top-color: #6366f1;
    border-right-color: rgba(99,102,241,0.2);
    animation: spin1 1.4s linear infinite;
}
.orbit-ring.r2 {
    inset: 10px;
    border-top-color: #0ea5e9;
    border-left-color: rgba(14,165,233,0.2);
    animation: spin2 1.0s linear infinite;
}
.orbit-ring.r3 {
    inset: 20px;
    border-bottom-color: #8b5cf6;
    border-right-color: rgba(139,92,246,0.2);
    animation: spin1 1.8s linear infinite;
}
.orbit-core {
    position: absolute; top: 50%; left: 50%;
    width: 16px; height: 16px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #0ea5e9);
    box-shadow: 0 0 20px rgba(99,102,241,0.8);
    animation: corePulse 1.6s ease-in-out infinite;
}
.loader-title { font-size: 1.15rem; font-weight: 700; color: #f1f5f9; text-align: center; font-family: 'Outfit', sans-serif; letter-spacing: -0.02em; }
.loader-sub   { font-size: 0.88rem; color: #64748b; text-align: center; margin-top: 6px; font-family: 'Space Grotesk', sans-serif; }
.progress-track {
    width: 180px; height: 2px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px; margin-top: 1.2rem;
    overflow: hidden; position: relative;
}
.progress-fill {
    position: absolute; top: 0; height: 100%; width: 60%;
    background: linear-gradient(90deg, transparent, #6366f1, #0ea5e9, transparent);
    animation: barFlow 1.3s ease-in-out infinite;
    border-radius: 99px;
}

/* ─── CONFLICT CARDS ───────────────────────────────────────── */
@keyframes cardIn {
    from { opacity: 0; transform: translateY(20px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.conflict-header {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.78rem; font-weight: 800; letter-spacing: 0.12em;
    text-transform: uppercase; color: #64748b;
    margin: 2rem 0 1.2rem 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-family: 'Outfit', sans-serif;
}

.ccard {
    position: relative; overflow: hidden;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 10px;
    border-left: 3px solid;
    backdrop-filter: blur(10px);
    animation: cardIn 0.45s ease both;
    transition: transform 0.25s, box-shadow 0.25s;
}
.ccard:hover {
    transform: translateX(5px);
    box-shadow: -4px 0 20px -4px currentColor;
}
.ccard.err  { border-color: #ef4444; background: linear-gradient(120deg, rgba(239,68,68,0.08), rgba(9,12,25,0.9)); }
.ccard.warn { border-color: #eab308; background: linear-gradient(120deg, rgba(234,179,8,0.07), rgba(9,12,25,0.9)); }
.ccard.ok   { border-color: #22c55e; background: linear-gradient(120deg, rgba(34,197,94,0.07), rgba(9,12,25,0.9)); }

.ctag {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.75rem; font-weight: 800; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 3px 11px;
    border-radius: 99px; margin-bottom: 10px;
    font-family: 'Outfit', sans-serif;
}
.ctag.err  { background: rgba(239,68,68,0.12); color: #fca5a5; border: 1px solid rgba(239,68,68,0.2); }
.ctag.warn { background: rgba(234,179,8,0.12);  color: #fde68a; border: 1px solid rgba(234,179,8,0.2); }
.ctag.ok   { background: rgba(34,197,94,0.12);  color: #86efac; border: 1px solid rgba(34,197,94,0.2); }

.ctext { font-size: 0.95rem; color: #cbd5e1; font-family: 'JetBrains Mono', monospace; line-height: 1.65; }

.divider-arrow {
    display: flex; align-items: center; gap: 8px;
    margin: 10px 0;
    color: #334155; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
}
.divider-arrow::before, .divider-arrow::after {
    content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.05);
}

.resolution-box {
    padding: 10px 14px;
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.12);
    border-radius: 10px;
    font-size: 0.85rem; color: #86efac; line-height: 1.5;
    font-family: 'JetBrains Mono', monospace;
}
.warn-box {
    padding: 10px 14px;
    background: rgba(234,179,8,0.06);
    border: 1px solid rgba(234,179,8,0.12);
    border-radius: 10px;
    font-size: 0.85rem; color: #fde68a; line-height: 1.5;
}

.summary-pills {
    display: flex; gap: 8px; flex-wrap: wrap;
    margin-bottom: 1.4rem; padding: 14px 18px;
    background: rgba(9,12,25,0.8);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
}
.pill {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 13px; border-radius: 99px;
    font-size: 0.78rem; font-weight: 600;
}
.pill.r { background: rgba(239,68,68,0.1);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.18); }
.pill.y { background: rgba(234,179,8,0.1);  color: #fde68a; border: 1px solid rgba(234,179,8,0.18); }
.pill.g { background: rgba(34,197,94,0.1);  color: #86efac; border: 1px solid rgba(34,197,94,0.18); }

/* ─── COMPLETE BANNER ──────────────────────────────────────── */
@keyframes completePop {
    0%   { opacity: 0; transform: scale(0.96) translateY(-8px); }
    60%  { transform: scale(1.01) translateY(0); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}
.complete-banner {
    display: flex; align-items: center; gap: 16px;
    padding: 1.2rem 1.8rem;
    background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(99,102,241,0.06));
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 18px;
    margin-bottom: 2rem;
    animation: completePop 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.complete-icon { font-size: 2.4rem; }
.complete-title { font-size: 1.3rem; font-weight: 800; color: #f1f5f9; font-family: 'Outfit', sans-serif; letter-spacing: -0.02em; }
.complete-sub   { font-size: 0.92rem; color: #64748b; margin-top: 4px; font-family: 'Space Grotesk', sans-serif; }

/* ─── HEALTH BADGE ─────────────────────────────────────────── */
.health-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 18px; border-radius: 99px;
    font-weight: 600; font-size: 0.9rem; margin-top: 1rem;
}
.hp-g { background: rgba(34,197,94,0.1);  color: #22c55e; border: 1px solid rgba(34,197,94,0.25); }
.hp-y { background: rgba(234,179,8,0.1);  color: #eab308; border: 1px solid rgba(234,179,8,0.25); }
.hp-o { background: rgba(249,115,22,0.1); color: #f97316; border: 1px solid rgba(249,115,22,0.25); }
.hp-r { background: rgba(239,68,68,0.1);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }

/* ─── FEATURE CARDS (empty state) ─────────────────────────── */
@keyframes featureIn {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fcard {
    background: rgba(9,12,25,0.65);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 22px; padding: 2rem;
    height: 100%;
    backdrop-filter: blur(14px);
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    animation: featureIn 0.6s ease both;
}
.fcard:nth-child(2) { animation-delay: 0.1s; }
.fcard:nth-child(3) { animation-delay: 0.2s; }
.fcard:hover {
    transform: translateY(-8px);
    border-color: rgba(99,102,241,0.35);
    box-shadow: 0 30px 60px -20px rgba(99,102,241,0.18), 0 0 0 1px rgba(99,102,241,0.1);
    background: rgba(16,20,40,0.85);
}
.fcard-icon {
    font-size: 2.2rem; margin-bottom: 1.1rem;
    display: block; line-height: 1;
    filter: drop-shadow(0 0 12px rgba(99,102,241,0.5));
}
.fcard h3 { font-size: 1.25rem; font-weight: 800; margin: 0 0 0.8rem 0; color: #f1f5f9; font-family: 'Outfit', sans-serif; letter-spacing: -0.02em; }
.fcard p  { font-size: 0.96rem; color: #64748b; line-height: 1.75; margin: 0; font-family: 'Space Grotesk', sans-serif; }

/* ─── FILE UPLOADER — all rules removed for debugging ──────── */
/* Fix Streamlit's default info text at top */
.stAlert p, .stAlert span { font-size: 0.95rem !important; }
/* Global text scale-up */
.stMarkdown p, .stMarkdown li { font-size: 1rem !important; line-height: 1.7 !important; }
[data-testid="stText"] { font-size: 1rem !important; }
[data-testid="stCaption"] { font-size: 0.9rem !important; }
label[data-testid="stWidgetLabel"] > div { font-size: 1rem !important; font-weight: 600 !important; }

/* ─── TABS ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    padding: 12px 24px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px 12px 0 0;
    border: 1px solid rgba(255,255,255,0.05); border-bottom: none;
    color: #64748b; font-size: 0.95rem; font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.01em;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.1) !important;
    color: #a5b4fc !important;
    border-color: rgba(99,102,241,0.22) !important;
}

/* ─── CODE BLOCKS ──────────────────────────────────────────── */
.stCodeBlock { border-radius: 14px !important; border: 1px solid rgba(255,255,255,0.07) !important; }
.stCodeBlock code { font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important; }



/* ─── PRIMARY BUTTON ───────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 1.8rem !important;
    box-shadow: 0 4px 24px -4px rgba(99,102,241,0.55) !important;
    transition: all 0.22s !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.01em !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 32px -4px rgba(99,102,241,0.7) !important;
}
</style>

<!-- ══ CURSOR AURA JS — writes directly to body background ══ -->
<script>
(function () {
    var mouseX = window.innerWidth / 2;
    var mouseY = window.innerHeight / 2;
    var curX   = mouseX;
    var curY   = mouseY;

    document.addEventListener('mousemove', function(e) {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    function lerp(a, b, t) { return a + (b - a) * t; }

    function updateAura() {
        curX = lerp(curX, mouseX, 0.07);
        curY = lerp(curY, mouseY, 0.07);

        var px = curX + 'px';
        var py = curY + 'px';

        /* Paint directly on body — visible through every Streamlit layer */
        document.body.style.background = [
            'radial-gradient(ellipse 700px 500px at ' + px + ' ' + py + ', rgba(99,102,241,0.13) 0%, rgba(14,165,233,0.05) 40%, transparent 70%)',
            'radial-gradient(ellipse 280px 200px at ' + px + ' ' + py + ', rgba(139,92,246,0.10) 0%, transparent 55%)',
            '#030712'
        ].join(', ');

        requestAnimationFrame(updateAura);
    }

    /* Wait for DOM — Streamlit loads content slightly after script */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', updateAura);
    } else {
        updateAura();
    }

    /* ── Floating particles ── */
    function spawnParticle() {
        var p = document.createElement('div');
        var size = Math.random() * 3.5 + 1.5;
        var x    = Math.random() * window.innerWidth;
        var dur  = Math.random() * 16 + 10;
        var delay= Math.random() * 4;
        var colors = [
            'rgba(99,102,241,0.55)',
            'rgba(139,92,246,0.5)',
            'rgba(14,165,233,0.5)',
        ];
        var color = colors[Math.floor(Math.random() * colors.length)];

        p.style.cssText = [
            'position:fixed',
            'width:' + size + 'px',
            'height:' + size + 'px',
            'left:' + x + 'px',
            'bottom:-10px',
            'border-radius:50%',
            'pointer-events:none',
            'z-index:0',
            'background:' + color,
            'box-shadow: 0 0 ' + (size * 3) + 'px ' + color,
            'opacity:0',
            'transition:none',
            'animation: _rise ' + dur + 's ' + delay + 's linear forwards'
        ].join(';');

        document.body.appendChild(p);
        setTimeout(function() { if (p.parentNode) p.parentNode.removeChild(p); },
            (dur + delay + 1) * 1000);
    }

    /* Inject particle keyframe once */
    if (!document.getElementById('_particle_style')) {
        var s = document.createElement('style');
        s.id  = '_particle_style';
        s.textContent = '@keyframes _rise { 0%{transform:translateY(0) scale(0);opacity:0} 8%{opacity:0.7} 92%{opacity:0.3} 100%{transform:translateY(-110vh) scale(1.3);opacity:0} }';
        document.head.appendChild(s);
    }

    setInterval(spawnParticle, 1600);
    for (var i = 0; i < 8; i++) setTimeout(spawnParticle, i * 250);
})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════════════

def render_loader(title: str, subtitle: str = "Please wait…", substeps: list = None, extra_html: str = "") -> str:
    substeps_html = ""
    if substeps:
        substeps_html = '<div style="margin-top:16px; display:flex; flex-direction:column; gap:6px; width:100%;">'
        for i, s in enumerate(substeps):
            delay = i * 0.18
            substeps_html += (
                f'<div style="font-size:0.9rem; color:#64748b; font-family:\'Space Grotesk\',sans-serif;'
                f' opacity:0; animation: fadeUp 0.5s ease {delay}s both; display:flex; align-items:center; gap:8px;">'
                f'<span style="color:#6366f1; flex-shrink:0;">&#9656;</span> {s}</div>'
            )
        substeps_html += '</div>'
    return (
        '<div class="loader-shell">'
        '<div class="orbit-wrap">'
        '<div class="orbit-ring r1"></div>'
        '<div class="orbit-ring r2"></div>'
        '<div class="orbit-ring r3"></div>'
        '<div class="orbit-core"></div>'
        '</div>'
        f'<div class="loader-title">{title}</div>'
        f'<div class="loader-sub">{subtitle}</div>'
        '<div class="progress-track"><div class="progress-fill"></div></div>'
        + substeps_html +
        '</div>'
        + extra_html
    )


def render_stepper(step_num, title, state="active", details="", extra_html=""):
    icons = {"active": str(step_num), "done": "✓", "error": "✕", "pending": "·"}
    icon  = icons.get(state, str(step_num))
    detail_block = f'<div class="step-detail">{details}</div>' if details else ''
    extra_block  = extra_html if extra_html else ''
    st.markdown(
        '<div class="step-card ' + state + '">'
        '<div class="step-icon">' + icon + '</div>'
        '<div style="flex:1;">'
        '<div class="step-title">' + title + '</div>'
        + detail_block
        + extra_block +
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


def render_conflict_cards(final_output: dict):
    conflicts = final_output.get("conflicts_found", [])
    warnings  = final_output.get("dep_warnings", [])
    resolved  = final_output.get("conflicts_resolved", False)
    resolved_reqs = final_output.get("resolved_requirements", "")

    if not conflicts and not warnings:
        st.markdown("""
        <div class="ccard ok">
            <span class="ctag ok">✓ Clean</span>
            <div class="ctext">No dependency conflicts or warnings detected — environment is clean and ready to build.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    nc = len(conflicts)
    nw = len(warnings)
    resolved_label = "✅ All Resolved" if resolved else "⏳ Pending"
    st.markdown(f"""
    <div class="summary-pills">
        <span class="pill r">🔴 {nc} Conflict{'s' if nc != 1 else ''}</span>
        <span class="pill y">⚠️ {nw} Warning{'s' if nw != 1 else ''}</span>
        <span class="pill g">{resolved_label}</span>
    </div>
    """, unsafe_allow_html=True)

    if conflicts:
        st.markdown('<div class="conflict-header">⚡ Hard Conflicts</div>', unsafe_allow_html=True)
        for i, conflict in enumerate(conflicts):
            pkg = conflict.split()[0] if conflict else ""
            resolution_html = ""
            if resolved:
                hint = ""
                for line in resolved_reqs.splitlines():
                    if pkg.lower() in line.lower():
                        hint = line.strip()
                        break
                if not hint:
                    hint = "Version constraint adjusted and pinned by conflict resolver."
                resolution_html = f"""
                <div class="divider-arrow">→ resolved as</div>
                <div class="resolution-box">🔧 {hint}</div>
                """
            st.markdown(f"""
            <div class="ccard err" style="animation-delay:{i*0.07:.2f}s">
                <span class="ctag err">⚡ Conflict #{i+1}</span>
                <div class="ctext">{conflict}</div>
                {resolution_html}
            </div>
            """, unsafe_allow_html=True)

    if warnings:
        st.markdown('<div class="conflict-header">⚠️ Dependency Warnings</div>', unsafe_allow_html=True)
        for i, warning in enumerate(warnings):
            delay = (len(conflicts) + i) * 0.07
            st.markdown(f"""
            <div class="ccard warn" style="animation-delay:{delay:.2f}s">
                <span class="ctag warn">⚠ Warning #{i+1}</span>
                <div class="ctext">{warning}</div>
                <div class="divider-arrow">→ recommendation</div>
                <div class="warn-box">🛠 Update this dependency to a recent stable release. The generated Dockerfile pins the best compatible version available.</div>
            </div>
            """, unsafe_allow_html=True)


def render_results(final_output: dict, paper_json: dict = None):
    if not paper_json:
        p = final_output["paper"]
        paper_json = {
            "title": p["title"],
            "arxiv_id": p.get("arxiv_id"),
            "authors": p.get("authors", []),
            "year": p.get("year"),
            "abstract": p.get("abstract", ""),
        }

    title = paper_json.get("title", "Unknown Paper")

    st.markdown(f"""
    <div class="complete-banner">
        <div class="complete-icon">✨</div>
        <div>
            <div class="complete-title">Analysis Complete</div>
            <div class="complete-sub">{title}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_summary, tab_conflicts, tab_docker, tab_json, tab_dl = st.tabs([
        "Summary", "Conflicts & Warnings", "Dockerfile", "Raw JSON", "Export",
    ])

    with tab_summary:
        st.markdown(f"### {title}")
        st.info(format_summary(final_output))

        score = final_output.get("repo_health_score")
        if score is not None:
            emoji = final_output.get("health_emoji", "⚪")
            label = final_output.get("health_label", "Unknown")
            cls   = "g" if score >= 80 else "y" if score >= 60 else "o" if score >= 30 else "r"
            st.markdown(f'<div class="health-pill hp-{cls}">{emoji} Repo Health: {label} ({score}/100)</div>', unsafe_allow_html=True)
        elif final_output.get("health"):
            h     = final_output["health"]
            score = h.get("health_score", -1)
            emoji = h.get("health_emoji", "⚪")
            label = h.get("health_label", "Unknown")
            cls   = "g" if score >= 80 else "y" if score >= 60 else "o" if score >= 30 else "r"
            st.markdown(f'<div class="health-pill hp-{cls}">{emoji} Repo Health: {label} ({score}/100)</div>', unsafe_allow_html=True)

    with tab_conflicts:
        render_conflict_cards(final_output)

    with tab_docker:
        if final_output.get("dockerfile"):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Base Image", final_output.get("selected_base_image", "N/A"))
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
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("Paper JSON",      json.dumps(paper_json, indent=2),    "paper.json",          "application/json", use_container_width=True)
        with c2:
            st.download_button("Result JSON",     json.dumps(final_output, indent=2),  "result.json",         "application/json", use_container_width=True)
        with c3:
            if final_output.get("dockerfile"):
                st.download_button("Dockerfile",  final_output["dockerfile"],           "Dockerfile",          "text/plain",        use_container_width=True, type="primary")
        with c4:
            if final_output.get("implementation_script"):
                st.download_button("Implementation", final_output["implementation_script"], "implementation.py", "text/x-python", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sys-label">System Status</div>', unsafe_allow_html=True)

    gemini_ok = bool(os.environ.get("GEMINI_API_KEY"))
    groq_ok   = bool(os.environ.get("GROQ_API_KEY"))
    github_ok = bool(os.environ.get("GITHUB_TOKEN"))

    coral = get_coral_client()
    pwc   = get_s2_client()

    st.markdown(f"""
    <div class="status-row"><div class="led {'g' if gemini_ok else 'r'}"></div>Gemini API</div>
    <div class="status-row"><div class="led {'g' if groq_ok else 'r'}"></div>Groq API</div>
    <div class="status-row"><div class="led {'g' if github_ok else 'y'}"></div>GitHub Token</div>
    <div class="status-row"><div class="led {'g' if coral.available else 'y'}"></div>Coral CLI {'(Ready)' if coral.available else '(Fallback)'}</div>
    <div class="status-row"><div class="led {'g' if pwc.available else 'r'}"></div>Semantic Scholar</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sys-label" style="margin-top:24px;">Activity</div>', unsafe_allow_html=True)

    if "coral_queries" not in st.session_state: st.session_state.coral_queries = 0
    if "api_calls"     not in st.session_state: st.session_state.api_calls = 0

    metric_placeholder = st.empty()

    def update_sidebar_metrics(flash=True):
        cq = st.session_state.get('coral_queries', 0)
        ac = st.session_state.get('api_calls', 0)
        flash_class = 'class="metric-num flash"' if flash else 'class="metric-num"'
        metric_placeholder.markdown(f'''
        <div style="display:flex; gap:10px;">
            <div class="metric-box" style="flex:1;"><div {flash_class}>{cq}</div><div class="metric-label">Coral SQL</div></div>
            <div class="metric-box" style="flex:1;"><div {flash_class}>{ac}</div><div class="metric-label">API Calls</div></div>
        </div>
        ''', unsafe_allow_html=True)

    update_sidebar_metrics()


# ══════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">
        <div class="dot-pulse"></div>
        General Research Environment
    </div>
    <h1 class="hero-title">Coralaus</h1>
    <h3 style="font-weight: 600; color: #818cf8; font-style: italic; margin-top: -10px; margin-bottom: 20px; font-size: 1.8rem; letter-spacing: 0.02em;">fucking done right !</h3>
    <p class="hero-sub" style="font-size: 1.3rem;">
        From any research paper to a fully reproducible Docker environment — 
        automatically discovered, validated, and conflict-free.
    </p>
    <div class="hero-chips">
        <span class="chip">⚡ Gemini 2.0 Flash</span>
        <span class="chip">GitHub</span>
        <span class="chip">🐳 Docker-Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  MAIN FLOW
# ══════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Drop your research paper PDF here",
    type=["pdf"],
    help="Upload an ML/AI research paper PDF to begin analysis",
)

if uploaded_file:
    os.makedirs("output", exist_ok=True)
    pdf_path = os.path.join("output", uploaded_file.name)

    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        for key in ("final_output", "paper_json"):
            st.session_state.pop(key, None)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("⚡ Analyze & Generate Environment", type="primary", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if analyze_clicked:
        for key in ("final_output", "paper_json"):
            st.session_state.pop(key, None)

        coral.reset_query_count()
        pwc.reset_call_count()
        total_llm_calls = {"gemini_flash": 0, "groq_llama": 0}

        # ── Step 1 ──────────────────────────────────────────────
        ph1, ld1 = st.empty(), st.empty()
        with ph1: render_stepper(1, "Parsing PDF Structure", "active",
            "Extracting text, metadata, sections & references via GROBID · Building paper knowledge graph")
        with ld1: st.markdown(render_loader("Parsing PDF…", "GROBID structural extraction running",
            substeps=["Tokenizing raw PDF bytes", "Detecting document structure", "Extracting sections & abstract", "Parsing author metadata", "Identifying citations & references"]
        ), unsafe_allow_html=True)

        try:
            paper_json = parse_paper(pdf_path, os.path.join("output", "current_paper.json"))
            ld1.empty()
            authors = paper_json.get('authors', [])
            author_str = ", ".join([f"{a.get('first','')} {a.get('last','')}" for a in authors[:2]]).strip() if authors else "Unknown authors"
            with ph1: render_stepper(1, f"Paper Parsed ✓", "done",
                f"“{paper_json.get('title','Unknown')}”  ·  {author_str}  ·  {len(paper_json.get('sections',[]))} sections extracted")
        except Exception as e:
            ld1.empty()
            with ph1: render_stepper(1, "PDF parsing failed", "error", str(e))
            st.stop()

        # ── Step 2 ──────────────────────────────────────────────
        ph2, ld2 = st.empty(), st.empty()
        with ph2: render_stepper(2, "Discovering Implementations", "active",
            "Querying Semantic Scholar · Searching GitHub via Coral SQL · Cross-referencing arxiv IDs")
        with ld2: st.markdown(render_loader("Searching databases…", "Cross-referencing Semantic Scholar & GitHub",
            substeps=["Querying Semantic Scholar API", "Running Coral SQL on GitHub index", "Matching arxiv IDs to repositories", "Scoring candidate relevance", "Filtering duplicates & forks"]
        ), unsafe_allow_html=True)

        search_result = search_implementation(paper_json)
        st.session_state.api_calls = pwc.get_call_count()
        update_sidebar_metrics(flash=True)
        ld2.empty()

        candidates = search_result.get('candidates', [])
        n_found = len(candidates)
        if search_result["found"]:
            # Build animated repo list HTML
            repo_rows = ""
            for i, c in enumerate(candidates[:8]):
                url   = c.get('repo_url', '?')
                name  = url.split('/')[-1] if url != '?' else '?'
                owner = url.split('/')[-2] if '/' in url else ''
                stars = c.get('stars', 0)
                delay = i * 0.12
                repo_rows += f'<div class="repo-row" style="animation-delay:{delay}s"><span style="opacity:0.5">🐙</span> <span style="color:#e2e8f0">{owner}/</span><strong>{name}</strong><span class="repo-stars">⭐ {stars:,}</span></div>'
            extra = f'<div class="repo-list-wrap">{repo_rows}</div>' if repo_rows else ''
            with ph2: render_stepper(2, f"Found {n_found} Candidate Repositories", "done",
                f"Semantic Scholar · GitHub · arXiv  ·  Sorted by relevance", extra_html=extra)
        else:
            with ph2: render_stepper(2, "No Existing Implementation Found", "done",
                "No public implementation discovered — will synthesize reference code from scratch")

        health_result = compat_result = resolver_result = generator_result = None

        if search_result["found"] and search_result.get("candidates"):
            # ── Step 2.5 ─ Validator ─────────────────────────────
            ph25, ld25 = st.empty(), st.empty()
            with ph25: render_stepper(3, "Running RAG Validation Pipeline", "active",
                "Semantic similarity · Concept keyword matching · Dependency alignment · Codebase scan")

            rag_html = (
                '<div class="rag-pipeline">'
                '<div class="rag-stage active"> Semantic Embedding (abstract &#8596; README)<div class="rag-bar" style="margin-left:auto;"><div class="rag-bar-fill"></div></div></div>'
                '<div class="rag-stage active"> Concept Keyword Fuzzy Match<div class="rag-bar" style="margin-left:auto;"><div class="rag-bar-fill"></div></div></div>'
                '<div class="rag-stage active"> Dependency Framework Alignment<div class="rag-bar" style="margin-left:auto;"><div class="rag-bar-fill"></div></div></div>'
                '<div class="rag-stage active"> Codebase Architecture Scan<div class="rag-bar" style="margin-left:auto;"><div class="rag-bar-fill"></div></div></div>'
                '</div>'
            )
            with ld25: st.markdown(
                render_loader(
                    "Validating candidates\u2026",
                    f"Running 4-stage RAG pipeline across {n_found} repositories",
                    substeps=["Loading SentenceTransformer embeddings", "Encoding paper abstract", "Fetching README from GitHub", "Computing cosine similarity", "Running fuzzy concept match", "Checking framework dependencies"],
                    extra_html=rag_html,
                ),
                unsafe_allow_html=True,
            )

            from agents.repo_validator import validate_and_rank_candidates
            best_match, all_matches = validate_and_rank_candidates(paper_json, search_result["candidates"])
            st.session_state.coral_queries = coral.get_query_count()
            update_sidebar_metrics(flash=True)
            ld25.empty()

            if best_match["confidence_score"] < 0.20:
                search_result.update({"found": False, "repo_url": None, "validation": best_match})
                with ph25: render_stepper(3, "Low Confidence — Skipping", "done",
                    f"Best confidence {best_match['confidence_score']:.1%} across all candidates — threshold not met · Falling back to scratch generation")
            else:
                search_result.update({
                    "repo_url":      best_match["repo_url"],
                    "stars":         best_match.get("stars", 0),
                    "is_official":   best_match.get("is_official", False),
                    "search_method": best_match.get("search_method"),
                    "validation":    best_match,
                })
                scores = best_match.get('validator_scores', {})
                score_str = f"Sem: {scores.get('semantic',0):.0%}  Concept: {scores.get('concept',0):.0%}  Dep: {scores.get('dependency',0):.0%}  Code: {scores.get('codebase',0):.0%}"
                with ph25: render_stepper(3, f"Best Match: {best_match['repo_url'].split('/')[-1]}", "done",
                    f"Confidence {best_match['confidence_score']:.1%} · {best_match['classification']}  ·  {score_str}")

        if search_result["found"]:
            # ── Step 3 ─ Health Check ───────────────────────────
            ph3, ld3 = st.empty(), st.empty()
            repo_name = search_result["repo_url"].split("/")[-1]
            with ph3: render_stepper(4, f"Checking Repository Health  —  {repo_name}", "active",
                "Analyzing commit activity · Star count · Open issues · Last commit date · Maintenance signals via Coral SQL")
            with ld3: st.markdown(render_loader("Health check running…", "Coral SQL queries on GitHub data",
                substeps=["Fetching commit history", "Counting open issues & PRs", "Measuring release cadence", "Checking star growth trend", "Computing health score"]
            ), unsafe_allow_html=True)

            health_result = check_repo_health(search_result["repo_url"])
            st.session_state.coral_queries = coral.get_query_count()
            update_sidebar_metrics(flash=True)
            ld3.empty()

            sc  = health_result.get("health_score", -1)
            sig = health_result.get("signals", {})
            extra_health = ""
            if sig:
                days  = sig.get('last_commit_days_ago')
                stars = sig.get('stars')
                parts = []
                if days  is not None: parts.append(f"Last commit {days}d ago")
                if stars is not None: parts.append(f"⭐ {stars:,} stars")
                if parts: extra_health = f"  ·  {' · '.join(parts)}"
            with ph3: render_stepper(4, f"Health Score: {sc}/100", "done",
                f"{health_result.get('health_emoji','⚪')} {health_result.get('health_label','Unknown')}{extra_health}")

            # ── Step 4 ─ Dependency Analysis ────────────────────
            ph4, ld4 = st.empty(), st.empty()
            with ph4: render_stepper(5, "Dependency & Compatibility Analysis", "active",
                "Fetching requirements.txt · setup.py · pyproject.toml · Detecting version conflicts")
            with ld4: st.markdown(render_loader("Scanning dependencies…", "Detecting version conflicts & compatibility issues",
                substeps=["Fetching requirements.txt via Coral SQL", "Parsing version constraints", "Detecting Python version mismatches", "Identifying CUDA/framework conflicts", "Building conflict graph"]
            ), unsafe_allow_html=True)

            compat_result = check_compatibility(search_result["repo_url"], health_result.get("owner"), health_result.get("repo"))
            st.session_state.coral_queries = coral.get_query_count()
            update_sidebar_metrics(flash=True)
            ld4.empty()

            n_conflicts = len(compat_result.get('conflicts', []))
            n_warnings  = len(compat_result.get('warnings', []))
            if compat_result.get("clean"):
                status_lbl = "No conflicts detected — environment is clean"
            else:
                status_lbl = f"{n_conflicts} conflict(s) · {n_warnings} warning(s) detected — proceeding to resolution"
            with ph4: render_stepper(5, "Dependency Scan Complete", "done", status_lbl)

            # ── Step 5 ─ Dockerfile Generation ──────────────────
            ph5, ld5 = st.empty(), st.empty()
            n_dep_files = len(compat_result.get('dep_files', {}))
            with ph5: render_stepper(6, "Generating Dockerfile & Resolving Conflicts", "active",
                f"LLM selecting optimal base image · Pinning dependencies · Resolving {n_conflicts} conflict(s) across {n_dep_files} dep file(s)")
            with ld5: st.markdown(render_loader("Groq Llama generating Dockerfile…", "LLM drafting optimised Docker environment",
                substeps=["Selecting base image (CUDA / Python version)", "Pinning conflicting package versions", "Resolving transitive dependencies", "Writing COPY & RUN layers", "Validating Dockerfile syntax"]
            ), unsafe_allow_html=True)

            req_content = ""
            for name, content in compat_result.get("dep_files", {}).items():
                if "requirement" in name.lower() and name.endswith(".txt"):
                    req_content = content; break

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
            update_sidebar_metrics(flash=True)
            ld5.empty()

            base_img = resolver_result.get('selected_base_image', 'Unknown')
            with ph5: render_stepper(6, "Dockerfile Generated ✓", "done",
                f"Base image: {base_img}  ·  {n_conflicts} conflict(s) resolved  ·  Requirements pinned & locked")

        else:
            # ── Step 4 (scratch) ─ Synthesise from scratch ──────
            ph6, ld6 = st.empty(), st.empty()
            with ph6: render_stepper(4, "Synthesising Reference Implementation", "active",
                "No public repo found — Gemini 2.0 Flash generating code from related repos via Coral SQL JOINs")
            with ld6: st.markdown(render_loader("Synthesising…", "Gemini generating implementation from related repos",
                substeps=["Querying related repositories via Coral SQL", "Extracting common patterns", "Building abstract syntax scaffold", "Generating model architecture code", "Writing training loop & config", "Generating Dockerfile from scratch"]
            ), unsafe_allow_html=True)

            generator_result = generate_from_scratch(paper_json)
            st.session_state.coral_queries = coral.get_query_count()
            update_sidebar_metrics(flash=True)
            total_llm_calls.update(generator_result.get("llm_calls", {}))
            ld6.empty()

            n_related = len(generator_result.get('related_repos', []))
            with ph6: render_stepper(4, "Reference Code & Dockerfile Generated ✓", "done",
                f"Synthesised from {n_related} related repos  ·  Gemini calls: {total_llm_calls.get('gemini_flash', 0)}")

        # ── Final Build ──────────────────────────────────────
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
        update_sidebar_metrics()
        st.session_state.llm_calls     = total_llm_calls
        st.session_state.final_output  = final_output
        st.session_state.paper_json    = paper_json

    if "final_output" in st.session_state:
        render_results(st.session_state.final_output, st.session_state.get("paper_json"))

else:
    if "final_output" in st.session_state:
        st.info(f"Showing results from last analyzed paper: **{st.session_state.final_output['paper']['title']}**")
        render_results(st.session_state.final_output, st.session_state.get("paper_json"))
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(3)
        features = [
            ("Intelligent Discovery",
             "Automatically cross-references databases and GitHub to find the most official, highest-rated implementation for any research paper."),
            ("Conflict Resolution",
             "Advanced RAG + LLM reasoning detects and resolves dependency hell before you ever run docker build."),
            ("Production Ready",
             "Generates optimized Dockerfiles with precise base images, proper caching layers, and minimal attack surfaces."),
        ]
        for col, (title, desc) in zip(cols, features):
            with col:
                st.markdown(f"""
                <div class="fcard">
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)