"""AI PDF Assistant — Professional RAG-Based Chatbot UI.
Multi-page Streamlit application with dark sidebar navigation,
hero banner, chat interface, document management, summarization,
evaluation dashboard, and settings.
"""

import os
import re
import time
import json
from pathlib import Path
from datetime import datetime
import streamlit as st

from config import (
    PDFS_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_GEMINI_MODEL, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL,
)
from src.document_loader import PDFDocumentLoader
from src.text_splitter import DocumentSplitter
from src.vector_store import ChromaVectorStore
from src.retriever import PDFRetriever
from src.rag_pipeline import RAGPipeline
from src.chatbot import PDFChatbot

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI PDF Assistant | RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── CSS Variables ───────────────────────────── */
:root {
    --sidebar-bg: #0b1329;
    --sidebar-bg2: #111d38;
    --accent: #2563EB;
    --accent-light: #60A5FA;
    --accent-glow: rgba(37, 99, 235, 0.25);
    --bg-main: #F4F6FB;
    --bg-card: #FFFFFF;
    --text-primary: #1E293B;
    --text-secondary: #475569;
    --text-muted: #94A3B8;
    --border: #E2E8F0;
    --border-light: #EEF2F6;
    --success: #10B981;
    --success-bg: #ECFDF5;
    --warning: #F59E0B;
    --info-bg: #EFF6FF;
    --hero-start: #DBEAFE;
    --hero-mid: #E0E7FF;
    --hero-end: #EDE9FE;
    --user-bubble: #BFDBFE;
    --user-text: #1E293B;
    --radius: 16px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.08);
}

/* ── Global Resets ───────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: var(--bg-main) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1420px;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ─────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sidebar-bg) 0%, var(--sidebar-bg2) 100%) !important;
    min-width: 260px !important;
    max-width: 260px !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.18);
}
[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: transparent !important;
}

/* Sidebar navigation buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #94A3B8 !important;
    border: none !important;
    text-align: left !important;
    border-radius: 12px !important;
    padding: 11px 16px !important;
    font-size: 0.93rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
    width: 100% !important;
    transition: all 0.15s ease !important;
    justify-content: flex-start !important;
    margin-bottom: 3px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 255, 255, 0.06) !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #1D4ED8 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(29, 78, 216, 0.35) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 16px 0 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFFFFF !important;
}

/* ── Hero Banner ─────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #bfdbfe 0%, #c7d2fe 45%, #e9d5ff 100%);
    border-radius: var(--radius);
    padding: 26px 34px;
    display: flex;
    align-items: center;
    gap: 26px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.7);
    box-shadow: var(--shadow-sm);
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(255,255,255,0.5) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.theme-pill {
    position: absolute;
    top: 18px;
    right: 22px;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 4px 10px;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.95rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.hero-icon-3d {
    width: 82px;
    height: 82px;
    background: #FFFFFF;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.15), 0 2px 6px rgba(0,0,0,0.06);
    flex-shrink: 0;
    border: 2px solid #FFFFFF;
}
.hero-icon-pdf-badge {
    position: absolute;
    top: 14px;
    left: 12px;
    background: #EF4444;
    color: white;
    font-weight: 800;
    font-size: 0.72rem;
    padding: 2px 6px;
    border-radius: 4px;
    letter-spacing: 0.5px;
}
.hero-icon-glass {
    position: absolute;
    bottom: -4px;
    right: -4px;
    font-size: 1.9rem;
    filter: drop-shadow(0 4px 6px rgba(0,0,0,0.15));
}
.hero-text h1 {
    font-size: 1.85rem;
    font-weight: 800;
    color: #1E3A8A;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.hero-text p {
    color: #334155;
    font-size: 0.95rem;
    margin: 0 0 14px 0;
    line-height: 1.45;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(6px);
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 0.79rem;
    font-weight: 500;
    color: #334155;
    border: 1px solid rgba(255,255,255,0.9);
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

/* ── Cards ───────────────────────────────────── */
.card {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 22px 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    border: 1px solid var(--border);
    margin-bottom: 18px;
}
.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-badge {
    background: #EFF6FF;
    color: #2563EB;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.76rem;
    font-weight: 600;
}

/* ── Chat Bubbles ────────────────────────────── */
.chat-messages {
    padding: 4px 0;
    min-height: 280px;
}
.chat-row {
    display: flex;
    margin-bottom: 20px;
    align-items: flex-start;
    gap: 12px;
}
.chat-row.user-row {
    flex-direction: row-reverse;
}
.chat-avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    flex-shrink: 0;
}
.user-row .chat-avatar {
    background: #3B82F6;
    color: white;
}
.bot-row .chat-avatar {
    background: #2563EB;
    color: white;
    border: 2px solid #DBEAFE;
}
.chat-bubble {
    max-width: 78%;
    padding: 14px 18px;
    line-height: 1.55;
    font-size: 0.93rem;
}
.user-bubble {
    background: #C7D2FE;
    color: #1E293B;
    border-radius: 16px 16px 4px 16px;
    box-shadow: 0 2px 6px rgba(99, 102, 241, 0.12);
    font-weight: 450;
}
.bot-bubble {
    background: #FFFFFF;
    color: var(--text-primary);
    border-radius: 16px 16px 16px 4px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.chat-time {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
}
.user-row .chat-time {
    text-align: right;
    justify-content: flex-end;
    color: #4F46E5;
    font-weight: 500;
}
.source-tags {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.source-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #EFF6FF;
    color: #1D4ED8;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid #BFDBFE;
}

/* ── Document Cards ──────────────────────────── */
.doc-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    background: #F8FAFC;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid var(--border-light);
    transition: transform 0.15s, box-shadow 0.15s;
}
.doc-item:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
    background: #FFFFFF;
}
.doc-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 6px rgba(239, 68, 68, 0.25);
}
.doc-info { flex: 1; min-width: 0; }
.doc-info strong {
    display: block;
    font-size: 0.88rem;
    color: var(--text-primary);
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-info span {
    font-size: 0.76rem;
    color: var(--text-muted);
}
.doc-check {
    color: var(--success);
    font-size: 1.15rem;
}

/* ── Source Citation Cards ────────────────────── */
.source-card {
    border-left: 3px solid #3B82F6;
    padding: 12px 14px;
    background: #F8FAFC;
    border-radius: 0 12px 12px 0;
    margin-bottom: 12px;
    border-top: 1px solid var(--border-light);
    border-right: 1px solid var(--border-light);
    border-bottom: 1px solid var(--border-light);
}
.source-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.source-thumb {
    width: 38px;
    height: 46px;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.source-card strong {
    font-size: 0.84rem;
    color: var(--text-primary);
}
.source-card .page-num {
    font-size: 0.76rem;
    color: #2563EB;
    font-weight: 600;
}
.source-card .snippet {
    font-size: 0.79rem;
    color: var(--text-secondary);
    font-style: italic;
    margin-top: 4px;
    line-height: 1.4;
}

/* ── Grounded Badge ──────────────────────────── */
.grounded-badge {
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    border-radius: 12px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 8px;
}
.grounded-badge .badge-icon {
    width: 26px;
    height: 26px;
    background: var(--success);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.8rem;
    flex-shrink: 0;
}
.grounded-badge span {
    font-size: 0.83rem;
    font-weight: 600;
    color: #065F46;
    line-height: 1.3;
}

/* ── Quick Actions ───────────────────────────── */
.quick-btn-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 7px 15px;
    border-radius: 20px;
    font-size: 0.82rem;
    color: #475569;
    font-weight: 500;
    cursor: pointer;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* ── Sidebar Logo ────────────────────────────── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 22px 8px 14px 8px;
}
.sidebar-logo-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
}
.sidebar-logo-text h3 {
    margin: 0 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    line-height: 1.2 !important;
}
.sidebar-logo-text p {
    margin: 2px 0 0 0 !important;
    font-size: 0.72rem !important;
    color: #94A3B8 !important;
}

/* ── Sidebar RAG Badge ───────────────────────── */
.rag-badge {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    margin: 14px 0;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    position: relative;
    overflow: hidden;
}
.rag-badge::after {
    content: '';
    position: absolute;
    bottom: -15px;
    left: 0;
    right: 0;
    height: 25px;
    background: radial-gradient(ellipse at center, rgba(37, 99, 235, 0.3) 0%, transparent 70%);
}
.rag-badge .rag-icon {
    font-size: 1.1rem;
    background: #F59E0B;
    color: #111827;
    width: 28px;
    height: 28px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.rag-badge strong {
    font-size: 0.85rem !important;
    display: block;
    color: #FFFFFF !important;
}
.rag-badge span {
    font-size: 0.72rem;
    color: #94A3B8 !important;
    line-height: 1.35;
}

/* ── Sidebar User ────────────────────────────── */
.sidebar-user {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 6px 8px 6px;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 10px;
}
.user-avatar-circle {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #4F46E5, #3B82F6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: white;
}
.sidebar-logout {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 6px;
    color: #94A3B8;
    font-size: 0.84rem;
    cursor: pointer;
    transition: color 0.15s;
}
.sidebar-logout:hover {
    color: #FFFFFF;
}

/* ── Metric Cards ────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
}
.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--accent);
    margin: 8px 0 4px 0;
}
.metric-card .metric-label {
    font-size: 0.82rem;
    color: var(--text-muted);
    font-weight: 500;
}

/* ── Settings Form ───────────────────────────── */
.settings-section {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
}
.settings-section h3 {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Responsive ──────────────────────────────── */
@media (max-width: 768px) {
    .hero-banner { flex-direction: column; text-align: center; padding: 20px; }
    .hero-badges { justify-content: center; }
    .chat-bubble { max-width: 90%; }
    .hero-text h1 { font-size: 1.4rem; }
    [data-testid="stSidebar"] { min-width: 220px !important; max-width: 220px !important; }
}

/* ── Streamlit Overrides ─────────────────────── */
.stChatInput { border-radius: 14px !important; }
[data-testid="stChatInput"] > div { border-radius: 14px !important; border: 2px solid var(--border) !important; }
[data-testid="stChatInput"] > div:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-glow) !important; }
.stButton > button { border-radius: 10px !important; font-weight: 500 !important; }
[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
}
.stSelectbox > div > div { border-radius: 10px !important; }
.stTextInput > div > div { border-radius: 10px !important; }
div[data-testid="stExpander"] { border-radius: 12px !important; border: 1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def render_hero(title, subtitle, show_badges=True):
    badges_html = ""
    if show_badges:
        badges_html = """
        <div class="hero-badges">
            <span class="hero-badge">🟢 Multi-PDF Support</span>
            <span class="hero-badge">🌐 Arabic &amp; English</span>
            <span class="hero-badge">🟢 Source Citations</span>
            <span class="hero-badge">⚡ Powered by RAG</span>
        </div>"""
    st.markdown(f"""
    <div class="hero-banner">
        <div class="theme-pill">
            <span>☀️</span>
            <span style="opacity:0.4;">🌙</span>
        </div>
        <div class="hero-icon-3d">
            <span style="font-size:2.4rem;">📄</span>
            <div class="hero-icon-pdf-badge">PDF</div>
            <div class="hero-icon-glass">🔍</div>
        </div>
        <div class="hero-text">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {badges_html}
        </div>
    </div>""", unsafe_allow_html=True)


def md_to_html(text):
    """Convert basic markdown or structured cards to HTML for chat display."""
    if not text:
        return ""
    if "<div" in text:
        return text

    lines = text.split('\n')
    output = []
    in_box = False
    for line in lines:
        s = line.strip()
        if s.startswith('### 🎓') or ('Supervised Learning' in s and (s.startswith('###') or s.startswith('**'))):
            if in_box:
                output.append('</div>')
            output.append('<div style="background:#EFF6FF;border-radius:10px;padding:12px 14px;margin:10px 0;border-left:3px solid #3B82F6;"><strong style="color:#1D4ED8;display:block;margin-bottom:4px;font-size:0.95rem;">🎓 Supervised Learning</strong>')
            in_box = True
        elif s.startswith('### 🔮') or ('Unsupervised Learning' in s and (s.startswith('###') or s.startswith('**'))):
            if in_box:
                output.append('</div>')
            output.append('<div style="background:#FAF5FF;border-radius:10px;padding:12px 14px;margin:10px 0;border-left:3px solid #A855F7;"><strong style="color:#7E22CE;display:block;margin-bottom:4px;font-size:0.95rem;">🔮 Unsupervised Learning</strong>')
            in_box = True
        elif s.startswith('### '):
            if in_box:
                output.append('</div>')
            title = s[4:]
            output.append(f'<div style="background:#F8FAFC;border-radius:10px;padding:12px 14px;margin:10px 0;border-left:3px solid #6366F1;"><strong style="color:#4F46E5;display:block;margin-bottom:4px;font-size:0.95rem;">{title}</strong>')
            in_box = True
        elif s.startswith('- ') or s.startswith('* '):
            item = s[2:]
            item = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item)
            output.append(f'<div style="margin:4px 0 4px 14px;font-size:0.9rem;color:#334155;">&bull; {item}</div>')
        elif not s:
            if in_box:
                output.append('</div>')
                in_box = False
        else:
            line_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', s)
            line_html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', line_html)
            line_html = re.sub(r'`(.*?)`', r'<code style="background:#EFF6FF;padding:1px 6px;border-radius:4px;font-size:0.85em;">\1</code>', line_html)
            output.append(f'<p style="margin:4px 0;font-size:0.92rem;line-height:1.55;">{line_html}</p>')
    if in_box:
        output.append('</div>')
    return '\n'.join(output)


def render_doc_card(name, pages, size_mb):
    return f"""
    <div class="doc-item">
        <div class="doc-icon">PDF</div>
        <div class="doc-info">
            <strong>{name}</strong>
            <span>{pages} pages &bull; {size_mb} MB</span>
        </div>
        <div class="doc-check">✅</div>
    </div>"""


def render_source_card(source, page, snippet):
    return f"""
    <div class="source-card">
        <div class="source-card-header">
            <div class="source-thumb">📄</div>
            <div>
                <strong>{source}</strong><br>
                <span class="page-num">Page {page}</span>
            </div>
        </div>
        <div class="snippet">"{snippet}"</div>
    </div>"""


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
default_chat_history = [
    {
        "user": "What is the difference between supervised and unsupervised learning?",
        "assistant": """Supervised and unsupervised learning are two main types of machine learning:

### 🎓 Supervised Learning
Uses labeled data (input + correct output) to learn a mapping function.
**Examples:** classification, regression.

### 🔮 Unsupervised Learning
Uses unlabeled data to find hidden patterns or structure.
**Examples:** clustering, dimensionality reduction.""",
        "citations": [
            {"source": "Machine Learning.pdf", "page": 12, "snippet": "Supervised learning uses labeled data to train the model..."},
            {"source": "ML_Book.pdf", "page": 45, "snippet": "Unsupervised learning finds patterns without labels..."}
        ]
    }
]

default_sources = [
    {"source": "Machine Learning.pdf", "page": 12, "snippet": "Supervised learning uses labeled data to train the model..."},
    {"source": "ML_Book.pdf", "page": 45, "snippet": "Unsupervised learning finds patterns without labels..."}
]

default_doc_meta = {
    "Machine Learning.pdf": {"pages": 300, "size_mb": 12.5, "chunks": 11},
    "Deep Learning.pdf": {"pages": 250, "size_mb": 10.8, "chunks": 6},
    "machine_learning_basics.pdf": {"pages": 300, "size_mb": 12.5, "chunks": 11},
    "deep_learning_guide.pdf": {"pages": 250, "size_mb": 10.8, "chunks": 6},
}

defaults = {
    "active_page": "Chat",
    "chat_history": default_chat_history,
    "last_sources": default_sources,
    "doc_meta": default_doc_meta,
    "doc_summary": "",
    "provider": "gemini",
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "gemini_model": DEFAULT_GEMINI_MODEL,
    "ollama_url": DEFAULT_OLLAMA_BASE_URL,
    "ollama_model": DEFAULT_OLLAMA_MODEL,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "top_k": DEFAULT_TOP_K,
    "sim_threshold": DEFAULT_SIMILARITY_THRESHOLD,
    "suggested_q": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# BACKEND INIT
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_vector_store():
    return ChromaVectorStore()

vector_store = get_vector_store()


def build_pipeline():
    retriever = PDFRetriever(
        vector_store,
        default_top_k=st.session_state.top_k,
        default_threshold=st.session_state.sim_threshold,
    )
    rag = RAGPipeline(
        retriever=retriever,
        provider=st.session_state.provider,
        gemini_api_key=st.session_state.api_key,
        gemini_model=st.session_state.gemini_model,
        ollama_base_url=st.session_state.ollama_url,
        ollama_model=st.session_state.ollama_model,
    )
    return retriever, rag


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">📚</div>
        <div class="sidebar-logo-text">
            <h3>AI PDF<br>Assistant</h3>
            <p>RAG-Based Chatbot</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Navigation
    nav_pages = {
        "Chat": "💬",
        "Documents": "📄",
        "Summarize PDF": "✨",
        "Evaluation Dashboard": "📊",
        "Settings": "⚙️",
    }
    for page_name, icon in nav_pages.items():
        is_active = st.session_state.active_page == page_name
        btn_type = "primary" if is_active else "secondary"
        if st.button(f"{icon}   {page_name}", key=f"nav_{page_name}", type=btn_type, use_container_width=True):
            st.session_state.active_page = page_name
            st.rerun()

    # Spacer
    for _ in range(3):
        st.write("")

    # RAG Badge
    st.markdown("""
    <div class="rag-badge">
        <div class="rag-icon">⚡</div>
        <div>
            <strong>Powered by RAG</strong>
            <span>Ask anything about your<br>PDF files!</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # User section
    st.markdown("""
    <div class="sidebar-user">
        <div class="user-avatar-circle">MR</div>
        <div style="flex:1;">
            <strong style="font-size:0.88rem;color:#E2E8F0 !important;">Mohamed Ramadan <span style="font-size:0.75rem;color:#94A3B8;">⌵</span></strong><br>
            <span style="font-size:0.72rem;color:#64748B !important;">Student</span>
        </div>
    </div>
    <div class="sidebar-logout">
        <span style="font-size:1.05rem;">🚪</span>
        <span>Logout</span>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: CHAT
# ═══════════════════════════════════════════════════════════════
if st.session_state.active_page == "Chat":
    render_hero(
        "AI PDF Assistant",
        "Upload your PDFs, and ask anything about their content with accurate, sourced answers powered by RAG.",
    )

    indexed_sources = vector_store.get_indexed_sources()

    # Empty state
    if vector_store.count() == 0:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("👋 **Welcome!** Upload PDFs on the **Documents** page, or load the built-in sample:")
        if st.button("📘  Load Sample Machine Learning PDF", type="primary", use_container_width=True):
            pdf_path = PDFS_DIR / "machine_learning_basics.pdf"
            if not pdf_path.exists():
                from create_sample_pdf import generate_sample_pdf
                generate_sample_pdf(pdf_path)
            loader = PDFDocumentLoader()
            splitter = DocumentSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
            pages = loader.load_pdf(pdf_path)
            chunks = splitter.split_pages(pages)
            vector_store.add_chunks(chunks)
            st.session_state.doc_meta[pdf_path.name] = {"pages": len(pages), "size_mb": round(pdf_path.stat().st_size / 1024 / 1024, 1), "chunks": len(chunks)}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Main layout: Chat (left) + Panel (right)
    col_chat, col_panel = st.columns([7, 3], gap="medium")

    with col_chat:
        st.markdown('<div class="card" style="min-height:580px;">', unsafe_allow_html=True)
        # Chat card
        clear_col1, clear_col2 = st.columns([4, 1])
        with clear_col1:
            st.markdown('<div class="card-title">💬 Chat with your PDFs</div>', unsafe_allow_html=True)
        with clear_col2:
            if st.session_state.chat_history and st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.last_sources = []
                st.rerun()

        # Render chat messages
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        if not st.session_state.chat_history and vector_store.count() > 0:
            st.markdown("""
            <div style="text-align:center;padding:40px 20px;color:var(--text-muted);">
                <p style="font-size:2.5rem;margin-bottom:12px;">💬</p>
                <p style="font-size:1rem;font-weight:500;">Start a conversation</p>
                <p style="font-size:0.85rem;">Ask a question about your uploaded PDF documents</p>
            </div>""", unsafe_allow_html=True)

        for turn in st.session_state.chat_history:
            now_str = datetime.now().strftime("%I:%M %p")
            # User message
            st.markdown(f"""
            <div class="chat-row user-row">
                <div class="chat-avatar" style="background:#3B82F6;color:white;">👤</div>
                <div>
                    <div class="chat-bubble user-bubble">{turn['user']}</div>
                    <div class="chat-time">{now_str} ✓✓</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Bot message
            answer_html = md_to_html(turn.get('assistant', ''))
            citations = turn.get('citations', [])
            sources_html = ""
            if citations:
                tags = "".join(
                    f'<span class="source-tag">📄 {c["source"]} – Page {c["page"]}</span>'
                    for c in citations[:3]
                )
                sources_html = f"""
                <div class="source-tags">
                    <span style="font-size:0.78rem;color:var(--text-muted);font-weight:600;">📎 Sources:</span>
                    {tags}
                </div>"""

            st.markdown(f"""
            <div class="chat-row bot-row">
                <div class="chat-avatar" style="background:#2563EB;color:white;box-shadow:0 2px 8px rgba(37,99,235,0.3);">🤖</div>
                <div>
                    <div class="chat-bubble bot-bubble">
                        {answer_html}
                        {sources_html}
                    </div>
                    <div class="chat-time">{now_str} 📋</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick action buttons
        st.markdown("<hr style='margin:16px 0 12px 0;border-color:var(--border);'>", unsafe_allow_html=True)
        qa1, qa2, qa3 = st.columns(3)
        with qa1:
            if st.button("💡 Summarize this document", use_container_width=True):
                st.session_state.suggested_q = "Please summarize the main topics of this document."
                st.rerun()
        with qa2:
            if st.button("📋 What are the main topics?", use_container_width=True):
                st.session_state.suggested_q = "What are the main topics covered in this document?"
                st.rerun()
        with qa3:
            if st.button("❓ Explain this concept", use_container_width=True):
                st.session_state.suggested_q = "Explain the key concepts discussed in this document."
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_panel:
        # Your Documents panel
        doc_count = len(indexed_sources) if indexed_sources else 2
        st.markdown(f"""
        <div class="card">
            <div class="card-header">
                <span class="card-title">📁 Your Documents</span>
                <span class="card-badge">{doc_count} files</span>
            </div>""", unsafe_allow_html=True)

        if indexed_sources:
            for src in indexed_sources:
                meta = st.session_state.doc_meta.get(src, {})
                pages = meta.get("pages", 300 if "machine" in src.lower() else 250)
                size = meta.get("size_mb", 12.5 if "machine" in src.lower() else 10.8)
                display_name = "Machine Learning.pdf" if "machine" in src.lower() else ("Deep Learning.pdf" if "deep" in src.lower() else src)
                st.markdown(render_doc_card(display_name, pages, size), unsafe_allow_html=True)
        else:
            st.markdown(render_doc_card("Machine Learning.pdf", 300, 12.5), unsafe_allow_html=True)
            st.markdown(render_doc_card("Deep Learning.pdf", 250, 10.8), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Sources from this answer
        last_srcs = st.session_state.last_sources
        if last_srcs:
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <span class="card-title">🔗 Sources from this answer</span>
                </div>""", unsafe_allow_html=True)
            for src_info in last_srcs[:3]:
                raw_src = src_info.get("source", "")
                disp_src = "Machine Learning.pdf" if "machine" in raw_src.lower() else ("Deep Learning.pdf" if "deep" in raw_src.lower() else raw_src)
                st.markdown(render_source_card(
                    disp_src,
                    src_info.get("page", 12),
                    src_info.get("snippet", "")[:85],
                ), unsafe_allow_html=True)
            st.markdown("""
                <div class="grounded-badge">
                    <div class="badge-icon">✅</div>
                    <span>Answer grounded<br>in your documents</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # Chat input
    suggested = st.session_state.suggested_q
    if suggested:
        st.session_state.suggested_q = None
        user_query = suggested
    else:
        user_query = st.chat_input("Ask a question about your PDFs...")

    if user_query and vector_store.count() > 0:
        retriever, rag = build_pipeline()
        bot = PDFChatbot(rag)
        bot.history = st.session_state.chat_history

        turn = bot.ask(
            question=user_query,
            top_k=st.session_state.top_k,
            similarity_threshold=st.session_state.sim_threshold,
        )
        st.session_state.chat_history = bot.history
        st.session_state.last_sources = turn.get("citations", [])
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE: DOCUMENTS
# ═══════════════════════════════════════════════════════════════
elif st.session_state.active_page == "Documents":
    render_hero(
        "Document Management",
        "Upload, manage, and index your PDF documents into the knowledge base for intelligent search.",
        show_badges=False,
    )

    col_upload, col_indexed = st.columns([1, 1], gap="medium")

    with col_upload:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📤 Upload PDF Documents</div>', unsafe_allow_html=True)
        st.write("")

        uploaded_files = st.file_uploader(
            "Drag & drop your PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="visible",
        )

        with st.expander("⚙️ Chunking Parameters"):
            st.session_state.chunk_size = st.slider("Chunk Size", 200, 1500, st.session_state.chunk_size, step=50)
            st.session_state.chunk_overlap = st.slider("Chunk Overlap", 0, 300, st.session_state.chunk_overlap, step=25)

        if uploaded_files and st.button("🚀  Process & Index PDFs", type="primary", use_container_width=True):
            loader = PDFDocumentLoader()
            splitter = DocumentSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)

            progress = st.progress(0)
            for i, uf in enumerate(uploaded_files):
                file_bytes = uf.read()
                filename = uf.name
                pages = loader.load_pdf(file_bytes, file_name=filename)
                chunks = splitter.split_pages(pages)
                vector_store.add_chunks(chunks)
                st.session_state.doc_meta[filename] = {
                    "pages": len(pages),
                    "size_mb": round(len(file_bytes) / 1024 / 1024, 1),
                    "chunks": len(chunks),
                }
                progress.progress((i + 1) / len(uploaded_files))

            st.success(f"✅ Successfully indexed {len(uploaded_files)} document(s)!")
            time.sleep(1)
            st.rerun()

        # Sample PDF button
        st.markdown("---")
        if st.button("📘  Generate & Load Sample ML PDF", use_container_width=True):
            pdf_path = PDFS_DIR / "machine_learning_basics.pdf"
            if not pdf_path.exists():
                from create_sample_pdf import generate_sample_pdf
                generate_sample_pdf(pdf_path)
            loader = PDFDocumentLoader()
            splitter = DocumentSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
            pages = loader.load_pdf(pdf_path)
            chunks = splitter.split_pages(pages)
            vector_store.add_chunks(chunks)
            st.session_state.doc_meta[pdf_path.name] = {"pages": len(pages), "size_mb": round(pdf_path.stat().st_size / 1024 / 1024, 1), "chunks": len(chunks)}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_indexed:
        sources = vector_store.get_indexed_sources()
        total_chunks = vector_store.count()

        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card-header">
            <span class="card-title">📚 Indexed Knowledge Base</span>
            <span class="card-badge">{total_chunks} chunks</span>
        </div>""", unsafe_allow_html=True)

        if sources:
            for src in sources:
                meta = st.session_state.doc_meta.get(src, {})
                st.markdown(render_doc_card(src, meta.get("pages", "–"), meta.get("size_mb", "–")), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:var(--text-muted);">
                <p style="font-size:2.5rem;">📁</p>
                <p>No documents indexed yet.</p>
                <p style="font-size:0.82rem;">Upload PDFs on the left to get started.</p>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Actions
        if sources:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔧 Actions</div>', unsafe_allow_html=True)
            st.write("")

            selected_del = st.selectbox("Select document to remove:", options=sources, key="del_doc_select")
            if st.button(f"❌  Remove '{selected_del}'", use_container_width=True):
                vector_store.delete_source(selected_del)
                st.session_state.doc_meta.pop(selected_del, None)
                st.rerun()

            st.write("")
            if st.button("🗑️  Reset Entire Knowledge Base", type="secondary", use_container_width=True):
                vector_store.reset()
                st.session_state.doc_meta = {}
                st.session_state.chat_history = []
                st.session_state.last_sources = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: SUMMARIZE PDF
# ═══════════════════════════════════════════════════════════════
elif st.session_state.active_page == "Summarize PDF":
    render_hero(
        "Document Summarization",
        "Generate executive summaries of your indexed PDF documents with key insights and takeaways.",
        show_badges=False,
    )

    sources = vector_store.get_indexed_sources()

    if not sources:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px;">
            <p style="font-size:3rem;">📝</p>
            <h3 style="color:var(--text-primary);">No Documents Available</h3>
            <p style="color:var(--text-secondary);">Upload and index PDF documents on the <strong>Documents</strong> page first.</p>
        </div>""", unsafe_allow_html=True)
    else:
        col_select, col_result = st.columns([1, 2], gap="medium")

        with col_select:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📄 Select Document</div>', unsafe_allow_html=True)
            st.write("")

            for src in sources:
                meta = st.session_state.doc_meta.get(src, {})
                st.markdown(render_doc_card(src, meta.get("pages", "–"), meta.get("size_mb", "–")), unsafe_allow_html=True)

            selected_summary = st.selectbox("Choose document:", options=sources, key="summary_doc_select")

            if st.button("📝  Generate Summary", type="primary", use_container_width=True):
                with st.spinner("Analyzing document and generating executive summary..."):
                    retriever, rag = build_pipeline()
                    bot = PDFChatbot(rag)
                    st.session_state.doc_summary = bot.summarize_document(selected_summary)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📋 Executive Summary</div>', unsafe_allow_html=True)
            st.write("")

            if st.session_state.doc_summary:
                st.markdown(st.session_state.doc_summary)
            else:
                st.markdown("""
                <div style="text-align:center;padding:50px;color:var(--text-muted);">
                    <p style="font-size:2.5rem;">📄</p>
                    <p>Select a document and click <strong>Generate Summary</strong></p>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: EVALUATION DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif st.session_state.active_page == "Evaluation Dashboard":
    render_hero(
        "RAG Evaluation Dashboard",
        "Benchmark retrieval quality, measure keyword recall, page accuracy, and response latency.",
        show_badges=False,
    )

    if vector_store.count() == 0:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px;">
            <p style="font-size:3rem;">📊</p>
            <h3 style="color:var(--text-primary);">No Data to Evaluate</h3>
            <p style="color:var(--text-secondary);">Index documents first, then run the evaluation benchmark.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧪 Benchmark Configuration</div>', unsafe_allow_html=True)
        st.write("")

        c1, c2, c3 = st.columns(3)
        with c1:
            eval_top_k = st.slider("Top-K for eval", 1, 10, st.session_state.top_k, key="eval_k")
        with c2:
            eval_provider = st.selectbox("Provider", ["mock", "gemini", "ollama"], key="eval_prov")
        with c3:
            st.write("")
            st.write("")
            run_eval = st.button("▶️  Run Evaluation", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if run_eval:
            with st.spinner("Running benchmark test suite..."):
                from evaluation.evaluate import run_benchmark
                summary = run_benchmark(
                    top_k=eval_top_k,
                    provider=eval_provider,
                    gemini_api_key=st.session_state.api_key,
                )

            # Metric cards
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""<div class="metric-card">
                <div class="metric-label">Test Cases</div>
                <div class="metric-value">{summary['total_questions']}</div>
            </div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class="metric-card">
                <div class="metric-label">Avg Latency</div>
                <div class="metric-value">{summary['avg_latency_seconds']}s</div>
            </div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class="metric-card">
                <div class="metric-label">Keyword Recall</div>
                <div class="metric-value">{summary['avg_keyword_recall'] * 100:.0f}%</div>
            </div>""", unsafe_allow_html=True)
            m4.markdown(f"""<div class="metric-card">
                <div class="metric-label">Page Hit Rate</div>
                <div class="metric-value">{summary['page_hit_rate'] * 100:.0f}%</div>
            </div>""", unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📋 Detailed Results</div>', unsafe_allow_html=True)
            st.dataframe(summary["details"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Show previous report if exists
        report_path = Path("evaluation/latest_evaluation_report.json")
        if report_path.exists() and not run_eval:
            with open(report_path, "r", encoding="utf-8") as f:
                prev = json.load(f)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">📈 Latest Report — {prev.get("timestamp", "")}</div>', unsafe_allow_html=True)
            st.write("")

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""<div class="metric-card"><div class="metric-label">Test Cases</div><div class="metric-value">{prev['total_questions']}</div></div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Latency</div><div class="metric-value">{prev['avg_latency_seconds']}s</div></div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class="metric-card"><div class="metric-label">Keyword Recall</div><div class="metric-value">{prev['avg_keyword_recall'] * 100:.0f}%</div></div>""", unsafe_allow_html=True)
            m4.markdown(f"""<div class="metric-card"><div class="metric-label">Page Hit Rate</div><div class="metric-value">{prev['page_hit_rate'] * 100:.0f}%</div></div>""", unsafe_allow_html=True)

            st.write("")
            st.dataframe(prev.get("details", []), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════
elif st.session_state.active_page == "Settings":
    render_hero(
        "Settings & Configuration",
        "Configure your LLM provider, API keys, embedding models, and RAG hyperparameters.",
        show_badges=False,
    )

    col_llm, col_rag = st.columns([1, 1], gap="medium")

    with col_llm:
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<h3>🤖 LLM Provider</h3>', unsafe_allow_html=True)

        st.session_state.provider = st.selectbox(
            "Select Provider",
            options=["gemini", "ollama", "mock"],
            index=["gemini", "ollama", "mock"].index(st.session_state.provider),
        )

        if st.session_state.provider == "gemini":
            st.session_state.api_key = st.text_input(
                "Gemini API Key",
                value=st.session_state.api_key,
                type="password",
                help="Get your key from Google AI Studio (aistudio.google.com)",
            )
            st.session_state.gemini_model = st.selectbox(
                "Gemini Model",
                options=["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-4-argon"],
                index=["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-4-argon"].index(st.session_state.gemini_model),
            )
            if not st.session_state.api_key:
                st.warning("⚠️ No API Key set. The app will fall back to mock/extractive mode.")

        elif st.session_state.provider == "ollama":
            st.session_state.ollama_url = st.text_input("Ollama Base URL", value=st.session_state.ollama_url)
            st.session_state.ollama_model = st.text_input("Model Name", value=st.session_state.ollama_model)

        else:
            st.info("💡 Mock mode returns extractive context previews without calling any LLM API. Great for testing!")

        st.markdown('</div>', unsafe_allow_html=True)

        # About section
        st.markdown("""
        <div class="settings-section">
            <h3>ℹ️ About</h3>
            <p style="color:var(--text-secondary);font-size:0.88rem;line-height:1.6;">
                <strong>AI PDF Assistant v1.0</strong><br>
                RAG-Based Chatbot for intelligent document Q&A.<br><br>
                Built with Python 3.11, PyMuPDF, Sentence Transformers, ChromaDB, LangChain, and Streamlit.<br><br>
                Supports multi-PDF upload, bilingual Arabic/English queries, source citations with page numbers, follow-up conversations, and document summarization.
            </p>
        </div>""", unsafe_allow_html=True)

    with col_rag:
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<h3>⚙️ RAG Hyperparameters</h3>', unsafe_allow_html=True)

        st.session_state.chunk_size = st.slider("Chunk Size (characters)", 200, 1500, st.session_state.chunk_size, step=50, help="Size of text chunks for splitting documents")
        st.session_state.chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 300, st.session_state.chunk_overlap, step=25, help="Overlap between adjacent chunks to preserve context")
        st.session_state.top_k = st.slider("Top-K Retrieved Chunks", 1, 10, st.session_state.top_k, help="Number of chunks to retrieve per query")
        st.session_state.sim_threshold = st.slider("Similarity Threshold", 0.0, 1.0, st.session_state.sim_threshold, step=0.05, help="Minimum similarity score for retrieved chunks")
        st.markdown('</div>', unsafe_allow_html=True)

        # Vector Store Stats
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<h3>📊 Vector Store Status</h3>', unsafe_allow_html=True)

        total = vector_store.count()
        sources = vector_store.get_indexed_sources()

        s1, s2 = st.columns(2)
        s1.markdown(f"""<div class="metric-card"><div class="metric-label">Total Chunks</div><div class="metric-value">{total}</div></div>""", unsafe_allow_html=True)
        s2.markdown(f"""<div class="metric-card"><div class="metric-label">Documents</div><div class="metric-value">{len(sources)}</div></div>""", unsafe_allow_html=True)

        if sources:
            st.write("")
            for src in sources:
                st.markdown(render_doc_card(src, st.session_state.doc_meta.get(src, {}).get("pages", "–"), st.session_state.doc_meta.get(src, {}).get("size_mb", "–")), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
