"""Streamlit User Interface for AI PDF Assistant (RAG-Based Chatbot).
Supports multi-PDF upload, real-time chunking, ChromaDB vector indexing,
conversational QA with Gemini/Ollama, source citations with page numbers,
document summaries, and a live RAG evaluation dashboard.
"""

import os
import io
import time
from pathlib import Path
import streamlit as st

from config import (
    PDFS_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    GEMINI_API_KEY,
)
from src.document_loader import PDFDocumentLoader
from src.text_splitter import DocumentSplitter
from src.vector_store import ChromaVectorStore
from src.retriever import PDFRetriever
from src.rag_pipeline import RAGPipeline
from src.chatbot import PDFChatbot
from evaluation.evaluate import calculate_keyword_recall

# Page Configuration
st.set_page_config(
    page_title="AI PDF Assistant | RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #EEF2FF;
        color: #4F46E5;
        margin-right: 0.4rem;
    }
    .citation-card {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_vector_store():
    """Initializes or retrieves the persistent Chroma vector store."""
    return ChromaVectorStore()


# Initialize state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = ""

vector_store = get_vector_store()
existing_sources = vector_store.get_indexed_sources()

# =========================================================================
# SIDEBAR: Document Management & System Configurations
# =========================================================================
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langchain_logo.png",
        width=48,
    )
    st.title("Settings & Documents")

    # Section 1: LLM Provider Configuration
    st.markdown("### 🤖 LLM Configuration")
    provider = st.selectbox(
        "LLM Provider",
        options=["gemini", "ollama", "mock"],
        index=0,
        help="Choose between Google Gemini API, local Ollama, or offline Mock/Extractive mode.",
    )

    api_key = ""
    gemini_model = DEFAULT_GEMINI_MODEL
    ollama_url = DEFAULT_OLLAMA_BASE_URL
    ollama_model = DEFAULT_OLLAMA_MODEL

    if provider == "gemini":
        default_key = os.getenv("GEMINI_API_KEY", "")
        api_key = st.text_input(
            "Gemini API Key",
            value=default_key,
            type="password",
            help="Enter your Google AI Studio API key.",
        )
        gemini_model = st.selectbox(
            "Gemini Model",
            options=["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-4-argon"],
            index=0,
        )
        if not api_key:
            st.info("💡 You can run in **'mock'** provider mode if you don't have an API key right now.")
    elif provider == "ollama":
        ollama_url = st.text_input("Ollama Base URL", value=DEFAULT_OLLAMA_BASE_URL)
        ollama_model = st.text_input("Ollama Model Name", value=DEFAULT_OLLAMA_MODEL)

    st.divider()

    # Section 2: PDF Upload & Processing
    st.markdown("### 📄 Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload lecture notes, textbooks, manuals, or research papers.",
    )

    # Advanced RAG Parameters
    with st.expander("⚙️ RAG Hyperparameters"):
        chunk_size = st.slider("Chunk Size", 200, 1500, DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 300, DEFAULT_CHUNK_OVERLAP, step=25)
        top_k = st.slider("Top-K Retrieved Chunks", 1, 10, DEFAULT_TOP_K)
        sim_threshold = st.slider(
            "Similarity Threshold", 0.0, 1.0, DEFAULT_SIMILARITY_THRESHOLD, step=0.05
        )

    # Process Uploaded PDFs Button
    if uploaded_files and st.button("🚀 Process & Index PDFs", use_container_width=True):
        loader = PDFDocumentLoader()
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        with st.status("Processing PDF documents...", expanded=True) as status:
            total_added = 0
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                filename = uploaded_file.name

                st.write(f"📖 Reading **{filename}** using PyMuPDF...")
                pages = loader.load_pdf(file_bytes, file_name=filename)

                st.write(f"✂️ Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})...")
                chunks = splitter.split_pages(pages)

                st.write(f"🧠 Generating embeddings & indexing into ChromaDB ({len(chunks)} chunks)...")
                added = vector_store.add_chunks(chunks)
                total_added += added

            status.update(
                label=f"✅ Successfully indexed {total_added} chunks into Vector DB!",
                state="complete",
                expanded=False,
            )
            st.rerun()

    st.divider()

    # Section 3: Indexed Documents Overview & Actions
    st.markdown("### 📚 Indexed Knowledge Base")
    current_sources = vector_store.get_indexed_sources()
    total_chunks = vector_store.count()

    st.metric("Total Indexed Chunks", total_chunks)

    if current_sources:
        st.write("Indexed Files:")
        for src in current_sources:
            col1, col2 = st.columns([0.75, 0.25])
            with col1:
                st.caption(f"📄 {src}")
            with col2:
                if st.button("❌", key=f"del_{src}", help=f"Delete {src}"):
                    vector_store.delete_source(src)
                    st.rerun()

        # Document Filter for search
        active_doc_filter = st.selectbox(
            "Filter Search by Document",
            options=["All Documents"] + current_sources,
            index=0,
            help="Restrict retrieval to a specific document or search across all.",
        )
        doc_filter = None if active_doc_filter == "All Documents" else active_doc_filter

        # Document Summarization Trigger
        st.markdown("#### 📝 Document Summarization")
        selected_for_summary = st.selectbox(
            "Choose Document to Summarize", options=current_sources, key="summary_select"
        )
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Generating executive summary..."):
                retriever = PDFRetriever(vector_store)
                pipe = RAGPipeline(
                    retriever=retriever,
                    provider=provider,
                    gemini_api_key=api_key,
                    gemini_model=gemini_model,
                    ollama_base_url=ollama_url,
                    ollama_model=ollama_model,
                )
                bot = PDFChatbot(pipe)
                summary_res = bot.summarize_document(selected_for_summary)
                st.session_state.doc_summary = summary_res

        if st.button("🗑️ Reset Entire Vector Store", type="secondary", use_container_width=True):
            vector_store.reset()
            st.session_state.chat_history = []
            st.session_state.doc_summary = ""
            st.rerun()
    else:
        doc_filter = None
        st.warning("No documents indexed yet. Upload a PDF or generate the sample document to begin.")


# =========================================================================
# MAIN APP BODY: Tabs for Chat, Document Explorer, and Evaluation
# =========================================================================
st.markdown('<div class="main-header">📚 AI PDF Assistant — RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Ask questions in English or Arabic, get grounded answers, and inspect exact PDF page citations.'
    '</div>',
    unsafe_allow_html=True,
)

# Badges
st.markdown(
    """
    <div>
        <span class="badge-pill">⚡ LangChain Orchestration</span>
        <span class="badge-pill">🔍 ChromaDB Vector Store</span>
        <span class="badge-pill">🧠 sentence-transformers</span>
        <span class="badge-pill">📑 PyMuPDF Text Extraction</span>
        <span class="badge-pill">🌐 Bilingual EN/AR</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# Construct Pipeline and Chatbot
retriever = PDFRetriever(
    vector_store,
    default_top_k=top_k,
    default_threshold=sim_threshold,
)

rag_pipeline = RAGPipeline(
    retriever=retriever,
    provider=provider,
    gemini_api_key=api_key,
    gemini_model=gemini_model,
    ollama_base_url=ollama_url,
    ollama_model=ollama_model,
)

chatbot = PDFChatbot(rag_pipeline)
# Sync history
chatbot.history = st.session_state.chat_history

tab_chat, tab_summary, tab_eval, tab_docs, tab_arch = st.tabs(
    ["💬 Chat with PDFs", "📝 Document Summary", "📊 RAG Evaluation", "📑 Chunk Explorer", "🏗️ Architecture"]
)

# -------------------------------------------------------------------------
# TAB 1: Chat Interface
# -------------------------------------------------------------------------
with tab_chat:
    # Check if knowledge base is empty
    if vector_store.count() == 0:
        st.info("👋 Welcome! To get started, upload a PDF in the sidebar, or click below to load the built-in Machine Learning sample PDF.")
        if st.button("📘 Load Sample Machine Learning PDF (Bilingual 3-Page Book)"):
            sample_pdf_path = PDFS_DIR / "machine_learning_basics.pdf"
            if not sample_pdf_path.exists():
                from create_sample_pdf import generate_sample_pdf
                generate_sample_pdf(sample_pdf_path)

            loader = PDFDocumentLoader()
            splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            pages = loader.load_pdf(sample_pdf_path)
            chunks = splitter.split_pages(pages)
            vector_store.add_chunks(chunks)
            st.success("Loaded and indexed 'machine_learning_basics.pdf'! Now ask your questions below.")
            st.rerun()

    # Quick Suggestion Prompts
    st.caption("Suggested Questions:")
    col_q1, col_q2, col_q3 = st.columns(3)
    suggested_q = None
    with col_q1:
        if st.button("What is the difference between supervised and unsupervised learning?"):
            suggested_q = "What is the difference between supervised and unsupervised learning?"
    with col_q2:
        if st.button("ما هو الفرق بين التعلم الخاضع للإشراف وغير الخاضع للإشراف؟"):
            suggested_q = "ما هو الفرق بين التعلم الخاضع للإشراف وغير الخاضع للإشراف؟"
    with col_q3:
        if st.button("How can overfitting be prevented?"):
            suggested_q = "How can overfitting be prevented?"

    # Display Conversation History
    for idx, turn in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(turn["user"])

        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])

            # Citations Expandable
            citations = turn.get("citations", [])
            if citations:
                with st.expander(f"📍 Referenced Sources & Pages ({len(citations)})", expanded=False):
                    for cite in citations:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <strong>📄 {cite['source']} — Page {cite['page']}</strong>
                                <span style="float: right; color: #4F46E5; font-size: 0.85rem;">
                                    Relevance: {int(cite.get('similarity', 1.0) * 100)}%
                                </span>
                                <p style="margin-top: 0.4rem; color: #475569; font-style: italic; font-size: 0.9rem;">
                                    "{cite['snippet']}"
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # User Input
    user_query = st.chat_input("Ask a question about your PDF documents...") or suggested_q

    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document vectors & generating grounded answer..."):
                turn_result = chatbot.ask(
                    question=user_query,
                    top_k=top_k,
                    similarity_threshold=sim_threshold,
                    source_filter=doc_filter,
                )

                st.markdown(turn_result["assistant"])

                citations = turn_result.get("citations", [])
                if citations:
                    with st.expander(f"📍 Referenced Sources & Pages ({len(citations)})", expanded=True):
                        for cite in citations:
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <strong>📄 {cite['source']} — Page {cite['page']}</strong>
                                    <span style="float: right; color: #4F46E5; font-size: 0.85rem;">
                                        Relevance: {int(cite.get('similarity', 1.0) * 100)}%
                                    </span>
                                    <p style="margin-top: 0.4rem; color: #475569; font-style: italic; font-size: 0.9rem;">
                                        "{cite['snippet']}"
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        # Update session state history
        st.session_state.chat_history = chatbot.history
        st.rerun()

    # Clear Chat Button
    if st.session_state.chat_history:
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            chatbot.clear_history()
            st.rerun()

# -------------------------------------------------------------------------
# TAB 2: Document Summary
# -------------------------------------------------------------------------
with tab_summary:
    st.subheader("📝 Executive Document Summarization")
    if st.session_state.doc_summary:
        st.markdown(st.session_state.doc_summary)
    else:
        st.info("Select a document in the sidebar and click **'Generate Summary'** to view an executive synthesis.")

# -------------------------------------------------------------------------
# TAB 3: RAG Evaluation Dashboard
# -------------------------------------------------------------------------
with tab_eval:
    st.subheader("📊 RAG Evaluation & Performance Dashboard")
    st.markdown(
        "Measure Context Precision, Keyword Recall, Page Hit Rate, and Response Latency across a standardized test suite."
    )

    if st.button("▶️ Run Evaluation Benchmark"):
        with st.spinner("Running test questions against current vector store..."):
            from evaluation.evaluate import run_benchmark
            summary = run_benchmark(
                top_k=top_k,
                provider=provider,
                gemini_api_key=api_key,
            )

            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Test Cases", summary["total_questions"])
            col2.metric("Avg Latency", f"{summary['avg_latency_seconds']} s")
            col3.metric("Keyword Recall", f"{summary['avg_keyword_recall'] * 100:.1f} %")
            col4.metric("Target Page Hit Rate", f"{summary['page_hit_rate'] * 100:.1f} %")

            st.write("### Test Results Breakdown")
            st.dataframe(summary["details"], use_container_width=True)

# -------------------------------------------------------------------------
# TAB 4: Chunk & Vector Explorer
# -------------------------------------------------------------------------
with tab_docs:
    st.subheader("📑 Chunk & Metadata Inspection")
    if vector_store.count() > 0:
        st.write(f"The vector store currently contains **{vector_store.count()}** active chunks.")
        sample_query = st.text_input("Test semantic search query:", value="supervised learning")
        if sample_query:
            results = vector_store.query_similarity(sample_query, n_results=6)
            for i, r in enumerate(results, 1):
                with st.expander(f"Chunk #{i} | {r['metadata'].get('source')} (Page {r['metadata'].get('page')}) — Sim: {r['similarity']}"):
                    st.json(r["metadata"])
                    st.markdown(f"**Content:**\n\n{r['content']}")
    else:
        st.info("No chunks indexed yet.")

# -------------------------------------------------------------------------
# TAB 5: Architecture & Educational Guide
# -------------------------------------------------------------------------
with tab_arch:
    st.subheader("🏗️ System Architecture & Workflow")
    st.markdown(
        """
        ```mermaid
        flowchart TD
            A["📄 PDF Documents"] --> B["PyMuPDF Loader\\n(Extract Text & Page Metadata)"]
            B --> C["Recursive Text Splitter\\n(Chunk Size: 500, Overlap: 100)"]
            C --> D["Embedding Model\\n(sentence-transformers/all-MiniLM-L6-v2)"]
            D --> E["ChromaDB Vector Store\\n(Persistent Vector Index)"]
            F["💬 User Question\\n(English or Arabic)"] --> G["Contextualizer & Retriever\\n(Top-K & Similarity Threshold)"]
            E --> G
            G --> H["Grounded Context Injection\\n(System Prompt Engineering)"]
            H --> I["LLM\\n(Gemini 3.8 Flash / Ollama)"]
            I --> J["🎯 Grounded Answer + Citations\\n([Source: file.pdf, Page: X])"]
        ```
        """
    )
    st.markdown(
        """
        ### 🔍 How RAG Solves LLM Hallucinations:
        1. **Document Loading**: PyMuPDF extracts text per page, retaining accurate 1-indexed page numbers.
        2. **Semantic Chunking**: Large texts are split into bite-sized segments with overlap to prevent context loss at chunk boundaries.
        3. **Dense Embeddings**: `all-MiniLM-L6-v2` encodes 384-dimensional semantic vectors.
        4. **Vector Retrieval**: Given a question, ChromaDB retrieves only the mathematically closest chunks using cosine similarity.
        5. **Strict Grounding**: The LLM prompt restricts answers exclusively to retrieved chunks, instructing it to admit missing info rather than hallucinating.
        """
    )
