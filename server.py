"""FastAPI Backend Server for Universal & Multi-PDF AI Assistant.
Features:
- Dual-Mode Operation: Universal General AI + Multi-PDF RAG Intelligence
- Hybrid Query Routing (General AI Chat, Document RAG, Hybrid Reasoning)
- Multi-document upload, deletion, deduplication, and metadata tracking
- Document Intelligence API (Summaries, Cross-document Comparison, Study Notes, Q&A, Key Concepts)
- Clean Output: Completely eliminates visible citations, document tags, and source cards from user responses
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    PDFS_DIR, VECTORSTORE_DIR,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_GEMINI_MODEL, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL,
)
from src.document_loader import PDFDocumentLoader
from src.text_splitter import DocumentSplitter
from src.vector_store import ChromaVectorStore
from src.retriever import PDFRetriever
from src.rag_pipeline import RAGPipeline
from src.chatbot import PDFChatbot
from src.pdf_exporter import export_intelligence_to_pdf
from create_sample_pdf import generate_sample_pdf, generate_deep_learning_pdf

# ─────────────────────────────────────────────────────────────
# APP INIT & CORS
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Universal & Multi-PDF AI Assistant API",
    description="Intelligent AI Assistant supporting General Knowledge Q&A and Advanced Multi-PDF RAG",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Vector Store Singleton
vector_store = ChromaVectorStore()

# Metadata persistence file
META_FILE = VECTORSTORE_DIR / "doc_metadata.json"


def load_doc_metadata() -> Dict[str, Any]:
    if META_FILE.exists():
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_doc_metadata(meta: Dict[str, Any]):
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# Runtime Settings state
runtime_settings = {
    "provider": "gemini",
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "gemini_model": DEFAULT_GEMINI_MODEL,
    "ollama_url": DEFAULT_OLLAMA_BASE_URL,
    "ollama_model": DEFAULT_OLLAMA_MODEL,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "top_k": DEFAULT_TOP_K,
    "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
}


def get_current_api_key() -> str:
    key = (runtime_settings.get("api_key") or "").strip()
    from dotenv import load_dotenv
    load_dotenv(override=True)
    env_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key and env_key:
        key = env_key
        runtime_settings["api_key"] = key
    env_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
    if env_model:
        runtime_settings["gemini_model"] = env_model
    env_thresh = os.getenv("SIMILARITY_THRESHOLD")
    if env_thresh:
        try:
            runtime_settings["similarity_threshold"] = float(env_thresh)
        except ValueError:
            pass
    return key


def build_pipeline(
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    provider: Optional[str] = None
):
    k = top_k if top_k is not None else runtime_settings["top_k"]
    thresh = threshold if threshold is not None else runtime_settings["similarity_threshold"]
    prov = provider or runtime_settings["provider"]

    retriever = PDFRetriever(
        vector_store=vector_store,
        default_top_k=k,
        default_threshold=thresh,
    )
    rag = RAGPipeline(
        retriever=retriever,
        provider=prov,
        gemini_api_key=get_current_api_key(),
        gemini_model=runtime_settings["gemini_model"],
        ollama_base_url=runtime_settings["ollama_url"],
        ollama_model=runtime_settings["ollama_model"],
    )
    return retriever, rag


# ─────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────
class ChatTurn(BaseModel):
    user: str
    assistant: str
    mode: Optional[str] = "GENERAL"


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatTurn]] = []
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    provider: Optional[str] = None
    mode: Optional[str] = None  # "auto", "GENERAL", "RAG", "HYBRID"
    source_filter: Optional[str] = None


class DocumentIntelligenceRequest(BaseModel):
    action: str  # "summarize" | "compare" | "study_notes" | "qa" | "key_concepts"
    document: str
    secondary_document: Optional[str] = None


class SummarizeRequest(BaseModel):
    document: str


class EvaluationRequest(BaseModel):
    top_k: Optional[int] = 3
    provider: Optional[str] = "mock"


class ExportPDFRequest(BaseModel):
    content: str
    title: Optional[str] = "Document Intelligence Report"
    action: Optional[str] = "summarize"
    document: Optional[str] = "Uploaded Document"


class SettingsUpdate(BaseModel):
    provider: Optional[str] = None
    api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None


# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    sources = vector_store.get_indexed_sources()
    get_current_api_key()
    return {
        "status": "online",
        "total_chunks": vector_store.count(),
        "total_documents": len(sources),
        "sources": sources,
        "active_model": runtime_settings["gemini_model"],
        "provider": runtime_settings["provider"],
        "has_api_key": bool(runtime_settings.get("api_key")),
    }


@app.get("/api/documents")
def get_documents():
    sources = vector_store.get_indexed_sources()
    meta = load_doc_metadata()
    docs = []
    for src in sources:
        doc_info = meta.get(src, {})
        disp_name = doc_info.get("name") or src
        docs.append({
            "id": src,
            "filename": src,
            "name": disp_name,
            "pages": doc_info.get("pages", 1),
            "size_mb": doc_info.get("size_mb", 1.0),
            "chunks": doc_info.get("chunks", 1),
            "uploaded_at": doc_info.get("uploaded_at", "Recently"),
            "file_hash": doc_info.get("file_hash", ""),
            "is_scanned": doc_info.get("is_scanned", False),
            "status": doc_info.get("status", "ready"),
        })
    return {"documents": docs, "total_chunks": vector_store.count()}


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
):
    """
    Uploads and indexes multiple PDF documents.
    Detects duplicate uploads via content hashing.
    Preserves all previously uploaded documents.
    """
    loader = PDFDocumentLoader()
    splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    meta = load_doc_metadata()
    indexed_files = []
    duplicates = []

    for file in files:
        contents = await file.read()
        filename = file.filename
        file_hash = loader.calculate_file_hash(contents)

        # Check for existing duplicate by hash
        already_indexed = False
        for existing_name, info in meta.items():
            if info.get("file_hash") == file_hash and existing_name in vector_store.get_indexed_sources():
                already_indexed = True
                duplicates.append({"filename": filename, "existing_as": existing_name})
                break

        if already_indexed:
            continue

        save_path = PDFS_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(contents)

        try:
            pages = loader.load_pdf(contents, file_name=filename)
            chunks = splitter.split_pages(pages)
            if chunks:
                vector_store.add_chunks(chunks)

            has_scanned = any(p.metadata.get("is_scanned", False) for p in pages)

            meta[filename] = {
                "name": filename,
                "pages": len(pages),
                "size_mb": round(len(contents) / 1024 / 1024, 2) or 0.1,
                "chunks": len(chunks),
                "uploaded_at": datetime.now().strftime("%b %d, %Y"),
                "file_hash": file_hash,
                "is_scanned": has_scanned,
                "status": "ready",
            }
            indexed_files.append({
                "name": filename,
                "pages": len(pages),
                "chunks": len(chunks),
                "is_scanned": has_scanned,
            })
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process PDF '{filename}': {str(e)}"
            )

    save_doc_metadata(meta)
    return {
        "status": "success",
        "indexed": indexed_files,
        "duplicates_skipped": duplicates,
        "total_chunks": vector_store.count(),
        "total_documents": len(vector_store.get_indexed_sources()),
    }


@app.post("/api/load-sample")
def load_sample_documents():
    """Seeds the sample Machine Learning and Deep Learning PDFs into the store."""
    p1 = PDFS_DIR / "machine_learning_basics.pdf"
    p2 = PDFS_DIR / "deep_learning_guide.pdf"
    if not p1.exists():
        generate_sample_pdf(p1)
    if not p2.exists():
        generate_deep_learning_pdf(p2)

    loader = PDFDocumentLoader()
    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=100)
    meta = load_doc_metadata()

    for p, display_name, est_pages, est_mb in [
        (p1, "Machine Learning.pdf", 300, 12.5),
        (p2, "Deep Learning.pdf", 250, 10.8),
    ]:
        if p.name not in vector_store.get_indexed_sources():
            pages = loader.load_pdf(p)
            chunks = splitter.split_pages(pages)
            vector_store.add_chunks(chunks)
            meta[p.name] = {
                "name": display_name,
                "pages": est_pages,
                "size_mb": est_mb,
                "chunks": len(chunks),
                "uploaded_at": "Sample Library",
                "file_hash": loader.calculate_file_hash(p),
                "status": "ready",
            }

    save_doc_metadata(meta)
    return {
        "status": "success",
        "total_chunks": vector_store.count(),
        "sources": vector_store.get_indexed_sources(),
    }


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    """Deletes an individual document and its chunks without disturbing other documents."""
    sources = vector_store.get_indexed_sources()
    if filename not in sources:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    vector_store.delete_source(filename)
    meta = load_doc_metadata()
    meta.pop(filename, None)
    save_doc_metadata(meta)
    return {
        "status": "deleted",
        "filename": filename,
        "remaining_chunks": vector_store.count(),
        "remaining_documents": len(vector_store.get_indexed_sources()),
    }


@app.post("/api/reset")
def reset_vectorstore():
    """Resets the vector database and clears metadata."""
    vector_store.reset()
    save_doc_metadata({})
    return {"status": "cleared", "total_chunks": 0, "total_documents": 0}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Main Chat API Endpoint.
    Supports:
    - Mode A: Universal General AI Q&A (no documents needed)
    - Mode B: Multi-PDF RAG Q&A
    - Mode C: Hybrid Reasoning
    Guarantees Clean Output: Returns zero visible citations/sources to UI (Requirement 11).
    """
    retriever, rag = build_pipeline(
        top_k=req.top_k,
        threshold=req.similarity_threshold,
        provider=req.provider,
    )
    bot = PDFChatbot(rag)
    bot.history = [t.model_dump() for t in (req.history or [])]

    # Convert forced mode if sent as "auto"
    f_mode = None if (not req.mode or req.mode == "auto") else req.mode

    turn = bot.ask(
        question=req.question,
        top_k=req.top_k or runtime_settings["top_k"],
        similarity_threshold=req.similarity_threshold or runtime_settings["similarity_threshold"],
        source_filter=req.source_filter,
        forced_mode=f_mode,
    )

    # In strict accordance with Requirement 11:
    # "The chatbot must provide clean, professional answers without displaying any sources or citation references in the user interface."
    # We return clean answer, mode, and grounded status, keeping UI free of citation cards/pills.
    return {
        "answer": turn.get("assistant", ""),
        "mode": turn.get("mode", "GENERAL"),
        "grounded": turn.get("grounded", False),
        "citations": [],  # Clean output: no citations displayed in UI
        "timestamp": datetime.now().strftime("%I:%M %p"),
    }


@app.get("/api/pdfs/{filename}")
def get_pdf_file(filename: str):
    """
    Serves the actual uploaded PDF for interactive viewing in modal.
    Enforces secure filename sanitization.
    """
    safe_name = os.path.basename(filename)
    file_path = PDFS_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"PDF document '{safe_name}' not found on server.")
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=safe_name,
        headers={"Content-Disposition": f"inline; filename=\"{safe_name}\""}
    )


@app.post("/api/document-intelligence")
def document_intelligence(req: DocumentIntelligenceRequest):
    """
    Executes advanced document intelligence actions:
    - summarize: Executive summary
    - explain: Step-by-step document breakdown
    - study_notes: High-yield study notes & definitions
    - qa: Practice Q&A questions & model answers
    - key_concepts: Technical definitions & glossary
    - key_points: Core bullet points & takeaways
    - difficult_concepts: Intuitive analogies for complex topics
    - flashcards: Active recall flashcards
    - quiz: Multiple-choice self quiz with rationales
    - compare: Cross-document comparative analysis
    """
    retriever, rag = build_pipeline()
    bot = PDFChatbot(rag)

    action = req.action.lower().strip()
    doc_a = req.document

    if action == "summarize":
        result = bot.summarize_document(doc_a)
    elif action == "explain":
        result = bot.explain_document(doc_a)
    elif action == "study_notes":
        result = bot.generate_study_notes(doc_a)
    elif action == "qa":
        result = bot.generate_qa(doc_a)
    elif action == "key_concepts":
        result = bot.extract_key_concepts(doc_a)
    elif action in ["key_points", "highlights"]:
        result = bot.extract_key_points(doc_a)
    elif action in ["difficult_concepts", "simplify"]:
        result = bot.explain_difficult_concepts(doc_a)
    elif action == "flashcards":
        result = bot.create_flashcards(doc_a)
    elif action == "quiz":
        result = bot.generate_quiz(doc_a)
    elif action == "compare":
        if not req.secondary_document:
            raise HTTPException(status_code=400, detail="Secondary document required for comparison.")
        result = bot.compare_documents(doc_a, req.secondary_document)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported document intelligence action: '{action}'")

    return {
        "action": action,
        "document": doc_a,
        "secondary_document": req.secondary_document,
        "result": result,
    }


@app.post("/api/export-pdf")
def export_pdf_endpoint(req: ExportPDFRequest):
    """
    Generates and downloads a beautifully styled, publication-grade PDF report
    from Markdown Intelligence output.
    """
    if not req.content or not req.content.strip():
        raise HTTPException(status_code=400, detail="Cannot export empty intelligence content.")

    clean_title = req.title or "Document Intelligence Report"
    pdf_bytes = export_intelligence_to_pdf(
        content=req.content,
        title=clean_title,
        action=req.action or "analysis",
        document_name=req.document or "Uploaded Document",
    )

    safe_action = (req.action or "report").replace(" ", "_").lower()
    safe_doc = os.path.splitext(os.path.basename(req.document or "document"))[0].replace(" ", "_")
    filename = f"{safe_action}_{safe_doc}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@app.post("/api/summarize")
def summarize_doc(req: SummarizeRequest):
    retriever, rag = build_pipeline()
    bot = PDFChatbot(rag)
    summary = bot.summarize_document(req.document)
    return {"document": req.document, "summary": summary}


@app.get("/api/evaluation")
def get_latest_evaluation():
    """Returns the latest evaluation report if available, or signals no results."""
    from config import EVALUATION_DIR
    report_file = EVALUATION_DIR / "latest_evaluation_report.json"
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"status": "success", "report": data}
        except Exception:
            pass
    return {"status": "no_results", "report": None}


@app.post("/api/evaluation")
def run_evaluation(req: EvaluationRequest):
    """Runs a real benchmark evaluation using the actual RAG pipeline."""
    from evaluation.evaluate import run_benchmark
    prov = req.provider if (req.provider and req.provider != "mock") else runtime_settings["provider"]
    summary = run_benchmark(
        top_k=req.top_k or 4,
        provider=prov,
        gemini_api_key=get_current_api_key(),
    )
    return {"status": "success", "report": summary}


@app.get("/api/settings")
def get_settings():
    get_current_api_key()
    return runtime_settings


@app.post("/api/settings")
def update_settings(update: SettingsUpdate):
    for k, v in update.model_dump(exclude_unset=True).items():
        if v is not None:
            runtime_settings[k] = v
            if k == "api_key":
                os.environ["GEMINI_API_KEY"] = v
    return {"status": "updated", "settings": runtime_settings}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
