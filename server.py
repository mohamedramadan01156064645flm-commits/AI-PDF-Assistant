"""FastAPI Backend Server for AI PDF Assistant.
Connects the RAG pipeline, ChromaDB vector store, document loaders,
and LLM providers (Gemini / Ollama / Mock) to the React frontend.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
from create_sample_pdf import generate_sample_pdf, generate_deep_learning_pdf

# ─────────────────────────────────────────────────────────────
# APP INIT & CORS
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="AI PDF Assistant API", version="1.0.0")

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
    # Default metadata for sample files if they exist
    defaults = {
        "machine_learning_basics.pdf": {
            "name": "Machine Learning.pdf",
            "pages": 300,
            "size_mb": 12.5,
            "chunks": 11,
            "uploaded_at": "Today",
        },
        "deep_learning_guide.pdf": {
            "name": "Deep Learning.pdf",
            "pages": 250,
            "size_mb": 10.8,
            "chunks": 6,
            "uploaded_at": "Today",
        },
    }
    return defaults


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


def build_pipeline(top_k: Optional[int] = None, threshold: Optional[float] = None, provider: Optional[str] = None):
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
        gemini_api_key=runtime_settings["api_key"],
        gemini_model=runtime_settings["gemini_model"],
        ollama_base_url=runtime_settings["ollama_url"],
        ollama_model=runtime_settings["ollama_model"],
    )
    return retriever, rag


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────
class ChatTurn(BaseModel):
    user: str
    assistant: str
    citations: Optional[List[Dict[str, Any]]] = []


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatTurn]] = []
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    provider: Optional[str] = None


class SummarizeRequest(BaseModel):
    document: str


class EvaluationRequest(BaseModel):
    top_k: Optional[int] = 3
    provider: Optional[str] = "mock"


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
# ENDPOINTS
# ─────────────────────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    sources = vector_store.get_indexed_sources()
    return {
        "status": "online",
        "total_chunks": vector_store.count(),
        "total_documents": len(sources),
        "sources": sources,
    }


@app.get("/api/documents")
def get_documents():
    sources = vector_store.get_indexed_sources()
    meta = load_doc_metadata()
    docs = []
    for src in sources:
        doc_info = meta.get(src, {})
        # Format display name nicely
        disp_name = doc_info.get("name")
        if not disp_name:
            if "machine" in src.lower():
                disp_name = "Machine Learning.pdf"
            elif "deep" in src.lower():
                disp_name = "Deep Learning.pdf"
            else:
                disp_name = src
        docs.append({
            "id": src,
            "filename": src,
            "name": disp_name,
            "pages": doc_info.get("pages", 1),
            "size_mb": doc_info.get("size_mb", 1.0),
            "chunks": doc_info.get("chunks", 1),
            "uploaded_at": doc_info.get("uploaded_at", "Recently"),
        })
    return {"documents": docs, "total_chunks": vector_store.count()}


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
):
    loader = PDFDocumentLoader()
    splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    meta = load_doc_metadata()
    indexed_files = []

    for file in files:
        contents = await file.read()
        filename = file.filename
        save_path = PDFS_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(contents)

        pages = loader.load_pdf(contents, file_name=filename)
        chunks = splitter.split_pages(pages)
        vector_store.add_chunks(chunks)

        meta[filename] = {
            "name": filename,
            "pages": len(pages),
            "size_mb": round(len(contents) / 1024 / 1024, 2) or 0.1,
            "chunks": len(chunks),
            "uploaded_at": datetime.now().strftime("%b %d, %Y"),
        }
        indexed_files.append({
            "name": filename,
            "pages": len(pages),
            "chunks": len(chunks),
        })

    save_doc_metadata(meta)
    return {"status": "success", "indexed": indexed_files, "total_chunks": vector_store.count()}


@app.post("/api/load-sample")
def load_sample_documents():
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
            }

    save_doc_metadata(meta)
    return {"status": "success", "total_chunks": vector_store.count(), "sources": vector_store.get_indexed_sources()}


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    sources = vector_store.get_indexed_sources()
    if filename not in sources:
        raise HTTPException(status_code=404, detail="Document not found")
    vector_store.delete_source(filename)
    meta = load_doc_metadata()
    meta.pop(filename, None)
    save_doc_metadata(meta)
    return {"status": "deleted", "filename": filename, "remaining_chunks": vector_store.count()}


@app.post("/api/reset")
def reset_vectorstore():
    vector_store.reset()
    save_doc_metadata({})
    return {"status": "cleared", "total_chunks": 0}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if vector_store.count() == 0:
        return {
            "answer": "No documents are currently indexed. Please upload a PDF or load the sample documents first.",
            "citations": [],
            "grounded": False,
            "timestamp": datetime.now().strftime("%I:%M %p"),
        }

    retriever, rag = build_pipeline(
        top_k=req.top_k,
        threshold=req.similarity_threshold,
        provider=req.provider,
    )
    bot = PDFChatbot(rag)
    bot.history = [t.model_dump() for t in (req.history or [])]

    turn = bot.ask(
        question=req.question,
        top_k=req.top_k or runtime_settings["top_k"],
        similarity_threshold=req.similarity_threshold or runtime_settings["similarity_threshold"],
    )

    # Clean up and normalize citation display names
    meta = load_doc_metadata()
    clean_citations = []
    for c in turn.get("citations", []):
        src_raw = c.get("source", "")
        doc_info = meta.get(src_raw, {})
        disp_name = doc_info.get("name")
        if not disp_name:
            if "machine" in src_raw.lower():
                disp_name = "Machine Learning.pdf"
            elif "deep" in src_raw.lower():
                disp_name = "Deep Learning.pdf"
            else:
                disp_name = src_raw
        clean_citations.append({
            "source": disp_name,
            "raw_source": src_raw,
            "page": c.get("page", 1),
            "snippet": c.get("snippet", "")[:120],
        })

    return {
        "answer": turn.get("assistant", ""),
        "citations": clean_citations,
        "grounded": len(clean_citations) > 0,
        "timestamp": datetime.now().strftime("%I:%M %p"),
    }


@app.post("/api/summarize")
def summarize_doc(req: SummarizeRequest):
    retriever, rag = build_pipeline()
    bot = PDFChatbot(rag)
    summary = bot.summarize_document(req.document)
    return {"document": req.document, "summary": summary}


@app.post("/api/evaluation")
def run_evaluation(req: EvaluationRequest):
    from evaluation.evaluate import run_benchmark
    summary = run_benchmark(
        top_k=req.top_k or 3,
        provider=req.provider or "mock",
        gemini_api_key=runtime_settings["api_key"],
    )
    return summary


@app.get("/api/settings")
def get_settings():
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
