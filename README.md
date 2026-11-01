# AI PDF Assistant — Full-Stack RAG Website

A modern, full-stack **RAG-based AI PDF Assistant** web application built with **React**, **Tailwind CSS v4**, **TypeScript**, and **FastAPI**.

Allows users to upload multiple PDF documents, ask complex questions in Arabic or English, and receive grounded answers with exact page citations, document previews, and zero hallucinations.

---

## 🚀 Features

- **Pixel-Perfect Modern UI:** Built with React 19, Tailwind CSS v4, and Lucide React icons.
- **5 Functional Interactive Pages:**
  1. 💬 **Chat:** Real-time conversational interface with multi-turn memory, suggestion chips, citations, and grounded answer verification.
  2. 📄 **Documents:** Multi-file drag & drop PDF upload, chunking sliders, real-time indexing status, sample library loader, and delete/reset controls.
  3. ✨ **Summarize PDF:** Executive document summarization with key topics and takeaways.
  4. 📊 **Evaluation Dashboard:** Automated RAG evaluation measuring Latency, Keyword Recall, and Page Hit Rate across test datasets.
  5. ⚙️ **Settings:** Model switching (Google Gemini API / Ollama / Mock mode), API key configuration, and RAG hyperparameter tuning (Chunk size, overlap, Top-K, threshold).
- **Production RAG Engine:** PyMuPDF, Sentence-Transformers (`all-MiniLM-L6-v2`), ChromaDB vector store, and Gemini / Ollama / Mock LLM generation.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React, TypeScript, Vite, Tailwind CSS v4, Lucide Icons |
| **Backend API** | FastAPI, Uvicorn, Python 3.11 |
| **Vector DB** | ChromaDB (Persistent storage) |
| **Embeddings** | HuggingFace Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **PDF Extraction**| PyMuPDF (PyMuPDF 2.0+) with Unicode Normalization |
| **LLM Orchestration** | Google GenAI SDK (`gemini-3.8-flash`), Ollama, Extractive Mock |

---

## ⚡ Quick Start

### 1. Prerequisites
- **Python 3.11+**
- **Node.js 18+** & npm

### 2. Run Both Frontend & Backend (One Click)
Double-click `start.bat` or run:
```bash
# Terminal 1: Backend
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

### 3. Open in Browser
- **Frontend App:** [http://localhost:5173](http://localhost:5173)
- **FastAPI Interactive Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
