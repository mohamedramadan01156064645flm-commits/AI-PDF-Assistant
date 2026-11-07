"""RAG Pipeline module connecting Retriever with LLM providers (Gemini API, Ollama, Mock).
Features:
- Dual Functionality: Universal AI Assistant + Multi-PDF RAG Assistant
- Intelligent Hybrid Query Router (Mode A: General, Mode B: PDF RAG, Mode C: Hybrid)
- Multi-document balanced semantic retrieval across all uploaded files
- Document Intelligence (Summarization, Comparison, Study Notes, Key Concepts, Q&A)
- Clean Answer Output: Strict removal of visible citations, source markers, and references
- Bilingual & Egyptian Arabic natural conversational support
"""

import os
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from src.retriever import PDFRetriever, RetrievedChunk
from config import (
    GEMINI_API_KEY,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
)

# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATES (STRICT CLEAN OUTPUT ENFORCED)
# ─────────────────────────────────────────────────────────────

SYSTEM_UNIVERSAL_PROMPT = """You are an expert, highly intelligent Universal AI Assistant with deep mastery across Artificial Intelligence, Machine Learning, Data Science, Software Engineering, Mathematics, Science, Economics, Education, and General Knowledge.

Follow these guidelines for every response:
1. Provide an accurate, comprehensive, logically structured, and professional answer.
2. Adapt answer depth to the complexity of the question: concise for simple queries, detailed and step-by-step for complex topics.
3. Use clean Markdown formatting: headings (##, ###), bullet points, bold key terms, and formatted code blocks where applicable.
4. Language Rules:
   - If the user asks in Arabic (including Egyptian Arabic), respond in natural, fluent, professional Arabic.
   - If the user asks in English, respond in clear, professional English.
5. CLEAN OUTPUT RULE: Do NOT include any source tags, file names, page numbers, or any 'Sources:' or 'References:' section. Output only the clean, final answer.

USER QUESTION:
{question}

PROFESSIONAL ANSWER:"""


SYSTEM_RAG_PROMPT = """You are an expert AI Research Assistant specialized in deep, accurate document understanding and multi-PDF synthesis.

DOCUMENT CONTEXT EXTRACTED FROM UPLOADED FILES:
{context}

ANSWERING RULES:
1. Answer the question based directly and factually on the DOCUMENT CONTEXT above. Do not hallucinate or invent document facts.
2. Directly address the specific question with thorough explanations, organized sections, and structured bullet points.
3. If the context discusses multiple uploaded documents, synthesize the information across documents coherently.
4. If the provided documents do not contain information to answer the question, state that limitation clearly and politely.
5. Language Rules:
   - If the user asks in Arabic, respond entirely in professional Arabic.
   - If the user asks in English, respond entirely in clear, professional English.
6. CRITICAL CLEAN OUTPUT RULE: Do NOT include any citations, source tags like [Source: ...], page numbers, document links, or any 'Sources:' or 'References:' section in your response. Provide only the clean, authoritative, well-structured answer.

USER QUESTION:
{question}

PROFESSIONAL GROUNDED ANSWER:"""


SYSTEM_HYBRID_PROMPT = """You are an advanced AI Assistant capable of synthesizing document-grounded findings with broad real-world knowledge and technical best practices.

DOCUMENT CONTEXT FROM UPLOADED FILES:
{context}

HYBRID REASONING RULES:
1. Ground the specific facts, metrics, and case findings in the DOCUMENT CONTEXT above.
2. Seamlessly enrich and connect those findings with broader industry standards, practical methodologies, and expert reasoning.
3. Clearly and logically distinguish between what the uploaded documents specifically establish and additional general knowledge or best practices.
4. Language Rules:
   - If the user asks in Arabic, respond in natural, professional Arabic.
   - If the user asks in English, respond in professional English.
5. CRITICAL CLEAN OUTPUT RULE: Do NOT include any citations, source tags like [Source: ...], page numbers, or any 'Sources:' or 'References:' section in your response. Output only the clean, authoritative answer.

USER QUESTION:
{question}

PROFESSIONAL HYBRID ANSWER:"""


# ─────────────────────────────────────────────────────────────
# POST-PROCESSING SANITIZER (REQUIREMENT 11 COMPLIANCE)
# ─────────────────────────────────────────────────────────────

def clean_answer_output(text: str) -> str:
    """
    Enforces Requirement 11 by completely stripping any visible citations,
    source markers, document names, page references, or trailing source lists.
    """
    if not text:
        return ""

    # Remove inline tags: [Source: ...], [Page: ...], [Doc: ...], (Source: ...)
    # Also catches inline context references like [machine_learning_basics.pdf, Page 1]
    patterns = [
        r"\[Source:\s*[^\]]*\]",
        r"\[المصدر:\s*[^\]]*\]",
        r"\[Page:\s*[^\]]*\]",
        r"\[الصفحة:\s*[^\]]*\]",
        r"\[Document\s*\d+[^\]]*\]",
        r"\[Document:\s*[^\]]*\]",
        r"\(Source:\s*[^\)]*\)",
        r"\(المصدر:\s*[^\)]*\)",
        r"\[Source\s*\d+\]",
        r"\[\s*Page\s*\d+\s*\]",
        # Catch [filename.pdf, Page N] or [filename.pdf, Page N], [...]
        r"\[[^\]]*\.pdf\s*,\s*Page\s*\d+[^\]]*\]",
        # Catch (filename.pdf, Page N)
        r"\([^\)]*\.pdf\s*,\s*Page\s*\d+[^\)]*\)",
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)

    # Remove trailing citation/reference sections
    section_patterns = [
        r"(?im)^\s*#{1,4}\s*(Sources|References|المصادر|المراجع|Sources Referenced)\s*:?.*$",
        r"(?im)^\s*\*\*(Sources|References|المصادر|المراجع|Sources Referenced)\s*:?\*\*.*$",
        r"(?im)^\s*(Sources|References|المصادر|المراجع|Sources Referenced)\s*:.*$",
    ]
    for sp in section_patterns:
        cleaned = re.sub(sp, "", cleaned)

    # Clean up any leftover double brackets or dangling empty bullet points
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    cleaned = re.sub(r"^\s*[\*\-]\s*$", "", cleaned, flags=re.MULTILINE)

    # Clean multiple blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ─────────────────────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG Pipeline featuring:
    - Mode A: Universal General AI Chat
    - Mode B: PDF-Based RAG
    - Mode C: Hybrid Reasoning
    - Document Intelligence actions
    - Clean Output without visible citations
    """

    def __init__(
        self,
        retriever: PDFRetriever,
        provider: str = "gemini",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ):
        self.retriever = retriever
        self.provider = provider
        self.gemini_api_key = gemini_api_key or GEMINI_API_KEY
        self.gemini_model = gemini_model
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model

    def _call_gemini(self, prompt: str) -> str:
        """
        Invokes Google Gemini with high-resilience REST transport and automatic model fallback.
        Uses direct HTTPS REST queries to avoid Windows IPv6 socket conflicts (WinError 10051),
        with fallback to google-genai SDK.
        """
        key = (self.gemini_api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        if not key:
            from dotenv import load_dotenv
            load_dotenv(override=True)
            key = (os.getenv("GEMINI_API_KEY") or "").strip()

        if not key:
            raise ValueError("GEMINI_API_KEY_MISSING")

        models_to_try = [self.gemini_model]
        for fallback in [
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-flash-lite-latest",
        ]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        last_error = None
        # 1. Primary path: Direct REST API via requests (immune to WinError 10051 IPv6 glitches)
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            try:
                res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=25)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts)
                        if text.strip():
                            return text.strip()
                elif res.status_code in [429, 503]:
                    # High demand or quota spike, try next fallback model immediately
                    last_error = Exception(f"HTTP {res.status_code} on {model_name}: {res.text[:120]}")
                    continue
                elif res.status_code in [401, 403]:
                    raise ValueError("API_KEY_INVALID")
                else:
                    last_error = Exception(f"HTTP {res.status_code} on {model_name}: {res.text[:120]}")
            except (ValueError, requests.exceptions.RequestException) as re_err:
                if isinstance(re_err, ValueError):
                    raise
                last_error = re_err
                continue

        # 2. Secondary path: SDK fallback
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1beta", timeout=20.0))
            for model_name in ["gemini-3.1-flash-lite", "gemini-3.8-flash"]:
                try:
                    resp = client.models.generate_content(model=model_name, contents=prompt)
                    if resp and resp.text:
                        return resp.text.strip()
                except Exception as sdk_e:
                    last_error = sdk_e
                    continue
        except Exception:
            pass

        if last_error:
            raise last_error
        return "No response generated by model."

    def _call_ollama(self, prompt: str) -> str:
        """Invokes local Ollama instance."""
        url = f"{self.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "No response from Ollama.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.ollama_base_url}: {e}"
            ) from e

    def _call_llm(self, prompt: str) -> str:
        """Invokes active LLM provider."""
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "ollama":
            return self._call_ollama(prompt)
        else:
            return "Mock provider active."

    def _translate_to_english_keywords(self, text: str) -> str:
        """Translates an Arabic search query into concise English keywords for semantic vector retrieval."""
        prompt = (
            "Translate the following Arabic query into concise English search keywords for document retrieval. "
            "Return ONLY the English search phrase with no quotes or conversational filler:\n"
            f"{text}"
        )
        try:
            res = self._call_llm(prompt)
            clean = res.strip().split("\n")[0].strip('"').strip()
            return clean if clean else text
        except Exception:
            return text

    def classify_intent(self, question: str) -> Tuple[str, List[RetrievedChunk]]:
        """
        Intelligently determines whether the query requires:
        - Mode A ("GENERAL"): Universal AI Assistant (no document context needed)
        - Mode B ("RAG"): Document-grounded Q&A
        - Mode C ("HYBRID"): Combination of document findings and general AI reasoning
        """
        total_chunks = self.retriever.vector_store.count()
        if total_chunks == 0:
            return "GENERAL", []

        q_lower = question.lower().strip()
        is_arabic = any("\u0600" <= c <= "\u06FF" for c in question)

        # 1. Explicit document cue check
        doc_cues = [
            "document", "documents", "pdf", "pdfs", "file", "files", "uploaded", "paper",
            "resume", "cv", "author", "page", "pages", "chapter", "section",
            "المستند", "المستندات", "الملف", "الملفات", "المرفوعة", "الملف المرفوع",
            "السيرة الذاتية", "الصفحة", "الفصل", "في الملف", "في المستند"
        ]
        has_doc_cue = any(cue in q_lower for cue in doc_cues)

        # Also check if any indexed document title or filename is mentioned
        indexed_sources = self.retriever.vector_store.get_indexed_sources()
        for src in indexed_sources:
            stem = src.split(".")[0].lower()
            clean_stem = stem.replace("_", " ").replace("-", " ")
            if clean_stem in q_lower or src.lower() in q_lower:
                has_doc_cue = True
                break

        # 2. Hybrid cue check (asking to connect document content to general practices/extensions)
        hybrid_cues = [
            "how to improve", "best practices", "how would you implement", "industry standard",
            "in production", "in the real world", "how does this compare to", "alternatives to",
            "ازاي اطور", "افضل الممارسات", "في بيئة العمل", "مقارنة بالسوق"
        ]
        has_hybrid_cue = any(cue in q_lower for cue in hybrid_cues)

        # 3. Retrieve chunks with flexible threshold
        search_query = question
        if is_arabic:
            search_query = self._translate_to_english_keywords(question)

        chunks = self.retriever.retrieve(
            query=search_query,
            top_k=4,
            similarity_threshold=0.15,
        )

        if not chunks and is_arabic and search_query != question:
            chunks = self.retriever.retrieve(
                query=question,
                top_k=4,
                similarity_threshold=0.15,
            )

        # Decision matrix
        if has_doc_cue:
            if has_hybrid_cue and chunks:
                return "HYBRID", chunks
            return "RAG", chunks

        # If no explicit document cue was given:
        if chunks:
            # Check maximum similarity score
            max_sim = max(c.similarity for c in chunks)
            if max_sim >= 0.22:
                if has_hybrid_cue:
                    return "HYBRID", chunks
                return "RAG", chunks

        # Otherwise, question is general knowledge
        return "GENERAL", []

    def generate_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
        forced_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end question answering according to routed mode.
        Returns clean answer text without visible citations, while preserving
        internal metadata for the RAG pipeline.
        """
        is_arabic = any("\u0600" <= c <= "\u06FF" for c in question)

        # Determine mode and retrieved chunks
        if forced_mode in ["GENERAL", "RAG", "HYBRID"]:
            mode = forced_mode
            if mode != "GENERAL":
                search_q = self._translate_to_english_keywords(question) if is_arabic else question
                chunks = self.retriever.retrieve(
                    query=search_q,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    source_filter=source_filter,
                )
            else:
                chunks = []
        else:
            mode, chunks = self.classify_intent(question)

        raw_answer = ""
        error_msg = None

        try:
            if mode == "GENERAL":
                prompt = SYSTEM_UNIVERSAL_PROMPT.format(question=question)
                raw_answer = self._call_llm(prompt)

            elif mode == "HYBRID":
                context_str = self.retriever.format_context_for_prompt(chunks)
                prompt = SYSTEM_HYBRID_PROMPT.format(
                    context=context_str,
                    question=question,
                )
                raw_answer = self._call_llm(prompt)

            else:  # Mode B: RAG
                if not chunks:
                    indexed_sources = self.retriever.vector_store.get_indexed_sources()
                    sources_str = ", ".join(indexed_sources) if indexed_sources else "None"
                    if is_arabic:
                        raw_answer = (
                            f"لم أتمكن من العثور على معلومات محددة حول هذا السؤال في المستندات المرفوعة حالياً.\n\n"
                            f"المستندات المتاحة تغطي موضوعات أخرى. يمكنك سؤالي عن محتويات الملفات أو طرح أي سؤال عام آخر."
                        )
                    else:
                        raw_answer = (
                            f"I could not find specific information answering this question in the currently uploaded documents.\n\n"
                            f"You are welcome to ask questions directly about the uploaded content or ask any general AI question."
                        )
                else:
                    context_str = self.retriever.format_context_for_prompt(chunks)
                    prompt = SYSTEM_RAG_PROMPT.format(
                        context=context_str,
                        question=question,
                    )
                    raw_answer = self._call_llm(prompt)

        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()

            # Classify error type for user-friendly messaging
            if "GEMINI_API_KEY_MISSING" in error_msg:
                if is_arabic:
                    raw_answer = (
                        "⚠️ **مفتاح الـ API غير مُعدّ**\n\n"
                        "لم يتم العثور على مفتاح Gemini API. يرجى إضافة المفتاح في إعدادات التطبيق أو ملف `.env` للحصول على إجابات ذكية.\n\n"
                        "يمكنك الحصول على مفتاح مجاني من [Google AI Studio](https://aistudio.google.com/)."
                    )
                else:
                    raw_answer = (
                        "⚠️ **API Key Not Configured**\n\n"
                        "No Gemini API key was found. Please add your API key in the application settings or `.env` file to enable AI-powered answers.\n\n"
                        "You can get a free key from [Google AI Studio](https://aistudio.google.com/)."
                    )
            elif any(kw in error_lower for kw in ["unreachable network", "connecterror", "connection refused", "timed out", "timeout"]):
                if is_arabic:
                    raw_answer = (
                        "⚠️ **مشكلة في الاتصال بالإنترنت**\n\n"
                        "لم أتمكن من الاتصال بخدمة Gemini AI حالياً. يرجى التحقق من اتصالك بالإنترنت والمحاولة مرة أخرى."
                    )
                else:
                    raw_answer = (
                        "⚠️ **Network Connection Issue**\n\n"
                        "Unable to reach the Gemini AI service at this time. Please check your internet connection and try again."
                    )
            elif any(kw in error_lower for kw in ["429", "quota", "rate limit", "resource exhausted"]):
                if is_arabic:
                    raw_answer = (
                        "⚠️ **تم تجاوز حد الاستخدام**\n\n"
                        "تم الوصول إلى الحد الأقصى لعدد الطلبات المسموح بها. يرجى الانتظار بضع ثوان والمحاولة مرة أخرى."
                    )
                else:
                    raw_answer = (
                        "⚠️ **Rate Limit Exceeded**\n\n"
                        "The API request limit has been reached. Please wait a few seconds and try again."
                    )
            elif any(kw in error_lower for kw in ["401", "403", "invalid", "api_key_invalid", "permission"]):
                if is_arabic:
                    raw_answer = (
                        "⚠️ **مفتاح الـ API غير صالح**\n\n"
                        "مفتاح Gemini API المُستخدم غير صالح أو منتهي الصلاحية. يرجى التحقق من المفتاح في الإعدادات."
                    )
                else:
                    raw_answer = (
                        "⚠️ **Invalid API Key**\n\n"
                        "The Gemini API key is invalid or expired. Please verify your API key in the settings."
                    )
            else:
                # Generic fallback — still user-friendly, no raw error dump
                if is_arabic:
                    raw_answer = (
                        "⚠️ **حدث خطأ مؤقت**\n\n"
                        "حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى. إذا استمرت المشكلة، تحقق من إعدادات التطبيق."
                    )
                else:
                    raw_answer = (
                        "⚠️ **Temporary Error**\n\n"
                        "An error occurred while processing your question. Please try again. If the issue persists, check your application settings."
                    )

        # Clean answer to guarantee Requirement 11 (No visible citations/sources in output)
        final_clean_answer = clean_answer_output(raw_answer)

        # Internal citation records kept purely for logging/analytics/diagnostics
        internal_citations = []
        seen = set()
        for chunk in chunks:
            cite_key = (chunk.source, chunk.page)
            if cite_key not in seen:
                seen.add(cite_key)
                internal_citations.append(
                    {
                        "source": chunk.source,
                        "page": chunk.page,
                        "similarity": chunk.similarity,
                        "snippet": chunk.content[:150],
                    }
                )

        return {
            "question": question,
            "answer": final_clean_answer,
            "mode": mode,
            "citations": internal_citations,
            "retrieved_chunks": [c.__dict__ for c in chunks],
            "grounded": len(chunks) > 0,
            "error": error_msg,
        }

    # ─────────────────────────────────────────────────────────
    # DOCUMENT INTELLIGENCE ACTIONS
    # ─────────────────────────────────────────────────────────

    def _retrieve_for_action(self, source_name: str, query: str, top_k: int = 8) -> List[RetrievedChunk]:
        """Robust multi-pass retriever for document intelligence actions."""
        # 1. Action-specific query
        chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=0.10,
            source_filter=source_name,
        )
        if chunks:
            return chunks

        # 2. Broad fallback query with lower threshold
        chunks = self.retriever.retrieve(
            query="summary overview content details introduction",
            top_k=top_k,
            similarity_threshold=0.05,
            source_filter=source_name,
        )
        if chunks:
            return chunks

        # 3. Direct document chunk retrieval directly from vector store
        raw_chunks = self.retriever.vector_store.get_chunks_by_source(source_name, limit=top_k)
        if raw_chunks:
            return [
                RetrievedChunk(
                    content=c["content"],
                    source=c["metadata"].get("source", source_name),
                    page=c["metadata"].get("page", 1),
                    chunk_index=c["metadata"].get("chunk_index", 0),
                    similarity=1.0,
                )
                for c in raw_chunks
            ]
        return []

    def summarize_document(self, source_name: str) -> str:
        """Generates an executive, structured summary of a specific document."""
        chunks = self._retrieve_for_action(
            source_name=source_name,
            query="summary overview main points key topics executive summary",
            top_k=8,
        )
        if not chunks:
            return f"No content found for document '{source_name}' to summarize."

        context = self.retriever.format_context_for_prompt(chunks)
        prompt = (
            f"You are an expert document summarizer. Provide a comprehensive, professional executive summary "
            f"of the following document content.\n\n"
            f"Structure the summary with:\n"
            f"1. Executive Overview (2-3 sentences)\n"
            f"2. Core Topics & Key Takeaways (bullet points)\n"
            f"3. Important Methodologies / Findings / Highlights\n"
            f"4. Conclusion\n\n"
            f"CRITICAL RULE: Do NOT include any citations, source tags like [Source: ...], page numbers, or references.\n\n"
            f"DOCUMENT CONTENT:\n{context}\n\n"
            f"EXECUTIVE SUMMARY:"
        )
        res = self._call_llm(prompt)
        return clean_answer_output(res)

    def compare_documents(self, source_a: str, source_b: str) -> str:
        """Compares two uploaded documents, highlighting similarities, differences, and complementary insights."""
        chunks_a = self._retrieve_for_action(
            source_name=source_a,
            query="key topics overview main concepts",
            top_k=5,
        )
        chunks_b = self._retrieve_for_action(
            source_name=source_b,
            query="key topics overview main concepts",
            top_k=5,
        )

        ctx_a = self.retriever.format_context_for_prompt(chunks_a)
        ctx_b = self.retriever.format_context_for_prompt(chunks_b)

        prompt = (
            f"Compare and contrast the content and focus of the following two documents.\n\n"
            f"Structure your comparison clearly:\n"
            f"1. Overview of Document 1 and Document 2\n"
            f"2. Common Themes & Similarities\n"
            f"3. Key Differences & Distinctions\n"
            f"4. How they complement each other\n\n"
            f"CRITICAL RULE: Do NOT include any citations, source tags, page numbers, or references.\n\n"
            f"DOCUMENT 1 CONTENT:\n{ctx_a}\n\n"
            f"DOCUMENT 2 CONTENT:\n{ctx_b}\n\n"
            f"COMPARATIVE ANALYSIS:"
        )
        res = self._call_llm(prompt)
        return clean_answer_output(res)

    def generate_study_notes(self, source_name: str) -> str:
        """Extracts high-yield study notes, definitions, and key takeaways."""
        chunks = self._retrieve_for_action(
            source_name=source_name,
            query="definitions concepts principles formulas methodologies",
            top_k=8,
        )
        if not chunks:
            return f"No content found for document '{source_name}' to create study notes."

        context = self.retriever.format_context_for_prompt(chunks)
        prompt = (
            f"Create an organized, high-yield Study Guide based on the following document content.\n\n"
            f"Structure:\n"
            f"1. Core Concepts & Formal Definitions (Bold terms with clear explanations)\n"
            f"2. Important Rules, Principles & Workflows\n"
            f"3. Key Highlights & Review Points (Bullet list)\n\n"
            f"CRITICAL RULE: Do NOT include any citations, source tags, page numbers, or references.\n\n"
            f"CONTENT:\n{context}\n\n"
            f"STUDY GUIDE:"
        )
        res = self._call_llm(prompt)
        return clean_answer_output(res)

    def generate_qa(self, source_name: str) -> str:
        """Generates practice Q&A questions with detailed answers based on document content."""
        chunks = self._retrieve_for_action(
            source_name=source_name,
            query="important questions key points concepts definitions",
            top_k=6,
        )
        if not chunks:
            return f"No content found for document '{source_name}' to generate questions."

        context = self.retriever.format_context_for_prompt(chunks)
        prompt = (
            f"Generate 5 practice questions and comprehensive model answers based on the following document content.\n\n"
            f"For each question:\n"
            f"- Question: [Clear question testing understanding]\n"
            f"- Model Answer: [Detailed, structured explanation]\n\n"
            f"CRITICAL RULE: Do NOT include any citations, source tags, page numbers, or references.\n\n"
            f"CONTENT:\n{context}\n\n"
            f"PRACTICE Q&A:"
        )
        res = self._call_llm(prompt)
        return clean_answer_output(res)

    def extract_key_concepts(self, source_name: str) -> str:
        """Extracts a glossary of key concepts, technical terms, and definitions."""
        chunks = self._retrieve_for_action(
            source_name=source_name,
            query="terminology definitions glossary concepts technical terms",
            top_k=6,
        )
        if not chunks:
            return f"No content found for document '{source_name}' to extract concepts."

        context = self.retriever.format_context_for_prompt(chunks)
        prompt = (
            f"Extract a comprehensive Glossary of Key Terms and Technical Concepts from the following document content.\n\n"
            f"Format each term as:\n"
            f"- **[Term Name]**: [Precise definition and its practical context]\n\n"
            f"CRITICAL RULE: Do NOT include any citations, source tags, page numbers, or references.\n\n"
            f"CONTENT:\n{context}\n\n"
            f"KEY CONCEPTS GLOSSARY:"
        )
        res = self._call_llm(prompt)
        return clean_answer_output(res)

    def explain_document(self, source_name: str) -> str:
        """Explains the document's purpose, background, core narrative, and practical significance."""
        chunks = self._retrieve_for_action(
            source_name=source_name,
            query="overview introduction methodology conclusion findings purpose",
            top_k=8,
        )
        if not chunks:
            return f"No content found for document '{source_name}' to explain."

        context = self.retriever.format_context_for_prompt(chunks)
        prompt = (
            f"Provide a clear, engaging, step-by-step explanation of the following document content.\n\n"
            f"Structure:\n"
            f"1. **Core Purpose & Big Picture**: What problem does this document address?\n"
            f"2. **Step-by-Step Breakdown**: Walk through the principal sections, methods, or arguments in logical sequence.\n"
            f"3. **Practical Implications**: Why this matters and how it applies in the real world.\n\n"
            f"CRITICAL RULE: Do NOT include any citations, source tags, page numbers, or references.\n\n"
            f"CONTENT:\n{context}\n\n"
            f"DOCUMENT EXPLANATION:"
        )
        res = self._call_llm(prompt)
        return clean_answer_output(res)

    def extract_key_points(self, source_name: str) -> str:
        """Extracts the critical bullet points and key findings from the document."""
        chunks = self._retrieve_for_action(
            source_name=source_name,
            query="key points findings highlights results takeaways conclusions",
            top_k=8,
        )
        if not chunks:
            return f"No content found for document '{source_name}' to extract key points."

        context = self.retriever.format_context_for_prompt(chunks)
        prompt = (
            f"Extract the most important Key Takeaways & Highlights from the following document content.\n\n"
            f"Structure as an organized list of high-impact bullet points:\n"
            f"- Group points by theme or section if applicable.\n"
            f"- Bold critical metrics, milestones, or findings.\n\n"
            f"CRITICAL RULE: Do NOT include any citations, source tags, page numbers, or references.\n\n"
            f"CONTENT:\n{context}\n\n"
            f"KEY POINTS & HIGHLIGHTS:"
        )
        res = self._call_llm(prompt)
        return clean_answer_output(res)

    def explain_difficult_concepts(self, source_name: str) -> str:
        """Simplifies complex technical concepts found in the document using intuitive analogies."""
        chunks = self._retrieve_for_action(
            source_name=source_name,
            query="complex technical concepts algorithms equations architecture theory",
            top_k=7,
        )
        if not chunks:
            return f"No content found for document '{source_name}' to explain concepts."

        context = self.retriever.format_context_for_prompt(chunks)
        prompt = (
            f"Identify the most complex or challenging concepts within the following document content, and explain each one using intuitive language, real-world analogies, and clear step-by-step logic.\n\n"
            f"For each complex concept:\n"
            f"### [Concept Name]\n"
            f"- **What it is in plain English**: [Simple explanation]\n"
            f"- **Intuitive Analogy**: [Relatable metaphor or real-world comparison]\n"
            f"- **Technical Nuance**: [Why it matters in practice]\n\n"
            f"CRITICAL RULE: Do NOT include any citations, source tags, page numbers, or references.\n\n"
            f"CONTENT:\n{context}\n\n"
            f"EXPLAINING DIFFICULT CONCEPTS:"
        )
        res = self._call_llm(prompt)
        return clean_answer_output(res)

    def create_flashcards(self, source_name: str) -> str:
        """Generates active recall flashcards (Front / Back) from the document content."""
        chunks = self._retrieve_for_action(
            source_name=source_name,
            query="definitions principles facts questions answers flashcards key terms",
            top_k=7,
        )
        if not chunks:
            return f"No content found for document '{source_name}' to create flashcards."

        context = self.retriever.format_context_for_prompt(chunks)
        prompt = (
            f"Create a set of 6-8 Active Recall Study Flashcards based on the document content below.\n\n"
            f"Format each flashcard clearly as:\n"
            f"#### Flashcard [Number]\n"
            f"- **Front (Prompt / Question)**: [Concise question or concept test]\n"
            f"- **Back (Answer / Explanation)**: [Direct, crisp, authoritative answer]\n\n"
            f"CRITICAL RULE: Do NOT include any citations, source tags, page numbers, or references.\n\n"
            f"CONTENT:\n{context}\n\n"
            f"STUDY FLASHCARDS:"
        )
        res = self._call_llm(prompt)
        return clean_answer_output(res)

    def generate_quiz(self, source_name: str) -> str:
        """Generates a self-assessment quiz with multiple-choice questions, answer keys, and rationales."""
        chunks = self._retrieve_for_action(
            source_name=source_name,
            query="test questions quiz multiple choice assessment review problems",
            top_k=7,
        )
        if not chunks:
            return f"No content found for document '{source_name}' to generate quiz."

        context = self.retriever.format_context_for_prompt(chunks)
        prompt = (
            f"Generate a 5-question Self-Assessment Multiple-Choice Quiz based on the document content below.\n\n"
            f"For each question:\n"
            f"**Question [Number]**: [Realistic question testing comprehension]\n"
            f"A) [Option A]\n"
            f"B) [Option B]\n"
            f"C) [Option C]\n"
            f"D) [Option D]\n\n"
            f"> **Correct Answer**: [Letter & Explanation of why it is correct]\n\n"
            f"CRITICAL RULE: Do NOT include any citations, source tags, page numbers, or references.\n\n"
            f"CONTENT:\n{context}\n\n"
            f"SELF-ASSESSMENT QUIZ:"
        )
        res = self._call_llm(prompt)
        return clean_answer_output(res)
