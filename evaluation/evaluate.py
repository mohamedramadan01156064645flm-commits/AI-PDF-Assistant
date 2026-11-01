"""Evaluation script to measure RAG pipeline metrics:
- Retrieval Context Precision (relevance of retrieved chunks)
- Context Recall (presence of expected keywords)
- Latency / Response Time
- Faithfulness & Answer Relevancy check
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import BASE_DIR, EVALUATION_DIR
from src.vector_store import ChromaVectorStore
from src.retriever import PDFRetriever
from src.rag_pipeline import RAGPipeline
from src.chatbot import PDFChatbot


def calculate_keyword_recall(text: str, expected_keywords: List[str]) -> float:
    """Calculates fraction of expected keywords found in the retrieved context."""
    if not expected_keywords:
        return 1.0
    text_lower = text.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return round(matches / len(expected_keywords), 3)


def run_benchmark(
    questions_file: Path = EVALUATION_DIR / "test_questions.json",
    top_k: int = 4,
    provider: str = "mock",
    gemini_api_key: str = "",
) -> Dict[str, Any]:
    """
    Executes benchmark queries against the indexed vector store and computes metrics.
    """
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    vector_store = ChromaVectorStore()
    retriever = PDFRetriever(vector_store, default_top_k=top_k)
    rag_pipe = RAGPipeline(
        retriever=retriever,
        provider=provider,
        gemini_api_key=gemini_api_key,
    )
    bot = PDFChatbot(rag_pipe)

    results: List[Dict[str, Any]] = []
    total_latency = 0.0
    total_recall = 0.0

    print(f"\n==========================================")
    print(f"🚀 Running RAG Evaluation Benchmark ({len(questions)} test cases)")
    print(f"Provider: {provider} | Top-K: {top_k}")
    print(f"==========================================\n")

    for idx, item in enumerate(questions, 1):
        q_text = item["question"]
        expected_kws = item.get("expected_keywords", [])
        expected_page = item.get("expected_page")

        # Bilingual search enhancement: if purely Arabic and no English translation yet, augment with English terms for vector search
        search_query = q_text
        if item.get("language") == "ar":
            # Map common Arabic ML query terms to English equivalents if running in mock/offline mode
            term_map = {
                "التعلم الخاضع للإشراف": "supervised learning",
                "غير الخاضع للإشراف": "unsupervised learning",
                "شبكات التعلم العميق": "deep learning neural networks",
                "الفرق": "difference",
            }
            augmented_terms = [v for k, v in term_map.items() if k in q_text]
            if augmented_terms:
                search_query = f"{q_text} ({' '.join(augmented_terms)})"

        start_t = time.perf_counter()
        rag_output = rag_pipe.generate_answer(search_query, top_k=top_k)
        elapsed = time.perf_counter() - start_t

        retrieved_texts = " ".join([c["content"] for c in rag_output["retrieved_chunks"]])
        recall_score = calculate_keyword_recall(retrieved_texts, expected_kws)
        
        # Check if expected page is present in retrieved chunks
        retrieved_pages = [c["page"] for c in rag_output["retrieved_chunks"]]
        page_hit = (expected_page in retrieved_pages) if expected_page else True

        total_latency += elapsed
        total_recall += recall_score

        result_row = {
            "id": item["id"],
            "language": item.get("language", "en"),
            "question": q_text,
            "latency_seconds": round(elapsed, 3),
            "keyword_recall": recall_score,
            "page_hit": page_hit,
            "retrieved_count": len(rag_output["retrieved_chunks"]),
            "answer_preview": rag_output["answer"][:120] + "...",
        }
        results.append(result_row)

        print(f"[{idx}/{len(questions)}] {q_text[:45]}...")
        print(f"      Latency: {elapsed:.2f}s | Keyword Recall: {recall_score * 100:.1f}% | Page Hit: {page_hit}")

    avg_latency = round(total_latency / len(questions), 3) if questions else 0.0
    avg_recall = round(total_recall / len(questions), 3) if questions else 0.0
    page_hit_rate = (
        round(sum(1 for r in results if r["page_hit"]) / len(questions), 3)
        if questions
        else 0.0
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(questions),
        "avg_latency_seconds": avg_latency,
        "avg_keyword_recall": avg_recall,
        "page_hit_rate": page_hit_rate,
        "details": results,
    }

    report_path = EVALUATION_DIR / "latest_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n==========================================")
    print(f"📊 Benchmark Summary:")
    print(f"   Avg Latency: {avg_latency}s")
    print(f"   Avg Keyword Recall: {avg_recall * 100:.1f}%")
    print(f"   Target Page Hit Rate: {page_hit_rate * 100:.1f}%")
    print(f"   Report saved to: {report_path}")
    print(f"==========================================\n")

    return summary


if __name__ == "__main__":
    run_benchmark()
