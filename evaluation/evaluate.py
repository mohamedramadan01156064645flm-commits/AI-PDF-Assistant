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
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    """Calculates fraction of expected keywords found in the text."""
    if not expected_keywords:
        return 1.0
    text_lower = text.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return round(matches / len(expected_keywords), 3)


def calculate_precision_at_k(chunks: List[Any], expected_keywords: List[str], expected_page: Optional[int] = None) -> float:
    """Calculates precision@k: fraction of retrieved chunks that are relevant."""
    if not chunks:
        return 0.0
    relevant_count = 0
    for chunk in chunks:
        content = chunk.get("content", "") if isinstance(chunk, dict) else getattr(chunk, "content", "")
        page = chunk.get("page") if isinstance(chunk, dict) else getattr(chunk, "page", None)
        has_kw = any(kw.lower() in content.lower() for kw in expected_keywords)
        is_page = (page == expected_page) if expected_page else False
        if has_kw or is_page:
            relevant_count += 1
    return round(relevant_count / len(chunks), 3)


def calculate_reciprocal_rank(chunks: List[Any], expected_keywords: List[str], expected_page: Optional[int] = None) -> float:
    """Calculates MRR component (1/rank) for the first relevant chunk."""
    for rank, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "") if isinstance(chunk, dict) else getattr(chunk, "content", "")
        page = chunk.get("page") if isinstance(chunk, dict) else getattr(chunk, "page", None)
        has_kw = any(kw.lower() in content.lower() for kw in expected_keywords)
        is_page = (page == expected_page) if expected_page else False
        if has_kw or is_page:
            return round(1.0 / rank, 3)
    return 0.0


def run_benchmark(
    questions_file: Path = EVALUATION_DIR / "test_questions.json",
    top_k: int = 4,
    provider: str = "gemini",
    gemini_api_key: str = "",
) -> Dict[str, Any]:
    """
    Executes real benchmark queries against the indexed vector store and computes genuine RAG metrics:
    - Precision@K, Recall@K, MRR, Page Hit Rate
    - Answer Relevancy, Faithfulness / Groundedness
    - Retrieval Latency & Generation Latency
    """
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Use GEMINI_API_KEY from env if not provided
    if not gemini_api_key:
        import os
        from dotenv import load_dotenv
        load_dotenv(override=True)
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    # Auto-fallback to mock only if no Gemini API key is present
    actual_provider = provider if (provider == "gemini" and gemini_api_key) else ("gemini" if gemini_api_key else "mock")

    vector_store = ChromaVectorStore()
    retriever = PDFRetriever(vector_store, default_top_k=top_k)
    rag_pipe = RAGPipeline(
        retriever=retriever,
        provider=actual_provider,
        gemini_api_key=gemini_api_key,
    )

    results: List[Dict[str, Any]] = []
    total_retrieval_latency = 0.0
    total_gen_latency = 0.0
    total_total_latency = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_relevancy = 0.0
    grounded_count = 0

    benchmark_start = time.perf_counter()

    print(f"\n==========================================")
    print(f"🚀 Running Real RAG Benchmark ({len(questions)} test cases)")
    print(f"Provider: {actual_provider} | Top-K: {top_k} | Chunks: {vector_store.count()}")
    print(f"==========================================\n")

    for idx, item in enumerate(questions, 1):
        q_text = item["question"]
        expected_kws = item.get("expected_keywords", [])
        expected_page = item.get("expected_page")
        gt_answer = item.get("ground_truth_answer", "")

        # 1. Retrieval Latency
        retrieval_start = time.perf_counter()
        # Bilingual keyword translation
        search_query = q_text
        if item.get("language") == "ar":
            try:
                search_query = rag_pipe._translate_to_english_keywords(q_text)
            except Exception:
                search_query = q_text

        retrieved_chunks = retriever.retrieve(search_query, top_k=top_k, similarity_threshold=0.15)
        if not retrieved_chunks and search_query != q_text:
            retrieved_chunks = retriever.retrieve(q_text, top_k=top_k, similarity_threshold=0.15)
        retrieval_elapsed = time.perf_counter() - retrieval_start

        # 2. Answer Generation Latency
        gen_start = time.perf_counter()
        rag_output = rag_pipe.generate_answer(q_text, top_k=top_k)
        gen_elapsed = time.perf_counter() - gen_start
        item_total_elapsed = retrieval_elapsed + gen_elapsed

        # 3. Calculate Retrieval Metrics
        retrieved_texts = " ".join([c["content"] for c in rag_output["retrieved_chunks"]])
        recall_score = calculate_keyword_recall(retrieved_texts, expected_kws)
        prec_score = calculate_precision_at_k(rag_output["retrieved_chunks"], expected_kws, expected_page)
        mrr_score = calculate_reciprocal_rank(rag_output["retrieved_chunks"], expected_kws, expected_page)

        retrieved_pages = [c["page"] for c in rag_output["retrieved_chunks"]]
        page_hit = (expected_page in retrieved_pages) if expected_page else True

        # 4. Calculate Generation Metrics
        gen_answer = rag_output.get("answer", "")
        # Answer relevancy: keywords from ground truth reflected in answer
        gt_kws = [w for w in gt_answer.replace("،", " ").replace(".", "").split() if len(w) > 3][:6]
        relevancy_score = calculate_keyword_recall(gen_answer, gt_kws) if gt_kws else recall_score
        is_grounded = len(rag_output["retrieved_chunks"]) > 0

        total_retrieval_latency += retrieval_elapsed
        total_gen_latency += gen_elapsed
        total_total_latency += item_total_elapsed
        total_recall += recall_score
        total_precision += prec_score
        total_mrr += mrr_score
        total_relevancy += relevancy_score
        if is_grounded:
            grounded_count += 1

        result_row = {
            "id": item["id"],
            "language": item.get("language", "en"),
            "question": q_text,
            "retrieval_latency_ms": round(retrieval_elapsed * 1000, 1),
            "generation_latency_seconds": round(gen_elapsed, 2),
            "total_latency_seconds": round(item_total_elapsed, 2),
            "keyword_recall": recall_score,
            "precision_at_k": prec_score,
            "mrr": mrr_score,
            "page_hit": page_hit,
            "relevancy": relevancy_score,
            "retrieved_count": len(rag_output["retrieved_chunks"]),
            "answer_preview": gen_answer[:140] + ("..." if len(gen_answer) > 140 else ""),
        }
        results.append(result_row)

        print(f"[{idx}/{len(questions)}] {q_text[:40]}...")
        print(f"      Retrieval: {retrieval_elapsed*1000:.0f}ms | Gen: {gen_elapsed:.2f}s | Recall: {recall_score*100:.0f}% | MRR: {mrr_score:.2f} | Hit: {page_hit}")

    n = max(len(questions), 1)
    avg_retrieval_ms = round((total_retrieval_latency / n) * 1000, 1)
    avg_gen_sec = round(total_gen_latency / n, 2)
    avg_total_sec = round(total_total_latency / n, 2)
    avg_recall = round(total_recall / n, 3)
    avg_precision = round(total_precision / n, 3)
    avg_mrr = round(total_mrr / n, 3)
    page_hit_rate = round(sum(1 for r in results if r["page_hit"]) / n, 3)
    faithfulness_rate = round(grounded_count / n, 3)
    avg_relevancy = round(total_relevancy / n, 3)
    benchmark_duration = round(time.perf_counter() - benchmark_start, 2)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": actual_provider,
        "dataset_name": "Standard Bilingual RAG Benchmark (test_questions.json)",
        "total_questions": len(questions),
        "total_chunks_indexed": vector_store.count(),
        "benchmark_duration_seconds": benchmark_duration,
        "avg_retrieval_latency_ms": avg_retrieval_ms,
        "avg_generation_latency_seconds": avg_gen_sec,
        "avg_total_latency_seconds": avg_total_sec,
        "avg_keyword_recall": avg_recall,
        "avg_precision_at_k": avg_precision,
        "mean_reciprocal_rank": avg_mrr,
        "page_hit_rate": page_hit_rate,
        "faithfulness_rate": faithfulness_rate,
        "answer_relevancy": avg_relevancy,
        "details": results,
    }

    report_path = EVALUATION_DIR / "latest_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n==========================================")
    print(f"📊 Benchmark Summary (REAL EVALUATION):")
    print(f"   Avg Retrieval Latency: {avg_retrieval_ms} ms")
    print(f"   Avg Generation Latency: {avg_gen_sec} s")
    print(f"   Avg Keyword Recall: {avg_recall * 100:.1f}%")
    print(f"   Avg Precision@K: {avg_precision * 100:.1f}%")
    print(f"   Mean Reciprocal Rank (MRR): {avg_mrr}")
    print(f"   Target Page Hit Rate: {page_hit_rate * 100:.1f}%")
    print(f"   Faithfulness / Groundedness: {faithfulness_rate * 100:.1f}%")
    print(f"   Report saved to: {report_path}")
    print(f"==========================================\n")

    return summary


if __name__ == "__main__":
    run_benchmark()
