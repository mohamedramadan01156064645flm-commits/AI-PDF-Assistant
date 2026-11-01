"""Retriever module providing semantic search, similarity threshold filtering,
metadata constraints, and formatted context construction for LLM grounding.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from src.vector_store import ChromaVectorStore
from config import DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD


@dataclass
class RetrievedChunk:
    """Structured representation of a retrieved context chunk."""
    content: str
    source: str
    page: int
    total_pages: int
    similarity: float
    chunk_index: int
    metadata: Dict[str, Any]

    def citation_str(self) -> str:
        """Human-readable citation string."""
        return f"{self.source} (Page {self.page})"


class PDFRetriever:
    """High-level semantic search retriever over PDF vector store."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        default_top_k: int = DEFAULT_TOP_K,
        default_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieves top relevant chunks for a user query.

        Args:
            query: The user's question or search phrase.
            top_k: Max chunks to retrieve.
            similarity_threshold: Minimum similarity score required.
            source_filter: Optional filename to restrict search to a single PDF.

        Returns:
            List of filtered and ranked RetrievedChunk objects.
        """
        k = top_k if top_k is not None else self.default_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.default_threshold
        )

        where_clause = {"source": source_filter} if source_filter else None

        raw_results = self.vector_store.query_similarity(
            query=query,
            n_results=k,
            where_filter=where_clause,
        )

        retrieved: List[RetrievedChunk] = []

        for item in raw_results:
            sim = item.get("similarity", 0.0)
            if sim < threshold:
                continue

            meta = item.get("metadata", {})
            retrieved.append(
                RetrievedChunk(
                    content=item.get("content", ""),
                    source=meta.get("source", "Unknown"),
                    page=int(meta.get("page", 1)),
                    total_pages=int(meta.get("total_pages", 1)),
                    similarity=sim,
                    chunk_index=int(meta.get("chunk_index", 0)),
                    metadata=meta,
                )
            )

        return retrieved

    @staticmethod
    def format_context_for_prompt(chunks: List[RetrievedChunk]) -> str:
        """
        Formats retrieved chunks into clear, numbered context blocks with citations.
        """
        if not chunks:
            return "No relevant context found in documents."

        formatted_blocks = []
        for i, chunk in enumerate(chunks, 1):
            block = (
                f"[Document {i} | Source: {chunk.source} | Page: {chunk.page}]\n"
                f"{chunk.content.strip()}\n"
            )
            formatted_blocks.append(block)

        return "\n---\n".join(formatted_blocks)
