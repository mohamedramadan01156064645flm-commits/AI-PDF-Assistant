"""Vector database interface using ChromaDB for persistent storage and retrieval
of document chunks and metadata.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.text_splitter import TextChunk
from src.embeddings import EmbeddingManager
from config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME


class ChromaVectorStore:
    """Manages storage and retrieval of embedded chunks using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = COLLECTION_NAME,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager or EmbeddingManager()

        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AI PDF Assistant Document Store"},
        )

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """
        Computes embeddings and adds chunks to ChromaDB with metadata.
        Returns the count of added chunks.
        """
        if not chunks:
            return 0

        documents = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [
            c.metadata.get("global_chunk_id") or f"chunk_{i}"
            for i, c in enumerate(chunks)
        ]

        # Generate vectors
        embeddings = self.embedding_manager.embed_documents(documents)

        # Batch upsert to ChromaDB
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = i + batch_size
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

        return len(chunks)

    def query_similarity(
        self,
        query: str,
        n_results: int = 4,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against the collection.
        Returns matches with distance and similarity score.
        """
        if self.count() == 0:
            return []

        query_embedding = self.embedding_manager.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                # Chroma's cosine distance = 1 - cosine_similarity.
                # Convert to similarity score between 0 and 1
                similarity = max(0.0, 1.0 - (dist / 2.0)) if dist is not None else 1.0

                formatted_results.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "similarity": round(similarity, 4),
                    }
                )

        return formatted_results

    def get_indexed_sources(self) -> List[str]:
        """Returns unique list of document source names indexed in the collection."""
        if self.count() == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        sources = set()
        if data and data.get("metadatas"):
            for m in data["metadatas"]:
                if m and "source" in m:
                    sources.add(m["source"])
        return sorted(list(sources))

    def delete_source(self, source_name: str) -> None:
        """Deletes all chunks belonging to a specific file."""
        self.collection.delete(where={"source": source_name})

    def count(self) -> int:
        """Returns total number of chunks stored."""
        return self.collection.count()

    def reset(self) -> None:
        """Clears all entries from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AI PDF Assistant Document Store"},
        )
