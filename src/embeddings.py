"""Embedding module using Sentence Transformers (HuggingFace) for converting
text chunks and queries into dense numerical vectors.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DEFAULT_EMBEDDING_MODEL


class EmbeddingManager:
    """Manages text embeddings using Hugging Face Sentence Transformers."""

    _instances = {}

    def __new__(cls, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu"):
        """Singleton per model_name to avoid reloading heavy weights repeatedly."""
        key = (model_name, device)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
            instance._initialized = False
        return cls._instances[key]

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu"):
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dimension = self.model.get_embedding_dimension()
        else:
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self._initialized = True

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embeds a list of document strings into float vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalized for cosine similarity
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single search query string.
        """
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @staticmethod
    def cosine_similarity(vec_a: Union[List[float], np.ndarray], vec_b: Union[List[float], np.ndarray]) -> float:
        """Computes cosine similarity between two normalized vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
