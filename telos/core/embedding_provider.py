"""
Embedding providers for TELOS.

Provides two implementations:
1. DeterministicEmbeddingProvider - Fast, deterministic (for testing)
2. SentenceTransformerProvider - Real semantic embeddings (for production)
"""

from __future__ import annotations
import hashlib
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DeterministicEmbeddingProvider:
    """
    Deterministic embedding provider for testing.

    Uses hash-based embeddings that are:
    - Deterministic (same text → same embedding)
    - Fast (no model loading)
    - Suitable for unit tests and rapid iteration

    NOT suitable for production (no semantic meaning).
    """

    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.expected_dim = dimension

    def _is_valid_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding contains no NaN/Inf and has correct shape.

        Args:
            embedding: Vector to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if embedding is None or embedding.size == 0:
            return False

        # Check for NaN or Inf values
        if not np.all(np.isfinite(embedding)):
            return False

        # Check dimension matches expected
        if embedding.shape[0] != self.expected_dim:
            return False

        return True

    def encode(self, text: str) -> np.ndarray:
        """
        Generate deterministic embedding from text hash with validation.

        Args:
            text: Input text

        Returns:
            Deterministic embedding vector

        Raises:
            ValueError: If generated embedding is invalid
        """
        # Create hash of text
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()

        # Expand hash to desired dimension
        embedding = []
        for i in range(self.dimension):
            byte_index = i % len(hash_bytes)
            value = hash_bytes[byte_index] / 255.0  # Normalize to [0, 1]
            embedding.append(value)

        embedding = np.array(embedding, dtype=np.float32)

        # Validate embedding before returning
        if not self._is_valid_embedding(embedding):
            raise ValueError(
                f"Invalid embedding generated: shape={embedding.shape}, "
                f"has_nan={np.any(np.isnan(embedding))}, "
                f"has_inf={np.any(np.isinf(embedding))}"
            )

        return embedding

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate batch embeddings with validation.

        Args:
            texts: List of input texts

        Returns:
            List of validated embedding vectors

        Raises:
            ValueError: If any embedding is invalid
        """
        embeddings = []
        for i, text in enumerate(texts):
            try:
                emb = self.encode(text)
                embeddings.append(emb)
            except ValueError as e:
                raise ValueError(f"Invalid embedding at index {i}: {e}")

        return embeddings


class SentenceTransformerProvider:
    """
    Production embedding provider using sentence-transformers.

    Provides semantic embeddings suitable for:
    - Real governance applications
    - Semantic similarity comparisons
    - Production deployments
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace model identifier
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.expected_dim = self.dimension

    def _is_valid_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding contains no NaN/Inf and has correct shape.

        Args:
            embedding: Vector to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if embedding is None or embedding.size == 0:
            return False

        # Check for NaN or Inf values
        if not np.all(np.isfinite(embedding)):
            return False

        # Check dimension matches expected
        if embedding.shape[0] != self.expected_dim:
            return False

        return True

    def encode(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding for text with comprehensive validation.

        Args:
            text: Input text

        Returns:
            Semantic embedding vector

        Raises:
            ValueError: If embedding is invalid (NaN/Inf/wrong dimension)
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            embedding = embedding.astype(np.float32)

            # Validate embedding before returning
            if not self._is_valid_embedding(embedding):
                raise ValueError(
                    f"Invalid embedding received: shape={embedding.shape}, "
                    f"has_nan={np.any(np.isnan(embedding))}, "
                    f"has_inf={np.any(np.isinf(embedding))}"
                )

            return embedding

        except Exception as e:
            logger.error(f"Embedding API failed for text: '{text[:50]}...': {e}")
            raise  # Re-raise for upstream handling

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate batch embeddings with validation.

        Args:
            texts: List of input texts

        Returns:
            List of validated embedding vectors

        Raises:
            ValueError: If any embedding is invalid
        """
        try:
            # Use batch encoding for efficiency
            embeddings = self.model.encode(texts, convert_to_numpy=True)

            # Convert to list of arrays with proper dtype
            embeddings = [emb.astype(np.float32) for emb in embeddings]

            # Validate all embeddings
            for i, emb in enumerate(embeddings):
                if not self._is_valid_embedding(emb):
                    raise ValueError(
                        f"Invalid embedding at index {i}: shape={emb.shape}, "
                        f"has_nan={np.any(np.isnan(emb))}, "
                        f"has_inf={np.any(np.isinf(emb))}"
                    )

            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding API failed: {e}")
            raise  # Re-raise for upstream handling


def EmbeddingProvider(deterministic: bool = False, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Factory function to create appropriate embedding provider.

    Args:
        deterministic: If True, use fast deterministic embeddings (testing).
                      If False, use real semantic embeddings (production).
        model_name: Model name for SentenceTransformer (ignored if deterministic=True)

    Returns:
        Embedding provider instance
    """
    if deterministic:
        return DeterministicEmbeddingProvider()
    else:
        return SentenceTransformerProvider(model_name=model_name)
