"""
Embedding providers for TELOS.

Provides two implementations:
1. DeterministicEmbeddingProvider - Fast, deterministic (for testing)
2. SentenceTransformerProvider - Real semantic embeddings (for production)
"""

from __future__ import annotations
import hashlib
import numpy as np
from typing import Optional


class DeterministicEmbeddingProvider:
    """
    Deterministic embedding provider for testing.

    Uses hash-based embeddings that are:
    - Deterministic (same text -> same embedding)
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

    def encode(self, text: str) -> np.ndarray:
        """
        Generate deterministic embedding from text hash.

        Args:
            text: Input text

        Returns:
            Deterministic embedding vector
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

        return np.array(embedding, dtype=np.float32)


class MistralEmbeddingProvider:
    """
    Mistral embedding provider using mistral-embed API.

    Provides high-quality 1024-dimensional embeddings via Mistral's API.
    Requires MISTRAL_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Mistral API key (optional, will use env var if not provided)
        """
        import os
        from mistralai import Mistral

        self.api_key = api_key or os.environ.get('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment or parameters")

        self.client = Mistral(api_key=self.api_key)
        self.dimension = 1024  # mistral-embed produces 1024-dimensional vectors
        self.model_name = "mistral-embed"

    def encode(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding using Mistral API.

        Args:
            text: Input text

        Returns:
            Semantic embedding vector (1024 dimensions)
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=[text]
            )
            # Extract embedding from response
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Mistral embedding API error: {e}")

    def batch_encode(self, texts: list) -> list:
        """
        Generate semantic embeddings for multiple texts in a single API call.

        Args:
            texts: List of input texts

        Returns:
            List of semantic embedding vectors (1024 dimensions each)
        """
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=texts
            )
            embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Mistral batch embedding API error: {e}")


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
        import logging

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name

        try:
            self.model = SentenceTransformer(model_name)
            logging.info(f"SentenceTransformer loaded successfully: {model_name}")
        except Exception as e:
            logging.warning(f"Primary load failed: {e}, trying alternative approach")
            try:
                import torch
                with torch.inference_mode():
                    self.model = SentenceTransformer(model_name, device='cpu')
            except Exception as e2:
                logging.error(f"All SentenceTransformer loading approaches failed: {e2}")
                raise RuntimeError(f"Failed to load SentenceTransformer model: {e2}")

        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding for text.

        Args:
            text: Input text

        Returns:
            Semantic embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)


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


# =============================================================================
# FIDELITY RESCALING FOR SENTENCETRANSFORMER
# =============================================================================

def rescale_sentence_transformer_fidelity(raw_score: float) -> float:
    """
    Rescale SentenceTransformer cosine similarity to TELOS fidelity range.

    Formula: fidelity = 0.25 + raw_score * 1.8, clamped to [0.0, 1.0]

    This maps:
        raw 0.00 -> 0.25 (RED - clearly off-topic)
        raw 0.15 -> 0.52 (ORANGE - borderline)
        raw 0.25 -> 0.70 (GREEN - on-topic threshold)
        raw 0.35 -> 0.88 (GREEN - strongly aligned)
        raw 0.50 -> 1.00 (GREEN - perfect alignment)

    Args:
        raw_score: Raw cosine similarity from SentenceTransformer (-1.0 to 1.0)

    Returns:
        Rescaled fidelity score for TELOS (0.0 to 1.0)
    """
    rescaled = 0.25 + raw_score * 1.8
    return max(0.0, min(1.0, rescaled))


# =============================================================================
# CACHED PROVIDER SINGLETONS
# =============================================================================

_cached_minilm_provider = None


def get_cached_minilm_provider() -> SentenceTransformerProvider:
    """
    Return a cached singleton SentenceTransformerProvider (all-MiniLM-L6-v2).

    Avoids reloading the model on every call. The provider exposes an
    ``encode(text) -> np.ndarray`` method suitable for use as ``embed_fn``
    throughout the governance pipeline.

    Returns:
        Cached SentenceTransformerProvider instance
    """
    global _cached_minilm_provider
    if _cached_minilm_provider is None:
        _cached_minilm_provider = SentenceTransformerProvider(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _cached_minilm_provider
